// src/kernels/cuda/deepseek4_attn.cu - P3: attention primitives on the GPU.
//
// RoPE is the reference's kernel geometry exactly (cuda/ds41_attn.cuh:65-77): one 64-thread block per
// (head, token), thread i < n_rot/2 owning one complex pair in the last n_rot values of the head.  The
// frequency comes from the header's shared inline function, so the host oracle and this kernel compute the
// identical f32 angle; the two rotated values are bf16-rounded once, matching the reference's write-back into
// its bf16 tensor.
//
// The quantized-KV group keeps the reference's geometry (cuda/ds41_attn.cuh:86-284): one 256-thread block per
// pooled group, one 32-thread warp per quant block for act_quant and the packs, and a flat grid-stride pass
// for the unpack direction.  The encode conversions are the header's shared inline functions, so a packed
// byte is the same byte the host oracle writes.
//
// Sparse attention is the reference's scalar kernel (cuda/ds41_attn.cuh:304-370) with its geometry intact: one
// block per (query, 8-head group), four warps x two heads, keys staged through shared eight at a time, the
// 32-lane xor dot, the bf16-rounded probability (acc_s_cast), invalid slots zeroed, and the sink in the
// denominator only.
//
// Attention's remaining primitives (the window ring and the indexer) join this file as they land; each keeps the
// same entry convention: device pointers, an optional stream (null = default stream, the call synchronizes), bad
// shapes refused before any launch, false on a launch error.
#include "strata/kernels/deepseek4_attn.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace strata::kernels {

namespace {

using strata::deepseek4::engram_bf16r;

__global__ void attn_rope_kernel(float* x, const int32_t* pos, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                                 float theta, uint32_t original_seq_len, float factor, float beta_fast,
                                 float beta_slow, int inverse) {
    const uint32_t t = blockIdx.y, h = blockIdx.x, i = threadIdx.x;
    if (i >= n_rot / 2u) return;
    float* xr = x + ((uint64_t) t * n_head + h) * head_dim + (head_dim - n_rot) + 2u * i;
    const float ang = (float) pos[t] * deepseek4_rope_freq(i, n_rot, theta, original_seq_len, factor, beta_fast,
                                                           beta_slow);
    float c = cosf(ang), s = sinf(ang);
    if (inverse) s = -s;
    const float a = xr[0], b = xr[1];
    xr[0] = engram_bf16r(a * c - b * s);
    xr[1] = engram_bf16r(a * s + b * c);
}

__global__ void attn_compress_pool_kernel(float* out, const float* kv, const float* sc, uint32_t ratio,
                                          uint32_t dim) {
    const uint32_t g = blockIdx.x;
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float mx = -INFINITY;
        for (uint32_t t = 0; t < ratio; t++) mx = fmaxf(mx, sc[((uint64_t) g * ratio + t) * dim + d]);
        float den = 0.0f, acc = 0.0f;
        for (uint32_t t = 0; t < ratio; t++) {
            const float e = expf(sc[((uint64_t) g * ratio + t) * dim + d] - mx);
            den += e;
            acc += e * kv[((uint64_t) g * ratio + t) * dim + d];
        }
        out[(uint64_t) g * dim + d] = engram_bf16r(acc / den);
    }
}

__global__ void attn_act_quant_kernel(float* x, uint32_t dim, uint32_t block, int mode, uint32_t nb_row) {
    const uint32_t row = blockIdx.x / nb_row, b = blockIdx.x % nb_row;
    const uint32_t lane = threadIdx.x;
    float* p = x + (uint64_t) row * dim + (uint64_t) b * block;
    const float v = lane < block ? p[lane] : 0.0f;
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float s = deepseek4_quant_scale(a, mode);
    if (lane < block) p[lane] = deepseek4_quant_value(v, s, mode);
}

__global__ void attn_kv_pack_kernel(uint8_t* cache, const float* rows, uint32_t g0, uint32_t dim, uint32_t blk,
                                    uint32_t row_bytes, uint32_t nib_bytes, uint32_t nb_row, int mode) {
    const uint32_t r = blockIdx.x / nb_row, b = blockIdx.x % nb_row, lane = threadIdx.x;
    const float* p = rows + (uint64_t) r * dim + (uint64_t) b * blk;
    const float v = lane < blk ? p[lane] : 0.0f;
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float s = deepseek4_quant_scale(a, mode);
    uint8_t* row = cache + (uint64_t)(g0 + r) * row_bytes;
    if (lane == 0) {
        row[nib_bytes + b] = mode == kDeepseek4QuantFp4E4m3 ? deepseek4_e4m3_byte(s) : deepseek4_e8m0_byte(s);
        if (b == 0)
            for (uint32_t k = nib_bytes + nb_row; k < row_bytes; k++) row[k] = 0;   // index key's pad
    }
    const uint8_t nib = lane < blk ? deepseek4_fp4_to_nibble(v / s) : 0u;
    const uint8_t odd = (uint8_t) __shfl_down_sync(0xffffffffu, (uint32_t) nib, 1);
    if (lane < blk && (lane & 1u) == 0u) row[(b * blk + lane) >> 1] = (uint8_t)(nib | (odd << 4));
}

__global__ void attn_kv_unpack_kernel(float* out, const uint8_t* cache, uint64_t n, int idxk) {
    const uint32_t dim = idxk ? kDeepseek4IdxkDim : kDeepseek4CkvDim;
    const uint32_t row_bytes = idxk ? kDeepseek4IdxkBytes : kDeepseek4CkvBytes;
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t) gridDim.x * blockDim.x) {
        const uint8_t* row = cache + (i / dim) * row_bytes;
        out[i] = idxk ? deepseek4_idxk_get(row, (uint32_t)(i % dim)) : deepseek4_ckv_get(row, (uint32_t)(i % dim));
    }
}

// The compressed row's scale decoded once into shared (cuda/ds41_attn.cuh:247-251): same value, same multiply as
// deepseek4_ckv_get's per-element decode, so the kernel and the host oracle stay bit-identical.
__device__ __forceinline__ float attn_ckv_get_s(const uint8_t* row, uint32_t d, const float* scales) {
    const uint8_t by = row[d >> 1];
    const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
    return engram_bf16r(deepseek4_fp4_from_nibble(nib) * scales[d >> 4]);
}

// Sparse attention over the causal SWA window plus the topk compressed keys (cuda/ds41_attn.cuh:304-370).
//
// PER is the lane's share of the head (hd/32) as a compile-time constant, or 0 for the runtime fallback.  The
// loops over it must unroll: with a runtime bound the arrays below (qa, acc, acc_add, s) are dynamically
// indexed and ptxas puts every one of them in local memory — the first build answered with REG:64 and a
// 336-byte stack frame, and each key's dot paid local-memory round trips instead of register reads.  Only
// hd 64 (PER 2) and hd 512 (PER 16, this model) are instantiated; other head dims keep the dynamic kernel.
template <uint32_t PER>
__global__ __launch_bounds__(kDeepseek4AttnHeadsPerBlock * 16u, 4) void attn_sparse_kernel(
        float* o, const float* q, const float* kvw, const uint8_t* kvc, const int32_t* idx, const float* sink,
        uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head, uint32_t hd, float scale,
        uint32_t full_block, uint32_t ring, uint32_t win_lo) {
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t per = PER ? PER : hd / 32u;
    // A lane's hd/32 values sit at an odd stride, not back to back: at stride 16 every lane*16+e addresses
    // only two of the 32 shared banks, so each of the 32 loads a (key, head) pair makes replays 16 ways and the
    // kernel runs on the LSU instead of the FP pipe.  An odd stride keeps lane -> bank one-to-one; the lane's
    // values are still dims lane*per..lane*per+per-1, so the dot's reduction order (and every bit of the
    // result) is the reference's.
    const uint32_t sw = (per & 1u) ? per + 2u : per + 1u;
    __shared__ float ks[kDeepseek4AttnKeyTile][576];
    __shared__ int kok[kDeepseek4AttnKeyTile];
    __shared__ float ksc[kDeepseek4AttnKeyTile][32];
    float qa[2][16], acc[2][16], mx[2], sum[2];
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * kDeepseek4AttnHeadsPerBlock + warp * 2u + hh;
        for (uint32_t e = 0; e < per; e++) {
            qa[hh][e] = q[((uint64_t) i * n_head + h) * hd + lane * per + e];
            acc[hh][e] = 0.f;
        }
        mx[hh] = -1e30f;
        sum[hh] = 0.f;
    }
    const uint32_t p = pos0 + i;
    const uint32_t last = full_block ? pos0 + full_block - 1u : p;
    uint32_t lo = full_block ? (pos0 > window ? pos0 - window : 0u)
                             : (p + 1u > window ? p + 1u - window : 0u);
    if (lo < win_lo) lo = win_lo;
    const uint32_t nwin = last - lo + 1u;
    const uint32_t nkeys = nwin + topk;
    for (uint32_t base = 0; base < nkeys; base += kDeepseek4AttnKeyTile) {
        const uint32_t nt = (nkeys - base) < kDeepseek4AttnKeyTile ? (nkeys - base) : kDeepseek4AttnKeyTile;
        __syncthreads();   // the previous tile is read out before it is overwritten
        for (uint32_t t = threadIdx.x / 32u; t < nt; t += blockDim.x / 32u) {
            const uint32_t kk = base + t;
            const float* krow = NULL;
            const uint8_t* cpk = NULL;
            if (kk < nwin) {
                krow = kvw + (uint64_t) deepseek4_win_row((int64_t) lo + kk, pos0, window, ring) * hd;
            } else {
                const int32_t g = idx[(uint64_t) i * topk + (kk - nwin)];
                if (g >= 0 && (uint32_t) g < ng) cpk = kvc + (uint64_t) g * kDeepseek4CkvBytes;
            }
            if (lane == 0) kok[t] = (krow != NULL || cpk != NULL) ? 1 : 0;
            if (cpk) ksc[t][lane] = strata::deepseek4::engram_e4m3_to_f32(cpk[kDeepseek4CkvNibbleBytes + lane]);
            __syncwarp();
            // An invalid slot must be zeroed: 0 x stale shared bits is not 0 when those bits are NaN or Inf.
            for (uint32_t d = lane; d < hd; d += 32u)
                ks[t][(d / per) * sw + (d % per)] =
                    krow ? krow[d] : (cpk ? attn_ckv_get_s(cpk, d, ksc[t]) : 0.f);
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            float s[kDeepseek4AttnKeyTile];
            float tm = -1e30f;
#pragma unroll
            for (uint32_t t = 0; t < kDeepseek4AttnKeyTile; t++) {
                if (t >= nt) {   // the tile edge: staged slots past nt hold stale bytes, never read them
                    s[t] = -1e30f;
                    continue;
                }
                float d = 0.f;
#pragma unroll
                for (uint32_t e = 0; e < per; e++) d += qa[hh][e] * ks[t][lane * sw + e];
                for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
                s[t] = kok[t] ? d * scale : -1e30f;
                tm = fmaxf(tm, s[t]);
            }
            const float nm = fmaxf(mx[hh], tm);
            const float rs = expf(mx[hh] - nm);   // one rescale per tile
            float acc_add[16];
#pragma unroll
            for (uint32_t e = 0; e < per; e++) acc_add[e] = 0.f;
            float ps = 0.f;
#pragma unroll
            for (uint32_t t = 0; t < kDeepseek4AttnKeyTile; t++) {
                if (t >= nt) continue;
                const float pv = expf(s[t] - nm);
                ps += pv;
                const float pb = engram_bf16r(pv);   // the official acc_s_cast
#pragma unroll
                for (uint32_t e = 0; e < per; e++) acc_add[e] += pb * ks[t][lane * sw + e];
            }
            sum[hh] = sum[hh] * rs + ps;
#pragma unroll
            for (uint32_t e = 0; e < per; e++) acc[hh][e] = acc[hh][e] * rs + acc_add[e];
            mx[hh] = nm;
        }
    }
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * kDeepseek4AttnHeadsPerBlock + warp * 2u + hh;
        const float den = sum[hh] + expf(sink[h] - mx[hh]);   // the sink enters the denominator only
        for (uint32_t e = 0; e < per; e++)
            o[((uint64_t) i * n_head + h) * hd + lane * per + e] = engram_bf16r(acc[hh][e] / den);
    }
}

// The window ring commit and its spec-decode snapshot (cuda/ds41_attn.cuh:382-425).
__global__ void attn_win_commit_kernel(float* win, uint32_t pos0, uint32_t i0, uint32_t window, uint32_t hd,
                                       const int32_t* posd) {
    const uint32_t i = i0 + blockIdx.x;
    if (posd) pos0 = (uint32_t) posd[0];
    const float* src = win + (uint64_t)(window + i) * hd;
    float* dst = win + (uint64_t)((pos0 + i) % window) * hd;
    for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = src[d];
}

__global__ void attn_win_ring_snap_kernel(float* win, float* snap, uint32_t pos0, uint32_t i0, uint32_t window,
                                          uint32_t hd, uint32_t back, const int32_t* posd) {
    const uint32_t i = i0 + blockIdx.x;
    if (posd) pos0 = (uint32_t) posd[0];
    float* ring = win + (uint64_t)((pos0 + i) % window) * hd;
    float* sp = snap + (uint64_t) i * hd;
    if (back) {
        for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) ring[d] = sp[d];
    } else {
        for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) sp[d] = ring[d];
    }
}

// ---- the sparse indexer (cuda/ds41_indexer.cuh) ----
// One warp advances kDeepseek4IdxGroupsPerWarp groups at once (the reference's V41_IDX_G).  The per-(group,
// head) chain "MACs -> 5-level xor reduce -> round" is a serial dependency; the groups interleave it and each
// head's query row is read once.  The MAC order, the reduction tree, the bf16 rounding point and the head
// accumulation order all stay the reference's, so the score is bit-identical to one group per warp.
__global__ void attn_indexer_score_kernel(float* score, const float* q, const uint8_t* k, const float* w,
                                          const int32_t* cand, uint32_t pos0, uint32_t ng, uint32_t n_head,
                                          uint32_t dk, uint32_t ratio, const int32_t* posd, uint32_t cand_bs,
                                          uint32_t cand_cap) {
    constexpr uint32_t G = kDeepseek4IdxGroupsPerWarp;
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    if (posd) {   // graph: device position; the host ng is the bucket cap
        pos0 = (uint32_t) posd[0];
        ng = (pos0 + gridDim.x) / ratio;
    }
    const uint32_t vis = (pos0 + i + 1u) / ratio;   // visible groups at this absolute position
    const uint32_t per = dk / 32u;                 // dk=128 => 4 dims per lane
    // C2: with a list only the compact row [ns] is scored, item c maps to group g
    const int32_t* cl = cand ? cand + (uint64_t) i * (1u + cand_cap) : nullptr;
    const uint32_t nc = cl ? (uint32_t) cl[0] : 0u;
    const uint32_t ns = cl ? deepseek4_cand_ns(ng, cand_bs, cand_cap) : ng;
    const uint32_t gstride = nwarp * gridDim.y * G;
    for (uint32_t gb = (blockIdx.y * nwarp + warp) * G; gb < ns; gb += gstride) {
        float kv[G][4], acc[G];
        bool live[G];
#pragma unroll
        for (uint32_t j = 0; j < G; j++) {
            const uint32_t c = gb + j;
            uint32_t g = c;
            if (cl) {
                const uint32_t b = c / cand_bs;
                g = b < nc ? (uint32_t) cl[1u + b] * cand_bs + c % cand_bs : ng;   // off-list item = dead group
            }
            live[j] = c < ns && g < ng && g < vis;
            acc[j] = 0.0f;
            const uint8_t* kg = k + (uint64_t)(live[j] ? g : 0u) * kDeepseek4IdxkBytes;
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++)
                kv[j][e] = (live[j] && e < per) ? deepseek4_idxk_get(kg, lane * per + e) : 0.0f;
        }
        for (uint32_t h = 0; h < n_head; h++) {
            const float* qh = q + ((uint64_t) i * n_head + h) * dk;
            float qv[4];
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++) qv[e] = e < per ? qh[lane * per + e] : 0.0f;
            const float wh = w[(uint64_t) i * n_head + h];
            float d[G];
#pragma unroll
            for (uint32_t j = 0; j < G; j++) {
                d[j] = 0.0f;
#pragma unroll
                for (uint32_t e = 0; e < 4u; e++)
                    if (e < per) d[j] += qv[e] * kv[j][e];
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
                for (uint32_t j = 0; j < G; j++) d[j] += __shfl_xor_sync(0xffffffffu, d[j], o);
            }
#pragma unroll
            for (uint32_t j = 0; j < G; j++) {
                float dd = engram_bf16r(d[j]);   // einsum lands in bf16
                dd = fmaxf(dd, 0.0f);            // relu_
                acc[j] += engram_bf16r(dd * wh); // x weights (bf16), then sum
            }
        }
        if (lane == 0) {
#pragma unroll
            for (uint32_t j = 0; j < G; j++)
                if (gb + j < ns)
                    score[(uint64_t) i * ns + gb + j] = live[j] ? engram_bf16r(acc[j]) : -INFINITY;
        }
    }
}

// Candidate blocks: the block score is the in-block max, the block holding the newest position is pinned +inf,
// and topk_blocks blocks are kept.  kk >= nb selects every block that is not fully invisible; else a parallel
// radix select finds the kk-th key, keys above it all win and the tie group fills in ascending block order (the
// reference's strict `>` scan).  The selection compacts into an ascending list whose header is the block count.
__global__ void attn_candidate_kernel(int32_t* list, const float* score, uint32_t pos0, uint32_t ng, uint32_t ratio,
                                      uint32_t topk_blocks, uint32_t bs, const int32_t* posd, float* blk,
                                      uint32_t nb_cap) {
    const uint32_t i = blockIdx.x;
    if (posd) {   // graph: device position; the scratch is sized by the bucket cap
        pos0 = (uint32_t) posd[0];
        ng = (pos0 + gridDim.x) / ratio;
    }
    const uint32_t nb = (ng + bs - 1u) / bs;
    float* bsc = blk + (uint64_t) i * nb_cap;
    uint8_t* sel = (uint8_t*)(blk + (uint64_t) gridDim.x * nb_cap) + (uint64_t) i * nb_cap;
    __shared__ uint32_t hist[256], sh_bucket, sh_k, cnt[256];
    const uint32_t vis = (pos0 + i + 1u) / ratio;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        float m = -INFINITY;
        for (uint32_t j = b * bs; j < (b + 1u) * bs && j < ng; j++) m = fmaxf(m, score[(uint64_t) i * ng + j]);
        if (vis > 0u && b == (vis - 1u) / bs) m = INFINITY;   // pin the block holding the newest position
        bsc[b] = m;
        sel[b] = 0;
    }
    __syncthreads();
    const uint32_t kk = topk_blocks < nb ? topk_blocks : nb;
    if (kk >= nb) {   // select all except fully invisible blocks
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) sel[b] = deepseek4_topk_key(bsc[b]) != 0u;
    } else {          // parallel radix select for the threshold
        uint32_t prefix = 0, want = kk;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (uint32_t t = threadIdx.x; t < 256u; t += blockDim.x) hist[t] = 0;
            __syncthreads();
            for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
                const uint32_t key = deepseek4_topk_key(bsc[b]);
                if (key == 0u || (shift < 24 && (key >> (shift + 8)) != (prefix >> (shift + 8)))) continue;
                atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {   // accumulate from the top bucket down to the one holding the want-th key
                uint32_t acc = 0;
                int t = 255;
                for (; t > 0; t--) {
                    if (acc + hist[t] >= want) break;
                    acc += hist[t];
                }
                sh_bucket = (uint32_t) t;
                sh_k = want - acc;
            }
            __syncthreads();
            prefix |= sh_bucket << shift;
            want = sh_k;
            __syncthreads();
        }
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) sel[b] = deepseek4_topk_key(bsc[b]) > prefix;
        __syncthreads();
        if (threadIdx.x == 0) {   // keys == threshold fill in ascending block order (ties take the smaller block)
            uint32_t eq = want;
            for (uint32_t b = 0; b < nb && eq; b++)
                if (deepseek4_topk_key(bsc[b]) == prefix) {
                    sel[b] = 1;
                    eq--;
                }
        }
    }
    __syncthreads();
    // Compact the selection into an ascending list: each thread owns one contiguous block span, span counts ->
    // exclusive prefix -> spans written in order.
    const uint32_t chunk = (nb + blockDim.x - 1u) / blockDim.x;
    const uint32_t c0 = threadIdx.x * chunk, c1 = (c0 + chunk) < nb ? (c0 + chunk) : nb;
    uint32_t mine = 0;
    for (uint32_t b = c0; b < c1; b++) mine += sel[b];
    cnt[threadIdx.x] = mine;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t run = 0;
        for (uint32_t t = 0; t < blockDim.x; t++) {
            const uint32_t v = cnt[t];
            cnt[t] = run;
            run += v;
        }
        list[(uint64_t) i * (1u + topk_blocks)] = (int32_t) run;   // <= kk <= topk_blocks: the list always fits
    }
    __syncthreads();
    uint32_t wr = cnt[threadIdx.x];
    for (uint32_t b = c0; b < c1; b++)
        if (sel[b]) list[(uint64_t) i * (1u + topk_blocks) + 1u + wr++] = (int32_t) b;
}

// The topk selector: topk over min(index_topk, visible), output in position order, unreachable slots -1.
// Parallel radix select (8 bits per round, 1024 threads, reads batched) then a segmented write-out; the set,
// the order and the "ties take the smaller index" rule are the reference's serial scan's.
__device__ __forceinline__ int32_t attn_cand_map(const int32_t* cl, uint32_t c, uint32_t bs) {
    return cl ? cl[1u + c / bs] * (int32_t) bs + (int32_t)(c % bs) : (int32_t) c;
}

__global__ void __launch_bounds__(kDeepseek4TopkThreads) attn_topk_kernel(
    int32_t* idx, const float* score, uint32_t ng, uint32_t topk, uint32_t ratio, const int32_t* posd,
    const int32_t* cand, uint32_t cand_bs, uint32_t cand_cap) {
    constexpr uint32_t B = kDeepseek4TopkBatch;
    const uint32_t i = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    if (posd) {   // graph: ng and topk recomputed from the device position (same formula as the host's min)
        ng = ((uint32_t) posd[0] + gridDim.x) / ratio;
        if (ng < topk) topk = ng;
    }
    // C2: with a list the scan covers the compact row [ns]; writes map back; the topk cap still uses ng
    const int32_t* cl = cand ? cand + (uint64_t) i * (1u + cand_cap) : nullptr;
    const uint32_t ns = cl ? deepseek4_cand_ns(ng, cand_bs, cand_cap) : ng;
    const float* s = score + (uint64_t) i * ns;
    __shared__ uint32_t hist[256], suf[257], sh_bucket;
    __shared__ uint32_t cgt[kDeepseek4TopkThreads], ceq[kDeepseek4TopkThreads], sh_tgt, sh_teq;
    uint32_t prefix = 0, kk = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t b = tid; b < 256u; b += nt) hist[b] = 0;
        __syncthreads();
        for (uint32_t gb = tid; gb < ns; gb += B * nt) {
            float v[B];
#pragma unroll
            for (uint32_t j = 0; j < B; j++) {
                const uint32_t g = gb + j * nt;
                v[j] = g < ns ? s[g] : -INFINITY;
            }
#pragma unroll
            for (uint32_t j = 0; j < B; j++) {
                const uint32_t key = deepseek4_topk_key(v[j]);   // out-of-range -inf => key 0: not counted
                // shift=24 has no higher bits to compare; everything counts
                if (key == 0u || (shift < 24 && (key >> (shift + 8)) != (prefix >> (shift + 8)))) continue;
                atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();
        if (tid < 32u) {   // suffix sums suf[b] = sum_{b' >= b} hist[b']; lane l owns buckets [8l, 8l+8)
            uint32_t loc = 0;
            for (uint32_t b = 0; b < 8u; b++) loc += hist[tid * 8u + b];
            uint32_t inc = loc;
            for (int o = 1; o < 32; o <<= 1) {
                const uint32_t t = __shfl_down_sync(0xffffffffu, inc, o);
                if (tid + (uint32_t) o < 32u) inc += t;
            }
            uint32_t run = inc - loc;   // buckets above this lane
            for (int b = 7; b >= 0; b--) {
                run += hist[tid * 8u + (uint32_t) b];
                suf[tid * 8u + (uint32_t) b] = run;
            }
            if (tid == 0u) {
                suf[256] = 0u;
                sh_bucket = 0u;
            }
        }
        __syncthreads();
        if (shift == 24) {   // round 1's histogram covers every valid key => suf[0] = nvalid
            const uint32_t nval = suf[0];
            kk = topk < nval ? topk : nval;
            if (kk == 0u) break;   // original: kk == 0 never enters radix, prefix stays 0 (block-uniform branch)
        }
        for (uint32_t b = 1u + tid; b < 256u; b += nt)
            if (suf[b] >= kk) atomicMax(&sh_bucket, b);
        __syncthreads();
        const uint32_t bucket = sh_bucket;
        prefix |= bucket << shift;
        kk -= suf[bucket + 1u];   // how many more to take inside this bucket
        __syncthreads();          // everyone must finish reading hist/suf before the next round clears them
    }
    // Write-out segments: position = (count of > threshold before g) + min(count of == threshold before g, eq).
    if (tid == 0) {
        sh_tgt = 0;
        sh_teq = 0;
    }
    __syncthreads();
    const uint32_t chunk = (ns + nt - 1u) / nt;
    const uint32_t g0 = tid * chunk, g1 = (g0 + chunk) < ns ? (g0 + chunk) : ns;
    {   // (1) each span counts its own > threshold / == threshold (reads batched)
        uint32_t ngt = 0, neq = 0;
        for (uint32_t gb = g0; gb < g1; gb += B) {
            float v[B];
#pragma unroll
            for (uint32_t j = 0; j < B; j++) v[j] = gb + j < g1 ? s[gb + j] : -INFINITY;
#pragma unroll
            for (uint32_t j = 0; j < B; j++) {
                const uint32_t key = deepseek4_topk_key(v[j]);
                if (key == 0u) continue;
                if (key > prefix) ngt++;
                else if (key == prefix) neq++;
            }
        }
        cgt[tid] = ngt;
        ceq[tid] = neq;
        atomicAdd(&sh_tgt, ngt);   // totals: (4) uses them to fill the tail
        atomicAdd(&sh_teq, neq);
    }
    __syncthreads();
    {   // (2) two exclusive prefix sums (Hillis-Steele in place; every level needs its syncthreads, all outside branches)
        const uint32_t vg = tid ? cgt[tid - 1u] : 0u;
        const uint32_t ve = tid ? ceq[tid - 1u] : 0u;
        __syncthreads();
        cgt[tid] = vg;
        ceq[tid] = ve;
        __syncthreads();
        for (uint32_t off = 1u; off < nt; off <<= 1) {
            const uint32_t ag = tid >= off ? cgt[tid - off] : 0u;
            const uint32_t ae = tid >= off ? ceq[tid - off] : 0u;
            __syncthreads();
            cgt[tid] += ag;
            ceq[tid] += ae;
            __syncthreads();
        }
    }
    {   // (3) spans written ascending; eq = how many == threshold still to take, from the radix tail
        const uint32_t eq = kk;
        uint32_t wgt = cgt[tid], weq = ceq[tid];
        for (uint32_t gb = g0; gb < g1; gb += B) {
            float v[B];
#pragma unroll
            for (uint32_t j = 0; j < B; j++) v[j] = gb + j < g1 ? s[gb + j] : -INFINITY;
#pragma unroll
            for (uint32_t j = 0; j < B; j++) {
                const uint32_t g = gb + j;
                const uint32_t key = deepseek4_topk_key(v[j]);
                if (key == 0u) continue;
                if (key > prefix) {
                    const uint32_t pos = wgt + (weq < eq ? weq : eq);
                    if (pos < topk) idx[(uint64_t) i * topk + pos] = attn_cand_map(cl, g, cand_bs);
                    wgt++;
                } else if (key == prefix) {
                    if (weq < eq) {
                        const uint32_t pos = wgt + weq;
                        if (pos < topk) idx[(uint64_t) i * topk + pos] = attn_cand_map(cl, g, cand_bs);
                    }
                    weq++;
                }
            }
        }
    }
    __syncthreads();
    {   // (4) fill the tail with -1; total = all > threshold + min(all == threshold, eq), same as the original
        uint32_t total = sh_tgt + (sh_teq < kk ? sh_teq : kk);
        if (total > topk) total = topk;
        for (uint32_t w = total + tid; w < topk; w += nt) idx[(uint64_t) i * topk + w] = -1;
    }
}

bool finish(cudaStream_t stream, const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "deepseek4_attn: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    if (stream == nullptr) {
        const cudaError_t s = cudaDeviceSynchronize();
        if (s != cudaSuccess) {
            std::fprintf(stderr, "deepseek4_attn: %s (sync): %s\n", what, cudaGetErrorString(s));
            return false;
        }
    }
    return true;
}

unsigned unpack_blocks(uint64_t n) {
    const uint64_t blocks = (n + 255u) / 256u;
    return (unsigned)(blocks > 65535u ? 65535u : blocks);
}

}  // namespace

bool deepseek4_rope(float* d_x, const int32_t* d_pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                    uint32_t n_rot, float theta, uint32_t original_seq_len, float factor, float beta_fast,
                    float beta_slow, bool inverse, void* stream) {
    if (!d_x || !d_pos || n_tok == 0 || n_head == 0 || head_dim == 0 || (n_rot & 1u) != 0 || n_rot > 128 ||
        n_rot > head_dim || theta <= 0.0f || factor <= 0.0f)
        return false;
    if (n_rot == 0) return true;   // the reference launches and every thread returns
    attn_rope_kernel<<<dim3(n_head, n_tok), 64, 0, static_cast<cudaStream_t>(stream)>>>(
        d_x, d_pos, n_head, head_dim, n_rot, theta, original_seq_len, factor, beta_fast, beta_slow,
        inverse ? 1 : 0);
    return finish(static_cast<cudaStream_t>(stream), "rope");
}

bool deepseek4_compress_pool(float* d_out, const float* d_kv, const float* d_score, uint32_t n_tok, uint32_t ratio,
                             uint32_t dim, void* stream) {
    if (!d_out || !d_kv || !d_score || ratio == 0 || dim == 0) return false;
    const uint32_t ng = n_tok / ratio;
    if (ng == 0) return true;   // no complete group: the tail stays the caller's pending rows
    attn_compress_pool_kernel<<<ng, 256, 0, static_cast<cudaStream_t>(stream)>>>(d_out, d_kv, d_score, ratio, dim);
    return finish(static_cast<cudaStream_t>(stream), "compress pool");
}

bool deepseek4_act_quant(float* d_x, uint32_t n_rows, uint32_t dim, uint32_t block, int mode, void* stream) {
    if (!d_x || n_rows == 0 || dim == 0 || block == 0 || block > 32u || (dim % block) != 0 || mode < 0 ||
        mode > 2)
        return false;
    const uint32_t nb_row = dim / block;
    attn_act_quant_kernel<<<(unsigned)((uint64_t) n_rows * nb_row), 32, 0, static_cast<cudaStream_t>(stream)>>>(
        d_x, dim, block, mode, nb_row);
    return finish(static_cast<cudaStream_t>(stream), "act quant");
}

bool deepseek4_ckv_pack(uint8_t* d_cache, uint32_t g0, const float* d_rows, uint32_t n_rows, void* stream) {
    if (!d_cache || !d_rows || n_rows == 0) return false;
    const uint32_t nb_row = kDeepseek4CkvDim / kDeepseek4CkvBlock;
    attn_kv_pack_kernel<<<(unsigned)((uint64_t) n_rows * nb_row), 32, 0, static_cast<cudaStream_t>(stream)>>>(
        d_cache, d_rows, g0, kDeepseek4CkvDim, kDeepseek4CkvBlock, kDeepseek4CkvBytes, kDeepseek4CkvNibbleBytes,
        nb_row, kDeepseek4QuantFp4E4m3);
    return finish(static_cast<cudaStream_t>(stream), "ckv pack");
}

bool deepseek4_idxk_pack(uint8_t* d_cache, uint32_t g0, const float* d_rows, uint32_t n_rows, void* stream) {
    if (!d_cache || !d_rows || n_rows == 0) return false;
    const uint32_t nb_row = kDeepseek4IdxkDim / kDeepseek4IdxkBlock;
    attn_kv_pack_kernel<<<(unsigned)((uint64_t) n_rows * nb_row), 32, 0, static_cast<cudaStream_t>(stream)>>>(
        d_cache, d_rows, g0, kDeepseek4IdxkDim, kDeepseek4IdxkBlock, kDeepseek4IdxkBytes, kDeepseek4IdxkNibbleBytes,
        nb_row, kDeepseek4QuantFp4E8m0);
    return finish(static_cast<cudaStream_t>(stream), "idxk pack");
}

bool deepseek4_ckv_unpack(const uint8_t* d_cache, uint32_t n_rows, float* d_out, void* stream) {
    if (!d_cache || !d_out || n_rows == 0) return false;
    const uint64_t n = (uint64_t) n_rows * kDeepseek4CkvDim;
    attn_kv_unpack_kernel<<<unpack_blocks(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(d_out, d_cache, n, 0);
    return finish(static_cast<cudaStream_t>(stream), "ckv unpack");
}

bool deepseek4_idxk_unpack(const uint8_t* d_cache, uint32_t n_rows, float* d_out, void* stream) {
    if (!d_cache || !d_out || n_rows == 0) return false;
    const uint64_t n = (uint64_t) n_rows * kDeepseek4IdxkDim;
    attn_kv_unpack_kernel<<<unpack_blocks(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(d_out, d_cache, n, 1);
    return finish(static_cast<cudaStream_t>(stream), "idxk unpack");
}

bool deepseek4_sparse_attn(float* d_o, const float* d_q, const float* d_kv_win, const uint8_t* d_kv_comp,
                           const int32_t* d_idx, const float* d_sink, uint32_t n_tok, uint32_t pos0, uint32_t window,
                           uint32_t ng, uint32_t topk, uint32_t n_head, uint32_t head_dim, float scale,
                           uint32_t full_block, bool ring, uint32_t win_lo, void* stream) {
    if (!d_o || !d_q || !d_kv_win || !d_sink || n_tok == 0 || n_head == 0 || head_dim == 0 || head_dim > 512 ||
        (head_dim % 32u) != 0 || (n_head % kDeepseek4AttnHeadsPerBlock) != 0)
        return false;
    if ((d_kv_comp == nullptr) != (d_idx == nullptr)) return false;
    if (d_kv_comp == nullptr) topk = 0;
    if (full_block == 0 && !ring) return false;   // the main path's window must be a ring (the reference entry)
    if (full_block > n_tok) return false;
    if (win_lo > pos0) return false;              // the clamp would pass the last query: the caller's window is short
    // Opt-in tensor-core route for the prompt (--attn-mma); it refuses shapes it does not take, and the scalar
    // kernel below stays the reference and the fallback.
    if (full_block == 0u && ring && n_tok >= 64u && deepseek4_sparse_attn_mma_enabled() &&
        deepseek4_sparse_attn_mma(d_o, d_q, d_kv_win, d_kv_comp, d_idx, d_sink, n_tok, pos0, window, ng, topk,
                                  n_head, head_dim, scale, win_lo, stream))
        return true;
    if (head_dim == 512u)
        attn_sparse_kernel<16><<<dim3(n_tok, n_head / kDeepseek4AttnHeadsPerBlock), kDeepseek4AttnHeadsPerBlock * 16u,
                                  0, static_cast<cudaStream_t>(stream)>>>(
            d_o, d_q, d_kv_win, d_kv_comp, d_idx, d_sink, pos0, window, ng, topk, n_head, head_dim, scale,
            full_block, ring ? 1u : 0u, win_lo);
    else if (head_dim == 64u)
        attn_sparse_kernel<2><<<dim3(n_tok, n_head / kDeepseek4AttnHeadsPerBlock), kDeepseek4AttnHeadsPerBlock * 16u,
                                 0, static_cast<cudaStream_t>(stream)>>>(
            d_o, d_q, d_kv_win, d_kv_comp, d_idx, d_sink, pos0, window, ng, topk, n_head, head_dim, scale,
            full_block, ring ? 1u : 0u, win_lo);
    else
        attn_sparse_kernel<0><<<dim3(n_tok, n_head / kDeepseek4AttnHeadsPerBlock), kDeepseek4AttnHeadsPerBlock * 16u,
                                0, static_cast<cudaStream_t>(stream)>>>(
            d_o, d_q, d_kv_win, d_kv_comp, d_idx, d_sink, pos0, window, ng, topk, n_head, head_dim, scale,
            full_block, ring ? 1u : 0u, win_lo);
    return finish(static_cast<cudaStream_t>(stream), "sparse attn");
}

bool deepseek4_win_commit(float* d_win, uint32_t pos0, uint32_t n, uint32_t window, uint32_t head_dim,
                          const int32_t* d_pos, void* stream) {
    if (!d_win || n == 0 || window == 0 || head_dim == 0) return false;
    if (d_pos && n > 8u) return false;   // the reference's graph route takes only decode-sized batches
    const uint32_t i0 = n > window ? n - window : 0u, rows = n - i0;
    attn_win_commit_kernel<<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(d_win, pos0, i0, window, head_dim,
                                                                                d_pos);
    return finish(static_cast<cudaStream_t>(stream), "win commit");
}

bool deepseek4_win_ring_snap(float* d_win, float* d_snap, uint32_t pos0, uint32_t i0, uint32_t n, uint32_t window,
                             uint32_t head_dim, bool back, const int32_t* d_pos, void* stream) {
    if (!d_win || !d_snap || window == 0 || head_dim == 0) return false;
    if (n <= i0) return true;   // no rows to touch: success, the reference's rule
    attn_win_ring_snap_kernel<<<n - i0, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        d_win, d_snap, pos0, i0, window, head_dim, back ? 1u : 0u, d_pos);
    return finish(static_cast<cudaStream_t>(stream), back ? "win ring restore" : "win ring save");
}

bool deepseek4_indexer_score(float* d_score, const float* d_q, const uint8_t* d_idxk, const float* d_w,
                             const int32_t* d_cand, uint32_t pos0, uint32_t ng, uint32_t n_tok, uint32_t n_head,
                             uint32_t dk, uint32_t ratio, uint32_t cand_bs, uint32_t cand_cap,
                             const int32_t* d_pos, void* stream) {
    if (!d_score || !d_q || !d_idxk || !d_w || n_tok == 0 || ratio == 0 || (dk % 32u) != 0) return false;
    if (dk / 32u > 4u) return false;   // the reference refuses dk > 128: at most 4 dims per lane
    if (d_pos && n_tok > 8u) return false;   // the graph route takes only decode-sized batches
    if (ng == 0) return true;
    if (d_cand && (cand_bs == 0u || cand_cap == 0u)) return false;
    // One block = 8 warps x kDeepseek4IdxGroupsPerWarp groups; with a list the grid is sized by the compact width.
    const uint32_t ns = d_cand ? deepseek4_cand_ns(ng, cand_bs, cand_cap) : ng;
    uint32_t gblocks = (ns + 8u * kDeepseek4IdxGroupsPerWarp - 1u) / (8u * kDeepseek4IdxGroupsPerWarp);
    if (gblocks > 65535u) gblocks = 65535u;   // CUDA's grid.y cap; the kernel wraps by grid stride
    attn_indexer_score_kernel<<<dim3(n_tok, gblocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        d_score, d_q, d_idxk, d_w, d_cand, pos0, ng, n_head, dk, ratio, d_pos, cand_bs, cand_cap);
    return finish(static_cast<cudaStream_t>(stream), "indexer score");
}

bool deepseek4_candidate_blocks(int32_t* d_list, const float* d_score, float* d_scratch, uint32_t n_tok,
                                uint32_t pos0, uint32_t ng, uint32_t ratio, uint32_t topk_blocks,
                                uint32_t block_size, uint32_t nb_cap, const int32_t* d_pos, void* stream) {
    if (!d_list || !d_score || !d_scratch || n_tok == 0 || block_size == 0 || ratio == 0 || topk_blocks == 0)
        return false;
    if (d_pos && n_tok > 8u) return false;
    if (ng == 0) return true;
    const uint32_t nb = (ng + block_size - 1u) / block_size;
    if (nb > nb_cap) return false;   // the caller's scratch row stride
    attn_candidate_kernel<<<n_tok, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        d_list, d_score, pos0, ng, ratio, topk_blocks, block_size, d_pos, d_scratch, nb_cap);
    return finish(static_cast<cudaStream_t>(stream), "indexer candidate blocks");
}

bool deepseek4_indexer_topk(int32_t* d_idx, const float* d_score, uint32_t n_tok, uint32_t ng, uint32_t topk,
                            uint32_t ratio, const int32_t* d_cand, uint32_t cand_bs, uint32_t cand_cap,
                            const int32_t* d_pos, void* stream) {
    if (!d_idx || !d_score || n_tok == 0 || topk == 0) return false;
    if (d_pos && (n_tok > 8u || ratio == 0u)) return false;
    if (d_cand && (cand_bs == 0u || cand_cap == 0u)) return false;
    if (ng == 0) return true;
    attn_topk_kernel<<<n_tok, kDeepseek4TopkThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        d_idx, d_score, ng, topk, ratio, d_pos, d_cand, cand_bs, cand_cap);
    return finish(static_cast<cudaStream_t>(stream), "indexer topk");
}

}  // namespace strata::kernels
