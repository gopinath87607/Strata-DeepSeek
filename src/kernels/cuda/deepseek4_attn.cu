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
__global__ void attn_sparse_kernel(float* o, const float* q, const float* kvw, const uint8_t* kvc, const int32_t* idx,
                                   const float* sink, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                   uint32_t n_head, uint32_t hd, float scale, uint32_t full_block, uint32_t ring,
                                   uint32_t win_lo) {
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t per = hd / 32u;
    __shared__ float ks[kDeepseek4AttnKeyTile][512];
    __shared__ int kok[kDeepseek4AttnKeyTile];
    __shared__ float ksc[kDeepseek4AttnKeyTile][32];
    float qa[2][16], acc[2][16], mx[2], sum[2];
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
                ks[t][d] = krow ? krow[d] : (cpk ? attn_ckv_get_s(cpk, d, ksc[t]) : 0.f);
        }
        __syncthreads();
        for (int hh = 0; hh < 2; hh++) {
            float s[kDeepseek4AttnKeyTile];
            float tm = -1e30f;
            for (uint32_t t = 0; t < nt; t++) {
                float d = 0.f;
                for (uint32_t e = 0; e < per; e++) d += qa[hh][e] * ks[t][lane * per + e];
                for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
                s[t] = kok[t] ? d * scale : -1e30f;
                tm = fmaxf(tm, s[t]);
            }
            const float nm = fmaxf(mx[hh], tm);
            const float rs = expf(mx[hh] - nm);   // one rescale per tile
            float acc_add[16];
            for (uint32_t e = 0; e < per; e++) acc_add[e] = 0.f;
            float ps = 0.f;
            for (uint32_t t = 0; t < nt; t++) {
                const float pv = expf(s[t] - nm);
                ps += pv;
                const float pb = engram_bf16r(pv);   // the official acc_s_cast
                for (uint32_t e = 0; e < per; e++) acc_add[e] += pb * ks[t][lane * per + e];
            }
            sum[hh] = sum[hh] * rs + ps;
            for (uint32_t e = 0; e < per; e++) acc[hh][e] = acc[hh][e] * rs + acc_add[e];
            mx[hh] = nm;
        }
    }
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
    attn_sparse_kernel<<<dim3(n_tok, n_head / kDeepseek4AttnHeadsPerBlock), kDeepseek4AttnHeadsPerBlock * 16u, 0,
                         static_cast<cudaStream_t>(stream)>>>(
        d_o, d_q, d_kv_win, d_kv_comp, d_idx, d_sink, pos0, window, ng, topk, n_head, head_dim, scale, full_block,
        ring ? 1u : 0u, win_lo);
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

}  // namespace strata::kernels
