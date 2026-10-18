// src/kernels/cuda/deepseek4_attn_mma.cu - the sparse attention's tensor-core prefill route.
//
// Ported from the reference engine's single-pass prefill kernel
// (reference/ds4-dfm-rs/cuda/ds41_attn_mma.cuh:137-293, itself cuda_sparse_attn_mma.inc.cu:126-318): one block
// per (query, 16-head group), 8 warps, 16 keys per shared tile, QK and PV as mma.m16n8k16 bf16.  Q's A
// fragments live in registers; each warp takes 64 of the 512 dims for the K steps and for the output.
//
// This route is NOT bit-equal to the scalar kernel: the tensor-core accumulation order and the online-softmax
// tile grouping both differ (the reference's own header says so; its gate is the quality scale, not cmp).  The
// scalar kernel stays the reference for the tests and the fallback for every shape this refuses (n_tok < 64,
// head_dim != 512, n_head % 16, full_block, non-ring).  What IS kept exactly: the key order (window ascending,
// then the topk compressed rows), the compressed-row decode (same scale, nibble and multiply as
// deepseek4_ckv_get, so the bf16 key is bit-equal), the invalid-slot zeroing (0 x NaN residue is NaN if the
// row is not zeroed), the bf16-rounded probability, the sink in the denominator only, and the bf16 output.
//
// The mma.sync in this file needs sm_80 or newer, so it sits in its own static library with a pinned arch
// (86 SASS + 120, the cards the port is measured on), the same arrangement the batched MoE prefill and the
// dense tensor-core tiles use.
#include "strata/kernels/deepseek4_attn.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace strata::kernels {

namespace {

using strata::deepseek4::engram_bf16r;

constexpr uint32_t kMmaHeads = 16u;   // heads per block = one m16 tile
constexpr uint32_t kMmaKeys = 16u;    // keys per shared tile = one n16 step
constexpr uint32_t kMmaWarps = 8u;    // 8 warps: 4 k-steps each for QK, 64 output dims each for PV
constexpr uint32_t kMmaHd = 512u;     // head dim this route is bound to
// The key tile's row stride: 512 bf16 = 1024 B is an 8-way bank conflict on every load; 520 bf16 = 1040 B
// breaks it (cuda_sparse_attn_mma.inc.cu:129, the reference's 2026-09-29 form).
constexpr uint32_t kMmaLd = 520u;

__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void ldsm4t(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

// c += A[16x16] * B[16x8] in bf16 with an f32 accumulator: the A fragment is four packed bf16 pairs, the B
// fragment two (row/col pair layout of m16n8k16).
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two f32 as one packed bf16x2 with the RNE round the engine uses (v41_bf16r): low half = lo, high = hi.
__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
    return (__float_as_uint(engram_bf16r(lo)) >> 16) | (__float_as_uint(engram_bf16r(hi)) & 0xffff0000u);
}

// Gather 16 key rows into shared as bf16 (cuda_sparse_attn_mma.inc.cu:42-73).  Window rows first in ascending
// order, then the topk compressed rows -- the scalar kernel's key order exactly.  The compressed decode reads
// the row's 32 scales one per lane and the fp4 nibble one per lane, then resolves each element with two
// shuffles; the same scale, nibble and multiply as deepseek4_ckv_get, so every key value is bit-equal.
__device__ __forceinline__ void mma_gather(__nv_bfloat16* ks, int* valid, const float* kvw, const uint8_t* kvc,
                                           const int32_t* idx, uint32_t i, uint32_t base, uint32_t nt,
                                           uint32_t nwin, uint32_t lo, uint32_t pos0, uint32_t window,
                                           uint32_t ng, uint32_t topk) {
    const uint32_t lane = threadIdx.x & 31u;
    const float tv = deepseek4_fp4_from_nibble((uint8_t)(lane & 15u));   // lane l holds entry l&15
    for (uint32_t t = threadIdx.x / 32u; t < kMmaKeys; t += blockDim.x / 32u) {
        const uint32_t kk = base + t;
        const float* krow = NULL;
        const uint8_t* cpk = NULL;
        if (t < nt) {
            if (kk < nwin) {
                krow = kvw + (uint64_t) deepseek4_win_row((int64_t) lo + kk, pos0, window, 1u) * kMmaHd;
            } else if (kvc && idx) {
                const int32_t g = idx[(uint64_t) i * topk + (kk - nwin)];
                if (g >= 0 && (uint32_t) g < ng) cpk = kvc + (uint64_t) g * kDeepseek4CkvBytes;
            }
        }
        if (lane == 0) valid[t] = (krow != NULL || cpk != NULL) ? 1 : 0;
        __nv_bfloat16* kt = ks + (size_t) t * kMmaLd;
        if (krow) {
            for (uint32_t d = lane; d < kMmaHd; d += 32u) kt[d] = __float2bfloat16(krow[d]);
        } else if (cpk) {
            const float sc = strata::deepseek4::engram_e4m3_to_f32(cpk[kDeepseek4CkvNibbleBytes + lane]);
#pragma unroll
            for (uint32_t j = 0; j < kMmaHd / 32u; j++) {
                const uint32_t d = lane + 32u * j;
                const uint8_t by = cpk[d >> 1];
                const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
                const float s = __shfl_sync(0xffffffffu, sc, (int) (d >> 4));
                kt[d] = __float2bfloat16(__shfl_sync(0xffffffffu, tv, (int) nib) * s);
            }
        } else {
            for (uint32_t d = lane; d < kMmaHd; d += 32u) kt[d] = (__nv_bfloat16) 0.0f;
        }
    }
}

// 16 KB of key tile, 8 KB of warp partials, 2 KB of S/P tiles and the running stats: ~26 KB of static shared,
// 3 blocks per SM on the 3060's 100 KB and more on the newer cards.
__global__ __launch_bounds__(kMmaWarps * 32u) void attn_sparse_mma_kernel(
        float* o, const float* q, const float* kvw, const uint8_t* kvc, const int32_t* idx, const float* sink,
        uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head, float scale,
        uint32_t win_lo) {
    __shared__ __align__(16) __nv_bfloat16 ks[kMmaKeys * kMmaLd];
    __shared__ float spart[kMmaWarps * kMmaHeads * kMmaKeys];
    __shared__ float stile[kMmaHeads * kMmaKeys];
    __shared__ __align__(16) __nv_bfloat16 ptile[kMmaHeads * kMmaKeys];   // ldmatrix needs 16 B
    __shared__ float rmax[kMmaHeads], rsum[kMmaHeads], alpha[kMmaHeads];
    __shared__ int valid[kMmaKeys];
    const uint32_t i = blockIdx.x, h0 = blockIdx.y * kMmaHeads, lane = threadIdx.x & 31u,
                   warp = threadIdx.x >> 5;
    const uint32_t p = pos0 + i;
    uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    if (lo < win_lo) lo = win_lo;   // ring slots before win_lo were never written; do not read them
    const uint32_t nwin = p - lo + 1u, nkeys = nwin + topk;
    // Q's A fragments live in registers: warp w covers dims [64w, 64w+64) as 4 k16 steps (a0: row r col c..c+1;
    // a1: row r+8; a2: col +8; a3: row +8 col +8).
    uint32_t qa[4][4];
    {
        const uint32_t r = lane >> 2, c = (lane & 3u) * 2u;
#pragma unroll
        for (uint32_t s = 0; s < 4u; s++) {
            const float* q0 = q + ((uint64_t) i * n_head + h0 + r) * kMmaHd + warp * 64u + s * 16u + c;
            const float* q1 = q0 + 8u * kMmaHd;
            qa[s][0] = bf16x2(q0[0], q0[1]);
            qa[s][1] = bf16x2(q1[0], q1[1]);
            qa[s][2] = bf16x2(q0[8], q0[9]);
            qa[s][3] = bf16x2(q1[8], q1[9]);
        }
    }
    if (threadIdx.x < kMmaHeads) {
        rmax[threadIdx.x] = -1e30f;
        rsum[threadIdx.x] = 0.f;
    }
    float oacc[8][4];   // O[16 heads][this warp's 64 dims] = 8 n8 slices
#pragma unroll
    for (uint32_t j = 0; j < 8u; j++) oacc[j][0] = oacc[j][1] = oacc[j][2] = oacc[j][3] = 0.f;
    __syncthreads();
    for (uint32_t base = 0; base < nkeys; base += kMmaKeys) {
        const uint32_t nt = (nkeys - base) < kMmaKeys ? (nkeys - base) : kMmaKeys;
        __syncthreads();   // the previous tile's ks/ptile is fully consumed
        mma_gather(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        {   // (1) S partials: this warp's 64 dims x 16 heads x 16 keys (two n8 slices)
            float sp[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
            for (uint32_t s = 0; s < 4u; s++) {
                uint32_t b[4];
                const uint32_t mt = lane >> 3, key = (mt >> 1) * 8u + (lane & 7u),
                               col = warp * 64u + s * 16u + (mt & 1u) * 8u;
                ldsm4(b, ks + (size_t) key * kMmaLd + col);
                mma(sp[0], qa[s], b[0], b[1]);
                mma(sp[1], qa[s], b[2], b[3]);
            }
#pragma unroll
            for (uint32_t nb = 0; nb < 2u; nb++) {   // c0,c1 = row lane/4, cols (lane&3)*2+{0,1}; c2,c3 = row +8
                float* sw =
                    spart + (size_t) warp * kMmaHeads * kMmaKeys + (lane >> 2) * kMmaKeys + nb * 8u + (lane & 3u) * 2u;
                sw[0] = sp[nb][0];
                sw[1] = sp[nb][1];
                sw[8u * kMmaKeys] = sp[nb][2];
                sw[8u * kMmaKeys + 1u] = sp[nb][3];
            }
        }
        __syncthreads();
        {   // (2) fixed-order sum + scale + mask (an invalid slot scores -1e30, so its exp is 0)
            const uint32_t e = threadIdx.x, k = e % kMmaKeys;
            float v = 0.f;
#pragma unroll
            for (uint32_t w = 0; w < kMmaWarps; w++) v += spart[(size_t) w * kMmaHeads * kMmaKeys + e];
            stile[e] = (k < nt && valid[k]) ? v * scale : -1e30f;
        }
        __syncthreads();
        {   // (3) online max/sum, P tile (bf16, the official acc_s_cast) and alpha: one thread per head, keys serial
            if (threadIdx.x < kMmaHeads) {
                const uint32_t h = threadIdx.x;
                const float* sr = stile + h * kMmaKeys;
                const float m = rmax[h];
                float tm = m;
                for (uint32_t k = 0; k < kMmaKeys; k++) tm = fmaxf(tm, sr[k]);
                const float al = expf(m - tm);
                float sm = rsum[h] * al;
                for (uint32_t k = 0; k < kMmaKeys; k++) {
                    const float pv = expf(sr[k] - tm);
                    sm += pv;
                    ptile[h * kMmaKeys + k] = __float2bfloat16(pv);
                }
                rmax[h] = tm;
                rsum[h] = sm;
                alpha[h] = al;
            }
        }
        __syncthreads();
        {   // (4) O = O*alpha + P*V: A = ptile[16 heads][16 keys], B = ks[keys][dim] via .trans
            const float a0 = alpha[lane >> 2], a1 = alpha[(lane >> 2) + 8u];
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                oacc[j][0] *= a0;
                oacc[j][1] *= a0;
                oacc[j][2] *= a1;
                oacc[j][3] *= a1;
            }
            uint32_t pa[4];
            {
                const uint32_t mt = lane >> 3, row = (lane & 7u) + (mt & 1u) * 8u, col = (mt >> 1) * 8u;
                ldsm4(pa, ptile + (size_t) row * kMmaKeys + col);
            }
#pragma unroll
            for (uint32_t j = 0; j < 8u; j += 2u) {
                uint32_t b[4];
                const uint32_t mt = lane >> 3, key = (mt & 1u) * 8u + (lane & 7u),
                               col = warp * 64u + j * 8u + (mt >> 1) * 8u;
                ldsm4t(b, ks + (size_t) key * kMmaLd + col);
                mma(oacc[j], pa, b[0], b[1]);
                mma(oacc[j + 1], pa, b[2], b[3]);
            }
        }
    }
    __syncthreads();
    {   // exit: divide by (sum + exp(sink - max)), round bf16, write
        const uint32_t hA = lane >> 2, hB = hA + 8u;
        const float dA = rsum[hA] + expf(sink[h0 + hA] - rmax[hA]), dB = rsum[hB] + expf(sink[h0 + hB] - rmax[hB]);
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t d = warp * 64u + j * 8u + (lane & 3u) * 2u;
            float* oA = o + ((uint64_t) i * n_head + h0 + hA) * kMmaHd + d;
            float* oB = o + ((uint64_t) i * n_head + h0 + hB) * kMmaHd + d;
            oA[0] = engram_bf16r(oacc[j][0] / dA);
            oA[1] = engram_bf16r(oacc[j][1] / dA);
            oB[0] = engram_bf16r(oacc[j][2] / dB);
            oB[1] = engram_bf16r(oacc[j][3] / dB);
        }
    }
}

}  // namespace

namespace {
bool g_mma_enabled = false;   // opt-in: the scalar kernel's arithmetic is the reference
}  // namespace

void deepseek4_sparse_attn_set_mma(bool on) { g_mma_enabled = on; }
bool deepseek4_sparse_attn_mma_enabled() { return g_mma_enabled; }

// 1 = this route took the launch; 0 = shape unfit (the caller falls through to the scalar kernel).
bool deepseek4_sparse_attn_mma(float* d_o, const float* d_q, const float* d_kv_win, const uint8_t* d_kv_comp,
                               const int32_t* d_idx, const float* d_sink, uint32_t n_tok, uint32_t pos0,
                               uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head, uint32_t head_dim,
                               float scale, uint32_t win_lo, void* stream) {
    if (head_dim != kMmaHd || (n_head % kMmaHeads) != 0 || n_tok < 64u) return false;
    if ((d_kv_comp == nullptr) != (d_idx == nullptr)) return false;
    if (d_kv_comp == nullptr) topk = 0;   // same rule as the scalar entry: no cache, no compressed keys
    if (win_lo > pos0) return false;      // the clamp would pass the last query: the window is short
    attn_sparse_mma_kernel<<<dim3(n_tok, n_head / kMmaHeads), kMmaWarps * 32u, 0,
                             static_cast<cudaStream_t>(stream)>>>(d_o, d_q, d_kv_win, d_kv_comp, d_idx, d_sink,
                                                                 pos0, window, ng, topk, n_head, scale, win_lo);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "deepseek4_attn: sparse attn mma: %s\n", cudaGetErrorString(e));
        return false;
    }
    if (stream == nullptr) {
        const cudaError_t s = cudaDeviceSynchronize();
        if (s != cudaSuccess) {
            std::fprintf(stderr, "deepseek4_attn: sparse attn mma (sync): %s\n", cudaGetErrorString(s));
            return false;
        }
    }
    return true;
}

}  // namespace strata::kernels
