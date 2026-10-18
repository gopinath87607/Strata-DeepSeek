// Set by deepseek4_dense_set_tc (--tc-gemm): the batched matmuls call the tensor-core tiles in
// deepseek4_dense_tc.cu (--tc-gemm is opt-in; the default path keeps the kernels above).
// src/kernels/cuda/deepseek4_dense.cu - P3: DeepSeek-V4.1-Flash's dense primitives on the GPU.
//
// The semantics are stated in include/strata/kernels/deepseek4_dense.hpp and implemented twice: by the host
// oracle in src/kernels/deepseek4_dense_host.cpp and by these kernels.  The q4_K decoder itself is shared
// (`deepseek4_q4k_element` in the header), so it is one piece of code on both sides; the gate additionally
// checks it against `dequant_bf16`, the tree's independently validated decoder.
//
// These are correctness kernels, not the reference engine's performance kernels: a row of a matmul is one
// block with a shared tree reduction, the embedding is one thread per element, and the q4_K matmuls read each
// nibble from global memory instead of regenerating a block in registers per warp.  P7 replaces them with the
// reference's warp GEMV / dequant-to-bf16 GEMM / packed-dot layouts; nothing here decides what the forward
// pass computes, only how fast it runs.
//
// Every entry accepts a null stream (default stream, then the call synchronizes), refuses a null pointer or a
// shape the layout cannot express before launching, and reports a launch or execution failure by returning
// false rather than leaving the caller with silent garbage.
#include "strata/kernels/deepseek4_dense.hpp"
#include "strata/kernels/deepseek4_dense_tc.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace strata::kernels {

namespace {

using strata::deepseek4::engram_bf16r;

constexpr uint32_t kThreads = 256;

__device__ __forceinline__ void block_sum(float* sh, float v, uint32_t tid) {
    sh[tid] = v;
    __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) sh[tid] += sh[tid + k];
        __syncthreads();
    }
}

// One block per token row.  The sum is over the f32 row, eps is added after dividing by dim, and the weight
// multiplies inside the bf16 round, exactly as the oracle and v41_rms_norm do.  rsqrtf is the fast device
// reciprocal square root; its ~2^-22 error is far below bf16 and the gate's tolerance.
__global__ void dense_rms_norm_kernel(float* out, const float* x, const float* w, uint32_t dim, float eps) {
    const uint32_t t = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (size_t) t * dim;
    float* orow = out + (size_t) t * dim;
    float s = 0.0f;
    for (uint32_t d = tid; d < dim; d += blockDim.x) s = fmaf(xr[d], xr[d], s);
    __shared__ float sh[kThreads];
    block_sum(sh, s, tid);
    const float inv = rsqrtf(sh[0] / (float) dim + eps);
    for (uint32_t d = tid; d < dim; d += blockDim.x) orow[d] = engram_bf16r(w[d] * (xr[d] * inv));
}

__global__ void dense_add_kernel(float* a, const float* b, uint64_t n) {
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t) gridDim.x * blockDim.x)
        a[i] += b[i];
}

__global__ void dense_round_bf16_kernel(float* x, uint64_t n) {
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t) gridDim.x * blockDim.x)
        x[i] = engram_bf16r(x[i]);
}

__global__ void dense_scale_round_kernel(float* x, uint64_t n, float s) {
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t) gridDim.x * blockDim.x)
        x[i] = engram_bf16r(x[i] * s);
}

// The embedding's [n_tok][n_embd] row copied into every one of the n_hc lanes, layout [n_tok][n_hc][n_embd].
__global__ void dense_expand_hc_kernel(float* hc, const float* x, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    const uint32_t t = blockIdx.y;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd || t >= n_tok) return;
    const float v = x[(size_t) t * n_embd + d];
    for (uint32_t c = 0; c < n_hc; c++) hc[((size_t) t * n_hc + c) * n_embd + d] = v;
}

// up clamps both ways, gate only from above (the reference's swiglu: an expert gate below -limit is left
// alone), and the product is rounded once.
__global__ void dense_swiglu_kernel(float* h, const float* gate, const float* up, uint64_t n, float limit) {
    const bool do_clamp = limit > 0.0f;
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t) gridDim.x * blockDim.x) {
        float g = gate[i], u = up[i];
        if (do_clamp) {
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
        }
        h[i] = engram_bf16r((g / (1.0f + expf(-g))) * u);
    }
}

__global__ void dense_embed_q4k_kernel(float* out, const int32_t* tok, const uint8_t* w, uint32_t n_vocab,
                                       uint32_t n_tok, uint32_t dim) {
    const uint32_t blocks_per_row = dim / kDeepseek4Q4kBlock;
    const uint64_t total = (uint64_t) n_tok * dim;
    for (uint64_t idx = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (uint64_t) gridDim.x * blockDim.x) {
        const uint32_t t = (uint32_t) (idx / dim), d = (uint32_t) (idx % dim);
        const int32_t id = tok[t];
        float* orow = out + (uint64_t) t * dim;
        if (id < 0 || (uint32_t) id >= n_vocab) {
            orow[d] = 0.0f;
            continue;
        }
        const uint8_t* row = w + (uint64_t) id * blocks_per_row * kDeepseek4Q4kBytes;
        orow[d] = deepseek4_q4k_element(row + (uint64_t) (d / kDeepseek4Q4kBlock) * kDeepseek4Q4kBytes,
                                        d % kDeepseek4Q4kBlock);
    }
}

// One block per (output row, token); blockDim.x == kThreads.
__global__ void dense_matmul_f32_kernel(float* out, const float* w, const float* x, uint32_t in_dim,
                                        uint32_t out_dim, bool round_out) {
    const uint32_t o = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const float* wr = w + (size_t) o * in_dim;
    const float* xr = x + (size_t) t * in_dim;
    float acc = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) acc = fmaf(wr[i], xr[i], acc);
    __shared__ float sh[kThreads];
    block_sum(sh, acc, tid);
    if (tid == 0) {
        const float v = sh[0];
        out[(size_t) t * out_dim + o] = round_out ? engram_bf16r(v) : v;
    }
}

__global__ void dense_matmul_bf16_kernel(float* out, const uint16_t* w, const float* x, uint32_t in_dim,
                                         uint32_t out_dim, bool round_out) {
    const uint32_t o = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const uint16_t* wr = w + (size_t) o * in_dim;
    const float* xr = x + (size_t) t * in_dim;
    float acc = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) acc = fmaf(f32_from_bf16(wr[i]), xr[i], acc);
    __shared__ float sh[kThreads];
    block_sum(sh, acc, tid);
    if (tid == 0) {
        const float v = sh[0];
        out[(size_t) t * out_dim + o] = round_out ? engram_bf16r(v) : v;
    }
}

// blockDim.x == kThreads == one element per q4_K block, so thread `tid` reads element `tid` of every block
// in the row: the reduction is over blocks, and each thread walks its own element lane.
__global__ void dense_matmul_q4k_kernel(float* out, const uint8_t* w, const float* x, uint32_t blocks,
                                        uint32_t out_dim, bool round_out) {
    const uint32_t o = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const uint8_t* wrow = w + (size_t) o * blocks * kDeepseek4Q4kBytes;
    const float* xr = x + (size_t) t * blocks * kDeepseek4Q4kBlock;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++)
        acc = fmaf(deepseek4_q4k_element(wrow + (size_t) b * kDeepseek4Q4kBytes, tid),
                   xr[(size_t) b * kDeepseek4Q4kBlock + tid], acc);
    __shared__ float sh[kThreads];
    block_sum(sh, acc, tid);
    if (tid == 0) {
        const float v = sh[0];
        out[(size_t) t * out_dim + o] = round_out ? engram_bf16r(v) : v;
    }
}

__global__ void dense_matmul_q4k_grouped_kernel(float* out, const uint8_t* w, const float* x, uint32_t blocks,
                                                uint32_t group_dim, uint32_t rank, uint32_t n_groups,
                                                bool round_out) {
    const uint32_t o = blockIdx.x, t = blockIdx.y, g = blockIdx.z, tid = threadIdx.x;
    const size_t group_bytes = (size_t) rank * blocks * kDeepseek4Q4kBytes;
    const uint8_t* wrow = w + (size_t) g * group_bytes + (size_t) o * blocks * kDeepseek4Q4kBytes;
    const float* xg = x + ((size_t) t * n_groups + g) * group_dim;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++)
        acc = fmaf(deepseek4_q4k_element(wrow + (size_t) b * kDeepseek4Q4kBytes, tid),
                   xg[(size_t) b * kDeepseek4Q4kBlock + tid], acc);
    __shared__ float sh[kThreads];
    block_sum(sh, acc, tid);
    if (tid == 0) {
        const float v = sh[0];
        out[((size_t) t * n_groups + g) * rank + o] = round_out ? engram_bf16r(v) : v;
    }
}

// ---- token-batched tiles --------------------------------------------------
//
// The kernels above give one block to one (output row, token) pair, so a prefill-width matmul reads the whole
// activation matrix once per output row: out_dim x n_tok x in_dim floats of traffic where the arithmetic needs
// n_tok x in_dim.  At 5120x8192x1412 that is 236 GB for one matmul, and the block time was going to memory.
//
// A tile here is 64 output rows by 32 tokens, 256 threads as 4 (rows) x 2 (tokens) warps of 16 x 16.  Each lane
// holds 2 rows and all 16 of its warp's tokens, and one quarter of the k block, so a decoded weight scales 16
// tokens and no lane decodes a weight another lane also decodes; the four k quarters meet in two shuffles at the
// end.  The 32 activation rows are staged in shared once per k block (32 KiB) and read by every warp.
constexpr uint32_t kTileN = 64;   // output rows per block
constexpr uint32_t kTileT = 32;   // tokens per block
constexpr uint32_t kTileThreads = 256;

// GROUPED selects the wo_a layout: weight g is a [rank][group_dim] matrix of its own, token rows are n_groups
// apart in both x and out, and blockIdx.z picks the group.  The plain matmul is the same kernel with zero
// offsets and in_dim / out_dim strides.
template <bool GROUPED>
__global__ void dense_matmul_q4k_tile_kernel(float* __restrict__ out, const uint8_t* __restrict__ w,
                                             const float* __restrict__ x, uint32_t blocks, uint32_t out_dim,
                                             uint32_t n_tok, bool round_out) {
    __shared__ float xsh[kTileT][kDeepseek4Q4kBlock];
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t ow = warp & 3u, tw = warp >> 2;             // 4 row warps x 2 token warps
    const uint32_t rr = lane >> 2, cc = lane & 3u;             // 8 row pairs x 4 k quarters
    const uint32_t in_dim = blocks * kDeepseek4Q4kBlock;
    const uint32_t g = GROUPED ? blockIdx.z : 0u;
    const float* xg = GROUPED ? x + (size_t) g * in_dim : x;
    float* yg = GROUPED ? out + (size_t) g * out_dim : out;
    const size_t xr = GROUPED ? (size_t) in_dim * gridDim.z : (size_t) in_dim;
    const size_t yr = GROUPED ? (size_t) out_dim * gridDim.z : (size_t) out_dim;
    w += (size_t) g * out_dim * blocks * kDeepseek4Q4kBytes;
    const uint32_t o0 = blockIdx.x * kTileN + ow * 16u + rr;
    const uint32_t t0 = blockIdx.y * kTileT + tw * 16u;
    const uint32_t r0 = o0 < out_dim ? o0 : out_dim - 1u;
    const uint32_t r1 = o0 + 8u < out_dim ? o0 + 8u : out_dim - 1u;
    const uint8_t* w0 = w + (size_t) r0 * blocks * kDeepseek4Q4kBytes;
    const uint8_t* w1 = w + (size_t) r1 * blocks * kDeepseek4Q4kBytes;
    const uint32_t g0 = cc * 64u;                              // this lane's 64 elements: groups 2cc and 2cc+1
    float acc[2][16];
#pragma unroll
    for (uint32_t i = 0; i < 2; i++)
#pragma unroll
        for (uint32_t t = 0; t < 16; t++) acc[i][t] = 0.f;
    for (uint32_t b = 0; b < blocks; b++) {
        __syncthreads();
        for (uint32_t i = tid; i < kTileT * kDeepseek4Q4kBlock; i += kTileThreads) {
            const uint32_t tt = i >> 8, e = i & 255u;
            const uint32_t row = blockIdx.y * kTileT + tt;
            xsh[tt][e] = row < n_tok ? xg[(size_t) row * xr + b * kDeepseek4Q4kBlock + e] : 0.f;
        }
        __syncthreads();
        const uint8_t* b0 = w0 + (size_t) b * kDeepseek4Q4kBytes;
        const uint8_t* b1 = w1 + (size_t) b * kDeepseek4Q4kBytes;
        const float d0 = f32_from_f16((uint16_t) (b0[0] | (b0[1] << 8)));
        const float dm0 = f32_from_f16((uint16_t) (b0[2] | (b0[3] << 8)));
        const float d1 = f32_from_f16((uint16_t) (b1[0] | (b1[1] << 8)));
        const float dm1 = f32_from_f16((uint16_t) (b1[2] | (b1[3] << 8)));
        uint32_t s0 = 0, m0 = 0, s1 = 0, m1 = 0;
        deepseek4_q4k_scale_min(2u * cc, b0 + 4, s0, m0);
        deepseek4_q4k_scale_min(2u * cc + 1u, b0 + 4, s1, m1);
        uint32_t s0b = 0, m0b = 0, s1b = 0, m1b = 0;
        deepseek4_q4k_scale_min(2u * cc, b1 + 4, s0b, m0b);
        deepseek4_q4k_scale_min(2u * cc + 1u, b1 + 4, s1b, m1b);
        const float a00 = d0 * (float) s0, b00 = dm0 * (float) m0;
        const float a01 = d0 * (float) s1, b01 = dm0 * (float) m1;
        const float a10 = d1 * (float) s0b, b10 = dm1 * (float) m0b;
        const float a11 = d1 * (float) s1b, b11 = dm1 * (float) m1b;
        const uint8_t* p0 = b0 + 16u + 32u * cc;
        const uint8_t* p1 = b1 + 16u + 32u * cc;
#pragma unroll
        for (uint32_t l = 0; l < 32; l++) {
            const uint8_t y0 = p0[l], y1 = p1[l];
            const float q00 = a00 * (float) (y0 & 15u) - b00;
            const float q01 = a01 * (float) (y0 >> 4) - b01;
            const float q10 = a10 * (float) (y1 & 15u) - b10;
            const float q11 = a11 * (float) (y1 >> 4) - b11;
#pragma unroll
            for (uint32_t t = 0; t < 16; t++) {
                const float xa = xsh[tw * 16u + t][g0 + l];
                const float xb = xsh[tw * 16u + t][g0 + 32u + l];
                acc[0][t] = fmaf(q00, xa, acc[0][t]);
                acc[0][t] = fmaf(q01, xb, acc[0][t]);
                acc[1][t] = fmaf(q10, xa, acc[1][t]);
                acc[1][t] = fmaf(q11, xb, acc[1][t]);
            }
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < 2; i++)
#pragma unroll
        for (uint32_t t = 0; t < 16; t++) {
            acc[i][t] += __shfl_xor_sync(0xffffffffu, acc[i][t], 1u);
            acc[i][t] += __shfl_xor_sync(0xffffffffu, acc[i][t], 2u);
        }
    if (cc != 0) return;
#pragma unroll
    for (uint32_t i = 0; i < 2; i++) {
        const uint32_t row = i == 0 ? o0 : o0 + 8u;
        if (row >= out_dim) continue;
#pragma unroll
        for (uint32_t t = 0; t < 16; t++) {
            const uint32_t col = t0 + t;
            if (col >= n_tok) continue;
            const float v = round_out ? engram_bf16r(acc[i][t]) : acc[i][t];
            yg[(size_t) col * yr + row] = v;
        }
    }
}

// The bf16 twin of the same tile: a weight element is one half, so a lane reads 64 halves of each of its 2 rows
// per 256-element k block and scales 16 tokens with each.
__global__ void dense_matmul_bf16_tile_kernel(float* __restrict__ out, const uint16_t* __restrict__ w,
                                              const float* __restrict__ x, uint32_t in_dim, uint32_t out_dim,
                                              uint32_t n_tok, bool round_out) {
    __shared__ float xsh[kTileT][256];
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t ow = warp & 3u, tw = warp >> 2;
    const uint32_t rr = lane >> 2, cc = lane & 3u;
    const uint32_t o0 = blockIdx.x * kTileN + ow * 16u + rr;
    const uint32_t t0 = blockIdx.y * kTileT + tw * 16u;
    const uint32_t g0 = cc * 64u;
    const uint32_t r0 = o0 < out_dim ? o0 : out_dim - 1u;
    const uint32_t r1 = o0 + 8u < out_dim ? o0 + 8u : out_dim - 1u;
    const uint16_t* w0 = w + (size_t) r0 * in_dim;
    const uint16_t* w1 = w + (size_t) r1 * in_dim;
    float acc[2][16];
#pragma unroll
    for (uint32_t i = 0; i < 2; i++)
#pragma unroll
        for (uint32_t t = 0; t < 16; t++) acc[i][t] = 0.f;
    for (uint32_t kb = 0; kb < in_dim; kb += 256u) {
        __syncthreads();
        for (uint32_t i = tid; i < kTileT * 256u; i += kTileThreads) {
            const uint32_t tt = i >> 8, e = i & 255u;
            const uint32_t row = blockIdx.y * kTileT + tt;
            xsh[tt][e] = row < n_tok ? x[(size_t) row * in_dim + kb + e] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t l = 0; l < 64; l++) {
            const uint32_t k = kb + g0 + l;
            const float q0 = f32_from_bf16(w0[k]);
            const float q1 = f32_from_bf16(w1[k]);
#pragma unroll
            for (uint32_t t = 0; t < 16; t++) {
                const float xv = xsh[tw * 16u + t][g0 + l];
                acc[0][t] = fmaf(q0, xv, acc[0][t]);
                acc[1][t] = fmaf(q1, xv, acc[1][t]);
            }
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < 2; i++)
#pragma unroll
        for (uint32_t t = 0; t < 16; t++) {
            acc[i][t] += __shfl_xor_sync(0xffffffffu, acc[i][t], 1u);
            acc[i][t] += __shfl_xor_sync(0xffffffffu, acc[i][t], 2u);
        }
    if (cc != 0) return;
#pragma unroll
    for (uint32_t i = 0; i < 2; i++) {
        const uint32_t row = i == 0 ? o0 : o0 + 8u;
        if (row >= out_dim) continue;
#pragma unroll
        for (uint32_t t = 0; t < 16; t++) {
            const uint32_t col = t0 + t;
            if (col >= n_tok) continue;
            const float v = round_out ? engram_bf16r(acc[i][t]) : acc[i][t];
            out[(size_t) col * out_dim + row] = v;
        }
    }
}


// Set by deepseek4_dense_set_batched; the default path keeps the per-(row, token) kernels.
bool dense_batched = false;
bool dense_tc = false;
constexpr uint32_t kTileMinTok = 8;   // below this the per-row kernel is cheaper and the tile wastes lanes

unsigned grid_for(uint64_t n) {
    uint64_t g = (n + kThreads - 1) / kThreads;
    if (g == 0) g = 1;
    return (unsigned) (g > 2147483647ull ? 2147483647ull : g);
}

bool finish(cudaStream_t stream, const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "deepseek4_dense: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    if (stream == nullptr) {
        const cudaError_t s = cudaDeviceSynchronize();
        if (s != cudaSuccess) {
            std::fprintf(stderr, "deepseek4_dense: %s (sync): %s\n", what, cudaGetErrorString(s));
            return false;
        }
    }
    return true;
}

}  // namespace

void deepseek4_dense_set_batched(bool on) { dense_batched = on; }

void deepseek4_dense_set_tc(bool on) { dense_tc = on; }

bool deepseek4_rms_norm(float* d_out, const float* d_x, const float* d_w, uint32_t dim, uint32_t n_tok, float eps,
                        void* stream) {
    if (!d_out || !d_x || !d_w || dim == 0 || n_tok == 0) return false;
    dense_rms_norm_kernel<<<n_tok, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(d_out, d_x, d_w, dim, eps);
    return finish(static_cast<cudaStream_t>(stream), "rms_norm");
}

bool deepseek4_add(float* d_a, const float* d_b, uint64_t n, void* stream) {
    if (!d_a || !d_b || n == 0) return false;
    dense_add_kernel<<<grid_for(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(d_a, d_b, n);
    return finish(static_cast<cudaStream_t>(stream), "add");
}

bool deepseek4_round_bf16(float* d_x, uint64_t n, void* stream) {
    if (!d_x || n == 0) return false;
    dense_round_bf16_kernel<<<grid_for(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(d_x, n);
    return finish(static_cast<cudaStream_t>(stream), "round_bf16");
}

bool deepseek4_scale_round(float* d_x, uint64_t n, float s, void* stream) {
    if (!d_x || n == 0) return false;
    dense_scale_round_kernel<<<grid_for(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(d_x, n, s);
    return finish(static_cast<cudaStream_t>(stream), "scale_round");
}

bool deepseek4_expand_hc(float* d_hc, const float* d_x, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok,
                         void* stream) {
    if (!d_hc || !d_x || n_embd == 0 || n_hc == 0 || n_tok == 0) return false;
    dense_expand_hc_kernel<<<dim3((n_embd + kThreads - 1) / kThreads, n_tok), kThreads, 0,
                             static_cast<cudaStream_t>(stream)>>>(d_hc, d_x, n_embd, n_hc, n_tok);
    return finish(static_cast<cudaStream_t>(stream), "expand_hc");
}

bool deepseek4_swiglu(float* d_h, const float* d_gate, const float* d_up, uint64_t n, float limit, void* stream) {
    if (!d_h || !d_gate || !d_up || n == 0) return false;
    dense_swiglu_kernel<<<grid_for(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(d_h, d_gate, d_up, n,
                                                                                         limit);
    return finish(static_cast<cudaStream_t>(stream), "swiglu");
}

bool deepseek4_embed_q4k(float* d_out, const int32_t* d_tokens, const void* d_w, uint32_t n_vocab, uint32_t n_tok,
                         uint32_t dim, void* stream) {
    if (!d_out || !d_tokens || !d_w || n_vocab == 0 || n_tok == 0 || dim == 0 || dim % kDeepseek4Q4kBlock != 0)
        return false;
    dense_embed_q4k_kernel<<<grid_for((uint64_t) n_tok * dim), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        d_out, d_tokens, static_cast<const uint8_t*>(d_w), n_vocab, n_tok, dim);
    return finish(static_cast<cudaStream_t>(stream), "embed_q4k");
}

bool deepseek4_matmul_f32(float* d_out, const float* d_w, const float* d_x, uint32_t in_dim, uint32_t out_dim,
                          uint32_t n_tok, bool round_out, void* stream) {
    if (!d_out || !d_w || !d_x || in_dim == 0 || out_dim == 0 || n_tok == 0) return false;
    dense_matmul_f32_kernel<<<dim3(out_dim, n_tok), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        d_out, d_w, d_x, in_dim, out_dim, round_out);
    return finish(static_cast<cudaStream_t>(stream), "matmul_f32");
}

bool deepseek4_matmul_bf16(float* d_out, const uint16_t* d_w, const float* d_x, uint32_t in_dim, uint32_t out_dim,
                           uint32_t n_tok, bool round_out, void* stream) {
    if (!d_out || !d_w || !d_x || in_dim == 0 || out_dim == 0 || n_tok == 0) return false;
    if (dense_tc && n_tok >= kTileMinTok && (in_dim % 256u) == 0 &&
        deepseek4_matmul_tc_bf16(d_out, d_w, d_x, in_dim, out_dim, n_tok, round_out, stream))
        return true;
    if (dense_batched && n_tok >= kTileMinTok && (in_dim % 256u) == 0) {
        dense_matmul_bf16_tile_kernel<<<dim3((out_dim + kTileN - 1) / kTileN, (n_tok + kTileT - 1) / kTileT),
                                        kTileThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            d_out, d_w, d_x, in_dim, out_dim, n_tok, round_out);
        return finish(static_cast<cudaStream_t>(stream), "matmul_bf16 tile");
    }
    dense_matmul_bf16_kernel<<<dim3(out_dim, n_tok), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        d_out, d_w, d_x, in_dim, out_dim, round_out);
    return finish(static_cast<cudaStream_t>(stream), "matmul_bf16");
}

bool deepseek4_matmul_q4k(float* d_out, const void* d_w, const float* d_x, uint32_t in_dim, uint32_t out_dim,
                          uint32_t n_tok, bool round_out, void* stream) {
    if (!d_out || !d_w || !d_x || in_dim == 0 || in_dim % kDeepseek4Q4kBlock != 0 || out_dim == 0 || n_tok == 0)
        return false;
    if (dense_tc && n_tok >= kTileMinTok &&
        deepseek4_matmul_tc_q4k(d_out, d_w, d_x, in_dim, out_dim, n_tok, round_out, stream))
        return true;
    if (dense_batched && n_tok >= kTileMinTok) {
        dense_matmul_q4k_tile_kernel<false>
            <<<dim3((out_dim + kTileN - 1) / kTileN, (n_tok + kTileT - 1) / kTileT), kTileThreads, 0,
               static_cast<cudaStream_t>(stream)>>>(d_out, static_cast<const uint8_t*>(d_w), d_x,
                                                    in_dim / kDeepseek4Q4kBlock, out_dim, n_tok, round_out);
        return finish(static_cast<cudaStream_t>(stream), "matmul_q4k tile");
    }
    dense_matmul_q4k_kernel<<<dim3(out_dim, n_tok), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        d_out, static_cast<const uint8_t*>(d_w), d_x, in_dim / kDeepseek4Q4kBlock, out_dim, round_out);
    return finish(static_cast<cudaStream_t>(stream), "matmul_q4k");
}

bool deepseek4_matmul_q4k_grouped(float* d_out, const void* d_w, const float* d_x, uint32_t n_groups,
                                  uint32_t group_dim, uint32_t rank, uint32_t n_tok, bool round_out, void* stream) {
    if (!d_out || !d_w || !d_x || n_groups == 0 || group_dim == 0 || group_dim % kDeepseek4Q4kBlock != 0 ||
        rank == 0 || n_tok == 0)
        return false;
    if (dense_tc && n_tok >= kTileMinTok &&
        deepseek4_matmul_tc_q4k_grouped(d_out, d_w, d_x, n_groups, group_dim, rank, n_tok, round_out, stream))
        return true;
    if (dense_batched && n_tok >= kTileMinTok) {
        dense_matmul_q4k_tile_kernel<true>
            <<<dim3((rank + kTileN - 1) / kTileN, (n_tok + kTileT - 1) / kTileT, n_groups), kTileThreads, 0,
               static_cast<cudaStream_t>(stream)>>>(d_out, static_cast<const uint8_t*>(d_w), d_x, group_dim / 256u,
                                                    rank, n_tok, round_out);
        return finish(static_cast<cudaStream_t>(stream), "matmul_q4k_grouped tile");
    }
    dense_matmul_q4k_grouped_kernel<<<dim3(rank, n_tok, n_groups), kThreads, 0,
                                      static_cast<cudaStream_t>(stream)>>>(
        d_out, static_cast<const uint8_t*>(d_w), d_x, group_dim / kDeepseek4Q4kBlock, group_dim, rank, n_groups,
        round_out);
    return finish(static_cast<cudaStream_t>(stream), "matmul_q4k_grouped");
}

}  // namespace strata::kernels
