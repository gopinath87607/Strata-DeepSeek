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
    dense_matmul_bf16_kernel<<<dim3(out_dim, n_tok), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        d_out, d_w, d_x, in_dim, out_dim, round_out);
    return finish(static_cast<cudaStream_t>(stream), "matmul_bf16");
}

bool deepseek4_matmul_q4k(float* d_out, const void* d_w, const float* d_x, uint32_t in_dim, uint32_t out_dim,
                          uint32_t n_tok, bool round_out, void* stream) {
    if (!d_out || !d_w || !d_x || in_dim == 0 || in_dim % kDeepseek4Q4kBlock != 0 || out_dim == 0 || n_tok == 0)
        return false;
    dense_matmul_q4k_kernel<<<dim3(out_dim, n_tok), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        d_out, static_cast<const uint8_t*>(d_w), d_x, in_dim / kDeepseek4Q4kBlock, out_dim, round_out);
    return finish(static_cast<cudaStream_t>(stream), "matmul_q4k");
}

bool deepseek4_matmul_q4k_grouped(float* d_out, const void* d_w, const float* d_x, uint32_t n_groups,
                                  uint32_t group_dim, uint32_t rank, uint32_t n_tok, bool round_out, void* stream) {
    if (!d_out || !d_w || !d_x || n_groups == 0 || group_dim == 0 || group_dim % kDeepseek4Q4kBlock != 0 ||
        rank == 0 || n_tok == 0)
        return false;
    dense_matmul_q4k_grouped_kernel<<<dim3(rank, n_tok, n_groups), kThreads, 0,
                                      static_cast<cudaStream_t>(stream)>>>(
        d_out, static_cast<const uint8_t*>(d_w), d_x, group_dim / kDeepseek4Q4kBlock, group_dim, rank, n_groups,
        round_out);
    return finish(static_cast<cudaStream_t>(stream), "matmul_q4k_grouped");
}

}  // namespace strata::kernels
