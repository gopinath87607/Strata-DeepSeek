// src/kernels/cuda/deepseek4_attn.cu - P3: attention primitives on the GPU.
//
// RoPE is the reference's kernel geometry exactly (cuda/ds41_attn.cuh:65-77): one 64-thread block per
// (head, token), thread i < n_rot/2 owning one complex pair in the last n_rot values of the head.  The
// frequency comes from the header's shared inline function, so the host oracle and this kernel compute the
// identical f32 angle; the two rotated values are bf16-rounded once, matching the reference's write-back into
// its bf16 tensor.
//
// Attention's remaining primitives (the compressor pool, the window ring, sparse attention, the indexer) join
// this file as they land; each keeps the same entry convention: device pointers, an optional stream (null =
// default stream, the call synchronizes), bad shapes refused before any launch, false on a launch error.
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

}  // namespace strata::kernels
