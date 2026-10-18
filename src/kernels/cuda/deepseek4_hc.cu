// src/kernels/cuda/deepseek4_hc.cu - P3: DeepSeek-V4.1-Flash hyper-connections on the GPU.
//
// The semantics are stated once in include/strata/kernels/deepseek4_hc.hpp and implemented twice: by the host
// oracle in src/kernels/deepseek4_hc_host.cpp and by these kernels.  This file follows the reference engine's
// cuda_v41_hc.inc.cu element for element - the same sigmoid, the same max-subtracted row softmax, the same
// +hc_eps on the softmax and on every normalization, and the same alternating row/column Sinkhorn passes - but
// with its own launch geometry (the reference folds the mix's three launches and the split/pre/norm pair into
// fused kernels for latency; correctness does not depend on that and P7 can add it).
//
// The bf16 round is engram_bf16r, the engine's v41_bf16r: the P4 gate already proved it bit-exact.
#include "strata/kernels/deepseek4_hc.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#include "strata/artifact/deepseek4_engram_bits.hpp"

namespace strata::kernels {

namespace {

using strata::deepseek4::engram_bf16r;

__device__ __forceinline__ float hc_sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

__device__ __forceinline__ void block_sum(float* sh, float v, uint32_t tid) {
    sh[tid] = v;
    __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) sh[tid] += sh[tid + k];
        __syncthreads();
    }
}

// The Sinkhorn split, thread 0 serially: the reference's order over the 4x4 matrix (row softmax + eps, column
// pass, then iters-1 alternating row/column passes).  A 4x4 matrix does not need a block; the serial form is
// also the one the oracle implements.
__device__ void hc_split_serial(float* c, uint32_t hc, uint32_t iters, float eps) {
    for (uint32_t j = 0; j < hc; j++) {
        float mx = -INFINITY;
        for (uint32_t k = 0; k < hc; k++) mx = fmaxf(mx, c[j * hc + k]);
        float s = 0.0f;
        for (uint32_t k = 0; k < hc; k++) {
            c[j * hc + k] = expf(c[j * hc + k] - mx);
            s += c[j * hc + k];
        }
        for (uint32_t k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / s + eps;
    }
    for (uint32_t k = 0; k < hc; k++) {
        float s = 0.0f;
        for (uint32_t j = 0; j < hc; j++) s += c[j * hc + k];
        for (uint32_t j = 0; j < hc; j++) c[j * hc + k] /= (s + eps);
    }
    for (uint32_t it = 1; it < iters; it++) {
        for (uint32_t j = 0; j < hc; j++) {
            float s = 0.0f;
            for (uint32_t k = 0; k < hc; k++) s += c[j * hc + k];
            for (uint32_t k = 0; k < hc; k++) c[j * hc + k] /= (s + eps);
        }
        for (uint32_t k = 0; k < hc; k++) {
            float s = 0.0f;
            for (uint32_t j = 0; j < hc; j++) s += c[j * hc + k];
            for (uint32_t j = 0; j < hc; j++) c[j * hc + k] /= (s + eps);
        }
    }
}

__global__ void hc_rms_kernel(float* inv, const float* x, uint32_t dim, float eps) {
    const uint32_t t = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + static_cast<uint64_t>(t) * dim;
    float s = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float sh[256];
    block_sum(sh, s, tid);
    if (tid == 0) inv[t] = rsqrtf(sh[0] / static_cast<float>(dim) + eps);
}

__global__ void hc_mix_kernel(float* mix, const float* fn, const float* hc, const float* inv, uint32_t dim,
                              uint32_t mix_hc) {
    const uint32_t o = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const float* w = fn + static_cast<uint64_t>(o) * dim;
    const float* x = hc + static_cast<uint64_t>(t) * dim;
    float acc = 0.0f;
    for (uint32_t c = tid; c < dim; c += blockDim.x) acc += w[c] * x[c];
    __shared__ float sh[256];
    block_sum(sh, acc, tid);
    if (tid == 0) mix[static_cast<uint64_t>(t) * mix_hc + o] = sh[0] * inv[t];
}

__global__ void hc_split_kernel(float* pre, float* post, float* comb, const float* mix, const float* scale,
                                const float* base, uint32_t hc, uint32_t iters, float eps) {
    const uint32_t t = blockIdx.x, tid = threadIdx.x, mix_hc = 2u * hc + hc * hc;
    const float* m = mix + static_cast<uint64_t>(t) * mix_hc;
    __shared__ float c[64];
    if (tid < hc) {
        pre[static_cast<uint64_t>(t) * hc + tid] = hc_sigmoid(m[tid] * scale[0] + base[tid]) + eps;
        post[static_cast<uint64_t>(t) * hc + tid] = 2.0f * hc_sigmoid(m[hc + tid] * scale[1] + base[hc + tid]);
    }
    if (tid < hc * hc) c[tid] = m[2u * hc + tid] * scale[2] + base[2u * hc + tid];
    __syncthreads();
    if (tid == 0) hc_split_serial(c, hc, iters, eps);
    __syncthreads();
    if (tid < hc * hc) comb[static_cast<uint64_t>(t) * hc * hc + tid] = c[tid];
}

// Split + hc_pre + RMS norm for one token.  pre_in is the PREDECESSOR's pre (the reference's note: using this
// layer's own pre here shifts every layer's mixing and still looks plausible).
__global__ void hc_fused_kernel(float* pre, float* post, float* comb, float* x, float* xn, const float* mix,
                                const float* hc, const float* pre_in, const float* scale, const float* base,
                                const float* nw, uint32_t e, uint32_t hc_, uint32_t iters, float hc_eps,
                                float norm_eps) {
    const uint32_t t = blockIdx.x, tid = threadIdx.x, mix_hc = 2u * hc_ + hc_ * hc_;
    const float* m = mix + static_cast<uint64_t>(t) * mix_hc;
    const float* hcn = hc + static_cast<uint64_t>(t) * hc_ * e;
    const float* pin = pre_in + static_cast<uint64_t>(t) * hc_;
    float* xr = x + static_cast<uint64_t>(t) * e;
    float* on = xn + static_cast<uint64_t>(t) * e;
    __shared__ float c[64];
    __shared__ float sh[256];

    float acc = 0.0f;
    for (uint32_t d = tid; d < e; d += blockDim.x) {
        float v = 0.0f;
        for (uint32_t k = 0; k < hc_; k++) v += pin[k] * hcn[static_cast<uint64_t>(k) * e + d];
        v = engram_bf16r(v);
        xr[d] = v;
        acc += v * v;
    }
    if (tid < hc_) {
        pre[static_cast<uint64_t>(t) * hc_ + tid] = hc_sigmoid(m[tid] * scale[0] + base[tid]) + hc_eps;
        post[static_cast<uint64_t>(t) * hc_ + tid] = 2.0f * hc_sigmoid(m[hc_ + tid] * scale[1] + base[hc_ + tid]);
    }
    if (tid < hc_ * hc_) c[tid] = m[2u * hc_ + tid] * scale[2] + base[2u * hc_ + tid];
    __syncthreads();
    if (tid == 0) hc_split_serial(c, hc_, iters, hc_eps);
    __syncthreads();
    if (tid < hc_ * hc_) comb[static_cast<uint64_t>(t) * hc_ * hc_ + tid] = c[tid];

    block_sum(sh, acc, tid);
    const float inv = rsqrtf(sh[0] / static_cast<float>(e) + norm_eps);
    for (uint32_t d = tid; d < e; d += blockDim.x) on[d] = engram_bf16r(nw[d] * (xr[d] * inv));
}

__global__ void hc_pre_kernel(float* x, float* xn, const float* hc, const float* pre, const float* nw, uint32_t e,
                              uint32_t hc_, float eps) {
    const uint32_t t = blockIdx.x, tid = threadIdx.x;
    const float* hcn = hc + static_cast<uint64_t>(t) * hc_ * e;
    const float* pin = pre + static_cast<uint64_t>(t) * hc_;
    float* xr = x + static_cast<uint64_t>(t) * e;
    float* on = xn + static_cast<uint64_t>(t) * e;
    float acc = 0.0f;
    for (uint32_t d = tid; d < e; d += blockDim.x) {
        float v = 0.0f;
        for (uint32_t k = 0; k < hc_; k++) v += pin[k] * hcn[static_cast<uint64_t>(k) * e + d];
        v = engram_bf16r(v);
        xr[d] = v;
        acc += v * v;
    }
    __shared__ float sh[256];
    block_sum(sh, acc, tid);
    const float inv = rsqrtf(sh[0] / static_cast<float>(e) + eps);
    for (uint32_t d = tid; d < e; d += blockDim.x) on[d] = engram_bf16r(nw[d] * (xr[d] * inv));
}

__global__ void hc_post_kernel(float* out, const float* y, const float* res, const float* post, const float* comb,
                               uint32_t e, uint32_t hc) {
    const uint32_t t = blockIdx.y;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= e) return;
    const float yv = y[static_cast<uint64_t>(t) * e + d];
    const float* pt = post + static_cast<uint64_t>(t) * hc;
    const float* ct = comb + static_cast<uint64_t>(t) * hc * hc;
    for (uint32_t k = 0; k < hc; k++) {
        float s = pt[k] * yv;
        for (uint32_t j = 0; j < hc; j++)
            s += ct[j * hc + k] * res[(static_cast<uint64_t>(t) * hc + j) * e + d];
        out[(static_cast<uint64_t>(t) * hc + k) * e + d] = engram_bf16r(s);
    }
}

bool finish(cudaStream_t stream, const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "deepseek4_hc: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    if (stream == nullptr) {
        const cudaError_t s = cudaDeviceSynchronize();
        if (s != cudaSuccess) {
            std::fprintf(stderr, "deepseek4_hc: %s (sync): %s\n", what, cudaGetErrorString(s));
            return false;
        }
    }
    return true;
}

}  // namespace

bool deepseek4_hc_mix(const float* d_fn, const float* d_hc, float* d_mix, uint32_t n_embd, uint32_t n_hc,
                      uint32_t n_tok, float eps, void* stream) {
    if (!d_fn || !d_hc || !d_mix || n_embd == 0 || n_hc == 0 || n_tok == 0 || n_hc > 8) return false;
    const uint32_t dim = n_embd * n_hc, mix_hc = 2u * n_hc + n_hc * n_hc;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    float* inv = nullptr;
    const cudaError_t me = cudaMalloc(&inv, static_cast<size_t>(n_tok) * 4);
    if (me != cudaSuccess) {
        std::fprintf(stderr, "deepseek4_hc: mix inv alloc: %s\n", cudaGetErrorString(me));
        return false;
    }
    hc_rms_kernel<<<n_tok, 256, 0, s>>>(inv, d_hc, dim, eps);
    if (!finish(s, "hc rms")) { cudaFree(inv); return false; }
    hc_mix_kernel<<<dim3(mix_hc, n_tok), 256, 0, s>>>(d_mix, d_fn, d_hc, inv, dim, mix_hc);
    const bool ok = finish(s, "hc mix");
    cudaFree(inv);
    return ok;
}

bool deepseek4_hc_split(const float* d_mix, const float* d_scale, const float* d_base, float* d_pre,
                        float* d_post, float* d_comb, uint32_t n_hc, uint32_t iters, float hc_eps, uint32_t n_tok,
                        void* stream) {
    if (!d_mix || !d_scale || !d_base || !d_pre || !d_post || !d_comb || n_hc == 0 || n_hc > 8 || n_tok == 0)
        return false;
    hc_split_kernel<<<n_tok, 64, 0, static_cast<cudaStream_t>(stream)>>>(d_pre, d_post, d_comb, d_mix, d_scale,
                                                                          d_base, n_hc, iters, hc_eps);
    return finish(static_cast<cudaStream_t>(stream), "hc split");
}

bool deepseek4_hc_fused(const float* d_mix, const float* d_scale, const float* d_base, const float* d_pre_in,
                        const float* d_hc, const float* d_norm_w, float* d_pre, float* d_post, float* d_comb,
                        float* d_x, float* d_xn, uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps,
                        float norm_eps, uint32_t n_tok, void* stream) {
    if (!d_mix || !d_scale || !d_base || !d_pre_in || !d_hc || !d_norm_w || !d_pre || !d_post || !d_comb ||
        !d_x || !d_xn || n_embd == 0 || n_hc == 0 || n_hc > 8 || n_tok == 0)
        return false;
    hc_fused_kernel<<<n_tok, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        d_pre, d_post, d_comb, d_x, d_xn, d_mix, d_hc, d_pre_in, d_scale, d_base, d_norm_w, n_embd, n_hc, iters,
        hc_eps, norm_eps);
    return finish(static_cast<cudaStream_t>(stream), "hc fused");
}

bool deepseek4_hc_pre(const float* d_hc, const float* d_pre, const float* d_norm_w, float* d_x, float* d_xn,
                      uint32_t n_embd, uint32_t n_hc, float norm_eps, uint32_t n_tok, void* stream) {
    if (!d_hc || !d_pre || !d_norm_w || !d_x || !d_xn || n_embd == 0 || n_hc == 0 || n_hc > 8 || n_tok == 0)
        return false;
    hc_pre_kernel<<<n_tok, 256, 0, static_cast<cudaStream_t>(stream)>>>(d_x, d_xn, d_hc, d_pre, d_norm_w, n_embd,
                                                                          n_hc, norm_eps);
    return finish(static_cast<cudaStream_t>(stream), "hc pre");
}

bool deepseek4_hc_post(const float* d_y, const float* d_res, const float* d_post, const float* d_comb,
                       float* d_out, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, void* stream) {
    if (!d_y || !d_res || !d_post || !d_comb || !d_out || d_out == d_res || n_embd == 0 || n_hc == 0 ||
        n_hc > 8 || n_tok == 0)
        return false;
    hc_post_kernel<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        d_out, d_y, d_res, d_post, d_comb, n_embd, n_hc);
    return finish(static_cast<cudaStream_t>(stream), "hc post");
}

}  // namespace strata::kernels
