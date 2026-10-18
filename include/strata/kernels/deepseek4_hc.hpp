// include/strata/kernels/deepseek4_hc.hpp - P3: DeepSeek-V4.1-Flash hyper-connections (mHC) on the GPU.
//
// The model replaces each residual stream with n_hc lanes and mixes them per sublayer: an f32 projection of the
// flattened lanes (`hc_mix`), a 4-route Sinkhorn split into a pre/post scale and a route-combination matrix, a
// pre-mix + RMS norm feeding the sublayer, and a post-mix writing the sublayer output back into every lane.
// The semantics are the reference engine's (ds4-dfm-rs cuda/ds41_hc.cuh, itself the official model.py
// hc_mixes / hc_split_sinkhorn / hc_pre / hc_post): f32 compute with a bf16 round at each module boundary.
//
// The host functions in this header are the test oracle AND the specification: the device kernels are gated
// against them, and both implement the same order of operations the reference documents:
//
//   hc_mix    inv = rsqrt(mean(hc^2) + eps); mix = inv * (hc @ fn^T),   fn is [2*hc + hc*hc][E*hc] f32
//   split     pre  = sigmoid(mix[0..hc) * scale[0] + base[0..hc)) + hc_eps
//             post = 2 * sigmoid(mix[hc..2hc) * scale[1] + base[hc..2hc))
//             comb = softmax_rows(mix[2hc..) * scale[2] + base[2hc..) + hc_eps, then column-normalized by
//                    (column sum + hc_eps), then `iters-1` alternating row/column normalizations.  Every
//                    normalization divides by (sum + hc_eps); the row softmax is max-subtracted.
//   hc_pre    x[d]  = bf16r(sum_k pre_in[k] * hc[k][d])
//             xn[d] = bf16r(norm_w[d] * (x[d] * rsqrt(mean(x^2) + norm_eps)))
//   hc_post   out[k][d] = bf16r(post[k] * y[d] + sum_j comb[j*hc + k] * res[j][d])
//
// `fused` computes split, pre and the norm in one pass, and takes the PREDECESSOR's pre (pre_in) for its hc_pre -
// the reference's warning: using this layer's own pre there does not error, it shifts every layer's mixing one
// step and still looks plausible.  The device entry points take device pointers and an optional stream; a null
// stream uses the default stream and synchronizes on error-free launches.
#pragma once

#include <cstdint>

namespace strata::kernels {

// ---- host reference (the gate's oracle) ----

/// bf16 round-to-nearest-even of an f32, Inf and NaN untouched (the engine's v41_bf16r).
float deepseek4_hc_bf16r(float x);

/// mix[t][o] = rsqrt(mean_d(hc[t][d]^2) + eps) * sum_d fn[o][d] * hc[t][d].
/// `fn` is f32 [2*n_hc + n_hc*n_hc][n_embd*n_hc], row-major; `hc` is f32 [n_tok][n_embd*n_hc].
void deepseek4_hc_mix_host(const float* fn, const float* hc, float* mix, uint32_t n_embd, uint32_t n_hc,
                           uint32_t n_tok, float eps);

/// Split only (the reference's sinkhorn path).
void deepseek4_hc_split_host(const float* mix, const float* scale, const float* base, uint32_t n_hc,
                             uint32_t iters, float hc_eps, float* pre, float* post, float* comb, uint32_t n_tok);

/// Split + hc_pre + rms norm in one pass (the reference's fused kernel).  `pre_in` is the predecessor's pre.
void deepseek4_hc_fused_host(const float* mix, const float* scale, const float* base, const float* pre_in,
                             const float* hc, const float* norm_w, uint32_t n_embd, uint32_t n_hc, uint32_t iters,
                             float hc_eps, float norm_eps, float* pre, float* post, float* comb, float* x,
                             float* xn, uint32_t n_tok);

/// Unfused hc_pre: x/xn from hc and pre (no split).
void deepseek4_hc_pre_host(const float* hc, const float* pre, const float* norm_w, float* x, float* xn,
                           uint32_t n_embd, uint32_t n_hc, float norm_eps, uint32_t n_tok);

/// hc_post: out[k][d] = bf16r(post[k] * y[d] + sum_j comb[j*hc + k] * res[j][d]).  `out` must not alias `res`.
void deepseek4_hc_post_host(const float* y, const float* res, const float* post, const float* comb, float* out,
                            uint32_t n_embd, uint32_t n_hc, uint32_t n_tok);

// ---- device entry points ----
// All pointers are device pointers; weights are f32 (uploaded from the GGUF's F32 tensors).  Each returns false
// on a bad shape or a launch error, launching nothing for a bad shape.  `stream` may be null (default stream).

bool deepseek4_hc_mix(const float* d_fn, const float* d_hc, float* d_mix, uint32_t n_embd, uint32_t n_hc,
                      uint32_t n_tok, float eps, void* stream);

bool deepseek4_hc_split(const float* d_mix, const float* d_scale, const float* d_base, float* d_pre,
                        float* d_post, float* d_comb, uint32_t n_hc, uint32_t iters, float hc_eps,
                        uint32_t n_tok, void* stream);

bool deepseek4_hc_fused(const float* d_mix, const float* d_scale, const float* d_base, const float* d_pre_in,
                        const float* d_hc, const float* d_norm_w, float* d_pre, float* d_post, float* d_comb,
                        float* d_x, float* d_xn, uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps,
                        float norm_eps, uint32_t n_tok, void* stream);

bool deepseek4_hc_pre(const float* d_hc, const float* d_pre, const float* d_norm_w, float* d_x, float* d_xn,
                      uint32_t n_embd, uint32_t n_hc, float norm_eps, uint32_t n_tok, void* stream);

bool deepseek4_hc_post(const float* d_y, const float* d_res, const float* d_post, const float* d_comb,
                       float* d_out, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, void* stream);

}  // namespace strata::kernels
