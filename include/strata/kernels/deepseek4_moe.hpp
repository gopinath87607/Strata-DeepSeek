// include/strata/kernels/deepseek4_moe.hpp - P3: the DeepSeek-V4.1-Flash MoE group on the GPU.
//
// The artifact's FFN, exactly as the reference engine composes it (ds4_ds41_forward.inc v41_moe,
// cuda/ds41_router.cuh, cuda/ds41_vq_decode.cuh):
//
//     scores = logits(x)                                   [n_tok][n_expert]      (the caller's matmul)
//     probs  = sqrt(softplus(scores))                      softplus threshold 20: z > 20 ? z : log1p(exp(z))
//     choose = topk(probs + bias)                          selection score; ties -> the smaller expert id
//     w      = probs[choose] / (sum + 1e-20) * route_scale weights from the UNbiased probs
//     y      = sum_k w[k] * Expert(choose[k], x) + SharedExpert(x)
//
// The routed experts are the VQ matrices of deepseek4_vq.hpp (one GGUF tensor per layer, 384 experts,
// gate/up/down each): gate and up are row-dotted directly off the packed index stream, SwiGLU folds them into
// the mid vector (bf16, clamp 10 on this artifact), and down row-dots the mid into this layer's partial.  The
// tail then folds sum_k w[k] * partial[k] + shared_output in ascending k and rounds once to bf16.
//
// THE SHARED EXPERT IS NOT HERE.  Its three matrices are q4_K, so it is composed from P3's dense primitives -
// deepseek4_matmul_q4k(round_out=true) for gate and up, deepseek4_swiglu for the mid, matmul_q4k for down -
// and the forward driver adds it via the tail below.  The reference does exactly that (the FFN composition
// calls the q4_K path for the shared expert, not the VQ path).
//
// WHAT IS AND IS NOT PORTED.  The kernels here are the reference's correctness arms: the router, and the
// router's top-k selection; the VQ worker's xpack, fused gate/up (rows from global codebook memory), down,
// reduce and tail.  The reference's performance arms for n_tok > 8 (the grouped multi-token kernel) and its
// shared-memory codebook copies are P7 work with identical arithmetic; they are not needed for a bit-exact
// gate.  n_tok is therefore accepted up to np = n_tok * topk <= 65535 (the 2D grid's y bound) with the same
// per-pair plain kernels; the reference's prefill arm is what lifts that bound, and it is P7's.
//
// The host functions below are the specification and the gate's oracle.  They emulate the device exactly:
// the activation is rounded to bf16 first, each row dot takes the reference's lane map (lane j takes index
// groups j, j + 32, ...), the 5-step xor tree is replayed in the same order, the gate and up dots are each
// rounded to bf16 before SwiGLU, the mid is stored as bf16, and the down dot is rounded to bf16.  The one
// expression they cannot share with the device is the libm's expf inside SwiGLU; the gate's tolerances cover
// that single rounding, and the reference fixture's one-hot cases make the chain single-term so it is exact.
#pragma once

#include <cstdint>

namespace strata::kernels {

// 384 experts is 12 per warp lane; the reference refuses more (V41_ROUTER_PER_LANE).
inline constexpr uint32_t kDeepseek4RouterPerLane = 12;
inline constexpr uint32_t kDeepseek4MoeMaxExperts = 32 * kDeepseek4RouterPerLane;
inline constexpr uint32_t kDeepseek4MoeMaxTopk = 16;

// ---- router -------------------------------------------------------------------------------------------

// sel[t][r], wts[t][r] from logits[t][n_expert] (the caller's matmul output, f32) and the text bias.
// bias_vl/vl_rows implement the official `torch.where(image_mask, bias_vl, bias)`: vl_rows[t] >= 0 selects
// bias_vl for token t; either pointer null (or a null row entry) selects bias.  route_scale is the
// artifact's expert_weights_scale (1.5).  Refuses topk > 16 or n_expert > 384 (nothing is written, false).
bool deepseek4_router_host(int32_t* sel, float* wts, const float* logits, const float* bias, const float* bias_vl,
                           const int32_t* vl_rows, uint32_t n_tok, uint32_t n_expert, uint32_t topk,
                           float route_scale);

// Device entry; all pointers in device memory.  Same guards.  n_tok == 0 is a no-op success.
bool deepseek4_router(int32_t* d_sel, float* d_wts, const float* d_logits, const float* d_bias,
                      const float* d_bias_vl, const int32_t* d_vl_rows, uint32_t n_tok, uint32_t n_expert,
                      uint32_t topk, float route_scale, void* stream);

// ---- the routed VQ experts ----------------------------------------------------------------------------

// out[t][o] = sum_k w[t][k] * partial[t*topk + k][o], where partial[pair][o] is the bf16-rounded down row of
// the expert sel[pair] (pair = t*topk + k), in ascending k.  A negative sel entry, or a payload the parser
// refuses, writes a ZERO mid row and a zero partial: the reference leaves the previous call's scratch in
// that slot; a valid model never produces either, so this is our determinism choice, made the same way by
// the oracle and the kernels so the gate can compare them.
//
// Device entry.  `d_blob`/`blob_size` is the layer's DQVL blob in device memory; `nc` is its codebook size
// and `blob_v3` its payload version (1 for DQV3, 0 for DQVQ), from the layer's GGUF metadata - the kernel
// specialization needs them before any expert payload is parsed, as the reference's worker takes nc.
// `d_out != nullptr`: the reduce is launched and d_partial may be null.  `d_out == nullptr`: the per-pair
// partials are LEFT in d_partial (np*out_dim f32) for deepseek4_moe_tail, and d_partial must be non-null.
// `d_gain_override` is the optional per-expert down-row gain override [n_expert][out_dim], f32, or null.
// The entry allocates and frees its own scratch (the packed activation and the mid), so it synchronizes the
// given stream before returning; it is a correctness entry, not P7's performance path.
bool deepseek4_routed_moe_host(float* out, const uint8_t* blob, uint64_t blob_size, uint32_t in_dim,
                               uint32_t mid_dim, uint32_t out_dim, const int32_t* sel, const float* w,
                               uint32_t topk, float clamp, const float* x, uint32_t n_tok,
                               const float* gain_override);

bool deepseek4_routed_moe(float* d_out, float* d_partial, const void* d_blob, uint64_t blob_size,
                          uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim, const int32_t* d_sel,
                          const float* d_w, uint32_t topk, float clamp, const float* d_x, uint32_t n_tok,
                          uint32_t nc, uint32_t blob_v3, const float* d_gain_override, void* stream);

// ---- the tail -----------------------------------------------------------------------------------------

// y[t][o] = bf16r(sum_k w[t][k] * partial[t*topk + k][o] + so[t][o]) - the reference's four launches folded
// into one, with the arithmetic untouched (same k order, then + so, then one bf16 round).
void deepseek4_moe_tail_host(float* y, const float* partial, const float* w, const float* so, uint32_t n_tok,
                             uint32_t topk, uint32_t out_dim);

bool deepseek4_moe_tail(float* d_y, const float* d_partial, const float* d_w, const float* d_so,
                        uint32_t n_tok, uint32_t topk, uint32_t out_dim, void* stream);

}  // namespace strata::kernels
