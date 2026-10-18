// include/strata/kernels/deepseek4_attn.hpp - P3: attention primitives.
//
// The first one is RoPE, which every attention path runs: the q path and the window kv at the layer's base,
// the compressor's pooled key and the indexer's key at the compressor base, and the attention output through
// the inverse rotation.  This is the reference engine's cuda/ds41_attn.cuh:52-94, which in turn is the official
// precompute_freqs_cis + apply_rotary_emb:
//
//   * adjacent elements are one complex number, and only the LAST `n_rot` values of each head rotate
//     (head_dim 512 with n_rot 64 leaves 448 untouched; the indexer key is head_dim 128 with the same 64);
//   * f(i) = theta^(-2i/n_rot), with the YaRN ramp applied when original_seq_len > 0 (compressor layers):
//     low/high corrected dimensions from ln(original_seq_len/(beta*2pi)), a linear ramp between them, and
//     f = f/factor outside the ramp, f inside it; the base is then 160000 rather than 10000;
//   * angle = position * f, and the rotated pair is written with one bf16 round:
//     (a*c - b*s, a*s + b*c), the inverse rotation negating s.
//
// The frequency function is shared between the host oracle and the kernel, so both compute the same f32 angle
// from the same position and only their libm `cosf`/`sinf` can differ.
#pragma once

#include <cmath>
#include <cstdint>

#include "strata/artifact/deepseek4_engram_bits.hpp"

namespace strata::kernels {

/// The reference's v41_rope_freq (cuda/ds41_attn.cuh:52-70).  `rot_dim` is n_rot, not head_dim: the exponent
/// and the YaRN ramp are both over the rotated width.  `osl == 0` disables YaRN.
STRATA_DSV4_HD inline float deepseek4_rope_freq(uint32_t i, uint32_t rot_dim, float theta, uint32_t original_seq_len,
                                                float factor, float beta_fast, float beta_slow) {
    constexpr float kPi = 3.14159265358979323846f;
    float f = 1.0f / powf(theta, (float) (2u * i) / (float) rot_dim);
    if (original_seq_len > 0u) {
        const float lt = 2.0f * logf(theta);
        float low = floorf((float) rot_dim * logf((float) original_seq_len / (beta_fast * 2.0f * kPi)) / lt);
        float high = ceilf((float) rot_dim * logf((float) original_seq_len / (beta_slow * 2.0f * kPi)) / lt);
        low = fmaxf(low, 0.0f);
        high = fminf(high, (float) (rot_dim - 1u));
        float ramp = ((float) i - low) / fmaxf(high - low, 1e-3f);
        ramp = fminf(fmaxf(ramp, 0.0f), 1.0f);
        const float smooth = 1.0f - ramp;
        f = f / factor * (1.0f - smooth) + f * smooth;
    }
    return f;
}

/// Rotates the last `n_rot` values of each head in place.  x is [n_tok][n_head][head_dim] f32 (values on the
/// bf16 grid, as every producer of this tensor rounds them), pos is [n_tok].  The rotated pair is bf16-rounded
/// once, matching the reference's write-back into its bf16 tensor.
void deepseek4_rope_host(float* x, const int32_t* pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                         uint32_t n_rot, float theta, uint32_t original_seq_len, float factor, float beta_fast,
                         float beta_slow, bool inverse);

/// The GPU form.  Refuses odd n_rot, n_rot > 128 or n_rot > head_dim (the reference's entry refuses the first
/// two; the third would write outside the head), and any null pointer or empty shape.  n_rot == 0 is a no-op.
bool deepseek4_rope(float* d_x, const int32_t* d_pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                    uint32_t n_rot, float theta, uint32_t original_seq_len, float factor, float beta_fast,
                    float beta_slow, bool inverse, void* stream);

}  // namespace strata::kernels
