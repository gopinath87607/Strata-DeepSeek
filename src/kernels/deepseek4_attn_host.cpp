// src/kernels/deepseek4_attn_host.cpp - P3: attention primitives' host oracles.
//
// RoPE's specification is the reference engine's v41_rope_kernel, written out here loop for loop: one complex
// pair at a time, the angle from the shared frequency function, the inverse rotation negating the sine, and the
// two outputs each bf16-rounded exactly once.  The frequency is the header's inline function, so host and device
// cannot disagree about the ramp; only their libm differs.
#include "strata/kernels/deepseek4_attn.hpp"

#include <cmath>

namespace strata::kernels {

namespace {

float rope_bf16r(float x) { return strata::deepseek4::engram_bf16r(x); }

}  // namespace

void deepseek4_rope_host(float* x, const int32_t* pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                         uint32_t n_rot, float theta, uint32_t original_seq_len, float factor, float beta_fast,
                         float beta_slow, bool inverse) {
    if (n_rot == 0) return;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t h = 0; h < n_head; h++) {
            float* xr = x + ((size_t) t * n_head + h) * head_dim + (head_dim - n_rot);
            for (uint32_t i = 0; i < n_rot / 2u; i++) {
                const float ang = (float) pos[t] * deepseek4_rope_freq(i, n_rot, theta, original_seq_len, factor,
                                                                       beta_fast, beta_slow);
                float c = std::cos(ang), s = std::sin(ang);
                if (inverse) s = -s;
                const float a = xr[2u * i], b = xr[2u * i + 1u];
                xr[2u * i] = rope_bf16r(a * c - b * s);
                xr[2u * i + 1u] = rope_bf16r(a * s + b * c);
            }
        }
    }
}

}  // namespace strata::kernels
