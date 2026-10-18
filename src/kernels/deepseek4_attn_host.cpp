// src/kernels/deepseek4_attn_host.cpp - P3: attention primitives' host oracles.
//
// RoPE's specification is the reference engine's v41_rope_kernel, written out here loop for loop: one complex
// pair at a time, the angle from the shared frequency function, the inverse rotation negating the sine, and the
// two outputs each bf16-rounded exactly once.  The frequency is the header's inline function, so host and device
// cannot disagree about the ramp; only their libm differs.
//
// The quantized-KV group's oracles are the same shape: the compressor pool's per-dimension softmax
// (cuda/ds41_attn.cuh:90-99) and the block quant, pack and unpack steps (cuda/ds41_attn.cuh:175-256) are
// written with the header's shared inline conversions, so only the pool's expf can differ between the sides.
#include "strata/kernels/deepseek4_attn.hpp"

#include <cmath>

namespace strata::kernels {

namespace {

float attn_bf16r(float x) { return strata::deepseek4::engram_bf16r(x); }

void kv_pack_rows(uint8_t* cache, uint32_t g0, const float* rows, uint32_t n_rows, uint32_t dim, uint32_t blk,
                  uint32_t nib_bytes, uint32_t row_bytes, int mode) {
    const uint32_t nb_row = dim / blk;
    for (uint32_t r = 0; r < n_rows; r++) {
        uint8_t* row = cache + (size_t)(g0 + r) * row_bytes;
        const float* p = rows + (size_t) r * dim;
        for (uint32_t b = 0; b < nb_row; b++) {
            const float* qb = p + (size_t) b * blk;
            float a = 0.0f;
            for (uint32_t l = 0; l < blk; l++) a = fmaxf(a, fabsf(qb[l]));
            const float s = deepseek4_quant_scale(a, mode);
            row[nib_bytes + b] = mode == kDeepseek4QuantFp4E4m3 ? deepseek4_e4m3_byte(s)
                                                               : deepseek4_e8m0_byte(s);
            for (uint32_t l = 0; l < blk; l += 2) {
                const uint8_t lo = deepseek4_fp4_to_nibble(qb[l] / s);
                const uint8_t hi = deepseek4_fp4_to_nibble(qb[l + 1] / s);
                row[(b * blk + l) >> 1] = (uint8_t)(lo | (hi << 4));
            }
        }
        for (uint32_t k = nib_bytes + nb_row; k < row_bytes; k++) row[k] = 0;   // index key's pad
    }
}

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
                xr[2u * i] = attn_bf16r(a * c - b * s);
                xr[2u * i + 1u] = attn_bf16r(a * s + b * c);
            }
        }
    }
}

void deepseek4_compress_pool_host(float* out, const float* kv, const float* score, uint32_t n_tok, uint32_t ratio,
                                  uint32_t dim) {
    if (!out || !kv || !score || ratio == 0 || dim == 0) return;
    const uint32_t ng = n_tok / ratio;
    for (uint32_t g = 0; g < ng; g++) {
        for (uint32_t d = 0; d < dim; d++) {
            float mx = -INFINITY;
            for (uint32_t t = 0; t < ratio; t++) mx = fmaxf(mx, score[((size_t) g * ratio + t) * dim + d]);
            float den = 0.0f, acc = 0.0f;
            for (uint32_t t = 0; t < ratio; t++) {
                const float e = expf(score[((size_t) g * ratio + t) * dim + d] - mx);
                den += e;
                acc += e * kv[((size_t) g * ratio + t) * dim + d];
            }
            out[(size_t) g * dim + d] = attn_bf16r(acc / den);
        }
    }
}

void deepseek4_act_quant_host(float* x, uint32_t n_rows, uint32_t dim, uint32_t block, int mode) {
    if (!x || block == 0 || block > 32u || (dim % block) != 0 || dim == 0 || mode < 0 || mode > 2) return;
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t b = 0; b < dim / block; b++) {
            float* p = x + (size_t) row * dim + (size_t) b * block;
            float a = 0.0f;
            for (uint32_t l = 0; l < block; l++) a = fmaxf(a, fabsf(p[l]));
            const float s = deepseek4_quant_scale(a, mode);
            for (uint32_t l = 0; l < block; l++) p[l] = deepseek4_quant_value(p[l], s, mode);
        }
    }
}

void deepseek4_ckv_pack_host(uint8_t* cache, uint32_t g0, const float* rows, uint32_t n_rows) {
    if (!cache || !rows || n_rows == 0) return;
    kv_pack_rows(cache, g0, rows, n_rows, kDeepseek4CkvDim, kDeepseek4CkvBlock, kDeepseek4CkvNibbleBytes,
                 kDeepseek4CkvBytes, kDeepseek4QuantFp4E4m3);
}

void deepseek4_idxk_pack_host(uint8_t* cache, uint32_t g0, const float* rows, uint32_t n_rows) {
    if (!cache || !rows || n_rows == 0) return;
    kv_pack_rows(cache, g0, rows, n_rows, kDeepseek4IdxkDim, kDeepseek4IdxkBlock, kDeepseek4IdxkNibbleBytes,
                 kDeepseek4IdxkBytes, kDeepseek4QuantFp4E8m0);
}

void deepseek4_ckv_unpack_host(const uint8_t* cache, uint32_t n_rows, float* out) {
    if (!cache || !out) return;
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint8_t* row = cache + (size_t) r * kDeepseek4CkvBytes;
        float* o = out + (size_t) r * kDeepseek4CkvDim;
        for (uint32_t d = 0; d < kDeepseek4CkvDim; d++) o[d] = deepseek4_ckv_get(row, d);
    }
}

void deepseek4_idxk_unpack_host(const uint8_t* cache, uint32_t n_rows, float* out) {
    if (!cache || !out) return;
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint8_t* row = cache + (size_t) r * kDeepseek4IdxkBytes;
        float* o = out + (size_t) r * kDeepseek4IdxkDim;
        for (uint32_t d = 0; d < kDeepseek4IdxkDim; d++) o[d] = deepseek4_idxk_get(row, d);
    }
}

}  // namespace strata::kernels
