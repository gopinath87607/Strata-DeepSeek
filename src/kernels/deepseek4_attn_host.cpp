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
#include <vector>

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

void deepseek4_sparse_attn_host(float* o, const float* q, const float* kvw, const uint8_t* kvc, const int32_t* idx,
                                const float* sink, uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng,
                                uint32_t topk, uint32_t n_head, uint32_t head_dim, float scale, uint32_t full_block,
                                bool ring, uint32_t win_lo) {
    if (!o || !q || !kvw || !sink || n_tok == 0 || n_head == 0 || head_dim == 0 || head_dim > 512 ||
        (head_dim % 32u) != 0 || (n_head % kDeepseek4AttnHeadsPerBlock) != 0)
        return;
    if ((kvc == nullptr) != (idx == nullptr)) return;
    if (kvc == nullptr) topk = 0;
    if (full_block == 0 && !ring) return;   // the entry's contract: the main path's window is a ring
    if (full_block > n_tok) return;
    if (win_lo > pos0) return;              // the clamp would pass the last query: the caller's window is short

    const uint32_t per = head_dim / 32u;
    const uint32_t ktile = kDeepseek4AttnKeyTile;
    std::vector<float> ks((size_t) ktile * head_dim);
    std::vector<float> acc(head_dim), acc_add(head_dim);
    std::vector<int> kok(ktile);

    for (uint32_t i = 0; i < n_tok; i++) {
        const uint32_t p = pos0 + i;
        const uint32_t last = full_block ? pos0 + full_block - 1u : p;
        uint32_t lo = full_block ? (pos0 > window ? pos0 - window : 0u)
                                 : (p + 1u > window ? p + 1u - window : 0u);
        if (lo < win_lo) lo = win_lo;
        const uint32_t nwin = last - lo + 1u;
        const uint32_t nkeys = nwin + topk;
        for (uint32_t h = 0; h < n_head; h++) {
            const float* qh = q + ((size_t) i * n_head + h) * head_dim;
            float mx = -1e30f, sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) acc[d] = 0.0f;
            for (uint32_t base = 0; base < nkeys; base += ktile) {
                const uint32_t nt = (nkeys - base) < ktile ? (nkeys - base) : ktile;
                for (uint32_t t = 0; t < nt; t++) {
                    const uint32_t kk = base + t;
                    const float* krow = nullptr;
                    const uint8_t* cpk = nullptr;
                    if (kk < nwin) {
                        krow = kvw + (size_t) deepseek4_win_row((int64_t) lo + kk, pos0, window, ring ? 1u : 0u) *
                                          head_dim;
                    } else {
                        const int32_t g = idx[(size_t) i * topk + (kk - nwin)];
                        if (g >= 0 && (uint32_t) g < ng) cpk = kvc + (size_t) g * kDeepseek4CkvBytes;
                    }
                    kok[t] = (krow != nullptr || cpk != nullptr) ? 1 : 0;
                    for (uint32_t d = 0; d < head_dim; d++)
                        ks[(size_t) t * head_dim + d] = krow ? krow[d] : (cpk ? deepseek4_ckv_get(cpk, d) : 0.0f);
                }
                float st[kDeepseek4AttnKeyTile];
                float tm = -1e30f;
                for (uint32_t t = 0; t < nt; t++) {
                    // The kernel's 32-lane dot: each lane sums `per` contiguous dims, then the xor tree.
                    float part[32], next[32];
                    for (uint32_t l = 0; l < 32u; l++) {
                        float a = 0.0f;
                        for (uint32_t e = 0; e < per; e++)
                            a += qh[l * per + e] * ks[(size_t) t * head_dim + l * per + e];
                        part[l] = a;
                    }
                    for (int off = 16; off > 0; off >>= 1) {
                        for (uint32_t l = 0; l < 32u; l++) next[l] = part[l] + part[l ^ (uint32_t) off];
                        for (uint32_t l = 0; l < 32u; l++) part[l] = next[l];
                    }
                    const float dot = part[0];
                    st[t] = kok[t] ? dot * scale : -1e30f;
                    tm = fmaxf(tm, st[t]);
                }
                const float nm = fmaxf(mx, tm);
                const float rs = expf(mx - nm);
                float ps = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) acc_add[d] = 0.0f;
                for (uint32_t t = 0; t < nt; t++) {
                    const float pv = expf(st[t] - nm);
                    ps += pv;
                    const float pb = attn_bf16r(pv);
                    for (uint32_t d = 0; d < head_dim; d++)
                        acc_add[d] += pb * ks[(size_t) t * head_dim + d];
                }
                sum = sum * rs + ps;
                for (uint32_t d = 0; d < head_dim; d++) acc[d] = acc[d] * rs + acc_add[d];
                mx = nm;
            }
            const float den = sum + expf(sink[h] - mx);
            float* oh = o + ((size_t) i * n_head + h) * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) oh[d] = attn_bf16r(acc[d] / den);
        }
    }
}

void deepseek4_win_commit_host(float* win, uint32_t pos0, uint32_t n, uint32_t window, uint32_t head_dim,
                               const int32_t* pos) {
    if (!win || n == 0 || window == 0 || head_dim == 0) return;
    if (pos) pos0 = (uint32_t) pos[0];
    const uint32_t i0 = n > window ? n - window : 0u;
    for (uint32_t i = i0; i < n; i++) {
        const float* src = win + (size_t)(window + i) * head_dim;
        float* dst = win + (size_t)((pos0 + i) % window) * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) dst[d] = src[d];
    }
}

void deepseek4_win_ring_snap_host(float* win, float* snap, uint32_t pos0, uint32_t i0, uint32_t n, uint32_t window,
                                  uint32_t head_dim, bool back, const int32_t* pos) {
    if (!win || !snap || window == 0 || head_dim == 0 || n <= i0) return;
    if (pos) pos0 = (uint32_t) pos[0];
    for (uint32_t i = i0; i < n; i++) {
        float* ring = win + (size_t)((pos0 + i) % window) * head_dim;
        float* sp = snap + (size_t) i * head_dim;
        if (back) {
            for (uint32_t d = 0; d < head_dim; d++) ring[d] = sp[d];
        } else {
            for (uint32_t d = 0; d < head_dim; d++) sp[d] = ring[d];
        }
    }
}

}  // namespace strata::kernels
