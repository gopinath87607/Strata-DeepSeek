// src/kernels/deepseek4_hc_host.cpp - P3: the hyper-connection host oracle (the CUDA gate's reference).
//
// Same semantics as the device kernels and the reference engine's cuda_v41_hc.inc.cu, in f32 with the bf16
// round at each module boundary.  The accumulation is a plain ascending loop: the device uses parallel trees,
// so the gate compares with a tolerance around this oracle rather than bit for bit (the reference's own fused
// and unfused kernels differ the same way and are gated by quality, not cmp).
#include "strata/kernels/deepseek4_hc.hpp"

#include <cmath>
#include <cstring>

#include "strata/artifact/deepseek4_engram_bits.hpp"

namespace strata::kernels {

namespace {

using strata::deepseek4::engram_bf16r;

float sigmoidf_(float z) { return 1.0f / (1.0f + std::exp(-z)); }

// The Sinkhorn split for one token: the reference's hc_split_sinkhorn, including the +eps on the softmax and on
// every normalization, and the alternating row/column passes that follow the initial column pass.
void split_one(const float* m, const float* scale, const float* base, uint32_t hc, uint32_t iters, float eps,
               float* pre, float* post, float* comb) {
    for (uint32_t k = 0; k < hc; k++) {
        pre[k] = sigmoidf_(m[k] * scale[0] + base[k]) + eps;
        post[k] = 2.0f * sigmoidf_(m[hc + k] * scale[1] + base[hc + k]);
    }
    float c[64];
    for (uint32_t i = 0; i < hc * hc; i++) c[i] = m[2u * hc + i] * scale[2] + base[2u * hc + i];
    for (uint32_t j = 0; j < hc; j++) {
        float mx = -INFINITY;
        for (uint32_t k = 0; k < hc; k++) mx = std::fmax(mx, c[j * hc + k]);
        float s = 0.0f;
        for (uint32_t k = 0; k < hc; k++) {
            c[j * hc + k] = std::exp(c[j * hc + k] - mx);
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
    for (uint32_t i = 0; i < hc * hc; i++) comb[i] = c[i];
}

// hc_pre + the RMS norm for one token.  The mean is over the bf16-ROUNDED x values (the reference's order: the
// round is part of hc_pre, and the norm consumes its output).
void pre_one(const float* hcn, const float* pin, const float* nw, uint32_t e, uint32_t hc, float eps, float* x,
             float* xn) {
    float acc = 0.0f;
    for (uint32_t d = 0; d < e; d++) {
        float v = 0.0f;
        for (uint32_t k = 0; k < hc; k++) v += pin[k] * hcn[(uint64_t)k * e + d];
        v = engram_bf16r(v);
        x[d] = v;
        acc += v * v;
    }
    const float inv = 1.0f / std::sqrt(acc / static_cast<float>(e) + eps);
    for (uint32_t d = 0; d < e; d++) xn[d] = engram_bf16r(nw[d] * (x[d] * inv));
}

}  // namespace

float deepseek4_hc_bf16r(float x) { return engram_bf16r(x); }

void deepseek4_hc_mix_host(const float* fn, const float* hc, float* mix, uint32_t n_embd, uint32_t n_hc,
                           uint32_t n_tok, float eps) {
    const uint32_t dim = n_embd * n_hc, mix_hc = 2u * n_hc + n_hc * n_hc;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float* x = hc + static_cast<uint64_t>(t) * dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < dim; i++) s += x[i] * x[i];
        const float inv = 1.0f / std::sqrt(s / static_cast<float>(dim) + eps);
        for (uint32_t o = 0; o < mix_hc; o++) {
            const float* w = fn + static_cast<uint64_t>(o) * dim;
            float acc = 0.0f;
            for (uint32_t c = 0; c < dim; c++) acc += w[c] * x[c];
            mix[static_cast<uint64_t>(t) * mix_hc + o] = acc * inv;
        }
    }
}

void deepseek4_hc_split_host(const float* mix, const float* scale, const float* base, uint32_t n_hc,
                             uint32_t iters, float hc_eps, float* pre, float* post, float* comb, uint32_t n_tok) {
    if (n_hc > 8) return;
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    for (uint32_t t = 0; t < n_tok; t++) {
        split_one(mix + static_cast<uint64_t>(t) * mix_hc, scale, base, n_hc, iters, hc_eps,
                  pre + static_cast<uint64_t>(t) * n_hc, post + static_cast<uint64_t>(t) * n_hc,
                  comb + static_cast<uint64_t>(t) * n_hc * n_hc);
    }
}

void deepseek4_hc_fused_host(const float* mix, const float* scale, const float* base, const float* pre_in,
                             const float* hc, const float* norm_w, uint32_t n_embd, uint32_t n_hc, uint32_t iters,
                             float hc_eps, float norm_eps, float* pre, float* post, float* comb, float* x,
                             float* xn, uint32_t n_tok) {
    if (n_hc > 8) return;
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    for (uint32_t t = 0; t < n_tok; t++) {
        split_one(mix + static_cast<uint64_t>(t) * mix_hc, scale, base, n_hc, iters, hc_eps,
                  pre + static_cast<uint64_t>(t) * n_hc, post + static_cast<uint64_t>(t) * n_hc,
                  comb + static_cast<uint64_t>(t) * n_hc * n_hc);
        pre_one(hc + static_cast<uint64_t>(t) * n_hc * n_embd, pre_in + static_cast<uint64_t>(t) * n_hc, norm_w,
                n_embd, n_hc, norm_eps, x + static_cast<uint64_t>(t) * n_embd,
                xn + static_cast<uint64_t>(t) * n_embd);
    }
}

void deepseek4_hc_pre_host(const float* hc, const float* pre, const float* norm_w, float* x, float* xn,
                           uint32_t n_embd, uint32_t n_hc, float norm_eps, uint32_t n_tok) {
    for (uint32_t t = 0; t < n_tok; t++)
        pre_one(hc + static_cast<uint64_t>(t) * n_hc * n_embd, pre + static_cast<uint64_t>(t) * n_hc, norm_w,
                n_embd, n_hc, norm_eps, x + static_cast<uint64_t>(t) * n_embd,
                xn + static_cast<uint64_t>(t) * n_embd);
}

void deepseek4_hc_post_host(const float* y, const float* res, const float* post, const float* comb, float* out,
                            uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    for (uint32_t t = 0; t < n_tok; t++) {
        const float* yt = y + static_cast<uint64_t>(t) * n_embd;
        const float* pt = post + static_cast<uint64_t>(t) * n_hc;
        const float* ct = comb + static_cast<uint64_t>(t) * n_hc * n_hc;
        for (uint32_t d = 0; d < n_embd; d++) {
            const float yv = yt[d];
            for (uint32_t k = 0; k < n_hc; k++) {
                float s = pt[k] * yv;
                for (uint32_t j = 0; j < n_hc; j++) s += ct[j * n_hc + k] * res[(static_cast<uint64_t>(t) * n_hc + j) * n_embd + d];
                out[(static_cast<uint64_t>(t) * n_hc + k) * n_embd + d] = engram_bf16r(s);
            }
        }
    }
}

}  // namespace strata::kernels
