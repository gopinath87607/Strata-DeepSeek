// src/kernels/deepseek4_dense_host.cpp - P3: the dense primitives' host oracles.
//
// This is the specification the CUDA kernels in src/kernels/cuda/deepseek4_dense.cu are compared against, and the
// place the reference engine's exact order of operations is written down in readable form:
//
//   * rms_norm computes the mean over the f32 row, adds eps, takes rsqrt, multiplies the weight *inside* the
//     bf16 round (`bf16r(w * (x * inv))`), exactly as v41_rms_norm does;
//   * swiglu clamps up to [-limit, +limit] and gate only from above, then rounds the product once;
//   * the matmuls accumulate in double here (the device accumulates f32 with a block reduction; the gate's
//     tolerance covers the order difference) and round to the bf16 grid only when the caller asks.
//
// The q4_K decode itself is not repeated here: it is the one inline decoder in deepseek4_dense.hpp that the
// device side also runs.
#include "strata/kernels/deepseek4_dense.hpp"

#include <cmath>
#include <cstring>

namespace strata::kernels {

namespace {

float row_sum_sq(const float* x, uint32_t dim) {
    double s = 0.0;
    for (uint32_t i = 0; i < dim; i++) s += (double) x[i] * (double) x[i];
    return (float) (s / (double) dim);
}

float silu(float g) { return g / (1.0f + std::exp(-g)); }

// The three full matmuls share one loop: `wval(o, i)` returns skeleton row o's element i in f32.  Double
// accumulation keeps the oracle's own error out of the comparison; the device's f32 result is what the
// tolerance is for.
template <typename WVal>
void matmul_host_impl(float* out, const float* x, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                      bool round_out, WVal wval) {
    for (uint32_t t = 0; t < n_tok; t++) {
        const float* xr = x + (size_t) t * in_dim;
        for (uint32_t o = 0; o < out_dim; o++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < in_dim; i++) acc += (double) wval(o, i) * (double) xr[i];
            const float v = (float) acc;
            out[(size_t) t * out_dim + o] = round_out ? deepseek4_dense_bf16r(v) : v;
        }
    }
}

}  // namespace

void deepseek4_rms_norm_host(float* out, const float* x, const float* w, uint32_t dim, uint32_t n_tok, float eps) {
    for (uint32_t t = 0; t < n_tok; t++) {
        const float* xr = x + (size_t) t * dim;
        float* orow = out + (size_t) t * dim;
        const float inv = 1.0f / std::sqrt(row_sum_sq(xr, dim) + eps);
        for (uint32_t d = 0; d < dim; d++) orow[d] = deepseek4_dense_bf16r(w[d] * (xr[d] * inv));
    }
}

void deepseek4_swiglu_host(float* h, const float* gate, const float* up, uint64_t n, float limit) {
    for (uint64_t i = 0; i < n; i++) {
        float g = gate[i], u = up[i];
        if (limit > 0.0f) {
            g = std::fmin(g, limit);
            u = std::fmin(std::fmax(u, -limit), limit);
        }
        h[i] = deepseek4_dense_bf16r(silu(g) * u);
    }
}

void deepseek4_q4k_row_host(const uint8_t* row_blocks, uint32_t dim, float* out) {
    for (uint32_t i = 0; i < dim; i++)
        out[i] = deepseek4_q4k_element(row_blocks + (size_t) (i / kDeepseek4Q4kBlock) * kDeepseek4Q4kBytes,
                                       i % kDeepseek4Q4kBlock);
}

void deepseek4_embed_q4k_host(float* out, const int32_t* tok, const uint8_t* w, uint32_t n_vocab, uint32_t n_tok,
                              uint32_t dim) {
    const size_t row_bytes = (size_t) (dim / kDeepseek4Q4kBlock) * kDeepseek4Q4kBytes;
    for (uint32_t t = 0; t < n_tok; t++) {
        float* orow = out + (size_t) t * dim;
        const int32_t id = tok[t];
        if (id < 0 || (uint32_t) id >= n_vocab) {
            std::memset(orow, 0, (size_t) dim * sizeof(float));
            continue;
        }
        deepseek4_q4k_row_host(w + (size_t) id * row_bytes, dim, orow);
    }
}

void deepseek4_matmul_f32_host(float* out, const float* w, const float* x, uint32_t in_dim, uint32_t out_dim,
                               uint32_t n_tok, bool round_out) {
    matmul_host_impl(out, x, in_dim, out_dim, n_tok, round_out,
                     [w, in_dim](uint32_t o, uint32_t i) { return w[(size_t) o * in_dim + i]; });
}

void deepseek4_matmul_bf16_host(float* out, const uint16_t* w, const float* x, uint32_t in_dim, uint32_t out_dim,
                                uint32_t n_tok, bool round_out) {
    matmul_host_impl(out, x, in_dim, out_dim, n_tok, round_out, [w, in_dim](uint32_t o, uint32_t i) {
        return f32_from_bf16(w[(size_t) o * in_dim + i]);
    });
}

void deepseek4_matmul_q4k_host(float* out, const uint8_t* w, const float* x, uint32_t in_dim, uint32_t out_dim,
                               uint32_t n_tok, bool round_out) {
    const uint32_t blocks = in_dim / kDeepseek4Q4kBlock;
    const size_t row_bytes = (size_t) blocks * kDeepseek4Q4kBytes;
    matmul_host_impl(out, x, in_dim, out_dim, n_tok, round_out, [w, row_bytes](uint32_t o, uint32_t i) {
        return deepseek4_q4k_element(w + (size_t) o * row_bytes + (size_t) (i / kDeepseek4Q4kBlock) * kDeepseek4Q4kBytes,
                                     i % kDeepseek4Q4kBlock);
    });
}

void deepseek4_matmul_q4k_grouped_host(float* out, const uint8_t* w, const float* x, uint32_t n_groups,
                                       uint32_t group_dim, uint32_t rank, uint32_t n_tok, bool round_out) {
    const uint32_t blocks = group_dim / kDeepseek4Q4kBlock;
    const size_t group_bytes = (size_t) rank * blocks * kDeepseek4Q4kBytes;
    const size_t row_bytes = (size_t) blocks * kDeepseek4Q4kBytes;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t g = 0; g < n_groups; g++) {
            const float* xg = x + ((size_t) t * n_groups + g) * group_dim;
            const uint8_t* wg = w + (size_t) g * group_bytes;
            for (uint32_t o = 0; o < rank; o++) {
                double acc = 0.0;
                for (uint32_t i = 0; i < group_dim; i++)
                    acc += (double) deepseek4_q4k_element(wg + (size_t) o * row_bytes +
                                                              (size_t) (i / kDeepseek4Q4kBlock) * kDeepseek4Q4kBytes,
                                                          i % kDeepseek4Q4kBlock) *
                           (double) xg[i];
                const float v = (float) acc;
                out[((size_t) t * n_groups + g) * rank + o] =
                    round_out ? deepseek4_dense_bf16r(v) : v;
            }
        }
    }
}

}  // namespace strata::kernels
