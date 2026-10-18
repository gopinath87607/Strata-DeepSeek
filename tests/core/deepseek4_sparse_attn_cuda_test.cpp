// tests/core/deepseek4_sparse_attn_cuda_test.cpp - P3's sparse attention gate.
//
//   deepseek4_sparse_attn_cuda_test
//
// The kernel is the reference engine's own scalar fallback (cuda/ds41_attn.cuh:304-370): the causal SWA window
// in ascending position order, then the topk compressed keys; keys staged through shared 8 at a time; the
// 32-lane xor dot; the bf16-rounded probability; invalid topk slots zeroed and scored -1e30; the sink in the
// denominator only.  The host oracle in src/kernels/deepseek4_attn_host.cpp writes the same loop out for one
// thread, including the device's per-lane partial sums and xor reduction tree, so only libm's expf may differ
// between the two sides.
//
// The exact fixtures make the arithmetic exact on both sides (q = 0 leaves one key's own value; the sink
// makes the denominator 1 or 2), so those comparisons are bitwise.  The window-ring and full-block fixtures
// compare against an independent double-precision softmax over the intended buffer rows, which checks the row
// mapping rather than just host-vs-device agreement.  Invalid topk entries are proved equivalent to topk = 0
// bitwise, in a fixture whose key count crosses the 8-key tile boundary.  The random fixtures compare the
// kernel to the host oracle with the usual bf16 tolerance (2e-2 relative, plus the bf16-grid check).
// No model, no timing.  No GPU: prints SKIP and exits 77.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/deepseek4_attn.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        g_checks++;                                                              \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_failures++;                                                        \
        }                                                                        \
    } while (0)

uint32_t g_rng = 0x9e3779b9u;
uint32_t urng() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
float urand() { return static_cast<float>(urng() & 0x00FFFFFFu) / 8388608.0f - 1.0f; }
float u01() { return 0.5f * (urand() + 1.0f); }
float bf16_value() { return strata::kernels::f32_from_bf16(strata::kernels::bf16_from_f32(urand())); }

void check_same_bits(const char* what, const std::vector<float>& got, const std::vector<float>& want) {
    CHECK(got.size() == want.size());
    size_t bad = 0;
    for (size_t i = 0; i < got.size() && i < want.size(); i++) {
        uint32_t a = 0, b = 0;
        std::memcpy(&a, &got[i], 4);
        std::memcpy(&b, &want[i], 4);
        if (a != b) {
            if (bad < 4)
                std::fprintf(stderr, "  %s value %zu: gpu %.9g (0x%08x) oracle %.9g (0x%08x)\n", what, i, got[i], a,
                             want[i], b);
            bad++;
        }
    }
    g_checks++;
    if (bad != 0) g_failures++;
    std::printf("  %-22s %zu values, %zu bits differ\n", what, got.size(), bad);
}

void check_close(const char* what, const std::vector<float>& got, const std::vector<float>& want, double rel,
                 double abs_tol) {
    CHECK(got.size() == want.size());
    size_t bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < got.size() && i < want.size(); i++) {
        const double a = got[i], b = want[i];
        const double diff = std::fabs(a - b);
        const double bound = abs_tol + rel * std::fmax(std::fabs(a), std::fabs(b));
        const double ratio = diff / bound;
        if (ratio > worst) worst = ratio;
        if (ratio > 1.0) {
            if (bad < 4)
                std::fprintf(stderr, "  %s value %zu: gpu %.8g oracle %.8g (diff %.3g, bound %.3g)\n", what, i, a,
                             b, diff, bound);
            bad++;
        }
    }
    g_checks++;
    if (bad != 0) g_failures++;
    std::printf("  %-22s %zu values, worst %.3f x tolerance, %zu over\n", what, got.size(), worst, bad);
}

void check_bf16_grid(const char* what, const std::vector<float>& v) {
    size_t off = 0;
    for (float x : v) {
        if (strata::deepseek4::engram_bf16r(x) != x) {
            if (off < 4) std::fprintf(stderr, "  %s value off the bf16 grid: %.9g\n", what, x);
            off++;
        }
    }
    g_checks++;
    if (off != 0) g_failures++;
    std::printf("  %-22s %zu values, %zu off the bf16 grid\n", what, v.size(), off);
}

float* dev_f(size_t n) {
    float* p = nullptr;
    CHECK(cudaMalloc(&p, n * sizeof(float)) == cudaSuccess);
    return p;
}
float* dev_f_from(const std::vector<float>& v) {
    float* p = dev_f(v.size());
    CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}
template <typename T>
T* dev_from(const std::vector<T>& v) {
    T* p = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&p), v.size() * sizeof(T)) == cudaSuccess);
    CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}
void to_host(const float* d, std::vector<float>& v) {
    CHECK(cudaMemcpy(v.data(), d, v.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
}

// One query's output against a double-precision softmax over the named buffer rows, the sink in the
// denominator - an independent check that the kernel gathered the rows the geometry says it should.
void check_double_query(const char* what, const float* o, const float* q, const float* kvw,
                        const std::vector<uint32_t>& rows, const float* sink, uint32_t n_head, uint32_t hd,
                        float scale) {
    size_t bad = 0;
    double worst = 0.0;
    for (uint32_t h = 0; h < n_head; h++) {
        std::vector<double> dot(rows.size());
        double mx = -INFINITY;
        for (size_t t = 0; t < rows.size(); t++) {
            double a = 0.0;
            for (uint32_t d = 0; d < hd; d++)
                a += static_cast<double>(q[(size_t) h * hd + d]) * static_cast<double>(kvw[(size_t) rows[t] * hd + d]);
            dot[t] = a * scale;
            mx = std::fmax(mx, dot[t]);
        }
        std::vector<double> w(rows.size());
        double den = std::exp(static_cast<double>(sink[h]) - mx);
        for (size_t t = 0; t < rows.size(); t++) {
            w[t] = std::exp(dot[t] - mx);
            den += w[t];
        }
        for (uint32_t d = 0; d < hd; d++) {
            double acc = 0.0;
            for (size_t t = 0; t < rows.size(); t++)
                acc += w[t] * static_cast<double>(kvw[(size_t) rows[t] * hd + d]);
            const double want = acc / den;
            const double a = o[(size_t) h * hd + d];
            const double diff = std::fabs(a - want);
            const double bound = 1e-6 + 2e-2 * std::fmax(std::fabs(a), std::fabs(want));
            const double ratio = diff / bound;
            if (ratio > worst) worst = ratio;
            if (ratio > 1.0) {
                if (bad < 4) std::fprintf(stderr, "  %s h %u d %u: gpu %.8g double %.8g\n", what, h, d, a, want);
                bad++;
            }
        }
    }
    g_checks++;
    if (bad != 0) g_failures++;
    std::printf("  %-22s %zu values, worst %.3f x double reference, %zu over\n", what, (size_t) n_head * hd, worst,
                bad);
}

// ---- exact fixtures ----

void single_window_key_case() {
    const uint32_t hd = 64, nh = 8, win = 1;
    std::vector<float> kvw((size_t)(win + 1) * hd);
    for (float& v : kvw) v = bf16_value();
    std::vector<float> q((size_t) nh * hd, 0.0f);
    std::vector<float> sink(nh, -100.0f);
    std::vector<float> got((size_t) nh * hd);

    float* d_kvw = dev_f_from(kvw);
    float* d_q = dev_f_from(q);
    float* d_sink = dev_f_from(sink);
    float* d_o = dev_f(got.size());

    // q = 0 and one window key: pv = 1, pb = 1, acc = the key, den = 1 + exp(-100) = 1.
    CHECK(strata::kernels::deepseek4_sparse_attn(d_o, d_q, d_kvw, nullptr, nullptr, d_sink, 1, 0, win, 0, 0, nh, hd,
                                                 0.1f, 0, true, 0, nullptr));
    to_host(d_o, got);
    std::vector<float> want((size_t) nh * hd);
    for (uint32_t h = 0; h < nh; h++)
        for (uint32_t d = 0; d < hd; d++) want[(size_t) h * hd + d] = kvw[(size_t) win * hd + d];
    check_same_bits("one key sink -100", got, want);

    // sink = 0: den = 2, so the output is exactly the key halved.
    std::vector<float> zero_sink(nh, 0.0f);
    CHECK(cudaMemcpy(d_sink, zero_sink.data(), zero_sink.size() * sizeof(float), cudaMemcpyHostToDevice) ==
          cudaSuccess);
    CHECK(strata::kernels::deepseek4_sparse_attn(d_o, d_q, d_kvw, nullptr, nullptr, d_sink, 1, 0, win, 0, 0, nh, hd,
                                                 0.1f, 0, true, 0, nullptr));
    to_host(d_o, got);
    for (uint32_t h = 0; h < nh; h++)
        for (uint32_t d = 0; d < hd; d++)
            want[(size_t) h * hd + d] = strata::deepseek4::engram_bf16r(kvw[(size_t) win * hd + d] * 0.5f);
    check_same_bits("one key sink 0", got, want);

    cudaFree(d_kvw);
    cudaFree(d_q);
    cudaFree(d_sink);
    cudaFree(d_o);
}

void single_compressed_key_case() {
    const uint32_t hd = strata::kernels::kDeepseek4CkvDim, nh = 8;
    std::vector<float> rows(strata::kernels::kDeepseek4CkvDim);
    for (float& v : rows) v = 0.75f + u01();
    std::vector<uint8_t> cache(strata::kernels::kDeepseek4CkvBytes, 0);
    strata::kernels::deepseek4_ckv_pack_host(cache.data(), 0, rows.data(), 1);
    std::vector<float> key(strata::kernels::kDeepseek4CkvDim);
    strata::kernels::deepseek4_ckv_unpack_host(cache.data(), 1, key.data());

    std::vector<float> q((size_t) nh * hd, 0.0f);
    std::vector<float> sink(nh, 0.0f);
    std::vector<int32_t> idx(1, 0);
    std::vector<float> got((size_t) nh * hd);

    // window = 0: the compressed keys are the whole key set; q = 0 and sink = 0 make den = 2.
    float* d_kvw = dev_f(1);   // window 0 needs no rows
    float* d_q = dev_f_from(q);
    float* d_sink = dev_f_from(sink);
    uint8_t* d_kvc = dev_from(cache);
    int32_t* d_idx = dev_from(idx);
    float* d_o = dev_f(got.size());
    CHECK(strata::kernels::deepseek4_sparse_attn(d_o, d_q, d_kvw, reinterpret_cast<const uint8_t*>(d_kvc),
                                                 reinterpret_cast<const int32_t*>(d_idx), d_sink, 1, 0, 0, 1, 1, nh,
                                                 hd, 0.1f, 0, true, 0, nullptr));
    to_host(d_o, got);
    std::vector<float> want((size_t) nh * hd);
    for (uint32_t h = 0; h < nh; h++)
        for (uint32_t d = 0; d < hd; d++) want[(size_t) h * hd + d] = strata::deepseek4::engram_bf16r(key[d] * 0.5f);
    check_same_bits("one compressed key", got, want);

    cudaFree(d_kvw);
    cudaFree(d_q);
    cudaFree(d_sink);
    cudaFree(d_kvc);
    cudaFree(d_idx);
    cudaFree(d_o);
}

// ---- geometry ----

void ring_map_case() {
    const uint32_t hd = 64, nh = 8, win = 4, pos0 = 6, n_tok = 1;
    std::vector<float> kvw((size_t)(win + n_tok) * hd);
    for (float& v : kvw) v = 0.75f + u01();
    std::vector<float> q((size_t) nh * hd);
    for (float& v : q) v = urand();
    std::vector<float> sink(nh);
    for (float& v : sink) v = urand();
    std::vector<float> got((size_t) nh * hd);

    float* d_kvw = dev_f_from(kvw);
    float* d_q = dev_f_from(q);
    float* d_sink = dev_f_from(sink);
    float* d_o = dev_f(got.size());
    CHECK(strata::kernels::deepseek4_sparse_attn(d_o, d_q, d_kvw, nullptr, nullptr, d_sink, n_tok, pos0, win, 0, 0,
                                                 nh, hd, 0.1f, 0, true, 0, nullptr));
    to_host(d_o, got);

    // p = 6, window 4: positions 3..6; a < pos0 keeps a % window, a >= pos0 lands at window + i.
    const std::vector<uint32_t> rows = {3, 0, 1, 4};
    check_double_query("ring map", got.data(), q.data(), kvw.data(), rows, sink.data(), nh, hd, 0.1f);

    std::vector<float> want = got;
    strata::kernels::deepseek4_sparse_attn_host(want.data(), q.data(), kvw.data(), nullptr, nullptr, sink.data(),
                                               n_tok, pos0, win, 0, 0, nh, hd, 0.1f, 0, true, 0);
    check_close("ring map oracle", got, want, 2e-2, 1e-6);
    check_bf16_grid("ring map grid", got);

    cudaFree(d_kvw);
    cudaFree(d_q);
    cudaFree(d_sink);
    cudaFree(d_o);
}

void full_block_case() {
    const uint32_t hd = 64, nh = 8, win = 133, pos0 = 10, n_tok = 5;
    std::vector<float> kvw((size_t)(win + n_tok) * hd);
    for (float& v : kvw) v = 0.75f + u01();
    std::vector<float> q1((size_t) nh * hd);
    for (float& v : q1) v = urand();
    std::vector<float> q((size_t) n_tok * nh * hd);
    for (uint32_t i = 0; i < n_tok; i++) std::memcpy(&q[(size_t) i * nh * hd], q1.data(), q1.size() * sizeof(float));
    std::vector<float> sink(nh);
    for (float& v : sink) v = urand();
    std::vector<float> got(q.size());

    float* d_kvw = dev_f_from(kvw);
    float* d_q = dev_f_from(q);
    float* d_sink = dev_f_from(sink);
    float* d_o = dev_f(got.size());
    // The DSpark draft block: full_block = n_tok, a linear (non-ring) window segment, every query sees the
    // same key set, so replicated q rows must produce bitwise-identical outputs.
    CHECK(strata::kernels::deepseek4_sparse_attn(d_o, d_q, d_kvw, nullptr, nullptr, d_sink, n_tok, pos0, win, 0, 0,
                                                 nh, hd, 0.1f, n_tok, false, 0, nullptr));
    to_host(d_o, got);

    size_t differ = 0;
    for (uint32_t i = 1; i < n_tok; i++)
        for (size_t k = 0; k < (size_t) nh * hd; k++)
            if (std::memcmp(&got[(size_t) i * nh * hd + k], &got[k], 4) != 0) differ++;
    g_checks++;
    if (differ != 0) g_failures++;
    std::printf("  %-22s %zu values, %zu differ between replicated queries\n", "full block rows",
                (size_t)(n_tok - 1) * nh * hd, differ);

    // Positions 0..pos0+n_tok-1: a < pos0 is at window + a - pos0, a >= pos0 at window + a - pos0 too (the
    // same linear segment), so the rows are window - pos0 .. window - pos0 + pos0 + n_tok - 1.
    std::vector<uint32_t> rows;
    for (uint32_t a = 0; a < pos0 + n_tok; a++) rows.push_back(win + a - pos0);
    for (uint32_t i = 0; i < n_tok; i++)
        check_double_query("full block", &got[(size_t) i * nh * hd], &q[(size_t) i * nh * hd], kvw.data(), rows,
                           sink.data(), nh, hd, 0.1f);
    check_bf16_grid("full block grid", got);

    cudaFree(d_kvw);
    cudaFree(d_q);
    cudaFree(d_sink);
    cudaFree(d_o);
}

void invalid_equiv_case() {
    const uint32_t hd = 64, nh = 8, win = 6, pos0 = 5, topk = 3, ng = 4;
    std::vector<float> kvw((size_t)(win + 1) * hd);
    for (float& v : kvw) v = 0.75f + u01();
    std::vector<float> q((size_t) nh * hd);
    for (float& v : q) v = urand();
    std::vector<float> sink(nh);
    for (float& v : sink) v = urand();
    std::vector<int32_t> idx = {-1, static_cast<int32_t>(ng), -7};   // negative, out of range, negative
    std::vector<uint8_t> cache((size_t) ng * strata::kernels::kDeepseek4CkvBytes, 0xAA);

    float* d_kvw = dev_f_from(kvw);
    float* d_q = dev_f_from(q);
    float* d_sink = dev_f_from(sink);
    float* d_o = dev_f((size_t) nh * hd);
    uint8_t* d_kvc = dev_from(cache);
    int32_t* d_idx = dev_from(idx);

    // window 6 at p = 5: six window keys and three invalid compressed ones, so a tile holds both kinds.
    std::vector<float> a((size_t) nh * hd), b((size_t) nh * hd);
    CHECK(strata::kernels::deepseek4_sparse_attn(d_o, d_q, d_kvw, reinterpret_cast<const uint8_t*>(d_kvc),
                                                 reinterpret_cast<const int32_t*>(d_idx), d_sink, 1, pos0, win, ng,
                                                 topk, nh, hd, 0.1f, 0, true, 0, nullptr));
    to_host(d_o, a);
    CHECK(strata::kernels::deepseek4_sparse_attn(d_o, d_q, d_kvw, nullptr, nullptr, d_sink, 1, pos0, win, 0, 0, nh,
                                                 hd, 0.1f, 0, true, 0, nullptr));
    to_host(d_o, b);
    check_same_bits("invalid == no topk", a, b);

    cudaFree(d_kvw);
    cudaFree(d_q);
    cudaFree(d_sink);
    cudaFree(d_o);
    cudaFree(d_kvc);
    cudaFree(d_idx);
}

// ---- host-vs-device ----

void random_case(const char* name, uint32_t hd, uint32_t nh, uint32_t n_tok, uint32_t pos0, uint32_t win,
                 uint32_t topk, uint32_t ng, uint32_t win_lo) {
    std::vector<float> kvw((size_t)(win + n_tok) * hd);
    for (float& v : kvw) v = 0.75f + u01();
    std::vector<float> q((size_t) n_tok * nh * hd);
    for (float& v : q) v = urand();
    std::vector<float> sink(nh);
    for (float& v : sink) v = urand();

    std::vector<float> comp((size_t) ng * strata::kernels::kDeepseek4CkvDim);
    for (float& v : comp) v = 0.75f + u01();
    std::vector<uint8_t> cache((size_t) ng * strata::kernels::kDeepseek4CkvBytes, 0);
    strata::kernels::deepseek4_ckv_pack_host(cache.data(), 0, comp.data(), ng);
    std::vector<int32_t> idx(topk ? (size_t) n_tok * topk : 0);
    for (int32_t& g : idx) {
        const uint32_t r = urng() % 5u;
        g = r == 0 ? -1 : (r == 1 ? static_cast<int32_t>(ng) + 3 : static_cast<int32_t>(urng() % ng));
    }

    std::vector<float> want(q.size());
    strata::kernels::deepseek4_sparse_attn_host(want.data(), q.data(), kvw.data(), topk ? cache.data() : nullptr,
                                               topk ? idx.data() : nullptr, sink.data(), n_tok, pos0, win, ng, topk,
                                               nh, hd, 0.05f, 0, true, win_lo);

    float* d_kvw = dev_f_from(kvw);
    float* d_q = dev_f_from(q);
    float* d_sink = dev_f_from(sink);
    float* d_o = dev_f(want.size());
    uint8_t* d_kvc = nullptr;
    int32_t* d_idx = nullptr;
    if (topk) {
        d_kvc = dev_from(cache);
        d_idx = dev_from(idx);
    }
    CHECK(strata::kernels::deepseek4_sparse_attn(d_o, d_q, d_kvw, d_kvc, d_idx, d_sink, n_tok, pos0, win, ng, topk,
                                                 nh, hd, 0.05f, 0, true, win_lo, nullptr));
    std::vector<float> got(want.size());
    to_host(d_o, got);

    check_close(name, got, want, 2e-2, 1e-6);
    char gn[64];
    std::snprintf(gn, sizeof(gn), "%s grid", name);
    check_bf16_grid(gn, got);

    cudaFree(d_kvw);
    cudaFree(d_q);
    cudaFree(d_sink);
    cudaFree(d_o);
    if (d_kvc) cudaFree(d_kvc);
    if (d_idx) cudaFree(d_idx);
}

// The opt-in tensor-core prefill route (--attn-mma, cuda/ds41_attn_mma.cuh:137-293).  It is NOT bit-equal to
// the scalar kernel by design (tensor-core accumulation order, bf16 keys, 16-key tiles), so it is checked
// against the same host oracle with the usual bf16 tolerance.  The fixture puts every output in [0.75, 1.75] (a
// convex combination of positive keys), so the relative tolerance is not hiding cancellation.  The direct call
// must return true: that is what proves the route ran, because a silent fall-through to the scalar kernel would
// pass the same comparison.
void mma_case(const char* name, uint32_t n_tok, uint32_t pos0, uint32_t win, uint32_t topk, uint32_t ng) {
    const uint32_t hd = 512, nh = 64;
    std::vector<float> kvw((size_t)(win + n_tok) * hd);
    for (float& v : kvw) v = 0.75f + u01();
    std::vector<float> q((size_t) n_tok * nh * hd);
    for (float& v : q) v = urand();
    std::vector<float> sink(nh);
    for (float& v : sink) v = urand();

    std::vector<float> comp((size_t) ng * strata::kernels::kDeepseek4CkvDim);
    for (float& v : comp) v = 0.75f + u01();
    std::vector<uint8_t> cache((size_t) ng * strata::kernels::kDeepseek4CkvBytes, 0);
    strata::kernels::deepseek4_ckv_pack_host(cache.data(), 0, comp.data(), ng);
    std::vector<int32_t> idx((size_t) n_tok * topk);
    for (int32_t& g : idx) {
        const uint32_t r = urng() % 5u;
        g = r == 0 ? -1 : (r == 1 ? static_cast<int32_t>(ng) + 3 : static_cast<int32_t>(urng() % ng));
    }

    std::vector<float> want(q.size());
    strata::kernels::deepseek4_sparse_attn_host(want.data(), q.data(), kvw.data(), cache.data(), idx.data(),
                                                sink.data(), n_tok, pos0, win, ng, topk, nh, hd, 0.05f, 0, true, 0);

    float* d_kvw = dev_f_from(kvw);
    float* d_q = dev_f_from(q);
    float* d_sink = dev_f_from(sink);
    float* d_o = dev_f(want.size());
    uint8_t* d_kvc = dev_from(cache);
    int32_t* d_idx = dev_from(idx);

    CHECK(strata::kernels::deepseek4_sparse_attn_mma(d_o, d_q, d_kvw, d_kvc, d_idx, d_sink, n_tok, pos0, win, ng,
                                                     topk, nh, hd, 0.05f, 0, nullptr));
    std::vector<float> got(want.size());
    to_host(d_o, got);
    check_close(name, got, want, 2e-2, 1e-6);
    char gn[64];
    std::snprintf(gn, sizeof(gn), "%s grid", name);
    check_bf16_grid(gn, got);

    // The opt-in switch: off by default, and the public entry then takes this route for the same shape.
    CHECK(!strata::kernels::deepseek4_sparse_attn_mma_enabled());
    strata::kernels::deepseek4_sparse_attn_set_mma(true);
    CHECK(strata::kernels::deepseek4_sparse_attn_mma_enabled());
    CHECK(strata::kernels::deepseek4_sparse_attn(d_o, d_q, d_kvw, d_kvc, d_idx, d_sink, n_tok, pos0, win, ng, topk,
                                                 nh, hd, 0.05f, 0, true, 0, nullptr));
    strata::kernels::deepseek4_sparse_attn_set_mma(false);
    std::vector<float> via_entry(want.size());
    to_host(d_o, via_entry);
    check_close("via entry", via_entry, want, 2e-2, 1e-6);

    // The route's floor: 63 tokens (one short) or another head width is refused, never mis-launched.
    CHECK(!strata::kernels::deepseek4_sparse_attn_mma(d_o, d_q, d_kvw, d_kvc, d_idx, d_sink, 63, pos0, win, ng,
                                                      topk, nh, hd, 0.05f, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn_mma(d_o, d_q, d_kvw, d_kvc, d_idx, d_sink, n_tok, pos0, win, ng,
                                                      topk, nh, 64, 0.05f, 0, nullptr));

    cudaFree(d_kvw);
    cudaFree(d_q);
    cudaFree(d_sink);
    cudaFree(d_o);
    cudaFree(d_kvc);
    cudaFree(d_idx);
}

void guards() {
    const uint32_t hd = 64, nh = 8;
    float* dq = dev_f((size_t) nh * hd);
    float* dkw = dev_f((size_t) 5 * hd);
    float* ds = dev_f(nh);
    float* dout = dev_f((size_t) nh * hd);
    uint8_t* dc = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&dc), strata::kernels::kDeepseek4CkvBytes) == cudaSuccess);
    int32_t* di = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&di), sizeof(int32_t)) == cudaSuccess);

    // A working shape first, so the refusals below are not all failing for one shared reason.
    CHECK(strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, nullptr, nullptr, ds, 1, 0, 4, 0, 0, nh, hd, 0.1f, 0,
                                                 true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(nullptr, dq, dkw, nullptr, nullptr, ds, 1, 0, 4, 0, 0, nh, hd, 0.1f,
                                                  0, true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, nullptr, dkw, nullptr, nullptr, ds, 1, 0, 4, 0, 0, nh, hd, 0.1f,
                                                  0, true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, nullptr, nullptr, nullptr, ds, 1, 0, 4, 0, 0, nh, hd, 0.1f,
                                                  0, true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, nullptr, nullptr, nullptr, 1, 0, 4, 0, 0, nh, hd,
                                                  0.1f, 0, true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, nullptr, nullptr, ds, 0, 0, 4, 0, 0, nh, hd, 0.1f, 0,
                                                  true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, nullptr, nullptr, ds, 1, 0, 4, 0, 0, nh, 0, 0.1f, 0,
                                                  true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, nullptr, nullptr, ds, 1, 0, 4, 0, 0, nh, 513, 0.1f,
                                                  0, true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, nullptr, nullptr, ds, 1, 0, 4, 0, 0, nh, 48, 0.1f,
                                                  0, true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, nullptr, nullptr, ds, 1, 0, 4, 0, 0, 4, hd, 0.1f, 0,
                                                  true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, dc, nullptr, ds, 1, 0, 4, 1, 1, nh, hd, 0.1f, 0,
                                                  true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, nullptr, di, ds, 1, 0, 4, 1, 1, nh, hd, 0.1f, 0,
                                                  true, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, nullptr, nullptr, ds, 1, 0, 4, 0, 0, nh, hd, 0.1f, 0,
                                                  false, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, nullptr, nullptr, ds, 1, 0, 4, 0, 0, nh, hd, 0.1f, 3,
                                                  false, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_sparse_attn(dout, dq, dkw, nullptr, nullptr, ds, 1, 0, 4, 0, 0, nh, hd, 0.1f, 0,
                                                  true, 1, nullptr));

    // The host oracle refuses the same shapes and leaves its output alone.
    std::vector<float> q((size_t) nh * hd, 0.0f), kvw((size_t) 5 * hd, 1.0f), sink(nh, 0.0f);
    std::vector<uint8_t> hc(strata::kernels::kDeepseek4CkvBytes, 0);
    std::vector<float> sent((size_t) nh * hd, 7.0f), back = sent;
    strata::kernels::deepseek4_sparse_attn_host(sent.data(), q.data(), kvw.data(), hc.data(), nullptr, sink.data(), 1,
                                               0, 4, 1, 1, nh, hd, 0.1f, 0, true, 0);
    CHECK(std::memcmp(sent.data(), back.data(), sent.size() * sizeof(float)) == 0);
    strata::kernels::deepseek4_sparse_attn_host(sent.data(), q.data(), kvw.data(), nullptr, nullptr, sink.data(), 1, 0,
                                               4, 0, 0, nh, hd, 0.1f, 0, false, 0);
    CHECK(std::memcmp(sent.data(), back.data(), sent.size() * sizeof(float)) == 0);

    cudaFree(dq);
    cudaFree(dkw);
    cudaFree(ds);
    cudaFree(dout);
    cudaFree(dc);
    cudaFree(di);
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }

    // Exact fixtures first: the same numbers on both sides, compared bit for bit.
    single_window_key_case();
    single_compressed_key_case();

    // Geometry: the ring mapping and the DSpark draft block against a double-precision softmax over the rows
    // the mapping names, plus the invalid-topk equivalence.
    ring_map_case();
    full_block_case();
    invalid_equiv_case();

    // Host-vs-device across head widths, head counts, token counts, a clamp that bites, and a window-less
    // (compressed-only) layer.
    random_case("rand hd64", 64, 8, 5, 40, 16, 8, 12, 0);
    random_case("rand hd512 h64", 512, 64, 5, 40, 16, 8, 12, 0);
    random_case("rand clamp", 512, 8, 5, 40, 16, 8, 12, 30);
    random_case("rand no shift", 64, 64, 3, 3, 8, 5, 4, 0);
    random_case("rand window 0", 512, 8, 2, 0, 0, 5, 6, 0);

    // The opt-in tensor-core prefill route: a 96-token batch with 64-key windows crosses several 16-key tiles,
    // and the second fixture is compressed-only (no window at all).
    mma_case("mma 96 tok", 96, 200, 64, 8, 24);
    mma_case("mma comp only", 96, 200, 0, 8, 24);

    guards();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
