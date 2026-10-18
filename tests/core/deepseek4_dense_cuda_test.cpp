// tests/core/deepseek4_dense_cuda_test.cpp - P3's dense-primitive gate.
//
//   deepseek4_dense_cuda_test
//
// The GPU kernels in src/kernels/cuda/deepseek4_dense.cu are compared against the host oracle in
// src/kernels/deepseek4_dense_host.cpp, which writes down the reference engine's semantics: the RMS norm with
// the weight inside the bf16 round, SwiGLU's one-sided gate clamp, the plain elementwise forms, the q4_K
// embedding row, and the three skeleton matmuls.  Three kinds of check:
//
//   decoder     the one __host__ __device__ q4_K decoder this module adds is checked against `dequant_bf16`
//               (src/kernels/cuda/dequant_bf16.cu), whose Q4_K path is transcribed from llama.cpp and has been
//               validated against it.  The exact fixture uses power-of-two d/dmin so every product and the
//               subtraction are exact in f32, and the two decoders must then produce the same bf16 bits.  The
//               varied fixture exercises arbitrary fp16 d under the 6-bit scale table; there the reference is
//               bf16-rounded, so the bound is one bf16 ulp (2^-9) relative.
//   tolerance   kernel vs oracle elsewhere.  Where an output lands on the bf16 grid, a 1-ulp difference in the
//               pre-round value (a different reduction order, rsqrtf vs sqrt, expf vs exp) can flip the round
//               by one bf16 ulp, so those checks use 2e-2 relative - a few bf16 ulps - and separately prove the
//               output sits on the grid at all.  Non-rounded outputs use tight bounds.
//   bitwise     the elementwise kernels that have no reduction (add, round, scale, expand) and the q4_K
//               embedding/matmul over exact fixture values run identical arithmetic on both sides and must
//               agree bit for bit.
//
// Shapes: the live widths (5120), an odd width (300) for the grid-stride tails, a live q4_K projection
// (5120 -> 64), the grouped attention output shape (2 groups x 512 -> 64) and the artifact's real grouped form
// is exercised at P3's attention stage; the decoder fixtures use six rows of two blocks.  No model, no timing.
// No GPU: prints SKIP and exits 77 (ctest's SKIP_RETURN_CODE).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "strata/kernels/deepseek4_dense.hpp"
#include "strata/kernels/dequant_bf16.hpp"

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

// A deterministic fixture across runs, with no dependence on a <random> implementation.
uint32_t g_rng = 0x9e3779b9u;
uint32_t urng() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
float urand() { return static_cast<float>(urng() & 0x00FFFFFFu) / 8388608.0f - 1.0f; }

// |got - want| <= abs_tol + rel * max(|got|, |want|); the bound is reported as a ratio so a pass shows how
// much room was left, not only that it passed.
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
    std::printf("  %-26s %zu values, worst %.3f x tolerance, %zu over\n", what, got.size(), worst, bad);
}

void check_exact(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    CHECK(a.size() == b.size());
    size_t bad = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        uint32_t x = 0, y = 0;
        std::memcpy(&x, &a[i], 4);
        std::memcpy(&y, &b[i], 4);
        if (x != y) {
            if (bad < 4) std::fprintf(stderr, "  %s value %zu: %08x vs %08x\n", what, i, x, y);
            bad++;
        }
    }
    g_checks++;
    if (bad != 0) g_failures++;
    std::printf("  %-26s %zu values, %zu differ\n", what, a.size(), bad);
}

// Every bf16-rounded output must round to itself; a finite value off the grid is a missing or misplaced round.
void check_bf16_grid(const char* what, const std::vector<float>& v) {
    size_t off = 0;
    for (float x : v) {
        if (strata::kernels::deepseek4_dense_bf16r(x) != x) {
            if (off < 4) std::fprintf(stderr, "  %s value off the bf16 grid: %.9g\n", what, x);
            off++;
        }
    }
    g_checks++;
    if (off != 0) g_failures++;
    std::printf("  %-26s %zu values, %zu off the bf16 grid\n", what, v.size(), off);
}

float* dev_f(size_t n) {
    float* p = nullptr;
    CHECK(cudaMalloc(&p, n * sizeof(float)) == cudaSuccess);
    return p;
}

float* dev_up(const std::vector<float>& v) {
    float* p = dev_f(v.size());
    CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}

std::vector<float> dev_down(const float* d, size_t n) {
    std::vector<float> v(n);
    CHECK(cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    return v;
}

uint8_t* dev_u8(const std::vector<uint8_t>& v) {
    uint8_t* p = nullptr;
    CHECK(cudaMalloc(&p, v.size()) == cudaSuccess);
    CHECK(cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}

int32_t* dev_i32(const std::vector<int32_t>& v) {
    int32_t* p = nullptr;
    CHECK(cudaMalloc(&p, v.size() * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}

uint16_t* dev_u16(const std::vector<uint16_t>& v) {
    uint16_t* p = nullptr;
    CHECK(cudaMalloc(&p, v.size() * sizeof(uint16_t)) == cudaSuccess);
    CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}

uint16_t f16_pow2(int e) { return static_cast<uint16_t>((e + 15) << 10); }

// A q4_K weight fixture, rows of `dim` values.  `exact` picks power-of-two d/dmin so the decode's products and
// subtraction are exact in f32; otherwise d is an arbitrary fp16 in [2^-8, 2) with a random mantissa and dmin is
// zero, so every value is one product and the reference's bf16 round is the only error.
std::vector<uint8_t> make_q4k(uint32_t rows, uint32_t dim, bool exact) {
    const uint32_t blocks = dim / 256;
    std::vector<uint8_t> w(static_cast<size_t>(rows) * blocks * 144);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t* blk = w.data() + (static_cast<size_t>(r) * blocks + b) * 144;
            uint16_t d = 0, dmin = 0;
            if (exact) {
                const int ed[3] = {-3, -4, -5}, em[3] = {-5, -6, -7};
                d = f16_pow2(ed[urng() % 3]);
                dmin = f16_pow2(em[urng() % 3]);
            } else {
                const int e = -8 + static_cast<int>(urng() % 10);
                d = static_cast<uint16_t>(((e + 15) << 10) | (urng() & 1023u));
            }
            blk[0] = static_cast<uint8_t>(d & 0xff);
            blk[1] = static_cast<uint8_t>(d >> 8);
            blk[2] = static_cast<uint8_t>(dmin & 0xff);
            blk[3] = static_cast<uint8_t>(dmin >> 8);
            for (int i = 4; i < 144; i++) blk[i] = static_cast<uint8_t>(urng() & 0xff);
        }
    }
    return w;
}

std::vector<float> host_decode(const std::vector<uint8_t>& w, uint32_t rows, uint32_t dim) {
    const size_t row_bytes = static_cast<size_t>(dim / 256) * 144;
    std::vector<float> out(static_cast<size_t>(rows) * dim);
    for (uint32_t r = 0; r < rows; r++)
        strata::kernels::deepseek4_q4k_row_host(w.data() + static_cast<size_t>(r) * row_bytes, dim,
                                                out.data() + static_cast<size_t>(r) * dim);
    return out;
}

// The shared decoder against the tree's validated one, through the GPU.  `dequant_bf16` writes bf16 bits.
void check_decoder_against_dequant_bf16() {
    std::printf("q4_K decoder vs dequant_bf16\n");
    for (int exact = 1; exact >= 0; exact--) {
        const uint32_t rows = 6, dim = 512;
        const std::vector<uint8_t> w = make_q4k(rows, dim, exact == 1);
        const std::vector<float> ours = host_decode(w, rows, dim);

        uint16_t* d_ref = nullptr;
        CHECK(cudaMalloc(&d_ref, ours.size() * sizeof(uint16_t)) == cudaSuccess);
        strata::kernels::dequant_bf16(12, w.data(), 0, rows, dim, d_ref, nullptr);
        std::vector<uint16_t> ref(ours.size());
        CHECK(cudaMemcpy(ref.data(), d_ref, ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess);
        cudaFree(d_ref);

        if (exact) {
            size_t differ = 0;
            for (size_t i = 0; i < ours.size(); i++)
                if (strata::kernels::bf16_from_f32(ours[i]) != ref[i]) {
                    if (differ < 4)
                        std::fprintf(stderr, "  value %zu: ours %08x, dequant_bf16 %04x\n", i,
                                     strata::kernels::bf16_from_f32(ours[i]), ref[i]);
                    differ++;
                }
            g_checks++;
            if (differ != 0) g_failures++;
            std::printf("  %-26s %zu values, %zu differ\n", "exact headers", ours.size(), differ);
        } else {
            // The reference is bf16: bound its quantization (2^-9 relative), not the layout.
            std::vector<float> ref_f(ours.size());
            for (size_t i = 0; i < ref.size(); i++) ref_f[i] = strata::kernels::f32_from_bf16(ref[i]);
            check_close("varied fp16 d", ours, ref_f, 0.004, 0.0);
        }
    }
}

void run_rms_norm() {
    std::printf("rms_norm\n");
    for (const uint32_t dim : {5120u, 300u}) {
        const uint32_t n_tok = 2;
        std::vector<float> x(static_cast<size_t>(n_tok) * dim), w(dim);
        for (float& v : x) v = 2.0f * urand();
        for (float& v : w) v = 1.0f + 0.1f * urand();
        const float eps = 1e-20f;
        std::vector<float> want(x.size());
        strata::kernels::deepseek4_rms_norm_host(want.data(), x.data(), w.data(), dim, n_tok, eps);

        float* d_x = dev_up(x);
        float* d_w = dev_up(w);
        float* d_out = dev_f(want.size());
        CHECK(strata::kernels::deepseek4_rms_norm(d_out, d_x, d_w, dim, n_tok, eps, nullptr));
        const std::vector<float> got = dev_down(d_out, want.size());
        char name[64];
        std::snprintf(name, sizeof(name), "rms_norm dim %u", dim);
        check_close(name, got, want, 2e-2, 1e-6);
        std::snprintf(name, sizeof(name), "rms_norm dim %u grid", dim);
        check_bf16_grid(name, got);
        cudaFree(d_x);
        cudaFree(d_w);
        cudaFree(d_out);
    }
}

void run_elementwise() {
    std::printf("elementwise\n");
    const uint64_t n = 1000;   // not a multiple of the block size: exercises the grid-stride tail
    std::vector<float> a(n), b(n);
    for (float& v : a) v = 2.0f * urand();
    for (float& v : b) v = 2.0f * urand();

    {
        std::vector<float> want = a;
        for (uint64_t i = 0; i < n; i++) want[i] += b[i];
        float* d_a = dev_up(a);
        float* d_b = dev_up(b);
        CHECK(strata::kernels::deepseek4_add(d_a, d_b, n, nullptr));
        check_exact("add", dev_down(d_a, n), want);
        cudaFree(d_a);
        cudaFree(d_b);
    }
    {
        std::vector<float> want(n);
        for (uint64_t i = 0; i < n; i++) want[i] = strata::kernels::deepseek4_dense_bf16r(a[i]);
        float* d_a = dev_up(a);
        CHECK(strata::kernels::deepseek4_round_bf16(d_a, n, nullptr));
        check_exact("round_bf16", dev_down(d_a, n), want);
        cudaFree(d_a);
    }
    {
        const float s = 0.75f;
        std::vector<float> want(n);
        for (uint64_t i = 0; i < n; i++) want[i] = strata::kernels::deepseek4_dense_bf16r(a[i] * s);
        float* d_a = dev_up(a);
        CHECK(strata::kernels::deepseek4_scale_round(d_a, n, s, nullptr));
        check_exact("scale_round", dev_down(d_a, n), want);
        cudaFree(d_a);
    }
}

void run_expand_hc() {
    std::printf("expand_hc\n");
    const uint32_t ne = 300, nh = 4, nt = 3;
    std::vector<float> x(static_cast<size_t>(nt) * ne);
    for (float& v : x) v = 2.0f * urand();
    std::vector<float> want(static_cast<size_t>(nt) * nh * ne);
    for (uint32_t t = 0; t < nt; t++)
        for (uint32_t c = 0; c < nh; c++)
            for (uint32_t d = 0; d < ne; d++) want[(static_cast<size_t>(t) * nh + c) * ne + d] = x[t * ne + d];

    float* d_x = dev_up(x);
    float* d_hc = dev_f(want.size());
    CHECK(strata::kernels::deepseek4_expand_hc(d_hc, d_x, ne, nh, nt, nullptr));
    check_exact("expand_hc", dev_down(d_hc, want.size()), want);
    cudaFree(d_x);
    cudaFree(d_hc);
}

void run_swiglu(float limit, const char* label) {
    const uint64_t n = 1000;
    std::vector<float> gate(n), up(n);
    for (float& v : gate) v = 8.0f * urand();
    for (float& v : up) v = 8.0f * urand();
    // Force all four clamp corners to be exercised whatever the fixture draw is.
    gate[0] = 50.0f;
    up[0] = 50.0f;
    gate[1] = -50.0f;
    up[1] = -50.0f;
    gate[2] = 10.0f;
    up[2] = -10.0f;

    std::vector<float> want(n);
    strata::kernels::deepseek4_swiglu_host(want.data(), gate.data(), up.data(), n, limit);

    float* d_gate = dev_up(gate);
    float* d_up = dev_up(up);
    float* d_h = dev_f(n);
    CHECK(strata::kernels::deepseek4_swiglu(d_h, d_gate, d_up, n, limit, nullptr));
    const std::vector<float> got = dev_down(d_h, n);
    check_close(label, got, want, 2e-2, 1e-6);
    char name[64];
    std::snprintf(name, sizeof(name), "%s grid", label);
    check_bf16_grid(name, got);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_h);
}

void run_embed() {
    std::printf("embed_q4k\n");
    const uint32_t n_vocab = 7, dim = 512, n_tok = 5;
    const std::vector<uint8_t> w = make_q4k(n_vocab, dim, true);
    const std::vector<int32_t> tok = {3, 0, 6, 2, 5};
    const std::vector<int32_t> tok_bad = {3, -1, 0, 7, 6};   // -1 and 7 are out of range

    std::vector<float> want(static_cast<size_t>(n_tok) * dim), want_bad(want.size());
    strata::kernels::deepseek4_embed_q4k_host(want.data(), tok.data(), w.data(), n_vocab, n_tok, dim);
    strata::kernels::deepseek4_embed_q4k_host(want_bad.data(), tok_bad.data(), w.data(), n_vocab, n_tok, dim);

    uint8_t* d_w = dev_u8(w);
    float* d_out = dev_f(want.size());
    int32_t* d_tok = dev_i32(tok);
    CHECK(strata::kernels::deepseek4_embed_q4k(d_out, d_tok, d_w, n_vocab, n_tok, dim, nullptr));
    check_exact("embed rows", dev_down(d_out, want.size()), want);
    cudaFree(d_tok);

    d_tok = dev_i32(tok_bad);
    CHECK(strata::kernels::deepseek4_embed_q4k(d_out, d_tok, d_w, n_vocab, n_tok, dim, nullptr));
    check_exact("embed out-of-range", dev_down(d_out, want.size()), want_bad);

    cudaFree(d_tok);
    cudaFree(d_w);
    cudaFree(d_out);
}

void run_matmul_f32() {
    std::printf("matmul_f32\n");
    const uint32_t in_dim = 512, out_dim = 256, n_tok = 2;
    std::vector<float> w(static_cast<size_t>(in_dim) * out_dim), x(static_cast<size_t>(n_tok) * in_dim);
    for (float& v : w) v = urand();
    for (float& v : x) v = urand();
    float* d_w = dev_up(w);
    float* d_x = dev_up(x);
    float* d_out = dev_f(static_cast<size_t>(n_tok) * out_dim);

    std::vector<float> want(static_cast<size_t>(n_tok) * out_dim);
    strata::kernels::deepseek4_matmul_f32_host(want.data(), w.data(), x.data(), in_dim, out_dim, n_tok, false);
    CHECK(strata::kernels::deepseek4_matmul_f32(d_out, d_w, d_x, in_dim, out_dim, n_tok, false, nullptr));
    check_close("matmul_f32", dev_down(d_out, want.size()), want, 3e-5, 1e-5);

    strata::kernels::deepseek4_matmul_f32_host(want.data(), w.data(), x.data(), in_dim, out_dim, n_tok, true);
    CHECK(strata::kernels::deepseek4_matmul_f32(d_out, d_w, d_x, in_dim, out_dim, n_tok, true, nullptr));
    const std::vector<float> got = dev_down(d_out, want.size());
    check_close("matmul_f32 round", got, want, 2e-2, 1e-6);
    check_bf16_grid("matmul_f32 round grid", got);

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
}

void run_matmul_bf16() {
    std::printf("matmul_bf16\n");
    const uint32_t in_dim = 512, out_dim = 256, n_tok = 2;
    std::vector<float> wf(static_cast<size_t>(in_dim) * out_dim), x(static_cast<size_t>(n_tok) * in_dim);
    for (float& v : wf) v = urand();
    for (float& v : x) v = urand();
    std::vector<uint16_t> w(wf.size());
    for (size_t i = 0; i < wf.size(); i++) w[i] = strata::kernels::bf16_from_f32(wf[i]);

    uint16_t* d_w = dev_u16(w);
    float* d_x = dev_up(x);
    float* d_out = dev_f(static_cast<size_t>(n_tok) * out_dim);

    std::vector<float> want(static_cast<size_t>(n_tok) * out_dim);
    strata::kernels::deepseek4_matmul_bf16_host(want.data(), w.data(), x.data(), in_dim, out_dim, n_tok, false);
    CHECK(strata::kernels::deepseek4_matmul_bf16(d_out, d_w, d_x, in_dim, out_dim, n_tok, false, nullptr));
    check_close("matmul_bf16", dev_down(d_out, want.size()), want, 3e-5, 1e-5);

    strata::kernels::deepseek4_matmul_bf16_host(want.data(), w.data(), x.data(), in_dim, out_dim, n_tok, true);
    CHECK(strata::kernels::deepseek4_matmul_bf16(d_out, d_w, d_x, in_dim, out_dim, n_tok, true, nullptr));
    const std::vector<float> got = dev_down(d_out, want.size());
    check_close("matmul_bf16 round", got, want, 2e-2, 1e-6);
    check_bf16_grid("matmul_bf16 round grid", got);

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
}

void run_matmul_q4k() {
    std::printf("matmul_q4k\n");
    const uint32_t in_dim = 512, out_dim = 128, n_tok = 2;
    const std::vector<uint8_t> w = make_q4k(out_dim, in_dim, true);
    std::vector<float> x(static_cast<size_t>(n_tok) * in_dim);
    for (float& v : x) v = urand();

    uint8_t* d_w = dev_u8(w);
    float* d_x = dev_up(x);
    float* d_out = dev_f(static_cast<size_t>(n_tok) * out_dim);
    std::vector<float> want(static_cast<size_t>(n_tok) * out_dim);

    strata::kernels::deepseek4_matmul_q4k_host(want.data(), w.data(), x.data(), in_dim, out_dim, n_tok, false);
    CHECK(strata::kernels::deepseek4_matmul_q4k(d_out, d_w, d_x, in_dim, out_dim, n_tok, false, nullptr));
    check_close("matmul_q4k", dev_down(d_out, want.size()), want, 1e-5, 1e-4);

    strata::kernels::deepseek4_matmul_q4k_host(want.data(), w.data(), x.data(), in_dim, out_dim, n_tok, true);
    CHECK(strata::kernels::deepseek4_matmul_q4k(d_out, d_w, d_x, in_dim, out_dim, n_tok, true, nullptr));
    const std::vector<float> got = dev_down(d_out, want.size());
    check_close("matmul_q4k round", got, want, 2e-2, 1e-6);
    check_bf16_grid("matmul_q4k round grid", got);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
}

// The live width: 20 q4_K blocks per row, which the block loop in the kernel walks 20 times.
void run_matmul_q4k_live_width() {
    std::printf("matmul_q4k live width\n");
    const uint32_t in_dim = 5120, out_dim = 64, n_tok = 1;
    const std::vector<uint8_t> w = make_q4k(out_dim, in_dim, true);
    std::vector<float> x(static_cast<size_t>(n_tok) * in_dim);
    for (float& v : x) v = urand();

    uint8_t* d_w = dev_u8(w);
    float* d_x = dev_up(x);
    float* d_out = dev_f(static_cast<size_t>(n_tok) * out_dim);
    std::vector<float> want(static_cast<size_t>(n_tok) * out_dim);
    strata::kernels::deepseek4_matmul_q4k_host(want.data(), w.data(), x.data(), in_dim, out_dim, n_tok, false);
    CHECK(strata::kernels::deepseek4_matmul_q4k(d_out, d_w, d_x, in_dim, out_dim, n_tok, false, nullptr));
    check_close("matmul_q4k 20 blocks", dev_down(d_out, want.size()), want, 1e-5, 1e-4);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
}

void run_matmul_q4k_grouped() {
    std::printf("matmul_q4k_grouped\n");
    const uint32_t n_groups = 2, group_dim = 512, rank = 64, n_tok = 2;
    const std::vector<uint8_t> w = make_q4k(n_groups * rank, group_dim, true);
    std::vector<float> x(static_cast<size_t>(n_tok) * n_groups * group_dim);
    for (float& v : x) v = urand();

    uint8_t* d_w = dev_u8(w);
    float* d_x = dev_up(x);
    float* d_out = dev_f(static_cast<size_t>(n_tok) * n_groups * rank);
    std::vector<float> want(static_cast<size_t>(n_tok) * n_groups * rank);

    strata::kernels::deepseek4_matmul_q4k_grouped_host(want.data(), w.data(), x.data(), n_groups, group_dim, rank,
                                                       n_tok, false);
    CHECK(strata::kernels::deepseek4_matmul_q4k_grouped(d_out, d_w, d_x, n_groups, group_dim, rank, n_tok, false,
                                                        nullptr));
    check_close("grouped", dev_down(d_out, want.size()), want, 1e-5, 1e-4);

    strata::kernels::deepseek4_matmul_q4k_grouped_host(want.data(), w.data(), x.data(), n_groups, group_dim, rank,
                                                       n_tok, true);
    CHECK(strata::kernels::deepseek4_matmul_q4k_grouped(d_out, d_w, d_x, n_groups, group_dim, rank, n_tok, true,
                                                        nullptr));
    const std::vector<float> got = dev_down(d_out, want.size());
    check_close("grouped round", got, want, 2e-2, 1e-6);
    check_bf16_grid("grouped round grid", got);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
}

void run_guards() {
    std::printf("guards\n");
    std::vector<float> z(256, 0.0f);
    float* d_a = dev_up(z);
    float* d_b = dev_up(z);
    CHECK(!strata::kernels::deepseek4_rms_norm(nullptr, d_a, d_a, 256, 1, 1e-20f, nullptr));
    CHECK(!strata::kernels::deepseek4_rms_norm(d_a, d_a, d_a, 0, 1, 1e-20f, nullptr));
    CHECK(!strata::kernels::deepseek4_add(d_a, nullptr, 256, nullptr));
    CHECK(!strata::kernels::deepseek4_round_bf16(nullptr, 256, nullptr));
    CHECK(!strata::kernels::deepseek4_scale_round(d_a, 0, 1.0f, nullptr));
    CHECK(!strata::kernels::deepseek4_expand_hc(d_a, d_a, 0, 4, 2, nullptr));
    CHECK(!strata::kernels::deepseek4_swiglu(d_a, d_a, nullptr, 256, 10.0f, nullptr));
    CHECK(!strata::kernels::deepseek4_embed_q4k(d_a, nullptr, d_a, 7, 1, 512, nullptr));
    CHECK(!strata::kernels::deepseek4_embed_q4k(d_a, reinterpret_cast<const int32_t*>(d_b), d_a, 7, 1, 300,
                                                nullptr));
    CHECK(!strata::kernels::deepseek4_matmul_f32(d_a, d_a, d_a, 0, 8, 1, false, nullptr));
    CHECK(!strata::kernels::deepseek4_matmul_bf16(d_a, nullptr, d_a, 8, 8, 1, false, nullptr));
    CHECK(!strata::kernels::deepseek4_matmul_q4k(d_a, d_a, d_a, 300, 8, 1, false, nullptr));
    CHECK(!strata::kernels::deepseek4_matmul_q4k_grouped(d_a, d_a, d_a, 2, 300, 8, 1, false, nullptr));
    cudaFree(d_a);
    cudaFree(d_b);
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }

    check_decoder_against_dequant_bf16();
    run_rms_norm();
    run_elementwise();
    run_expand_hc();
    run_swiglu(10.0f, "swiglu limit 10");
    run_swiglu(0.0f, "swiglu no clamp");
    run_embed();
    run_matmul_f32();
    run_matmul_bf16();
    run_matmul_q4k();
    run_matmul_q4k_live_width();
    run_matmul_q4k_grouped();
    run_guards();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
