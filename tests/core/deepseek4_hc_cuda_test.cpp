// tests/core/deepseek4_hc_cuda_test.cpp - P3's first gate: the hyper-connection chain on the GPU.
//
//   deepseek4_hc_cuda_test
//
// The GPU kernels are compared against the host oracle in src/kernels/deepseek4_hc_host.cpp, which implements
// the reference engine's order (cuda/ds41_hc.cuh): mix = rms(hc) * fn @ hc, the 4-route Sinkhorn split with
// hc_eps on the softmax and on every normalization, hc_pre with the bf16 round before the RMS mean, and hc_post
// writing every lane.  Two kinds of comparison:
//
//   tolerance   kernel vs oracle.  The device reductions sum in a different order from the oracle's ascending
//               loop (a 20480-term dot's two orders can differ by ~1e-5 near zero) and the device sigmoid uses
//               `expf`/FMA where the host uses `std::exp`/separate multiply-add, so the values are allowed to
//               differ by a few ulps (far below bf16).  The bf16 outputs must sit exactly on the bf16 grid: a
//               value off the grid means a kernel skipped or misplaced the round.
//   bitwise     fused vs unfused on the device, with hc_pre fed the same `pre_in` the fused kernel takes.  These
//               run the same loops with the same thread mapping, so they must agree bit for bit; a mismatch is a
//               split/kernel wiring bug (for example, the fused kernel consuming this layer's pre instead of the
//               predecessor's pre_in).
//
// Shapes: the live geometry (5120 x 4 routes) at one and five tokens, a non-multiple width to exercise the
// bounds, and the widest supported route count (8, which fills the 64-entry comb matrix).  No model, no timing.
// No GPU: prints SKIP and exits 77 (ctest's SKIP_RETURN_CODE).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "strata/kernels/deepseek4_hc.hpp"

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

// A deterministic uniform in [-1, 1): the same fixture on every run, no <random> implementation dependence.
uint32_t g_rng = 0x9e3779b9u;
float urand() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return static_cast<float>(g_rng & 0x00FFFFFFu) / 8388608.0f - 1.0f;
}

// |got - want| <= abs_tol + rel * max(|got|, |want|).  The bound is reported as a ratio so a pass shows how
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
    std::printf("  %-22s %zu values, worst %.3f x tolerance, %zu over\n", what, got.size(), worst, bad);
}

// Every bf16-rounded output must round to itself.  engram_bf16r is the engine's v41_bf16r (RNE, Inf/NaN pass
// through), so a finite output off this grid is a missing or misplaced round.
void check_bf16_grid(const char* what, const std::vector<float>& v) {
    size_t off = 0;
    for (float x : v) {
        if (strata::kernels::deepseek4_hc_bf16r(x) != x) {
            if (off < 4) std::fprintf(stderr, "  %s value off the bf16 grid: %.9g\n", what, x);
            off++;
        }
    }
    g_checks++;
    if (off != 0) g_failures++;
    std::printf("  %-22s %zu values, %zu off the bf16 grid\n", what, v.size(), off);
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
    std::printf("  %-22s %zu values, %zu differ\n", what, a.size(), bad);
}

float* dev_alloc(size_t n) {
    float* p = nullptr;
    CHECK(cudaMalloc(&p, n * sizeof(float)) == cudaSuccess);
    return p;
}

float* dev_up(const std::vector<float>& v) {
    float* p = dev_alloc(v.size());
    CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}

void dev_down(std::vector<float>* v, const float* d) {
    CHECK(cudaMemcpy(v->data(), d, v->size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
}

void run_case(const char* name, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    const uint32_t dim = n_embd * n_hc, mix_hc = 2u * n_hc + n_hc * n_hc;
    const float mix_eps = 1e-6f, hc_eps = 1e-6f, norm_eps = 1e-20f;
    const uint32_t iters = 20;
    const uint32_t nt = n_tok, ne = n_embd, nh = n_hc;

    // The fixtures are scaled so the mix lands near unit size, like learned hc_fn tables: fn entries of
    // 2/sqrt(dim) times a unit-variance lane give a dot of order one.
    const float fn_scale = 2.0f / std::sqrt(static_cast<float>(dim));
    std::vector<float> fn(static_cast<size_t>(mix_hc) * dim), hc(static_cast<size_t>(nt) * dim),
        scale = {0.9f, 1.1f, 1.0f}, base(mix_hc), nw(ne), pre_in(static_cast<size_t>(nt) * nh),
        y(static_cast<size_t>(nt) * ne), res(static_cast<size_t>(nt) * nh * ne);
    for (float& v : fn) v = fn_scale * urand();
    for (float& v : hc) v = urand();
    for (float& v : base) v = 0.25f * urand();
    for (float& v : nw) v = 1.0f + 0.1f * urand();
    for (float& v : pre_in) v = 0.5f + 0.5f * urand();
    for (float& v : y) v = 2.0f * urand();
    for (float& v : res) v = urand();

    std::printf("%s: embd %u, routes %u, tokens %u (dim %u, mix_hc %u)\n", name, ne, nh, nt, dim, mix_hc);

    // The GPU's mix, then the host split/oracle consumes that same mix so the split comparison isolates the
    // kernel, not the two accumulation orders.
    std::vector<float> mix_got(static_cast<size_t>(nt) * mix_hc);
    float* d_fn = dev_up(fn);
    float* d_hc = dev_up(hc);
    float* d_scale = dev_up(scale);
    float* d_base = dev_up(base);
    float* d_nw = dev_up(nw);
    float* d_pre_in = dev_up(pre_in);
    float* d_y = dev_up(y);
    float* d_res = dev_up(res);
    float* d_mix = dev_alloc(mix_got.size());
    float* d_pre = dev_alloc(static_cast<size_t>(nt) * nh);
    float* d_post = dev_alloc(static_cast<size_t>(nt) * nh);
    float* d_comb = dev_alloc(static_cast<size_t>(nt) * nh * nh);
    float* d_x = dev_alloc(static_cast<size_t>(nt) * ne);
    float* d_xn = dev_alloc(static_cast<size_t>(nt) * ne);
    float* d_out = dev_alloc(static_cast<size_t>(nt) * nh * ne);
    float* d_pre2 = dev_alloc(static_cast<size_t>(nt) * nh);
    float* d_post2 = dev_alloc(static_cast<size_t>(nt) * nh);
    float* d_comb2 = dev_alloc(static_cast<size_t>(nt) * nh * nh);
    float* d_x2 = dev_alloc(static_cast<size_t>(nt) * ne);
    float* d_xn2 = dev_alloc(static_cast<size_t>(nt) * ne);

    CHECK(strata::kernels::deepseek4_hc_mix(d_fn, d_hc, d_mix, ne, nh, nt, mix_eps, nullptr));
    dev_down(&mix_got, d_mix);
    std::vector<float> mix_want(mix_got.size());
    strata::kernels::deepseek4_hc_mix_host(fn.data(), hc.data(), mix_want.data(), ne, nh, nt, mix_eps);
    // The 20480-term dot's CPU/GPU sum orders differ by ~1e-5 near a zero mix, hence the absolute floor; the
    // relative bound covers mixes of ordinary size.
    check_close("mix", mix_got, mix_want, 2e-5, 3e-5);

    // split, against the oracle fed the GPU's own mix values.
    std::vector<float> pre_want(static_cast<size_t>(nt) * nh), post_want(pre_want.size()),
        comb_want(static_cast<size_t>(nt) * nh * nh);
    strata::kernels::deepseek4_hc_split_host(mix_got.data(), scale.data(), base.data(), nh, iters, hc_eps,
                                             pre_want.data(), post_want.data(), comb_want.data(), nt);
    std::vector<float> pre_got(pre_want.size()), post_got(post_want.size()), comb_got(comb_want.size());
    CHECK(strata::kernels::deepseek4_hc_split(d_mix, d_scale, d_base, d_pre, d_post, d_comb, nh, iters, hc_eps,
                                              nt, nullptr));
    dev_down(&pre_got, d_pre);
    dev_down(&post_got, d_post);
    dev_down(&comb_got, d_comb);
    check_close("split pre", pre_got, pre_want, 1e-5, 1e-6);
    check_close("split post", post_got, post_want, 1e-5, 1e-6);
    check_close("split comb", comb_got, comb_want, 1e-5, 1e-6);

    // hc_pre (unfused) + the RMS norm.
    std::vector<float> x_want(static_cast<size_t>(nt) * ne), xn_want(x_want.size());
    strata::kernels::deepseek4_hc_pre_host(hc.data(), pre_got.data(), nw.data(), x_want.data(), xn_want.data(),
                                           ne, nh, norm_eps, nt);
    std::vector<float> x_got(x_want.size()), xn_got(x_want.size());
    CHECK(strata::kernels::deepseek4_hc_pre(d_hc, d_pre, d_nw, d_x, d_xn, ne, nh, norm_eps, nt, nullptr));
    dev_down(&x_got, d_x);
    dev_down(&xn_got, d_xn);
    check_close("hc_pre x", x_got, x_want, 2e-2, 1e-6);
    check_close("hc_pre xn", xn_got, xn_want, 2e-2, 1e-6);
    check_bf16_grid("hc_pre x", x_got);
    check_bf16_grid("hc_pre xn", xn_got);

    // hc_post.
    std::vector<float> out_want(static_cast<size_t>(nt) * nh * ne);
    strata::kernels::deepseek4_hc_post_host(y.data(), res.data(), post_got.data(), comb_got.data(),
                                            out_want.data(), ne, nh, nt);
    std::vector<float> out_got(out_want.size());
    CHECK(strata::kernels::deepseek4_hc_post(d_y, d_res, d_post, d_comb, d_out, ne, nh, nt, nullptr));
    dev_down(&out_got, d_out);
    check_close("hc_post", out_got, out_want, 2e-2, 1e-6);
    check_bf16_grid("hc_post", out_got);

    // The fused path: split + hc_pre + norm with the predecessor's pre_in.
    std::vector<float> pre2_want(pre_want.size()), post2_want(post_want.size()), comb2_want(comb_want.size()),
        x2_want(x_want.size()), xn2_want(x_want.size());
    strata::kernels::deepseek4_hc_fused_host(mix_got.data(), scale.data(), base.data(), pre_in.data(), hc.data(),
                                             nw.data(), ne, nh, iters, hc_eps, norm_eps, pre2_want.data(),
                                             post2_want.data(), comb2_want.data(), x2_want.data(),
                                             xn2_want.data(), nt);
    std::vector<float> pre2_got(pre_want.size()), post2_got(post_want.size()), comb2_got(comb_want.size()),
        x2_got(x_want.size()), xn2_got(x_want.size());
    CHECK(strata::kernels::deepseek4_hc_fused(d_mix, d_scale, d_base, d_pre_in, d_hc, d_nw, d_pre2, d_post2,
                                              d_comb2, d_x2, d_xn2, ne, nh, iters, hc_eps, norm_eps, nt,
                                              nullptr));
    dev_down(&pre2_got, d_pre2);
    dev_down(&post2_got, d_post2);
    dev_down(&comb2_got, d_comb2);
    dev_down(&x2_got, d_x2);
    dev_down(&xn2_got, d_xn2);
    check_close("fused pre", pre2_got, pre2_want, 1e-5, 1e-6);
    check_close("fused post", post2_got, post2_want, 1e-5, 1e-6);
    check_close("fused comb", comb2_got, comb2_want, 1e-5, 1e-6);
    check_close("fused x", x2_got, x2_want, 2e-2, 1e-6);
    check_close("fused xn", xn2_got, xn2_want, 2e-2, 1e-6);
    check_bf16_grid("fused x", x2_got);
    check_bf16_grid("fused xn", xn2_got);

    // Fused and unfused run the same code on the same values: bitwise, not a tolerance.  The fused kernel takes
    // the predecessor's pre by design, so its x/xn partner is hc_pre fed that same pre_in, not the split output
    // used above.
    CHECK(strata::kernels::deepseek4_hc_pre(d_hc, d_pre_in, d_nw, d_x, d_xn, ne, nh, norm_eps, nt, nullptr));
    dev_down(&x_got, d_x);
    dev_down(&xn_got, d_xn);
    check_exact("split vs fused pre", pre_got, pre2_got);
    check_exact("split vs fused post", post_got, post2_got);
    check_exact("split vs fused comb", comb_got, comb2_got);
    check_exact("pre(pre_in) vs fused x", x_got, x2_got);
    check_exact("pre(pre_in) vs fused xn", xn_got, xn2_got);

    // Bad shapes are refused before any launch.
    CHECK(!strata::kernels::deepseek4_hc_mix(nullptr, d_hc, d_mix, ne, nh, nt, mix_eps, nullptr));
    CHECK(!strata::kernels::deepseek4_hc_split(d_mix, d_scale, d_base, d_pre, d_post, d_comb, 9, iters, hc_eps, nt,
                                               nullptr));
    CHECK(!strata::kernels::deepseek4_hc_pre(d_hc, nullptr, d_nw, d_x, d_xn, ne, nh, norm_eps, nt, nullptr));
    CHECK(!strata::kernels::deepseek4_hc_post(d_y, d_res, d_post, d_comb, d_res, ne, nh, nt, nullptr));

    float* const all[] = {d_fn, d_hc,  d_scale, d_base, d_nw,    d_pre_in, d_y,   d_res,  d_mix,
                          d_pre, d_post, d_comb,  d_x,    d_xn,    d_out,    d_pre2, d_post2, d_comb2,
                          d_x2,  d_xn2};
    for (float* p : all) cudaFree(p);
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }

    run_case("decode 5120x4", 5120, 4, 1);
    run_case("batch5 5120x4", 5120, 4, 5);
    run_case("odd 300x2", 300, 2, 3);
    run_case("wide routes 128x8", 128, 8, 2);

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
