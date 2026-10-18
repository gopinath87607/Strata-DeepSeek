// tests/core/deepseek4_rope_cuda_test.cpp - P3's RoPE gate.
//
//   deepseek4_rope_cuda_test
//
// The GPU kernel in src/kernels/cuda/deepseek4_attn.cu is compared against the host oracle in
// src/kernels/deepseek4_attn_host.cpp, which implements the reference engine's v41_rope_kernel: the last n_rot
// values of each head rotate as complex pairs, f(i) = theta^(-2i/n_rot) with the YaRN ramp when
// original_seq_len > 0, the inverse form negates the sine, and both outputs are bf16-rounded once.
//
// Both sides share the frequency function (deepseek4_rope_freq in the header), so the angle is bit-identical
// and only their libm cos/sin can differ by a few ulps.  Where an output is bf16, a 1-ulp pre-round difference
// can flip the round by one bf16 ulp, so the check uses 2e-2 relative - a few bf16 ulps - and separately proves
// the output is on the grid.  The values outside the rotated window must not move at all: that is checked
// bitwise, because a wrong offset would be exactly the kind of plausible-looking bug the tolerance would hide.
//
// Cases: the live q path (64 heads x 512, 64 rotated), the compressor's pooled key (base 160000, original
// context 65536, so the ramp is live), the indexer's key (head dim 128 with the same 64 rotated), the inverse
// rotation, a partial window (32 of 96) and a full head (64 of 64).  Positions span 0, 1, 7, 1000 and 65535, the
// last inside the model's original context.  No model, no timing.  No GPU: prints SKIP and exits 77.
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
int32_t* dev_i32(const std::vector<int32_t>& v) {
    int32_t* p = nullptr;
    CHECK(cudaMalloc(&p, v.size() * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}

struct RopeCase {
    const char* name;
    uint32_t n_tok, n_head, head_dim, n_rot;
    float theta;
    uint32_t osl;
    bool inverse;
};

void run_case(const RopeCase& c) {
    const int32_t positions[5] = {0, 1, 7, 1000, 65535};
    const uint32_t nt = c.n_tok;
    const size_t n = static_cast<size_t>(nt) * c.n_head * c.head_dim;

    std::vector<int32_t> pos(positions, positions + nt);
    std::vector<float> x(n);
    for (float& v : x)
        v = strata::kernels::f32_from_bf16(strata::kernels::bf16_from_f32(2.0f * urand()));

    std::vector<float> want = x;
    strata::kernels::deepseek4_rope_host(want.data(), pos.data(), nt, c.n_head, c.head_dim, c.n_rot, c.theta,
                                         c.osl, 16.0f, 32.0f, 1.0f, c.inverse);

    float* d_x = dev_f(n);
    CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    int32_t* d_pos = dev_i32(pos);
    CHECK(strata::kernels::deepseek4_rope(d_x, d_pos, nt, c.n_head, c.head_dim, c.n_rot, c.theta, c.osl, 16.0f,
                                          32.0f, 1.0f, c.inverse, nullptr));
    std::vector<float> got(n);
    CHECK(cudaMemcpy(got.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    check_close(c.name, got, want, 2e-2, 1e-6);
    char grid_name[64];
    std::snprintf(grid_name, sizeof(grid_name), "%s grid", c.name);
    check_bf16_grid(grid_name, got);

    // Everything before the rotated window must be untouched, bit for bit.
    const uint32_t prefix = c.head_dim - c.n_rot;
    size_t moved = 0;
    for (uint32_t t = 0; t < nt; t++)
        for (uint32_t h = 0; h < c.n_head; h++)
            for (uint32_t d = 0; d < prefix; d++) {
                const size_t i = (static_cast<size_t>(t) * c.n_head + h) * c.head_dim + d;
                uint32_t a = 0, b = 0;
                std::memcpy(&a, &x[i], 4);
                std::memcpy(&b, &got[i], 4);
                if (a != b) {
                    if (moved < 4) std::fprintf(stderr, "  %s prefix value %zu moved\n", c.name, i);
                    moved++;
                }
            }
    g_checks++;
    if (moved != 0) g_failures++;
    std::printf("  %-22s %zu fixed values, %zu moved\n", "prefix", static_cast<size_t>(nt) * c.n_head * prefix,
                moved);

    cudaFree(d_x);
    cudaFree(d_pos);
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }

    run_case({"q 64h 512/64", 4, 64, 512, 64, 10000.0f, 0, false});
    run_case({"compressor 512/64 yarn", 5, 1, 512, 64, 160000.0f, 65536, false});
    run_case({"indexer key 128/64", 3, 1, 128, 64, 160000.0f, 65536, false});
    run_case({"inverse q", 4, 64, 512, 64, 10000.0f, 0, true});
    run_case({"partial 96/32", 3, 2, 96, 32, 160000.0f, 0, false});
    run_case({"full head 64/64", 3, 2, 64, 64, 10000.0f, 0, false});

    // Bad shapes and a zero-width rotation.
    float* d_x = dev_f(64 * 512);
    const int32_t zeros[4] = {0, 1, 2, 3};
    int32_t* d_pos = dev_i32(std::vector<int32_t>(zeros, zeros + 4));
    CHECK(!strata::kernels::deepseek4_rope(nullptr, d_pos, 4, 64, 512, 64, 10000.0f, 0, 16.0f, 32.0f, 1.0f, false,
                                           nullptr));
    CHECK(!strata::kernels::deepseek4_rope(d_x, nullptr, 4, 64, 512, 64, 10000.0f, 0, 16.0f, 32.0f, 1.0f, false,
                                           nullptr));
    CHECK(!strata::kernels::deepseek4_rope(d_x, d_pos, 4, 64, 512, 63, 10000.0f, 0, 16.0f, 32.0f, 1.0f, false,
                                           nullptr));
    CHECK(!strata::kernels::deepseek4_rope(d_x, d_pos, 4, 64, 512, 132, 10000.0f, 0, 16.0f, 32.0f, 1.0f, false,
                                           nullptr));
    CHECK(!strata::kernels::deepseek4_rope(d_x, d_pos, 4, 64, 32, 64, 10000.0f, 0, 16.0f, 32.0f, 1.0f, false,
                                           nullptr));
    CHECK(!strata::kernels::deepseek4_rope(d_x, d_pos, 0, 64, 512, 64, 10000.0f, 0, 16.0f, 32.0f, 1.0f, false,
                                           nullptr));
    CHECK(strata::kernels::deepseek4_rope(d_x, d_pos, 4, 64, 512, 0, 10000.0f, 0, 16.0f, 32.0f, 1.0f, false,
                                          nullptr));
    cudaFree(d_x);
    cudaFree(d_pos);

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
