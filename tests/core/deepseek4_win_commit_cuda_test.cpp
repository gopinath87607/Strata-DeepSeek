// tests/core/deepseek4_win_commit_cuda_test.cpp - P3's window ring gate.
//
//   deepseek4_win_commit_cuda_test
//
// Two pure copies, so every comparison is bitwise:
//
//   win commit   batch row i moves from the window buffer's block region (row window + i) to ring cell
//                (pos0 + i) % window; only the last window rows move when n > window (cuda/ds41_attn.cuh:382-405)
//   ring snap    save/restore of the ring cells a spec-decode commit will overwrite, rows [i0, n), snap indexed
//                by batch row (cuda/ds41_attn.cuh:407-425)
//
// The host oracle in src/kernels/deepseek4_attn_host.cpp writes the reference loop out; the gate compares the
// whole buffer against it after each call, so an untouched row that moved, or a destination row that did not,
// is a failure.  One case asserts the destination rule by hand (n > window aliasing), one feeds the position
// through the device pointer, and every refusal path the reference entry has is checked.
// No model, no timing.  No GPU: prints SKIP and exits 77.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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
void to_host(const float* d, std::vector<float>& v) {
    CHECK(cudaMemcpy(v.data(), d, v.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
}

// A distinct value per (row, dim), so any misplaced row differs.
std::vector<float> make_rows(uint32_t rows, uint32_t hd) {
    std::vector<float> v((size_t) rows * hd);
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t d = 0; d < hd; d++) v[(size_t) r * hd + d] = (float) r + 0.001f * (float) d;
    return v;
}

// ---- window ring commit ----

void commit_case(const char* name, uint32_t pos0, uint32_t n, uint32_t window, uint32_t hd, bool with_pos,
                 int32_t pos_value) {
    const uint32_t rows = window + n;
    std::vector<float> start = make_rows(rows, hd), want = start, got = start;
    int32_t pos = pos_value;
    strata::kernels::deepseek4_win_commit_host(want.data(), pos0, n, window, hd, with_pos ? &pos : nullptr);

    float* d_win = dev_f_from(start);
    int32_t* d_pos = nullptr;
    if (with_pos) {
        CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pos), sizeof(int32_t)) == cudaSuccess);
        CHECK(cudaMemcpy(d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    CHECK(strata::kernels::deepseek4_win_commit(d_win, pos0, n, window, hd, d_pos, nullptr));
    to_host(d_win, got);
    check_same_bits(name, got, want);
    if (d_pos) cudaFree(d_pos);
    cudaFree(d_win);
}

void commit_hand_case() {
    // n > window: only rows [n - window, n) move; their cells are consecutive positions mod the window.
    const uint32_t window = 4, n = 6, pos0 = 7, hd = 8;
    std::vector<float> start = make_rows(window + n, hd), want = start;
    const uint32_t i0 = n - window;
    for (uint32_t i = i0; i < n; i++)
        std::memcpy(&want[(size_t)((pos0 + i) % window) * hd], &start[(size_t)(window + i) * hd],
                    hd * sizeof(float));
    std::vector<float> got = start;
    float* d_win = dev_f_from(start);
    CHECK(strata::kernels::deepseek4_win_commit(d_win, pos0, n, window, hd, nullptr, nullptr));
    to_host(d_win, got);
    check_same_bits("commit aliasing", got, want);
    cudaFree(d_win);
}

// ---- spec-decode ring snapshot ----

void snap_round_trip_case() {
    const uint32_t window = 8, i0 = 2, n = 6, pos0 = 4, hd = 16;
    const float sentinel = 999.0f;
    std::vector<float> win_start = make_rows(window, hd), snap_start((size_t) n * hd, sentinel);
    std::vector<float> win_want = win_start, snap_want = snap_start;
    std::vector<float> win_got = win_start, snap_got = snap_start;

    // save, then scribble over the saved cells, then restore: both buffers must come back bit for bit.
    strata::kernels::deepseek4_win_ring_snap_host(win_want.data(), snap_want.data(), pos0, i0, n, window, hd, false,
                                                  nullptr);
    float* d_win = dev_f_from(win_start);
    float* d_snap = dev_f_from(snap_start);
    CHECK(strata::kernels::deepseek4_win_ring_snap(d_win, d_snap, pos0, i0, n, window, hd, false, nullptr, nullptr));
    to_host(d_snap, snap_got);
    check_same_bits("snap save", snap_got, snap_want);

    // The save leaves rows [0, i0) alone and row i holds cell (pos0 + i) % window.
    size_t bad = 0;
    for (uint32_t i = 0; i < i0; i++)
        for (uint32_t d = 0; d < hd; d++)
            if (snap_got[(size_t) i * hd + d] != sentinel) bad++;
    for (uint32_t i = i0; i < n; i++)
        for (uint32_t d = 0; d < hd; d++)
            if (snap_got[(size_t) i * hd + d] != win_start[(size_t)((pos0 + i) % window) * hd + d]) bad++;
    g_checks++;
    if (bad != 0) g_failures++;
    std::printf("  %-22s %zu values, %zu misplaced\n", "snap mapping", (size_t) n * hd, bad);

    // Scribble on every ring cell, then restore.  Both sides take the same route.
    std::vector<float> scribble = make_rows(window, hd);
    for (uint32_t r = 0; r < window; r++)
        for (uint32_t d = 0; d < hd; d++) scribble[(size_t) r * hd + d] = -scribble[(size_t) r * hd + d] - 1.0f;
    CHECK(cudaMemcpy(d_win, scribble.data(), scribble.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    win_want = scribble;
    strata::kernels::deepseek4_win_ring_snap_host(win_want.data(), snap_want.data(), pos0, i0, n, window, hd, true,
                                                  nullptr);
    CHECK(strata::kernels::deepseek4_win_ring_snap(d_win, d_snap, pos0, i0, n, window, hd, true, nullptr, nullptr));
    to_host(d_win, win_got);
    check_same_bits("snap restore", win_got, win_want);

    cudaFree(d_win);
    cudaFree(d_snap);
}

void snap_pos_case() {
    // A device position overrides pos0 for both kernels.
    const uint32_t window = 8, n = 3, hd = 16;
    const int32_t pos_value = 12;
    std::vector<float> win_start = make_rows(window + n, hd), win_want = win_start, win_got = win_start;
    int32_t pos = pos_value;
    strata::kernels::deepseek4_win_commit_host(win_want.data(), 5, n, window, hd, &pos);

    float* d_win = dev_f_from(win_start);
    int32_t* d_pos = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pos), sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMemcpy(d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(strata::kernels::deepseek4_win_commit(d_win, 5, n, window, hd, d_pos, nullptr));
    to_host(d_win, win_got);
    check_same_bits("commit pos", win_got, win_want);

    // Ring snap reads it the same way.
    std::vector<float> snap_a((size_t) n * hd, 1.0f), snap_b((size_t) n * hd, 2.0f), snap_c((size_t) n * hd, 1.0f),
        snap_d((size_t) n * hd, 2.0f);
    float* d_snap = dev_f_from(snap_a);
    CHECK(strata::kernels::deepseek4_win_ring_snap(d_win, d_snap, 5, 0, n, window, hd, false, d_pos, nullptr));
    to_host(d_snap, snap_c);
    strata::kernels::deepseek4_win_ring_snap_host(win_want.data(), snap_b.data(), 5, 0, n, window, hd, false, &pos);
    check_same_bits("snap pos", snap_c, snap_b);

    cudaFree(d_win);
    cudaFree(d_snap);
    cudaFree(d_pos);
}

void guards() {
    float* dw = dev_f((size_t) 32 * 16);
    float* ds = dev_f((size_t) 32 * 16);
    int32_t* dp = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&dp), sizeof(int32_t)) == cudaSuccess);

    CHECK(!strata::kernels::deepseek4_win_commit(nullptr, 0, 1, 8, 16, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_win_commit(dw, 0, 0, 8, 16, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_win_commit(dw, 0, 1, 0, 16, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_win_commit(dw, 0, 1, 8, 0, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_win_commit(dw, 0, 9, 16, 16, dp, nullptr));   // the graph route's n <= 8
    CHECK(strata::kernels::deepseek4_win_commit(dw, 0, 8, 16, 16, dp, nullptr));

    CHECK(!strata::kernels::deepseek4_win_ring_snap(nullptr, ds, 0, 0, 4, 8, 16, false, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_win_ring_snap(dw, nullptr, 0, 0, 4, 8, 16, false, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_win_ring_snap(dw, ds, 0, 0, 4, 0, 16, false, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_win_ring_snap(dw, ds, 0, 0, 4, 8, 0, false, nullptr, nullptr));

    // n <= i0: success and neither buffer moves.
    std::vector<float> w((size_t) 16 * 16, 3.0f), s((size_t) 16 * 16, 4.0f), w0 = w, s0 = s, back((size_t) 16 * 16);
    CHECK(cudaMemcpy(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(ds, s.data(), s.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(strata::kernels::deepseek4_win_ring_snap(dw, ds, 0, 6, 6, 8, 16, false, nullptr, nullptr));
    to_host(dw, back);
    CHECK(std::memcmp(back.data(), w0.data(), w0.size() * sizeof(float)) == 0);
    to_host(ds, back);
    CHECK(std::memcmp(back.data(), s0.data(), s0.size() * sizeof(float)) == 0);

    // The host oracle refuses the same shapes and leaves its buffers alone.
    strata::kernels::deepseek4_win_commit_host(nullptr, 0, 1, 8, 16, nullptr);
    strata::kernels::deepseek4_win_commit_host(w.data(), 0, 1, 0, 16, nullptr);
    CHECK(std::memcmp(w.data(), w0.data(), w.size() * sizeof(float)) == 0);
    strata::kernels::deepseek4_win_ring_snap_host(w.data(), nullptr, 0, 0, 4, 8, 16, false, nullptr);
    strata::kernels::deepseek4_win_ring_snap_host(w.data(), s.data(), 0, 4, 4, 8, 16, false, nullptr);
    CHECK(std::memcmp(w.data(), w0.data(), w.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(s.data(), s0.data(), s.size() * sizeof(float)) == 0);

    cudaFree(dw);
    cudaFree(ds);
    cudaFree(dp);
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }

    commit_case("commit n<=window", 5, 3, 8, 16, false, 0);
    commit_case("commit n>window", 7, 10, 4, 8, false, 0);
    commit_case("commit pos", 3, 2, 8, 16, true, 11);
    commit_hand_case();
    snap_round_trip_case();
    snap_pos_case();
    guards();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
