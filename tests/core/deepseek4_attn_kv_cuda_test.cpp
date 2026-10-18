// tests/core/deepseek4_attn_kv_cuda_test.cpp - P3's quantized-KV gate.
//
//   deepseek4_attn_kv_cuda_test
//
// Four primitives, each checked against the host oracle in src/kernels/deepseek4_attn_host.cpp, which is the
// reference engine written out loop for loop:
//
//   compress pool  the per-dimension softmax over each complete group of `ratio` rows weighting that group's
//                  kv values, bf16-rounded once (cuda/ds41_attn.cuh:90-99)
//   act quant      the three block modes in place: fp8/e8m0, fp4/e8m0 and fp4/e4m3 (cuda/ds41_attn.cuh:175-193)
//   ckv pack      512 dims as 256 B nibbles + 32 B E4M3 scales = 288 B per group (cuda/ds41_attn.cuh:211-284)
//   idxk pack     128 dims as 64 B nibbles + 4 B E8M0 scales = 68 B padded to 72
//
// act_quant and both packs are pure encode arithmetic - table lookups, exact power-of-two scales, one exact f32
// multiply - so host and device must agree bit for bit; those comparisons are bitwise and name the first
// differing element.  The pool is a softmax, where CUDA's expf and glibc's can differ in the last bit, so it
// uses the usual bf16 tolerance (2e-2 relative, plus the bf16-grid check) against the host oracle and a
// separate double-precision reference; one equal-score case is compared bitwise because exp(0) is exactly 1 on
// both sides and then every remaining operation is exact or the same f32 division.
//
// The pack gate also proves the reference's stated invariant: reading a packed element back equals act_quant's
// own write-back for that value at the same block scale, compared as floats (the only exception is the sign of
// zero: a tiny negative rounds to -0 on the act_quant side and to nibble 0, +0, on the pack side - the two
// compare equal and contribute nothing to any sum).  The index key's four pad bytes are asserted zero, and a
// cache prefilled with 0xAA proves the pack writes exactly its row range.  No model, no timing.  No GPU:
// prints SKIP and exits 77.
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

uint32_t g_rng = 0x243f6a88u;
uint32_t urng() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
float urand() { return static_cast<float>(urng() & 0x00FFFFFFu) / 8388608.0f - 1.0f; }
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

void check_same_bytes(const char* what, const std::vector<uint8_t>& got, const std::vector<uint8_t>& want) {
    CHECK(got.size() == want.size());
    size_t bad = 0;
    for (size_t i = 0; i < got.size() && i < want.size(); i++) {
        if (got[i] != want[i]) {
            if (bad < 4)
                std::fprintf(stderr, "  %s byte %zu: gpu 0x%02x oracle 0x%02x\n", what, i, got[i], want[i]);
            bad++;
        }
    }
    g_checks++;
    if (bad != 0) g_failures++;
    std::printf("  %-22s %zu bytes, %zu differ\n", what, got.size(), bad);
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
uint8_t* dev_b(size_t n, int fill) {
    uint8_t* p = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&p), n) == cudaSuccess);
    CHECK(cudaMemset(p, fill, n) == cudaSuccess);
    return p;
}
void to_host(float* d, std::vector<float>& v) {
    CHECK(cudaMemcpy(v.data(), d, v.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
}
void to_host_b(uint8_t* d, std::vector<uint8_t>& v) {
    CHECK(cudaMemcpy(v.data(), d, v.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
}

// ---- compressor pool ----

void pool_double_reference(const char* what, const std::vector<float>& got, const std::vector<float>& kv,
                           const std::vector<float>& sc, uint32_t n_tok, uint32_t ratio, uint32_t dim) {
    const uint32_t ng = n_tok / ratio;
    size_t bad = 0;
    double worst = 0.0;
    for (uint32_t g = 0; g < ng; g++) {
        for (uint32_t d = 0; d < dim; d++) {
            double mx = -INFINITY;
            for (uint32_t t = 0; t < ratio; t++) mx = std::fmax(mx, (double) sc[((size_t) g * ratio + t) * dim + d]);
            double den = 0.0, acc = 0.0;
            for (uint32_t t = 0; t < ratio; t++) {
                const double e = std::exp((double) sc[((size_t) g * ratio + t) * dim + d] - mx);
                den += e;
                acc += e * (double) kv[((size_t) g * ratio + t) * dim + d];
            }
            const double a = got[(size_t) g * dim + d], b = acc / den;
            const double diff = std::fabs(a - b);
            const double bound = 1e-6 + 2e-2 * std::fmax(std::fabs(a), std::fabs(b));
            const double r = diff / bound;
            if (r > worst) worst = r;
            if (r > 1.0) {
                if (bad < 4) std::fprintf(stderr, "  %s (%u,%u): gpu %.8g double %.8g\n", what, g, d, a, b);
                bad++;
            }
        }
    }
    g_checks++;
    if (bad != 0) g_failures++;
    std::printf("  %-22s %zu values, worst %.3f x double reference, %zu over\n", what,
                (size_t) ng * dim, worst, bad);
}

void pool_case(const char* name, uint32_t n_tok, uint32_t ratio, uint32_t dim, bool equal_scores) {
    const size_t n = (size_t) n_tok * dim;
    std::vector<float> kv(n), sc(n);
    for (size_t i = 0; i < n; i++) {
        kv[i] = 3.0f * bf16_value();
        sc[i] = equal_scores ? 1.5f : 4.0f * urand();
    }
    const uint32_t ng = n_tok / ratio;
    std::vector<float> want((size_t) ng * dim), got((size_t) ng * dim);
    strata::kernels::deepseek4_compress_pool_host(want.data(), kv.data(), sc.data(), n_tok, ratio, dim);

    float* d_kv = dev_f_from(kv);
    float* d_sc = dev_f_from(sc);
    float* d_out = dev_f(want.size());
    CHECK(strata::kernels::deepseek4_compress_pool(d_out, d_kv, d_sc, n_tok, ratio, dim, nullptr));
    to_host(d_out, got);

    if (equal_scores) {
        check_same_bits(name, got, want);
    } else {
        check_close(name, got, want, 2e-2, 1e-6);
        char dn[64];
        std::snprintf(dn, sizeof(dn), "%s double", name);
        pool_double_reference(dn, got, kv, sc, n_tok, ratio, dim);
    }
    char gn[64];
    std::snprintf(gn, sizeof(gn), "%s grid", name);
    check_bf16_grid(gn, got);

    cudaFree(d_kv);
    cudaFree(d_sc);
    cudaFree(d_out);
}

// ---- block quant ----

void act_case(const char* name, uint32_t n_rows, uint32_t dim, uint32_t block, int mode) {
    std::vector<float> x((size_t) n_rows * dim);
    for (float& v : x) v = 7.0f * urand();
    std::vector<float> want = x;
    strata::kernels::deepseek4_act_quant_host(want.data(), n_rows, dim, block, mode);

    float* d_x = dev_f_from(x);
    CHECK(strata::kernels::deepseek4_act_quant(d_x, n_rows, dim, block, mode, nullptr));
    std::vector<float> got(want.size());
    to_host(d_x, got);

    check_same_bits(name, got, want);
    char gn[64];
    std::snprintf(gn, sizeof(gn), "%s grid", name);
    check_bf16_grid(gn, got);
    cudaFree(d_x);
}

void act_ties() {
    // Block scale is exactly 1 (amax 6), so the expected outputs are the tie rule in plain numbers: ties land
    // on the even table index (0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4).
    const std::vector<float> in = {0.25f,  0.75f,  1.25f,  1.75f, 2.5f,  3.5f,  5.0f,  -0.25f,
                                   -0.75f, -1.25f, -1.75f, -2.5f, -3.5f, -5.0f, 6.0f,  -6.0f};
    const std::vector<float> want = {0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 4.0f, 4.0f, -0.0f,
                                     -1.0f, -1.0f, -2.0f, -2.0f, -4.0f, -4.0f, 6.0f, -6.0f};

    float* d_x = dev_f_from(in);
    CHECK(strata::kernels::deepseek4_act_quant(d_x, 1, (uint32_t) in.size(), (uint32_t) in.size(),
                                               strata::kernels::kDeepseek4QuantFp4E8m0, nullptr));
    std::vector<float> got(in.size());
    to_host(d_x, got);
    check_same_bits("act fp4 ties", got, want);
    cudaFree(d_x);
}

void act_zeros() {
    for (int mode = 0; mode <= 2; mode++) {
        const uint32_t block = mode == 2 ? 16 : 32;
        std::vector<float> x(block, 0.0f);
        x[1] = -0.0f;
        std::vector<float> want = x;
        strata::kernels::deepseek4_act_quant_host(want.data(), 1, block, block, mode);
        float* d_x = dev_f_from(x);
        CHECK(strata::kernels::deepseek4_act_quant(d_x, 1, block, block, mode, nullptr));
        std::vector<float> got(block);
        to_host(d_x, got);
        char name[32];
        std::snprintf(name, sizeof(name), "act zeros m%d", mode);
        check_same_bits(name, got, want);
        cudaFree(d_x);
    }
}

// ---- packed caches ----

void pack_case(const char* name, uint32_t n_rows, uint32_t g0, bool idxk, cudaStream_t stream) {
    const uint32_t dim = idxk ? strata::kernels::kDeepseek4IdxkDim : strata::kernels::kDeepseek4CkvDim;
    const uint32_t row_bytes = idxk ? strata::kernels::kDeepseek4IdxkBytes : strata::kernels::kDeepseek4CkvBytes;
    const uint32_t total = g0 + n_rows + 2;

    std::vector<uint8_t> want((size_t) total * row_bytes, 0xAA);
    std::vector<float> rows((size_t) n_rows * dim);
    for (float& v : rows) v = 8.0f * bf16_value();
    // the first two rows: all zeros (scale floors) and alternating +-6 (table values at the top of the range)
    if (n_rows >= 2)
        for (uint32_t d = 0; d < dim; d++) {
            rows[d] = 0.0f;
            rows[dim + d] = (d & 1u) ? 6.0f : -6.0f;
        }

    if (idxk)
        strata::kernels::deepseek4_idxk_pack_host(want.data(), g0, rows.data(), n_rows);
    else
        strata::kernels::deepseek4_ckv_pack_host(want.data(), g0, rows.data(), n_rows);

    uint8_t* d_cache = dev_b(want.size(), 0xAA);
    float* d_rows = dev_f_from(rows);
    const bool packed = idxk ? strata::kernels::deepseek4_idxk_pack(d_cache, g0, d_rows, n_rows, stream)
                             : strata::kernels::deepseek4_ckv_pack(d_cache, g0, d_rows, n_rows, stream);
    CHECK(packed);
    if (stream != nullptr) CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<uint8_t> got(want.size());
    to_host_b(d_cache, got);

    char bn[64];
    std::snprintf(bn, sizeof(bn), "%s bytes", name);
    check_same_bytes(bn, got, want);

    // Unpack the packed rows on both sides.
    const size_t u = (size_t) n_rows * dim;
    std::vector<float> want_u(u), got_u(u);
    if (idxk)
        strata::kernels::deepseek4_idxk_unpack_host(want.data() + (size_t) g0 * row_bytes, n_rows, want_u.data());
    else
        strata::kernels::deepseek4_ckv_unpack_host(want.data() + (size_t) g0 * row_bytes, n_rows, want_u.data());
    float* d_u = dev_f(u);
    const bool unpacked =
        idxk ? strata::kernels::deepseek4_idxk_unpack(d_cache + (size_t) g0 * row_bytes, n_rows, d_u, stream)
             : strata::kernels::deepseek4_ckv_unpack(d_cache + (size_t) g0 * row_bytes, n_rows, d_u, stream);
    CHECK(unpacked);
    if (stream != nullptr) CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    to_host(d_u, got_u);
    char un[64];
    std::snprintf(un, sizeof(un), "%s unpack", name);
    check_same_bits(un, got_u, want_u);
    char gn[64];
    std::snprintf(gn, sizeof(gn), "%s unpack grid", name);
    check_bf16_grid(gn, got_u);

    // The reference's invariant: read-back == act_quant's write-back at the row's block scale.
    const int mode = idxk ? strata::kernels::kDeepseek4QuantFp4E8m0 : strata::kernels::kDeepseek4QuantFp4E4m3;
    const uint32_t blk = idxk ? strata::kernels::kDeepseek4IdxkBlock : strata::kernels::kDeepseek4CkvBlock;
    size_t differ = 0;
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t b = 0; b < dim / blk; b++) {
            float a = 0.0f;
            for (uint32_t l = 0; l < blk; l++) a = std::fmax(a, std::fabs(rows[(size_t) r * dim + b * blk + l]));
            const float s = strata::kernels::deepseek4_quant_scale(a, mode);
            for (uint32_t l = 0; l < blk; l++) {
                const size_t i = (size_t) r * dim + b * blk + l;
                const float v = rows[i];
                if (got_u[i] != strata::kernels::deepseek4_quant_value(v, s, mode)) {
                    if (differ < 4) std::fprintf(stderr, "  %s row %u elem %zu: %.9g vs %.9g\n", name, r, i, got_u[i],
                                                 strata::kernels::deepseek4_quant_value(v, s, mode));
                    differ++;
                }
            }
        }
    }
    g_checks++;
    if (differ != 0) g_failures++;
    std::printf("  %-22s %zu values, %zu differ from act_quant's write-back\n", "identity", u, differ);

    if (idxk) {
        size_t pad = 0;
        for (uint32_t r = 0; r < n_rows; r++) {
            const uint8_t* row = want.data() + (size_t)(g0 + r) * row_bytes;
            for (uint32_t k = strata::kernels::kDeepseek4IdxkNibbleBytes +
                              strata::kernels::kDeepseek4IdxkDim / strata::kernels::kDeepseek4IdxkBlock;
                 k < row_bytes; k++)
                if (row[k] != 0) pad++;
        }
        g_checks++;
        if (pad != 0) g_failures++;
        std::printf("  %-22s %zu pad bytes, %zu nonzero\n", "idxk pad", (size_t) n_rows * (row_bytes - 68), pad);
    }

    cudaFree(d_cache);
    cudaFree(d_rows);
    cudaFree(d_u);
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }

    // compressor pool: the live ratio 2 at 512 dims, ratio 1 (a plain round), an odd group size, a tail
    // shorter than one group, and the exact equal-score case.
    pool_case("pool r1 512", 5, 1, 512, false);
    pool_case("pool r2 512", 6, 2, 512, false);
    pool_case("pool r2 equal", 6, 2, 512, true);
    pool_case("pool r3 256", 8, 3, 256, false);
    pool_case("pool r4 128", 7, 4, 128, false);

    // block quant: every mode at its live geometry, a narrower block, the tie fixture and the zero block.
    act_case("act fp8 b32 d512", 4, 512, 32, strata::kernels::kDeepseek4QuantFp8E8m0);
    act_case("act fp4/e8m0 b32 d128", 8, 128, 32, strata::kernels::kDeepseek4QuantFp4E8m0);
    act_case("act fp4/e4m3 b16 d512", 3, 512, 16, strata::kernels::kDeepseek4QuantFp4E4m3);
    act_case("act fp4/e8m0 b8 d64", 2, 64, 8, strata::kernels::kDeepseek4QuantFp4E8m0);
    act_ties();
    act_zeros();

    // packed caches: a gap before the written rows proves only [g0, g0+n) moves, plus the stream path.
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreate(&stream) == cudaSuccess);
    pack_case("ckv", 5, 3, false, nullptr);
    pack_case("idxk", 5, 3, true, nullptr);
    pack_case("ckv stream", 2, 0, false, stream);
    pack_case("idxk stream", 2, 0, true, stream);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);

    // Guards: null pointers, zero sizes and bad shapes are refused; no complete pool group is a success.
    float* dx = dev_f(128);
    float* dy = dev_f(128);
    uint8_t* dc = dev_b(strata::kernels::kDeepseek4CkvBytes, 0xAA);
    CHECK(!strata::kernels::deepseek4_compress_pool(nullptr, dy, dy, 4, 2, 8, nullptr));
    CHECK(!strata::kernels::deepseek4_compress_pool(dx, nullptr, dy, 4, 2, 8, nullptr));
    CHECK(!strata::kernels::deepseek4_compress_pool(dx, dy, nullptr, 4, 2, 8, nullptr));
    CHECK(!strata::kernels::deepseek4_compress_pool(dx, dy, dy, 4, 0, 8, nullptr));
    {
        // n_tok < ratio: success and the output is untouched.
        std::vector<float> sent(8, 123.0f), back(8);
        CHECK(cudaMemcpy(dx, sent.data(), sent.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(strata::kernels::deepseek4_compress_pool(dx, dy, dy, 1, 2, 8, nullptr));
        to_host(dx, back);
        CHECK(std::memcmp(sent.data(), back.data(), sent.size() * sizeof(float)) == 0);
    }
    CHECK(!strata::kernels::deepseek4_act_quant(nullptr, 1, 64, 32, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_act_quant(dx, 1, 64, 0, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_act_quant(dx, 1, 64, 33, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_act_quant(dx, 1, 48, 32, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_act_quant(dx, 1, 64, 32, 3, nullptr));
    CHECK(!strata::kernels::deepseek4_act_quant(dx, 1, 64, 32, -1, nullptr));
    CHECK(!strata::kernels::deepseek4_ckv_pack(nullptr, 0, dy, 1, nullptr));
    CHECK(!strata::kernels::deepseek4_ckv_pack(dc, 0, nullptr, 1, nullptr));
    CHECK(!strata::kernels::deepseek4_ckv_pack(dc, 0, dy, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_idxk_pack(nullptr, 0, dy, 1, nullptr));
    CHECK(!strata::kernels::deepseek4_idxk_pack(dc, 0, dy, 0, nullptr));
    CHECK(!strata::kernels::deepseek4_ckv_unpack(nullptr, 1, dx, nullptr));
    CHECK(!strata::kernels::deepseek4_ckv_unpack(dc, 1, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_ckv_unpack(dc, 0, dx, nullptr));
    CHECK(!strata::kernels::deepseek4_idxk_unpack(dc, 0, dx, nullptr));
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dc);

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
