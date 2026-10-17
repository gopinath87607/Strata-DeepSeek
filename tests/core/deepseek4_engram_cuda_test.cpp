// tests/core/deepseek4_engram_cuda_test.cpp - P4's device gate: the engram rows, the gate and the wkv
// projection on the GPU, against the host oracle and the reference's f64 emulation.
//
//   deepseek4_engram_cuda_test [reference-dir] [engram-shard-dir]
//
// reference-dir (default "reference") is the reference runtime's fixture tree; the fixtures used here are
// ds4-dfm-rs/tests/fixtures/ds41/engram/:
//   rows.bin / rows.ref.f32     five raw 264-byte rows and their f32 decode - the rows kernel is compared
//                               BYTE FOR BYTE, because every product in the dequant is exact and P4 requires
//                               bit equality, not a tolerance
//   engram.img / engram.cases.txt / engram.ref.f32
//                               three gate cases (dense, absorb, zero) at the artifact's geometry and the
//                               reference's f64 emulation.  The device's f32 lane/tree sums and expf differ
//                               from f64, so dense/zero get the reference's bound - one bf16 grid step at the
//                               output's magnitude plus 1e-5 of the terms summed - and absorb must be exact.
//
// engram-shard-dir, when given, holds the artifact's two sidecar shards.  The test then runs the device chain
// on real bytes: every golden prompt row read through the host's O_DIRECT reader and dequantized on the GPU
// (bitwise against the host decode of the same bytes), and the layer's real 25600x6144 fp8 wkv against a
// float64 dot of the same weights (the reference's 1e-3-of-max gate), for one token and for two.
//
// A synthetic wkv (in_dim 64, out_dim 40, 9 tokens) always runs: it exercises both activation arms (n == 1
// reads f32, n > 1 reads the caller's bf16 copy) and the 8-row segment loop, with no model or fixtures.
//
// No GPU: prints SKIP and exits 77 (ctest's SKIP_RETURN_CODE).  No server, no timing.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include "strata/artifact/deepseek4_engram.hpp"
#include "strata/kernels/deepseek4_engram.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;
int g_sections = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        g_checks++;                                                              \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_failures++;                                                        \
        }                                                                        \
    } while (0)

using strata::deepseek4::EngramShard;
using strata::deepseek4::engram_bf16r;
using strata::deepseek4::engram_e4m3_to_f32;
using strata::deepseek4::engram_e8m0_to_f32;
using strata::deepseek4::engram_rows_f32;

bool read_file(const std::string& path, std::vector<uint8_t>* out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out->resize(n > 0 ? static_cast<size_t>(n) : 0);
    const bool ok = std::fread(out->data(), 1, out->size(), f) == out->size();
    std::fclose(f);
    return ok;
}

bool read_text(const std::string& path, std::string* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

// Reads `n` bytes at an absolute file offset (the wkv planes, unlike a table row, are one contiguous span).
bool read_at(std::FILE* f, uint64_t off, size_t n, uint8_t* dst) {
    if (std::fseek(f, static_cast<long>(off), SEEK_SET) != 0) return false;
    return std::fread(dst, 1, n, f) == n;
}

template <typename T, typename Parse>
const char* parse_values(const char* p, size_t n, int base, T* out, Parse parse) {
    for (size_t i = 0; i < n; i++) {
        while (*p == ' ' || *p == '\t' || *p == '\r') p++;
        if (*p == '\0') return nullptr;
        char* end = nullptr;
        out[i] = static_cast<T>(parse(p, &end, base));
        if (end == p) return nullptr;
        p = end;
    }
    return p;
}

const char* skip_word(const char* p) {
    while (*p == ' ') p++;
    while (*p != '\0' && *p != ' ' && *p != '\t' && *p != '\r') p++;
    return p;
}

int64_t parse_i64(const char* p, char** end, int base) { return std::strtoll(p, end, base); }
uint32_t parse_hex32(const char* p, char** end, int) { return static_cast<uint32_t>(std::strtoul(p, end, 16)); }

float hex_to_f32(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// The ds41 fixture directory is keyed by engram.cases.txt, the ds4-core hash directory by engram_ref.txt.
bool fixture_dir(const std::string& refdir, const char* rel, std::string* out) {
    *out = refdir + "/" + rel;
    for (const char* sentinel : {"/engram.cases.txt", "/engram_ref.txt"}) {
        std::FILE* f = std::fopen((*out + sentinel).c_str(), "rb");
        if (f != nullptr) {
            std::fclose(f);
            return true;
        }
    }
    return false;
}

// bf16 ulp at |x|: the grid spacing of the coarser neighbor of x (the reference test's helper).
float bf16_ulp_at(float x) {
    if (!(x > 0.0f)) return 0.0f;
    int e = 0;
    (void)frexpf(x, &e);
    return ldexpf(1.0f, e - 9);
}

// A value on the bf16 grid, exactly what the engine's activation path produces.
float bf16_value(float x) { return engram_bf16r(x); }

// ---- 1. rows: rows.bin -> rows.ref.f32, bit-exact on the GPU -------------------------------------------

void check_rows_fixture(const std::string& dir) {
    std::vector<uint8_t> raw, ref_bytes;
    if (!read_file(dir + "/rows.bin", &raw) || !read_file(dir + "/rows.ref.f32", &ref_bytes)) {
        std::printf("SKIP rows (no rows.bin / rows.ref.f32)\n");
        return;
    }
    CHECK(raw.size() == 5 * 264);
    CHECK(ref_bytes.size() == 5 * 256 * 4);
    const uint32_t n_rows = 5, hd = 256;

    std::vector<float> want(n_rows * hd);
    CHECK(engram_rows_f32(raw.data(), n_rows, hd, want.data()));
    size_t host_bad = 0;
    for (size_t i = 0; i < want.size(); i++) {
        uint32_t a, b;
        std::memcpy(&a, &want[i], 4);
        std::memcpy(&b, ref_bytes.data() + i * 4, 4);
        host_bad += a != b;
    }
    CHECK(host_bad == 0);

    uint8_t* d_raw = nullptr;
    float* d_out = nullptr;
    CHECK(cudaMalloc(&d_raw, raw.size()) == cudaSuccess);
    CHECK(cudaMalloc(&d_out, want.size() * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpy(d_raw, raw.data(), raw.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(strata::kernels::deepseek4_engram_rows(d_raw, d_out, n_rows, hd, nullptr));

    std::vector<float> got(want.size());
    CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
        uint32_t a, b, c;
        std::memcpy(&a, &got[i], 4);
        std::memcpy(&b, &want[i], 4);
        std::memcpy(&c, ref_bytes.data() + i * 4, 4);
        if (a != b || a != c) {
            if (bad < 4) std::fprintf(stderr, "  rows %zu: gpu %08x host %08x ref %08x\n", i, a, b, c);
            bad++;
        }
    }
    CHECK(bad == 0);
    std::printf("rows: %u x %u, gpu vs host vs reference.f32, %zu bit-diffs\n", n_rows, hd, bad);

    // A refused geometry launches nothing.
    CHECK(!strata::kernels::deepseek4_engram_rows(d_raw, d_out, n_rows, 250, nullptr));
    CHECK(!strata::kernels::deepseek4_engram_rows(nullptr, d_out, n_rows, hd, nullptr));

    cudaFree(d_out);
    cudaFree(d_raw);
    g_sections++;
}

// ---- 2. the gate: device against the reference's f64 emulation ----------------------------------------

struct GateCase {
    std::string kind;
    uint32_t n_tok = 0;
    std::vector<float> h, kv;
};

void check_gate_fixture(const std::string& dir) {
    std::vector<uint8_t> img, ref_bytes;
    std::string cases_text;
    if (!read_file(dir + "/engram.img", &img) || !read_file(dir + "/engram.ref.f32", &ref_bytes) ||
        !read_text(dir + "/engram.cases.txt", &cases_text)) {
        std::printf("SKIP gate (fixtures absent)\n");
        return;
    }
    uint64_t q_off = 0, k_off = 0;
    uint32_t e = 0, hc = 0;
    float eps = 0;
    std::vector<GateCase> cases;
    std::istringstream in(cases_text);
    std::string line;
    while (std::getline(in, line)) {
        const char* p = line.c_str();
        if (std::strncmp(p, "gate ", 5) == 0) {
            p += 5;
            while (*p != '\0') {
                char key[16] = {0}, val[64] = {0};
                if (std::sscanf(p, " %15[^=]=%63s", key, val) != 2) break;
                if (std::strcmp(key, "q_off") == 0) q_off = std::strtoull(val, nullptr, 10);
                else if (std::strcmp(key, "k_off") == 0) k_off = std::strtoull(val, nullptr, 10);
                else if (std::strcmp(key, "e") == 0) e = static_cast<uint32_t>(std::strtoul(val, nullptr, 10));
                else if (std::strcmp(key, "hc") == 0) hc = static_cast<uint32_t>(std::strtoul(val, nullptr, 10));
                else if (std::strcmp(key, "eps") == 0) eps = std::strtof(val, nullptr);
                p = skip_word(p);
            }
            continue;
        }
        if (std::strncmp(p, "case ", 5) == 0) {
            GateCase c;
            char kind[16] = {0};
            unsigned idx = 0;
            CHECK(std::sscanf(p, "case %u %15s n=%u", &idx, kind, &c.n_tok) == 3);
            c.kind = kind;
            cases.push_back(std::move(c));
            continue;
        }
        if (cases.empty()) continue;
        GateCase& c = cases.back();
        if (std::strncmp(p, "h ", 2) == 0) {
            c.h.resize(static_cast<size_t>(c.n_tok) * hc * e);
            std::vector<uint32_t> bits(c.h.size());
            CHECK(parse_values(p + 2, bits.size(), 16, bits.data(), parse_hex32) != nullptr);
            for (size_t i = 0; i < bits.size(); i++) c.h[i] = hex_to_f32(bits[i]);
        } else if (std::strncmp(p, "kv ", 3) == 0) {
            c.kv.resize(static_cast<size_t>(c.n_tok) * (hc + 1) * e);
            std::vector<uint32_t> bits(c.kv.size());
            CHECK(parse_values(p + 3, bits.size(), 16, bits.data(), parse_hex32) != nullptr);
            for (size_t i = 0; i < bits.size(); i++) c.kv[i] = hex_to_f32(bits[i]);
        }
    }
    CHECK(e == 5120 && hc == 4);
    CHECK(cases.size() == 3);
    CHECK(q_off + static_cast<uint64_t>(hc) * e * 4 * 2 <= img.size());
    CHECK(ref_bytes.size() == static_cast<size_t>(3) * 2 * hc * e * 4);

    uint8_t* d_img = nullptr;
    CHECK(cudaMalloc(&d_img, img.size()) == cudaSuccess);
    CHECK(cudaMemcpy(d_img, img.data(), img.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    const float* qw = reinterpret_cast<const float*>(d_img + q_off);
    const float* kw = reinterpret_cast<const float*>(d_img + k_off);
    const float* ref = reinterpret_cast<const float*>(ref_bytes.data());

    size_t ref_off = 0;
    size_t total_bad = 0;
    for (const GateCase& c : cases) {
        const size_t n = static_cast<size_t>(c.n_tok) * hc * e;
        float* d_hc = nullptr;
        float* d_kv = nullptr;
        CHECK(cudaMalloc(&d_hc, n * sizeof(float)) == cudaSuccess);
        CHECK(cudaMalloc(&d_kv, c.kv.size() * sizeof(float)) == cudaSuccess);
        CHECK(cudaMemcpy(d_hc, c.h.data(), n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(cudaMemcpy(d_kv, c.kv.data(), c.kv.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(strata::kernels::deepseek4_engram_gate(d_hc, d_kv, qw, kw, e, hc, c.n_tok, eps, nullptr));
        std::vector<float> got(n);
        CHECK(cudaMemcpy(got.data(), d_hc, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

        int bit_exact = 1, n_diff = 0, over_bound = 0;
        float max_ulp_dist = 0.0f;
        for (size_t i = 0; i < n; i++) {
            if (got[i] == ref[ref_off + i]) continue;
            bit_exact = 0;
            n_diff++;
            const uint64_t t = i / (static_cast<uint64_t>(hc) * e);
            const uint32_t d = static_cast<uint32_t>(i % e);
            const float hin = c.h[i];
            const float val = c.kv[(t * (hc + 1u) + hc) * e + d];
            const float out_m = std::fmax(std::fabs(got[i]), std::fabs(ref[ref_off + i]));
            const float term_m = std::fmax(std::fmax(out_m, std::fabs(hin)), std::fabs(val));
            const float bound = 1e-5f * term_m + bf16_ulp_at(out_m);
            const float diff = std::fabs(got[i] - ref[ref_off + i]);
            if (diff > bound) {
                if (over_bound < 3)
                    std::fprintf(stderr, "  gate %s value %zu: got %a want %a |d| %a bound %a\n", c.kind.c_str(), i,
                                 got[i], ref[ref_off + i], diff, bound);
                over_bound++;
            }
            const float ulp = bf16_ulp_at(out_m);
            const float dist = ulp > 0.0f ? diff / ulp : 0.0f;
            if (dist > max_ulp_dist) max_ulp_dist = dist;
        }
        // absorb must be bit-exact; dense/zero within the reference's bound.
        const bool pass = c.kind == "absorb" ? bit_exact != 0 : over_bound == 0;
        CHECK(pass);
        total_bad += static_cast<size_t>(over_bound);
        std::printf("gate case %-6s n=%u: %zu values, bit-exact %d, %d differ, %d over bound, max ulp dist %.3f\n",
                    c.kind.c_str(), c.n_tok, n, bit_exact, n_diff, over_bound, max_ulp_dist);
        ref_off += n;
        cudaFree(d_kv);
        cudaFree(d_hc);
    }
    CHECK(ref_off == ref_bytes.size() / 4);
    CHECK(strata::kernels::deepseek4_engram_gate(nullptr, nullptr, nullptr, nullptr, e, hc, 1, eps, nullptr) == false);
    std::printf("gate: %zu cases, %zu values over bound against the reference f64 emulation\n", cases.size(),
                total_bad);
    cudaFree(d_img);
    g_sections++;
}

// ---- 3. the synthetic wkv: both activation arms, the segment loop, the refusals -----------------------

void fill_wkv_synthetic(uint32_t in_dim, uint32_t out_dim, std::vector<uint8_t>* bytes) {
    const uint32_t sbc = (in_dim + 31u) / 32u, sbr = (out_dim + 31u) / 32u;
    bytes->assign(static_cast<size_t>(in_dim) * out_dim + sbr * sbc, 0);
    for (uint32_t r = 0; r < out_dim; r++) {
        for (uint32_t c = 0; c < in_dim; c++) {
            const uint8_t mag = static_cast<uint8_t>((r * 7u + c * 13u) % 126u);   // 0..125, never the NaN code
            (*bytes)[static_cast<size_t>(r) * in_dim + c] = static_cast<uint8_t>(mag | ((r & 1u) ? 0x80u : 0u));
        }
    }
    for (uint32_t r = 0; r < out_dim; r++) {
        for (uint32_t c = 0; c < in_dim; c++) {
            const size_t tile = static_cast<size_t>(r >> 5) * sbc + (c >> 5);
            (*bytes)[static_cast<size_t>(in_dim) * out_dim + tile] =
                static_cast<uint8_t>(120u + ((r / 32u) * 2u + c / 32u) % 8u);
        }
    }
}

// The f64 dot of the same weights: the exact decode (each weight is one E4M3 code times one power of two, so
// f32 holds it exactly) times the same bf16-grid activation, summed in double.
void wkv_reference(const std::vector<uint8_t>& wkv, const std::vector<float>& x, uint32_t in_dim, uint32_t out_dim,
                   uint32_t n_tok, std::vector<double>* ref) {
    float e4[256], e8[256];
    for (uint32_t i = 0; i < 256; i++) {
        e4[i] = engram_e4m3_to_f32(static_cast<uint8_t>(i));
        e8[i] = engram_e8m0_to_f32(static_cast<uint8_t>(i));
    }
    const uint32_t sbc = (in_dim + 31u) / 32u;
    const uint8_t* sc = wkv.data() + static_cast<uint64_t>(in_dim) * out_dim;
    ref->assign(static_cast<size_t>(n_tok) * out_dim, 0.0);
    for (uint32_t r = 0; r < out_dim; r++) {
        const uint8_t* wr = wkv.data() + static_cast<uint64_t>(r) * in_dim;
        const uint8_t* scr = sc + static_cast<uint64_t>(r >> 5) * sbc;
        for (uint32_t t = 0; t < n_tok; t++) {
            double acc = 0.0;
            const float* xt = x.data() + static_cast<size_t>(t) * in_dim;
            for (uint32_t c = 0; c < in_dim; c++) {
                const float wv = e4[wr[c]] * e8[scr[c >> 5]];
                acc += static_cast<double>(wv) * xt[c];
            }
            (*ref)[static_cast<size_t>(t) * out_dim + r] = acc;
        }
    }
}

size_t compare_wkv(const std::vector<float>& got, const std::vector<double>& ref, const char* label) {
    double max_abs = 0.0, max_ref = 0.0;
    size_t over = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
        if (std::fabs(ref[i]) > max_ref) max_ref = std::fabs(ref[i]);
        if (d > max_abs) max_abs = d;
    }
    const double bound = 1e-3 * max_ref;
    for (size_t i = 0; i < got.size(); i++) over += std::fabs(static_cast<double>(got[i]) - ref[i]) > bound;
    CHECK(max_ref > 0.0 && max_abs <= bound);
    std::printf("  wkv %s: %zu values, max rel %.3e, max abs %.3e, scale %.3e, %zu over 1e-3*scale\n", label,
                got.size(), max_ref > 0.0 ? max_abs / max_ref : 0.0, max_abs, max_ref, over);
    return over;
}

void check_synthetic_wkv() {
    const uint32_t in_dim = 64, out_dim = 40, n_tok = 9;
    std::vector<uint8_t> wkv;
    fill_wkv_synthetic(in_dim, out_dim, &wkv);
    std::vector<float> x(static_cast<size_t>(n_tok) * in_dim);
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t c = 0; c < in_dim; c++) {
            const float v = 0.5f * std::sin(0.017f * static_cast<float>(t * in_dim + c)) +
                            0.25f * std::cos(0.003f * static_cast<float>(c) + 0.11f * static_cast<float>(t));
            x[static_cast<size_t>(t) * in_dim + c] = bf16_value(v);
        }
    }
    std::vector<double> ref;
    wkv_reference(wkv, x, in_dim, out_dim, n_tok, &ref);

    uint8_t* d_wkv = nullptr;
    float* d_x = nullptr;
    uint16_t* d_x16 = nullptr;
    float* d_y = nullptr;
    CHECK(cudaMalloc(&d_wkv, wkv.size()) == cudaSuccess);
    CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_x16, x.size() * sizeof(uint16_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_y, static_cast<size_t>(n_tok) * out_dim * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpy(d_wkv, wkv.data(), wkv.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    // One token: the f32 activation arm (XB=0).
    std::vector<double> ref1(ref.begin(), ref.begin() + out_dim);
    std::vector<float> got1(out_dim);
    CHECK(strata::kernels::deepseek4_engram_wkv(d_wkv, d_x, nullptr, d_y, in_dim, out_dim, 1, nullptr));
    CHECK(cudaMemcpy(got1.data(), d_y, out_dim * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    compare_wkv(got1, ref1, "n=1 (f32 activations)");

    // Nine tokens: the bf16 arm (XB=1) and the 8-row segment loop (8 + 1).
    CHECK(strata::kernels::deepseek4_engram_x_to_bf16(d_x, d_x16, x.size(), nullptr));
    std::vector<float> got(n_tok * out_dim);
    CHECK(strata::kernels::deepseek4_engram_wkv(d_wkv, d_x, d_x16, d_y, in_dim, out_dim, n_tok, nullptr));
    CHECK(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    compare_wkv(got, ref, "n=9 (bf16 activations, 8+1 segments)");

    // The refusals: geometry, nulls, and a batch without its bf16 copy.
    CHECK(!strata::kernels::deepseek4_engram_wkv(d_wkv, d_x, d_x16, d_y, 48, out_dim, 1, nullptr));
    CHECK(!strata::kernels::deepseek4_engram_wkv(nullptr, d_x, d_x16, d_y, in_dim, out_dim, 1, nullptr));
    CHECK(!strata::kernels::deepseek4_engram_wkv(d_wkv, d_x, nullptr, d_y, in_dim, out_dim, 2, nullptr));
    CHECK(!strata::kernels::deepseek4_engram_x_to_bf16(nullptr, d_x16, x.size(), nullptr));

    cudaFree(d_y);
    cudaFree(d_x16);
    cudaFree(d_x);
    cudaFree(d_wkv);
    std::printf("synthetic wkv: in %u, out %u, %u tokens; both arms and the refusals\n", in_dim, out_dim, n_tok);
    g_sections++;
}

// ---- 4. the real artifact shards, when the directory is given -----------------------------------------

struct RealTable {
    const char* file;
    uint64_t weight_off, scale_off, rows;      // the 384M-row embed table (absolute file offsets)
    uint64_t wkv_weight_off, wkv_scale_off;    // the wkv planes (absolute; scales follow the weights)
    const char* golden;
};

const RealTable kTables[2] = {
    {"model-00047-of-00048.safetensors", 664, 98305579672ull, 384006168ull, 101377710936ull, 101534997336ull,
     "p1.erows_L01.txt"},
    {"model-00048-of-00048.safetensors", 672, 98308271264ull, 384016682ull, 101380486640ull, 101537773040ull,
     "p1.erows_L14.txt"},
};

void check_real_wkv(const std::string& shard_dir, const std::string& golden_dir, const RealTable& t,
                    const char* label) {
    const std::string path = shard_dir + "/" + t.file;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        std::printf("SKIP real %s (no %s)\n", label, path.c_str());
        return;
    }
    const uint32_t in_dim = 6144, out_dim = 25600, n_tok = 2;
    const uint32_t sbc = (in_dim + 31u) / 32u, sbr = (out_dim + 31u) / 32u;
    const uint64_t wbytes = static_cast<uint64_t>(in_dim) * out_dim;
    const uint64_t sbytes = static_cast<uint64_t>(sbr) * sbc;
    std::vector<uint8_t> wkv(static_cast<size_t>(wbytes + sbytes));
    const bool read_ok = read_at(f, t.wkv_weight_off, static_cast<size_t>(wbytes), wkv.data()) &&
                         read_at(f, t.wkv_scale_off, static_cast<size_t>(sbytes), wkv.data() + wbytes);
    std::fclose(f);
    if (!read_ok) {
        std::fprintf(stderr, "  %s: wkv planes could not be read\n", t.file);
        CHECK(false);
        return;
    }

    // Two activations on the bf16 grid, then the f64 dot of the same bytes.
    std::vector<float> x(static_cast<size_t>(n_tok) * in_dim);
    for (uint32_t tkn = 0; tkn < n_tok; tkn++) {
        for (uint32_t c = 0; c < in_dim; c++) {
            const float v = 0.5f * std::sin(0.0017f * static_cast<float>(tkn * in_dim + c)) +
                            0.25f * std::cos(0.0003f * static_cast<float>(c) + 0.07f * static_cast<float>(tkn));
            x[static_cast<size_t>(tkn) * in_dim + c] = bf16_value(v);
        }
    }
    std::vector<double> ref;
    wkv_reference(wkv, x, in_dim, out_dim, n_tok, &ref);

    uint8_t* d_wkv = nullptr;
    float* d_x = nullptr;
    uint16_t* d_x16 = nullptr;
    float* d_y = nullptr;
    CHECK(cudaMalloc(&d_wkv, wkv.size()) == cudaSuccess);
    CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_x16, x.size() * sizeof(uint16_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_y, static_cast<size_t>(n_tok) * out_dim * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpy(d_wkv, wkv.data(), wkv.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    std::vector<float> got1(out_dim);
    std::vector<double> ref1(ref.begin(), ref.begin() + out_dim);
    CHECK(strata::kernels::deepseek4_engram_wkv(d_wkv, d_x, nullptr, d_y, in_dim, out_dim, 1, nullptr));
    CHECK(cudaMemcpy(got1.data(), d_y, out_dim * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    compare_wkv(got1, ref1, label);

    CHECK(strata::kernels::deepseek4_engram_x_to_bf16(d_x, d_x16, x.size(), nullptr));
    std::vector<float> got(n_tok * out_dim);
    CHECK(strata::kernels::deepseek4_engram_wkv(d_wkv, d_x, d_x16, d_y, in_dim, out_dim, n_tok, nullptr));
    CHECK(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    compare_wkv(got, ref, "n=2 (bf16 activations)");

    cudaFree(d_y);
    cudaFree(d_x16);
    cudaFree(d_x);
    cudaFree(d_wkv);

    // The golden prompt rows through the host's O_DIRECT reader, dequantized on the GPU.
    std::string ids_text;
    if (golden_dir.empty() || !read_text(golden_dir + "/" + t.golden, &ids_text)) {
        std::printf("  %s: no %s, golden rows skipped\n", label, t.golden);
        return;
    }
    EngramShard shard;
    std::string err;
    if (!shard.open(path, err)) {
        std::fprintf(stderr, "  %s: %s\n", t.file, err.c_str());
        CHECK(false);
        return;
    }
    if (!shard.check_span(t.weight_off, t.scale_off, t.rows, 256, err)) {
        std::fprintf(stderr, "  %s: %s\n", t.file, err.c_str());
        CHECK(false);
        return;
    }
    std::vector<int64_t> ids;
    std::istringstream idin(ids_text);
    std::string idline;
    while (std::getline(idin, idline)) {
        if (idline.empty()) continue;
        int64_t line_ids[24];
        if (parse_values(idline.c_str(), 24, 10, line_ids, parse_i64) == nullptr) {
            CHECK(false);
            break;
        }
        ids.insert(ids.end(), line_ids, line_ids + 24);
    }
    CHECK(ids.size() == 192);
    if (ids.empty()) return;
    std::vector<uint8_t> raw(ids.size() * 264);
    uint8_t* d_raw = nullptr;
    float* d_rows = nullptr;
    CHECK(cudaMalloc(&d_raw, raw.size()) == cudaSuccess);
    CHECK(cudaMalloc(&d_rows, ids.size() * 256 * sizeof(float)) == cudaSuccess);
    for (size_t i = 0; i < ids.size(); i++) {
        uint8_t* row = raw.data() + i * 264;
        if (!shard.row(t.weight_off, t.scale_off, static_cast<uint64_t>(ids[i]), 256, row, row + 256, err)) {
            std::fprintf(stderr, "  %s golden id %lld: %s\n", t.file, static_cast<long long>(ids[i]), err.c_str());
            CHECK(false);
            break;
        }
    }
    CHECK(cudaMemcpy(d_raw, raw.data(), raw.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    std::vector<float> want(ids.size() * 256);
    CHECK(engram_rows_f32(raw.data(), static_cast<uint32_t>(ids.size()), 256, want.data()));
    CHECK(strata::kernels::deepseek4_engram_rows(d_raw, d_rows, static_cast<uint32_t>(ids.size()), 256, nullptr));
    std::vector<float> rows_got(want.size());
    CHECK(cudaMemcpy(rows_got.data(), d_rows, rows_got.size() * sizeof(float), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
    size_t bad = 0;
    for (size_t i = 0; i < want.size(); i++) {
        uint32_t a, b;
        std::memcpy(&a, &rows_got[i], 4);
        std::memcpy(&b, &want[i], 4);
        bad += a != b;
    }
    CHECK(bad == 0);
    std::printf("  %s: %zu golden rows dequantized on the GPU, %zu bit-diffs vs the host decode (reader %s)\n",
                t.golden, ids.size(), bad, shard.is_direct() ? "O_DIRECT" : "buffered");
    cudaFree(d_rows);
    cudaFree(d_raw);
    g_sections++;
}

void check_real(const std::string& shard_dir, const std::string& golden_dir) {
    const char* labels[2] = {"L01", "L14"};
    for (size_t i = 0; i < 2; i++) check_real_wkv(shard_dir, golden_dir, kTables[i], labels[i]);
}

}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    const std::string refdir = argc > 1 ? argv[1] : "reference";
    const std::string sharddir = argc > 2 ? argv[2] : "";
    std::string gatedir, hashdir;
    const bool have_gate = fixture_dir(refdir, "ds4-dfm-rs/tests/fixtures/ds41/engram", &gatedir);
    const bool have_hash = fixture_dir(refdir, "ds4-dfm-rs/crates/ds4-core/tests/fixtures/engram", &hashdir);
    if (have_gate) {
        check_rows_fixture(gatedir);
        check_gate_fixture(gatedir);
    } else {
        std::printf("SKIP rows/gate fixtures (no %s/ds4-dfm-rs/tests/fixtures/ds41/engram)\n", refdir.c_str());
    }
    check_synthetic_wkv();
    if (!sharddir.empty()) check_real(sharddir, have_hash ? hashdir : std::string());

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
