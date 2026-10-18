// tests/core/deepseek4_moe_cuda_test.cpp - P3's gate: the router and the routed VQ MoE on the GPU.
//
//   deepseek4_moe_cuda_test [reference-dir]
//
// THE ROUTER against its host oracle (src/kernels/deepseek4_moe_host.cpp, which emulates v41_router_kernel
// lane for lane) and against cases whose answers are known by construction:
//   all logits equal            experts 0..topk-1, in order - the tie rule;
//   a six-way tie at the edge   exactly those six, ascending;
//   z = 100                     softplus's torch threshold 20 (without it log1pf(expf(100)) is inf);
//   topk > n_expert             the unselected rounds are -1 with weight 0 exactly;
//   an image token              bias_vl for selection, the text bias for the text tokens.
// Selection ids must be identical.  The weights are compared within 1e-6 of their own scale: the device's
// expf and libm's may differ by an ulp, and that is the only place they can.
//
// THE ROUTED MOE against the reference engine's own fixture -
// reference/ds4-dfm-rs/tests/fixtures/ds41/vq/{moe.blob,moe.cases.txt,moe.ref.f32} - with the reference
// test's criterion (tests/test_ds41_moe.cu): the one-hot cases BIT-EXACT against the reference f32 dump
// (a one-hot activation makes every dot in the chain a single term), the random cases within 1e-3 of the
// vector's own scale.  Then against the host oracle on the same cases with the same criterion: the oracle
// replays the device's lane map, so drift beyond the reference's own tolerance is a port bug.  Finally a
// 2-token call over two of the cases at once, to exercise pair = t*topk + k, and the partials + tail path:
// with d_out == nullptr the entry leaves the partials, and the tail adds a zero shared output, so the tail's
// result must be the reduce's under one bf16 round.
//
// No GPU: prints SKIP and exits 77 (ctest's SKIP_RETURN_CODE).  No model, no server, no timing.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "strata/kernels/deepseek4_dense.hpp"
#include "strata/kernels/deepseek4_moe.hpp"
#include "vq_decode.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;
std::string g_refdir;

#define CHECK(cond)                                                              \
    do {                                                                         \
        g_checks++;                                                              \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_failures++;                                                        \
        }                                                                        \
    } while (0)

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

template <typename T>
T* dev_new(const std::vector<T>& host) {
    T* p = nullptr;
    if (host.empty()) return nullptr;
    if (cudaMalloc(&p, host.size() * sizeof(T)) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    if (cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFree(p);
        return nullptr;
    }
    return p;
}

template <typename T>
bool dev_read(const T* dev, size_t n, std::vector<T>* out) {
    out->resize(n);
    const bool ok = cudaMemcpy(out->data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!ok) (void) cudaGetLastError();
    return ok;
}

// Deterministic logits in [-2, 2]; the router's selection only needs spread, not a distribution.
uint64_t g_lcg = 0x9E3779B97F4A7C15ull;
float frand() {
    g_lcg = g_lcg * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t u = static_cast<uint32_t>((g_lcg >> 33) & 0xFFFFFFu);
    return static_cast<float>(u) / static_cast<float>(0xFFFFFFu) * 4.0f - 2.0f;
}

// ---- router -------------------------------------------------------------------------------------------

struct RouterInput {
    uint32_t n_tok = 1;
    uint32_t n_expert = 384;
    uint32_t topk = 6;
    float route_scale = 1.5f;
    std::vector<float> logits;
    std::vector<float> bias;
    std::vector<float> bias_vl;      // empty: no image bias
    std::vector<int32_t> vl_rows;    // empty: no image rows
};

// One case through the host oracle and the device entry; ids must match, weights within 1e-6 of their scale.
void check_router(const char* name, const RouterInput& in) {
    const size_t n = static_cast<size_t>(in.n_tok) * in.topk;
    std::vector<int32_t> hsel(n, 0);
    std::vector<float> hwts(n, 0.0f);
    const bool host_ok = strata::kernels::deepseek4_router_host(
        hsel.data(), hwts.data(), in.logits.data(), in.bias.data(),
        in.bias_vl.empty() ? nullptr : in.bias_vl.data(),
        in.vl_rows.empty() ? nullptr : in.vl_rows.data(), in.n_tok, in.n_expert, in.topk, in.route_scale);
    CHECK(host_ok);
    if (!host_ok) return;

    int32_t* d_sel = dev_new(std::vector<int32_t>(n, 0));
    float* d_wts = dev_new(std::vector<float>(n, 0.0f));
    float* d_logits = dev_new(in.logits);
    float* d_bias = dev_new(in.bias);
    float* d_bias_vl = dev_new(in.bias_vl);
    int32_t* d_vl_rows = dev_new(in.vl_rows);
    CHECK(d_sel != nullptr && d_wts != nullptr && d_logits != nullptr && d_bias != nullptr &&
          (in.bias_vl.empty() || d_bias_vl != nullptr) && (in.vl_rows.empty() || d_vl_rows != nullptr));
    bool ok = d_sel != nullptr && d_wts != nullptr && d_logits != nullptr && d_bias != nullptr &&
              (in.bias_vl.empty() || d_bias_vl != nullptr) && (in.vl_rows.empty() || d_vl_rows != nullptr);
    std::vector<int32_t> gsel;
    std::vector<float> gwts;
    if (ok)
        ok = strata::kernels::deepseek4_router(d_sel, d_wts, d_logits, d_bias, d_bias_vl, d_vl_rows, in.n_tok,
                                               in.n_expert, in.topk, in.route_scale, nullptr);
    CHECK(ok);
    if (ok) {
        CHECK(dev_read(d_sel, n, &gsel));
        CHECK(dev_read(d_wts, n, &gwts));
    }
    (void) cudaFree(d_sel);
    (void) cudaFree(d_wts);
    (void) cudaFree(d_logits);
    (void) cudaFree(d_bias);
    (void) cudaFree(d_bias_vl);
    (void) cudaFree(d_vl_rows);
    if (!ok || gsel.size() != n) return;

    int id_bad = 0;
    size_t first_bad = n;
    float max_w = 0.0f, w_diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (gsel[i] != hsel[i]) {
            id_bad++;
            if (first_bad == n) first_bad = i;
        }
        max_w = std::max(max_w, std::fabs(hwts[i]));
        w_diff = std::max(w_diff, std::fabs(gwts[i] - hwts[i]));
    }
    if (id_bad != 0)
        std::fprintf(stderr, "router %s: %d of %zu ids differ (first %zu: device %d, oracle %d)\n", name, id_bad,
                     n, first_bad, gsel[first_bad], hsel[first_bad]);
    CHECK(id_bad == 0);
    const float scale = max_w > 0.0f ? max_w : 1.0f;
    CHECK(w_diff <= 1e-6f * scale);
    const bool pass = id_bad == 0 && w_diff <= 1e-6f * scale;
    std::printf("router %-28s tok %u experts %u topk %u: %s (%d ids differ, weight max diff %.3e)\n", name,
                in.n_tok, in.n_expert, in.topk, pass ? "PASS" : "FAIL", id_bad, static_cast<double>(w_diff));
}

void test_router() {
    // All logits equal, 384 experts, topk 8: the ids are 0..7 and the weights are 1/8 (within an ulp of the
    // normalization's own round).
    {
        RouterInput in;
        in.n_tok = 2;
        in.n_expert = 384;
        in.topk = 8;
        in.route_scale = 1.0f;
        in.logits.assign(static_cast<size_t>(in.n_tok) * in.n_expert, 0.0f);
        in.bias.assign(in.n_expert, 0.0f);
        check_router("all-equal ties", in);
        std::vector<int32_t> sel(static_cast<size_t>(in.n_tok) * in.topk);
        std::vector<float> wts(sel.size());
        CHECK(strata::kernels::deepseek4_router_host(sel.data(), wts.data(), in.logits.data(), in.bias.data(),
                                                     nullptr, nullptr, in.n_tok, in.n_expert, in.topk,
                                                     in.route_scale));
        for (size_t i = 0; i < sel.size(); i++) {
            CHECK(sel[i] == static_cast<int32_t>(i % in.topk));
            CHECK(std::fabs(wts[i] - 0.125f) <= 1e-6f);
        }
    }

    // softplus threshold 20: z = 100 must be softplus's linear branch (pr = 10).  The else branch would take
    // log1pf(inf) = inf, and the single weight would be nan instead of 1.
    {
        RouterInput in;
        in.n_expert = 2;
        in.topk = 1;
        in.route_scale = 1.0f;
        in.logits = {100.0f, 1.0f};
        in.bias = {0.0f, 0.0f};
        check_router("threshold 20 (z = 100)", in);
        std::vector<int32_t> sel(1);
        std::vector<float> wts(1);
        CHECK(strata::kernels::deepseek4_router_host(sel.data(), wts.data(), in.logits.data(), in.bias.data(),
                                                     nullptr, nullptr, 1, in.n_expert, 1, 1.0f));
        CHECK(sel[0] == 0);
        CHECK(wts[0] == 1.0f);
    }

    // Six experts tied at the top, the rest lower: the topk boundary tie goes to the smaller ids.
    {
        RouterInput in;
        in.n_expert = 384;
        in.topk = 6;
        in.route_scale = 1.0f;
        in.logits.assign(in.n_expert, 1.0f);
        for (uint32_t e = 0; e < 6; e++) in.logits[e] = 4.0f;
        in.bias.assign(in.n_expert, 0.0f);
        check_router("tie at the top-k boundary", in);
        std::vector<int32_t> sel(in.topk);
        std::vector<float> wts(in.topk);
        CHECK(strata::kernels::deepseek4_router_host(sel.data(), wts.data(), in.logits.data(), in.bias.data(),
                                                     nullptr, nullptr, 1, in.n_expert, in.topk, 1.0f));
        for (uint32_t r = 0; r < in.topk; r++) CHECK(sel[r] == static_cast<int32_t>(r));
    }

    // topk over the expert count: the extra rounds select -1 with weight exactly 0.
    {
        RouterInput in;
        in.n_expert = 8;
        in.topk = 10;
        in.route_scale = 1.0f;
        in.logits.assign(in.n_expert, 0.0f);
        in.bias.assign(in.n_expert, 0.0f);
        check_router("topk > n_expert", in);
        std::vector<int32_t> sel(in.topk);
        std::vector<float> wts(in.topk);
        CHECK(strata::kernels::deepseek4_router_host(sel.data(), wts.data(), in.logits.data(), in.bias.data(),
                                                     nullptr, nullptr, 1, in.n_expert, in.topk, 1.0f));
        for (uint32_t r = 0; r < in.n_expert; r++) CHECK(sel[r] == static_cast<int32_t>(r));
        for (uint32_t r = in.n_expert; r < in.topk; r++) {
            CHECK(sel[r] == -1);
            CHECK(wts[r] == 0.0f);
        }
    }

    // An image token uses bias_vl for the selection; the text tokens use the text bias.
    {
        RouterInput in;
        in.n_tok = 3;
        in.n_expert = 8;
        in.topk = 2;
        in.route_scale = 1.0f;
        in.logits.assign(static_cast<size_t>(in.n_tok) * in.n_expert, 0.0f);
        in.bias.assign(in.n_expert, 0.0f);
        in.bias_vl.assign(in.n_expert, 0.0f);
        in.bias_vl[5] = 100.0f;
        in.vl_rows = {-1, 0, -1};
        check_router("bias_vl (image token)", in);
        std::vector<int32_t> sel(static_cast<size_t>(in.n_tok) * in.topk);
        std::vector<float> wts(sel.size());
        CHECK(strata::kernels::deepseek4_router_host(sel.data(), wts.data(), in.logits.data(), in.bias.data(),
                                                     in.bias_vl.data(), in.vl_rows.data(), in.n_tok, in.n_expert,
                                                     in.topk, 1.0f));
        CHECK(sel[0] == 0 && sel[1] == 1);
        CHECK(sel[2] == 5 && sel[3] == 0);
        CHECK(sel[4] == 0 && sel[5] == 1);
    }

    // Dense logits across the full 384 experts, route_scale 1.5, one token carrying a z > 20 expert.
    {
        RouterInput in;
        in.n_tok = 5;
        in.n_expert = 384;
        in.topk = 6;
        in.route_scale = 1.5f;
        in.logits.resize(static_cast<size_t>(in.n_tok) * in.n_expert);
        for (auto& z : in.logits) z = frand();
        in.logits[3 * in.n_expert + 7] = 100.0f;
        in.bias.resize(in.n_expert);
        for (auto& b : in.bias) b = frand() * 0.1f;
        check_router("dense 384 experts", in);
    }

    // Guards: the reference refuses topk > 16 and n_expert > 384, host and device alike.
    {
        RouterInput in;
        in.n_expert = 8;
        in.topk = 17;
        in.logits.assign(in.n_expert, 0.0f);
        in.bias.assign(in.n_expert, 0.0f);
        std::vector<int32_t> sel(64);
        std::vector<float> wts(64);
        CHECK(!strata::kernels::deepseek4_router_host(sel.data(), wts.data(), in.logits.data(), in.bias.data(),
                                                      nullptr, nullptr, 1, in.n_expert, in.topk, 1.0f));
        std::vector<int32_t> z(64, 0);
        int32_t* d_sel = dev_new(z);
        float* d_wts = dev_new(std::vector<float>(64, 0.0f));
        float* d_logits = dev_new(in.logits);
        float* d_bias = dev_new(in.bias);
        CHECK(d_sel != nullptr && d_wts != nullptr && d_logits != nullptr && d_bias != nullptr);
        if (d_sel && d_wts && d_logits && d_bias)
            CHECK(!strata::kernels::deepseek4_router(d_sel, d_wts, d_logits, d_bias, nullptr, nullptr, 1,
                                                     in.n_expert, in.topk, 1.0f, nullptr));
        (void) cudaFree(d_sel);
        (void) cudaFree(d_wts);
        (void) cudaFree(d_logits);
        (void) cudaFree(d_bias);
    }
    {
        std::vector<float> logits(385, 0.0f), bias(385, 0.0f);
        std::vector<int32_t> sel(8);
        std::vector<float> wts(8);
        CHECK(!strata::kernels::deepseek4_router_host(sel.data(), wts.data(), logits.data(), bias.data(),
                                                      nullptr, nullptr, 1, 385, 8, 1.0f));
    }
    // n_tok == 0 is a no-op success on the device.
    {
        std::vector<float> logits(8, 0.0f), bias(8, 0.0f);
        std::vector<int32_t> z(8, 0);
        int32_t* d_sel = dev_new(z);
        float* d_wts = dev_new(std::vector<float>(8, 0.0f));
        float* d_logits = dev_new(logits);
        float* d_bias = dev_new(bias);
        CHECK(d_sel != nullptr && d_wts != nullptr && d_logits != nullptr && d_bias != nullptr);
        if (d_sel && d_wts && d_logits && d_bias)
            CHECK(strata::kernels::deepseek4_router(d_sel, d_wts, d_logits, d_bias, nullptr, nullptr, 0, 8, 8,
                                                    1.0f, nullptr));
        (void) cudaFree(d_sel);
        (void) cudaFree(d_wts);
        (void) cudaFree(d_logits);
        (void) cudaFree(d_bias);
    }
}

// ---- the routed MoE fixture ---------------------------------------------------------------------------

struct MoeCase {
    std::string kind;
    uint32_t k = 0;
    std::vector<int32_t> sel;
    std::vector<float> w;
    std::vector<float> x;
};

bool parse_cases(const std::string& path, uint32_t in_dim, std::vector<MoeCase>* out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string tok;
        ls >> tok;
        if (tok == "case") {
            MoeCase c;
            uint32_t idx = 0;
            std::string ktok;
            if (!(ls >> idx >> c.kind >> ktok) || ktok.rfind("K=", 0) != 0) return false;
            c.k = static_cast<uint32_t>(std::stoul(ktok.substr(2)));
            out->push_back(std::move(c));
        } else if (!out->empty()) {
            MoeCase& c = out->back();
            if (tok == "sel") {
                c.sel.resize(c.k);
                for (auto& v : c.sel) {
                    std::string s;
                    ls >> s;
                    v = static_cast<int32_t>(std::stol(s));
                }
            } else if (tok == "w") {
                c.w.resize(c.k);
                for (auto& v : c.w) {
                    std::string s;
                    ls >> s;
                    const uint32_t bits = static_cast<uint32_t>(std::stoul(s, nullptr, 16));
                    std::memcpy(&v, &bits, 4);
                }
            } else if (tok == "x") {
                c.x.resize(in_dim);
                for (auto& v : c.x) {
                    std::string s;
                    ls >> s;
                    const uint32_t bits = static_cast<uint32_t>(std::stoul(s, nullptr, 16));
                    std::memcpy(&v, &bits, 4);
                }
            }
        }
    }
    return !out->empty();
}

uint32_t float_bits(float v) {
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}

// The reference test's criterion, per case.
struct Compare {
    bool bit_exact = true;
    float max_abs = 0.0f;
    float max_ref = 0.0f;
    int n_diff = 0;
};

Compare compare(const float* got, const float* want, uint32_t n) {
    Compare c;
    for (uint32_t i = 0; i < n; i++) {
        c.max_ref = std::max(c.max_ref, std::fabs(want[i]));
        if (float_bits(got[i]) == float_bits(want[i])) continue;
        c.bit_exact = false;
        c.n_diff++;
        c.max_abs = std::max(c.max_abs, std::fabs(got[i] - want[i]));
    }
    return c;
}

void test_moe_fixture() {
    const std::string dir = g_refdir + "/ds4-dfm-rs/tests/fixtures/ds41/vq/";
    std::vector<uint8_t> blob;
    std::vector<uint8_t> ref_bytes;
    if (!read_file(dir + "moe.blob", &blob) || !read_file(dir + "moe.ref.f32", &ref_bytes)) {
        std::printf("SKIP routed MoE fixture (no moe.blob / moe.ref.f32 under %s)\n", dir.c_str());
        return;
    }

    // The fixture's geometry and codebook sentence, from the blob itself - the same fields the reference
    // entry reads before it launches (version from the container, nc from the first payload).
    const uint64_t off0 = dsv4::blob_slot(blob.data(), 0, 0);
    const uint64_t off2 = dsv4::blob_slot(blob.data(), 0, 2);
    CHECK(off0 != 0 && off0 + 32 <= blob.size());
    CHECK(off2 != 0 && off2 + 32 <= blob.size());
    if (off0 == 0 || off2 == 0 || off0 + 32 > blob.size() || off2 + 32 > blob.size()) return;
    const uint32_t mid_dim = dsv4::rd_u32(blob.data(), off0 + 8);
    const uint32_t in_dim = dsv4::rd_u32(blob.data(), off0 + 12);
    const uint32_t out_dim = dsv4::rd_u32(blob.data(), off2 + 8);
    CHECK(dsv4::rd_u32(blob.data(), off2 + 12) == mid_dim);
    const uint32_t nc = dsv4::rd_u16(blob.data(), off0 + 6);
    const uint32_t payload_magic = dsv4::rd_u32(blob.data(), off0);
    const uint32_t blob_v3 = payload_magic == 0x33565144u ? 1u : 0u;
    CHECK(in_dim == 2048u && mid_dim == 2048u && out_dim == 512u);
    CHECK(nc == 8192u && blob_v3 == 1u);

    std::vector<MoeCase> cases;
    if (!parse_cases(dir + "moe.cases.txt", in_dim, &cases)) {
        CHECK(!"moe.cases.txt did not parse");
        return;
    }
    CHECK(ref_bytes.size() == cases.size() * out_dim * 4u);
    if (ref_bytes.size() != cases.size() * out_dim * 4u) return;
    const float* ref = reinterpret_cast<const float*>(ref_bytes.data());

    uint8_t* d_blob = nullptr;
    CHECK(cudaMalloc(&d_blob, blob.size()) == cudaSuccess);
    if (d_blob == nullptr) return;
    CHECK(cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice) == cudaSuccess);

    int case_fail = 0;
    float worst_rel = 0.0f;      // device vs the reference emulation
    float worst_oracle = 0.0f;   // device vs our host oracle
    for (uint32_t ci = 0; ci < cases.size(); ci++) {
        const MoeCase& c = cases[ci];
        const float* want = ref + static_cast<size_t>(ci) * out_dim;

        // The host oracle: the same chain with the device's lane map.  Relative to the fixture's own scale.
        std::vector<float> oracle(out_dim, 0.0f);
        const bool host_ok = strata::kernels::deepseek4_routed_moe_host(
            oracle.data(), blob.data(), blob.size(), in_dim, mid_dim, out_dim, c.sel.data(), c.w.data(), c.k,
            10.0f, c.x.data(), 1, nullptr);
        CHECK(host_ok);

        float* d_x = dev_new(c.x);
        int32_t* d_sel = dev_new(c.sel);
        float* d_w = dev_new(c.w);
        float* d_out = dev_new(std::vector<float>(out_dim, 0.0f));
        float* d_part = dev_new(std::vector<float>(static_cast<size_t>(c.k) * out_dim, 0.0f));
        CHECK(d_x != nullptr && d_sel != nullptr && d_w != nullptr && d_out != nullptr && d_part != nullptr);
        bool ok = d_x != nullptr && d_sel != nullptr && d_w != nullptr && d_out != nullptr && d_part != nullptr;
        std::vector<float> got;
        if (ok)
            ok = strata::kernels::deepseek4_routed_moe(d_out, d_part, d_blob, blob.size(), in_dim, mid_dim,
                                                       out_dim, d_sel, d_w, c.k, 10.0f, d_x, 1, nc, blob_v3,
                                                       nullptr, nullptr);
        CHECK(ok);
        if (ok) CHECK(dev_read(d_out, out_dim, &got));

        const bool onehot = c.kind == "onehot";
        bool pass = false;
        Compare vs_ref;
        Compare vs_oracle;
        if (ok) {
            vs_ref = compare(got.data(), want, out_dim);
            vs_oracle = compare(got.data(), oracle.data(), out_dim);
            // Reference criterion for the one-hot cases; the fixture's scale tolerance for the random ones.
            pass = onehot ? vs_ref.bit_exact : (vs_ref.max_abs <= 1e-3f * vs_ref.max_ref);
            const bool oracle_pass = vs_oracle.max_abs <= 1e-3f * vs_ref.max_ref;
            if (!oracle_pass) pass = false;
            const float rel = vs_ref.max_ref > 0.0f ? vs_ref.max_abs / vs_ref.max_ref : 0.0f;
            worst_rel = std::max(worst_rel, rel);
            const float orel = vs_ref.max_ref > 0.0f ? vs_oracle.max_abs / vs_ref.max_ref : 0.0f;
            worst_oracle = std::max(worst_oracle, orel);
            std::printf("moe case %u %-6s K=%u: %s%s (ref max abs %.3e = %.3e of scale, %d diffs; oracle "
                        "max abs %.3e)\n",
                        ci, c.kind.c_str(), c.k, pass ? "PASS" : "FAIL",
                        onehot ? " bit-exact" : "", static_cast<double>(vs_ref.max_abs), static_cast<double>(rel),
                        vs_ref.n_diff, static_cast<double>(vs_oracle.max_abs));
        } else {
            std::printf("moe case %u %-6s K=%u: FAIL (entry refused)\n", ci, c.kind.c_str(), c.k);
        }
        if (!pass) case_fail++;

        // The partials + tail path on the last random case: d_out == nullptr leaves partials; a zero shared
        // output must then reproduce the reduce's value under one bf16 round.
        if (ok && ci + 1u == cases.size()) {
            std::vector<float> y(out_dim, 0.0f);
            std::vector<float> so(out_dim, 0.0f);
            float* d_y = dev_new(y);
            float* d_so = dev_new(so);
            bool tail_ok = d_y != nullptr && d_so != nullptr;
            if (tail_ok)
                tail_ok = strata::kernels::deepseek4_routed_moe(nullptr, d_part, d_blob, blob.size(), in_dim,
                                                                mid_dim, out_dim, d_sel, d_w, c.k, 10.0f, d_x,
                                                                1, nc, blob_v3, nullptr, nullptr);
            if (tail_ok)
                tail_ok = strata::kernels::deepseek4_moe_tail(d_y, d_part, d_w, d_so, 1, c.k, out_dim, nullptr);
            CHECK(tail_ok);
            std::vector<float> got_y;
            if (tail_ok) CHECK(dev_read(d_y, out_dim, &got_y));
            int bad = 0;
            if (got_y.size() == out_dim)
                for (uint32_t o = 0; o < out_dim; o++)
                    if (float_bits(got_y[o]) != float_bits(strata::kernels::deepseek4_dense_bf16r(got[o]))) bad++;
            std::printf("moe tail (partials, so = 0): %s (%d of %u differ from bf16(reduce))\n",
                        bad == 0 ? "PASS" : "FAIL", bad, out_dim);
            CHECK(bad == 0);
            (void) cudaFree(d_y);
            (void) cudaFree(d_so);
        }
        (void) cudaFree(d_x);
        (void) cudaFree(d_sel);
        (void) cudaFree(d_w);
        (void) cudaFree(d_out);
        (void) cudaFree(d_part);
    }
    (void) cudaFree(d_blob);

    // Pair indexing: two tokens in one call.  The reference's MoE takes a single n_expert_used per call, so
    // the two tokens are two copies of the first K=6 case - pair = t*K + k must then address them apart.
    {
        const MoeCase& b = cases[6];
        CHECK(b.k == 6u);
        std::vector<float> x2(b.x);
        x2.insert(x2.end(), b.x.begin(), b.x.end());
        std::vector<int32_t> sel2(b.sel);
        sel2.insert(sel2.end(), b.sel.begin(), b.sel.end());
        std::vector<float> w2(b.w);
        w2.insert(w2.end(), b.w.begin(), b.w.end());
        std::vector<float> oracle2(static_cast<size_t>(out_dim) * 2, 0.0f);
        CHECK(strata::kernels::deepseek4_routed_moe_host(oracle2.data(), blob.data(), blob.size(), in_dim,
                                                         mid_dim, out_dim, sel2.data(), w2.data(), b.k, 10.0f,
                                                         x2.data(), 2, nullptr));
        uint8_t* d_blob2 = nullptr;
        CHECK(cudaMalloc(&d_blob2, blob.size()) == cudaSuccess);
        if (d_blob2 != nullptr) {
            CHECK(cudaMemcpy(d_blob2, blob.data(), blob.size(), cudaMemcpyHostToDevice) == cudaSuccess);
            float* d_x = dev_new(x2);
            int32_t* d_sel = dev_new(sel2);
            float* d_w = dev_new(w2);
            float* d_out = dev_new(std::vector<float>(static_cast<size_t>(out_dim) * 2, 0.0f));
            float* d_part = dev_new(std::vector<float>(static_cast<size_t>(out_dim) * 2 * b.k, 0.0f));
            CHECK(d_x != nullptr && d_sel != nullptr && d_w != nullptr && d_out != nullptr && d_part != nullptr);
            bool ok = d_x != nullptr && d_sel != nullptr && d_w != nullptr && d_out != nullptr && d_part != nullptr;
            std::vector<float> got;
            if (ok)
                ok = strata::kernels::deepseek4_routed_moe(d_out, d_part, d_blob2, blob.size(), in_dim, mid_dim,
                                                           out_dim, d_sel, d_w, b.k, 10.0f, d_x, 2, nc, blob_v3,
                                                           nullptr, nullptr);
            CHECK(ok);
            if (ok) CHECK(dev_read(d_out, static_cast<size_t>(out_dim) * 2, &got));
            Compare cmp;
            if (got.size() == oracle2.size()) cmp = compare(got.data(), oracle2.data(), 2 * out_dim);
            float max_ref = 0.0f;
            for (float v : oracle2) max_ref = std::max(max_ref, std::fabs(v));
            const bool pass = got.size() == oracle2.size() && cmp.max_abs <= 1e-3f * max_ref;
            std::printf("moe 2-token call (K=6 twice): %s (max abs %.3e of scale %.3e)\n", pass ? "PASS" : "FAIL",
                        static_cast<double>(cmp.max_abs), static_cast<double>(max_ref));
            CHECK(pass);
            (void) cudaFree(d_x);
            (void) cudaFree(d_sel);
            (void) cudaFree(d_w);
            (void) cudaFree(d_out);
            (void) cudaFree(d_part);
            (void) cudaFree(d_blob2);
        }
    }

    std::printf("routed MoE: %s (%zu cases, worst %.3e of scale vs the reference emulation, %.3e vs the "
                "oracle; %d case failures)\n",
                case_fail == 0 ? "PASS" : "FAIL", cases.size(), static_cast<double>(worst_rel),
                static_cast<double>(worst_oracle), case_fail);
    CHECK(case_fail == 0);
}

// ---- the tail -----------------------------------------------------------------------------------------

void test_tail() {
    const uint32_t n_tok = 3, topk = 4, out_dim = 64;
    std::vector<float> partial(static_cast<size_t>(n_tok) * topk * out_dim), w(static_cast<size_t>(n_tok) * topk),
        so(static_cast<size_t>(n_tok) * out_dim);
    for (auto& v : partial) v = frand() * 2.0f;
    for (auto& v : w) v = frand() * 0.5f;
    for (auto& v : so) v = frand();
    std::vector<float> want(static_cast<size_t>(n_tok) * out_dim, 0.0f);
    strata::kernels::deepseek4_moe_tail_host(want.data(), partial.data(), w.data(), so.data(), n_tok, topk,
                                             out_dim);
    float* d_partial = dev_new(partial);
    float* d_w = dev_new(w);
    float* d_so = dev_new(so);
    float* d_y = dev_new(std::vector<float>(want.size(), 0.0f));
    CHECK(d_partial != nullptr && d_w != nullptr && d_so != nullptr && d_y != nullptr);
    bool ok = d_partial != nullptr && d_w != nullptr && d_so != nullptr && d_y != nullptr;
    std::vector<float> got;
    if (ok) ok = strata::kernels::deepseek4_moe_tail(d_y, d_partial, d_w, d_so, n_tok, topk, out_dim, nullptr);
    CHECK(ok);
    if (ok) CHECK(dev_read(d_y, want.size(), &got));
    int bad = 0, off_grid = 0;
    if (got.size() == want.size()) {
        for (size_t i = 0; i < want.size(); i++) {
            if (float_bits(got[i]) != float_bits(want[i]) && std::fabs(got[i] - want[i]) > 1e-6f) bad++;
            if ((float_bits(got[i]) & 0xFFFFu) != 0u) off_grid++;
        }
    }
    std::printf("moe tail: %s (%d of %zu differ, %d off the bf16 grid)\n", bad == 0 && off_grid == 0 ? "PASS" : "FAIL",
                bad, want.size(), off_grid);
    CHECK(bad == 0);
    CHECK(off_grid == 0);
    // Guards: a null buffer is refused, an empty call is a success.
    CHECK(!strata::kernels::deepseek4_moe_tail(nullptr, d_partial, d_w, d_so, n_tok, topk, out_dim, nullptr));
    CHECK(strata::kernels::deepseek4_moe_tail(d_y, d_partial, d_w, d_so, 0, topk, out_dim, nullptr));
    (void) cudaFree(d_partial);
    (void) cudaFree(d_w);
    (void) cudaFree(d_so);
    (void) cudaFree(d_y);
}

// The routed entry's guards, without a blob: shape and null refusals come before any device read.
void test_entry_guards() {
    float* d = nullptr;
    CHECK(cudaMalloc(&d, 4096 * sizeof(float)) == cudaSuccess);
    if (d == nullptr) return;
    const int32_t* sel = reinterpret_cast<const int32_t*>(d);
    const float* w = d;
    const float* x = d;
    CHECK(!strata::kernels::deepseek4_routed_moe(d, d, d, 64, 8, 8, 8, sel, w, 0, 10.0f, x, 1, 4096, 1, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_routed_moe(d, d, d, 64, 7, 8, 8, sel, w, 1, 10.0f, x, 1, 4096, 1, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_routed_moe(d, d, d, 64, 8, 8, 8, nullptr, w, 1, 10.0f, x, 1, 4096, 1, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_routed_moe(nullptr, nullptr, d, 64, 8, 8, 8, sel, w, 1, 10.0f, x, 1, 4096, 1, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_routed_moe(d, d, nullptr, 64, 8, 8, 8, sel, w, 1, 10.0f, x, 1, 4096, 1, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_routed_moe(d, d, d, 64, 8, 8, 8, sel, w, 1, 10.0f, x, 0, 4096, 1, nullptr, nullptr));
    (void) cudaFree(d);
}

}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    g_refdir = argc > 1 ? argv[1] : "reference";
    test_router();
    test_entry_guards();
    test_tail();
    test_moe_fixture();
    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
