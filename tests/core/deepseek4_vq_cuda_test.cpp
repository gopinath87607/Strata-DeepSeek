// tests/core/deepseek4_vq_cuda_test.cpp - P2's gate: the GPU VQ decode against the CPU decoder.
//
//   deepseek4_vq_cuda_test [reference-dir]
//
// The CPU decoder (tools/dsv4/vq_decode.hpp) is the P0 artifact - it was checked against the engine's own v2
// fixtures and the real layer-0 payload.  This test runs the CUDA kernel over the same inputs and requires
// equality for the decoded values, BYTE FOR BYTE, because the decode path has no floating-point judgment in
// it: every value is one exact codebook lookup times one exact gain.  A mismatch is a layout or conversion
// bug, not accumulation drift.
//
// Three input sets, strongest last:
//   v2_nc256 / v2_nc4096   the engine's v2 fixtures (16x64 matrices) and their reference f32 dumps, so the
//                          gate is against the engine itself, not only our decoder
//   expert 0, layer 0      the real v3 13-bit payloads (2304x5120 gate/up, 5120x2304 down) from the 8 MiB
//                          layer-0 head; CPU decode vs GPU decode, bitwise, all 35.4M values
//
// The row-dot kernel is checked there too, on the same matrices: one warp per row over the packed index
// stream, against a float64 dot of the decoded values and the same bf16 activation.  That comparison is a
// tolerance, not bit equality - the GPU's per-lane float accumulation is allowed to differ from float64 -
// and the bit-exact claim for the dot is P3's, against the reference engine, where the accumulation order
// and contraction can be compared like for like.
//
// No GPU: prints SKIP and exits 77 (ctest's SKIP_RETURN_CODE).  No model, no server, no timing.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "strata/kernels/deepseek4_vq.hpp"
#include "vq_decode.hpp"

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

// The descriptor's pointers are addresses in the host blob; the kernels read the device copy.  Re-base by
// offset - the same step the engine takes when it uploads a blob - and keep a null pointer null.
strata::kernels::Deepseek4VqMatrix rebase(const strata::kernels::Deepseek4VqMatrix& m, const uint8_t* host,
                                          const uint8_t* device) {
    strata::kernels::Deepseek4VqMatrix d = m;
    const auto off = [host, device](const uint8_t* p) {
        return p == nullptr ? nullptr : device + (p - host);
    };
    d.codebook = off(m.codebook);
    d.gains = off(m.gains);
    d.stream = off(m.stream);
    d.plane = off(m.plane);
    d.gain_override = nullptr;   // a host pointer has no device meaning; the engine rebases it the same way
    return d;
}

// One decoded matrix on the GPU vs the CPU decoder, bitwise, plus the reference dump when given.
void check_decode(const char* name, const std::vector<uint8_t>& blob, uint32_t e, uint32_t which,
                  uint32_t rows, uint32_t cols, const std::vector<uint8_t>* ref_bytes) {
    dsv4::VqMatrix cpu;
    const char* err = nullptr;
    CHECK(dsv4::VqMatrix::open(blob.data(), blob.size(), e, which, rows, cols, &cpu, &err));
    strata::kernels::Deepseek4VqMatrix gpu;
    CHECK(strata::kernels::deepseek4_vq_open(blob.data(), blob.size(), e, which, rows, cols, &gpu, &err));

    // The two openers must agree on the geometry before any value is compared.
    CHECK(cpu.rows == gpu.rows && cpu.cols == gpu.cols && cpu.nidx_row() == gpu.nidx_row &&
          cpu.nbit == gpu.nbit && cpu.nc == gpu.nc && static_cast<bool>(cpu.v3) == (gpu.v3 != 0));

    std::vector<float> want(static_cast<size_t>(rows) * cols);
    cpu.dequant_f32(want.data());

    uint8_t* d_blob = nullptr;
    float* d_out = nullptr;
    CHECK(cudaMalloc(&d_blob, blob.size()) == cudaSuccess);
    CHECK(cudaMalloc(&d_out, want.size() * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice) == cudaSuccess);

    strata::kernels::Deepseek4VqMatrix dm = rebase(gpu, blob.data(), d_blob);
    strata::kernels::deepseek4_vq_dequant(dm, d_out, nullptr);   // null stream: the call synchronizes

    std::vector<float> got(want.size());
    CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    size_t bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
        uint32_t a, b;
        std::memcpy(&a, &got[i], 4);
        std::memcpy(&b, &want[i], 4);
        if (a != b) {
            if (bad < 4) std::fprintf(stderr, "  %s value %zu: gpu %08x cpu %08x\n", name, i, a, b);
            bad++;
        }
    }
    CHECK(bad == 0);
    std::printf("%s: %u x %u (nc %u, %u-bit, v%u), %zu values, gpu vs cpu %zu mismatches\n", name, rows, cols,
                gpu.nc, gpu.nbit, gpu.v3 ? 3u : 2u, got.size(), bad);

    if (ref_bytes != nullptr) {
        CHECK(ref_bytes->size() == got.size() * sizeof(float));
        size_t rbad = 0;
        for (size_t i = 0; i < got.size() && (i + 1) * 4 <= ref_bytes->size(); i++) {
            float r;
            std::memcpy(&r, ref_bytes->data() + i * 4, 4);
            uint32_t a, b;
            std::memcpy(&a, &got[i], 4);
            std::memcpy(&b, &r, 4);
            if (a != b) rbad++;
        }
        CHECK(rbad == 0);
        std::printf("  against the engine dump: %zu mismatches\n", rbad);
    }

    cudaFree(d_out);
    cudaFree(d_blob);
}

// bf16 round-to-nearest-even, the engine's v41_bf16r; the activation the dot sees sits on this grid.
uint16_t bf16_of(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    if ((bits & 0x7F800000u) != 0x7F800000u) {
        bits += 0x7FFFu + ((bits >> 16) & 1u);
    }
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_f32(uint16_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// GPU row dot vs float64 over the decoded matrix and the same bf16 activation.
void check_gemv(const char* name, const std::vector<uint8_t>& blob, uint32_t e, uint32_t which, uint32_t rows,
                uint32_t cols) {
    dsv4::VqMatrix cpu;
    strata::kernels::Deepseek4VqMatrix gpu;
    const char* err = nullptr;
    CHECK(dsv4::VqMatrix::open(blob.data(), blob.size(), e, which, rows, cols, &cpu, &err));
    CHECK(strata::kernels::deepseek4_vq_open(blob.data(), blob.size(), e, which, rows, cols, &gpu, &err));

    std::vector<float> w(static_cast<size_t>(rows) * cols);
    cpu.dequant_f32(w.data());

    // A deterministic activation, rounded to bf16 exactly as the engine's activation path does.
    std::vector<uint32_t> x(static_cast<size_t>(cols + 1) / 2, 0u);
    std::vector<float> xf(cols);
    for (uint32_t j = 0; j < cols; j++) {
        const float v = 0.5f * std::sin(0.017f * static_cast<float>(j)) + 0.25f * std::cos(0.003f * j);
        const uint16_t b = bf16_of(v);
        xf[j] = bf16_to_f32(b);
        if ((j & 1u) == 0) x[j >> 1] = b;
        else x[j >> 1] |= static_cast<uint32_t>(b) << 16;
    }

    uint8_t* d_blob = nullptr;
    uint32_t* d_x = nullptr;
    float* d_y = nullptr;
    CHECK(cudaMalloc(&d_blob, blob.size()) == cudaSuccess);
    CHECK(cudaMalloc(&d_x, x.size() * sizeof(uint32_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_y, static_cast<size_t>(rows) * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess);

    strata::kernels::Deepseek4VqMatrix dm = rebase(gpu, blob.data(), d_blob);
    CHECK(strata::kernels::deepseek4_vq_gemv_bf16(dm, d_x, d_y, nullptr));

    std::vector<float> got(rows);
    CHECK(cudaMemcpy(got.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    double worst_rel = 0.0;
    size_t bad = 0;
    for (uint32_t r = 0; r < rows; r++) {
        double want = 0.0, scale = 1.0;
        const float* wr = w.data() + static_cast<size_t>(r) * cols;   // already scaled by the row gain
        for (uint32_t j = 0; j < cols; j++) {
            const double term = static_cast<double>(wr[j]) * xf[j];
            want += term;
            scale += std::fabs(term);
        }
        // A dot cancels: a row's sum can be near zero while its terms are not, so the bound is relative to
        // the sum of |terms| (the standard forward-error scale), not to the result.
        const double err = std::fabs(static_cast<double>(got[r]) - want);
        const double rel = err / scale;
        if (rel > worst_rel) worst_rel = rel;
        if (!(rel <= 1e-5)) bad++;
    }
    CHECK(bad == 0);
    std::printf("  row dot %s: %u rows, worst |gpu-f64|/(1+sum|term|) %.2e, %zu rows over 1e-5\n", name, rows,
                worst_rel, bad);

    cudaFree(d_y);
    cudaFree(d_x);
    cudaFree(d_blob);
}

void print_error_names(const std::vector<uint8_t>& v2) {
    // The host wrapper must name the same refusals the CPU decoder does; spot-check two, not all.
    strata::kernels::Deepseek4VqMatrix m;
    const char* err = nullptr;
    CHECK(!strata::kernels::deepseek4_vq_open(v2.data(), v2.size(), 0, 1, 16, 64, &m, &err));
    CHECK(err != nullptr && std::strcmp(err, "expert matrix absent") == 0);
    std::vector<uint8_t> bad = v2;
    bad[0] = 'X';
    CHECK(!strata::kernels::deepseek4_vq_open(bad.data(), bad.size(), 0, 0, 16, 64, &m, &err));
    CHECK(err != nullptr && std::strcmp(err, "not a DQVL v2/v3 blob") == 0);
}

}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    const std::string refdir = argc > 1 ? argv[1] : "reference";
    int sets = 0;

    // --- the engine's v2 fixtures ---------------------------------------------------------------------
    for (const char* name : {"v2_nc256", "v2_nc4096"}) {
        std::vector<uint8_t> blob, ref;
        if (!read_file(refdir + "/ds4-raw/" + name + ".blob", &blob) ||
            !read_file(refdir + "/ds4-raw/" + name + ".ref.f32", &ref)) {
            std::printf("SKIP fixture %s (not in %s/ds4-raw)\n", name, refdir.c_str());
            continue;
        }
        sets++;
        check_decode(name, blob, 0, 0, 16, 64, &ref);
        check_gemv(name, blob, 0, 0, 16, 64);
        if (std::strcmp(name, "v2_nc256") == 0) print_error_names(blob);
    }

    // --- real expert 0 of the layer-0 blob ------------------------------------------------------------
    std::vector<uint8_t> head;
    if (!read_file(refdir + "/layer0-blob-head.bin", &head)) {
        std::printf("SKIP real expert 0 (no %s/layer0-blob-head.bin)\n", refdir.c_str());
    } else {
        sets++;
        const char* names[3] = {"w1 gate", "w3 up", "w2 down"};
        for (uint32_t which = 0; which < 3; which++) {
            const uint64_t off = dsv4::blob_slot(head.data(), 0, which);
            CHECK(off != 0);
            if (off == 0 || off + 16 > head.size()) continue;
            const uint8_t* pay = head.data() + off;
            const uint32_t rows = dsv4::rd_u32(pay, 8);
            const uint32_t cols = dsv4::rd_u32(pay, 12);
            check_decode(names[which], head, 0, which, rows, cols, nullptr);
            check_gemv(names[which], head, 0, which, rows, cols);
        }
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    if (sets == 0) {
        std::printf("SKIP: no fixtures under %s (reference/ is research material, not in git)\n", refdir.c_str());
        return 77;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
