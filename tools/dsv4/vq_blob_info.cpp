// tools/dsv4/vq_blob_info.cpp - print a VQ expert blob's geometry and decode
// one expert, to check a real file against the format (and later against the
// reference runtime's own dump).
//
// Build:
//   g++ -O2 -std=c++17 tools/dsv4/vq_blob_info.cpp -o /tmp/dsv4_vq_info
// Run:
//   /tmp/dsv4_vq_info <blob-file> [expert] [--dump-w1 out.f32]
//
// The file may be a prefix of the layer blob: the codebook sits right after
// the slot table, so a few MB of the start can be checked without the whole
// 2.7 GB.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vq_decode.hpp"

namespace {

void print_stats(const char* which, const float* v, size_t n) {
    float lo = v[0], hi = v[0];
    double sum = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (!(v[i] == v[i])) {
            bad++;
            continue;
        }
        if (v[i] < lo) lo = v[i];
        if (v[i] > hi) hi = v[i];
        sum += v[i];
    }
    std::printf("  %s: min %.6g max %.6g mean %.6g finite %zu/%zu\n", which, lo, hi,
                sum / static_cast<double>(n - bad), n - bad, n);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <blob-file> [expert] [--dump-w1 out.f32]\n", argv[0]);
        return 2;
    }
    uint32_t expert = 0;
    const char* dump_w1 = nullptr;
    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--dump-w1") == 0 && i + 1 < argc) {
            dump_w1 = argv[++i];
        } else {
            expert = static_cast<uint32_t>(std::strtoul(argv[i], nullptr, 10));
        }
    }

    std::FILE* f = std::fopen(argv[1], "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blob(n > 0 ? static_cast<size_t>(n) : 0);
    if (std::fread(blob.data(), 1, blob.size(), f) != blob.size()) {
        std::fclose(f);
        return 1;
    }
    std::fclose(f);

    if (!dsv4::blob_ok(blob.data(), blob.size())) {
        std::fprintf(stderr, "not a DQVL v2/v3 blob (or truncated before the slot table)\n");
        return 1;
    }
    const uint32_t ver = dsv4::blob_version(blob.data(), blob.size());
    const uint32_t nexp = dsv4::blob_nexp(blob.data(), blob.size());
    std::printf("DQVL v%u, layer %u, %u experts, %zu bytes read\n", ver,
                dsv4::rd_u32(blob.data(), 8), nexp, blob.size());
    const uint32_t nc = dsv4::rd_u16(blob.data(), dsv4::blob_slot(blob.data(), expert, 0) + 6);
    std::printf("expert %u: nc %u -> %u-bit indices\n", expert, nc, dsv4::index_bits(nc));

    const char* names[3] = {"w1 gate", "w3 up", "w2 down"};
    float* dump = nullptr;
    size_t dump_n = 0;
    for (uint32_t which = 0; which < 3; which++) {
        const uint64_t off = dsv4::blob_slot(blob.data(), expert, which);
        if (off == 0) {
            std::printf("  %s: absent\n", names[which]);
            continue;
        }
        if (off > blob.size() || blob.size() - off < 16) {
            std::printf("  %s: slot offset %llu outside this file\n", names[which],
                        static_cast<unsigned long long>(off));
            continue;
        }
        const uint8_t* pay = blob.data() + off;
        const uint32_t magic = dsv4::rd_u32(pay, 0);
        const uint32_t rows = dsv4::rd_u32(pay, 8);
        const uint32_t cols = dsv4::rd_u32(pay, 12);
        std::printf("  %s: offset %llu magic %08x dim %u nc %u rows %u cols %u",
                    names[which], static_cast<unsigned long long>(off), magic,
                    dsv4::rd_u16(pay, 4), dsv4::rd_u16(pay, 6), rows, cols);
        if (magic == dsv4::kMat3Magic) {
            std::printf(" flags %u mnb %u cb_off %llu", dsv4::rd_u32(pay, 16),
                        dsv4::rd_u32(pay, 20), static_cast<unsigned long long>(dsv4::rd_u64(pay, 24)));
        }
        std::printf("\n");

        dsv4::VqMatrix m;
        const char* err = nullptr;
        if (!dsv4::VqMatrix::open(blob.data(), blob.size(), expert, which, rows, cols, &m, &err)) {
            std::printf("    open failed: %s\n", err);
            continue;
        }
        const size_t count = static_cast<size_t>(rows) * cols;
        std::vector<float> out(count);
        m.dequant_f32(out.data());
        print_stats("values", out.data(), count);
        std::printf("    gain[0..3]: %.6g %.6g %.6g %.6g; first 8: %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n",
                    m.gain(0), m.gain(1), m.gain(2), m.gain(3), out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6], out[7]);
        if (which == 0) {
            dump_n = count;
            if (dump_w1 != nullptr && ver == 3) {
                if (dump == nullptr) dump = new float[count];
                std::memcpy(dump, out.data(), count * sizeof(float));
            }
        }
    }
    if (dump_w1 != nullptr) {
        if (dump == nullptr || dump_n == 0) {
            std::fprintf(stderr, "--dump-w1 needs a v3 blob\n");
            return 1;
        }
        std::FILE* o = std::fopen(dump_w1, "wb");
        if (o == nullptr) {
            std::fprintf(stderr, "cannot write %s\n", dump_w1);
            return 1;
        }
        std::fwrite(dump, sizeof(float), dump_n, o);
        std::fclose(o);
        std::printf("wrote %zu f32 values to %s\n", dump_n, dump_w1);
    }
    delete[] dump;
    return 0;
}
