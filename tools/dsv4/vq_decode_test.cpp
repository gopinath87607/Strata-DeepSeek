// tools/dsv4/vq_decode_test.cpp - tests for the DeepSeek V4.1 VQ decoder.
//
// The synthetic cases are the ones in Baekpica/ds4-dfm-rs tests/vq.rs, which
// are built byte by byte from the format the C engine defines; the fixture
// cases are the engine's own v2 decoder output (tests/fixtures/vq/*.blob with
// *.ref.f32), so a parsing slip shows up as a value mismatch.
//
// Build and run from the fork root:
//   g++ -O2 -std=c++17 tools/dsv4/vq_decode_test.cpp -o /tmp/dsv4_vq_test
//   /tmp/dsv4_vq_test <path-to-ds4-dfm-rs>/crates/ds4-core/tests/fixtures/vq

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vq_decode.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

using dsv4::VqMatrix;

void push_u16(std::vector<uint8_t>* b, uint16_t v) {
    b->push_back(static_cast<uint8_t>(v));
    b->push_back(static_cast<uint8_t>(v >> 8));
}

void push_u32(std::vector<uint8_t>* b, uint32_t v) {
    for (int i = 0; i < 4; i++) b->push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void push_u64(std::vector<uint8_t>* b, uint64_t v) {
    for (int i = 0; i < 8; i++) b->push_back(static_cast<uint8_t>(v >> (8 * i)));
}

uint16_t f16_of(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint16_t exp = static_cast<uint16_t>(((bits >> 23) & 0xFFu) - 127 + 15);
    const uint16_t man = static_cast<uint16_t>((bits >> 13) & 0x3FFu);
    return sign | (exp << 10) | man;
}

std::vector<uint8_t> blob_header(uint32_t ver, uint32_t nexp, const std::vector<uint64_t>& slots) {
    std::vector<uint8_t> b;
    push_u32(&b, dsv4::kBlobMagic);
    push_u32(&b, ver);
    push_u32(&b, 0);
    push_u32(&b, nexp);
    for (uint64_t s : slots) push_u64(&b, s);
    return b;
}

std::vector<uint8_t> v2_payload(uint16_t nc, uint32_t rows, uint32_t cols, const std::vector<uint8_t>& stream) {
    std::vector<uint8_t> b;
    push_u32(&b, dsv4::kMatMagic);
    push_u16(&b, 8);
    push_u16(&b, nc);
    push_u32(&b, rows);
    push_u32(&b, cols);
    for (uint32_t w = 0; w < nc; w++) {
        for (uint32_t d = 0; d < 8; d++) push_u16(&b, f16_of(static_cast<float>((w + 1) * (d + 1))));
    }
    for (uint32_t r = 0; r < rows; r++) push_u16(&b, f16_of(r == 0 ? 0.5f : 2.0f));
    b.insert(b.end(), stream.begin(), stream.end());
    return b;
}

uint8_t cb_byte(size_t idx) { return static_cast<uint8_t>(0x38 + (idx % 4)); }

std::vector<uint8_t> v3_blob(uint16_t nc, uint32_t rows, uint32_t cols, uint32_t flags,
                             const std::vector<uint8_t>& stream) {
    std::vector<uint8_t> payload;
    push_u32(&payload, dsv4::kMat3Magic);
    push_u16(&payload, 8);
    push_u16(&payload, nc);
    push_u32(&payload, rows);
    push_u32(&payload, cols);
    push_u32(&payload, flags);
    push_u32(&payload, 12);
    push_u64(&payload, 0);  // cb_off, patched below
    for (uint32_t r = 0; r < rows; r++) push_u16(&payload, f16_of(1.0f));
    payload.insert(payload.end(), stream.begin(), stream.end());

    std::vector<uint8_t> blob = blob_header(3, 1, {40, 0, 0});
    blob.insert(blob.end(), payload.begin(), payload.end());
    const uint64_t cb_off = blob.size();
    for (size_t idx = 0; idx < nc; idx++) {
        for (int d = 0; d < 8; d++) blob.push_back(cb_byte(idx));
    }
    for (int i = 0; i < 8; i++) blob[40 + 24 + i] = static_cast<uint8_t>(cb_off >> (8 * i));
    return blob;
}

bool close_to(float a, float b) {
    if (a == b) return true;
    const float diff = a > b ? a - b : b - a;
    const float scale = (a > 0 ? a : -a) + (b > 0 ? b : -b);
    return diff <= 1e-6f * (scale > 1.0f ? scale : 1.0f);
}

void test_v2_synthetic() {
    std::vector<uint8_t> stream = {0x87, 0x00, 0x00};
    std::vector<uint8_t> blob = blob_header(2, 1, {40, 0, 0});
    std::vector<uint8_t> pay = v2_payload(4, 2, 16, stream);
    blob.insert(blob.end(), pay.begin(), pay.end());

    CHECK(dsv4::blob_ok(blob.data(), blob.size()));
    VqMatrix m;
    const char* err = nullptr;
    CHECK(VqMatrix::open(blob.data(), blob.size(), 0, 0, 2, 16, &m, &err));
    std::vector<float> got(2 * 16);
    m.dequant_f32(got.data());

    const uint32_t picks[2][2] = {{3, 1}, {0, 2}};
    const float gains[2] = {0.5f, 2.0f};
    for (uint32_t r = 0; r < 2; r++) {
        for (uint32_t i = 0; i < 2; i++) {
            for (uint32_t d = 0; d < 8; d++) {
                const float want = static_cast<float>((picks[r][i] + 1) * (d + 1)) * gains[r];
                CHECK(close_to(got[r * 16 + i * 8 + d], want));
            }
        }
    }
}

void test_v3_synthetic() {
    // 12-bit: two indices in one 12-bit-per-group stream, no plane.
    {
        std::vector<uint8_t> stream = {0x05, 0xE0, 0xFF, 0x00};
        std::vector<uint8_t> blob = v3_blob(4096, 1, 16, dsv4::kV3FlagE4M3, stream);
        CHECK(dsv4::blob_ok(blob.data(), blob.size()));
        VqMatrix m;
        const char* err = nullptr;
        CHECK(VqMatrix::open(blob.data(), blob.size(), 0, 0, 1, 16, &m, &err));
        CHECK(m.nbit == 12);
        std::vector<float> got(16);
        m.dequant_f32(got.data());
        const size_t idxs[2] = {5, 4094};
        for (size_t i = 0; i < 2; i++) {
            const float want = dsv4::e4m3fn_to_f32(cb_byte(idxs[i]));
            for (uint32_t d = 0; d < 8; d++) CHECK(close_to(got[i * 8 + d], want));
        }
    }
    // 13-bit: the plane supplies bit 12 of each index.
    {
        std::vector<uint8_t> stream = {0x05, 0xE0, 0xFF, 0x00, 0x03};
        std::vector<uint8_t> blob = v3_blob(8192, 1, 16, dsv4::kV3FlagE4M3 | dsv4::kV3FlagPlane, stream);
        CHECK(dsv4::blob_ok(blob.data(), blob.size()));
        VqMatrix m;
        const char* err = nullptr;
        CHECK(VqMatrix::open(blob.data(), blob.size(), 0, 0, 1, 16, &m, &err));
        CHECK(m.nbit == 13);
        std::vector<float> got(16);
        m.dequant_f32(got.data());
        const size_t idxs[2] = {0x1005, 0x1FFE};
        for (size_t i = 0; i < 2; i++) {
            const float want = dsv4::e4m3fn_to_f32(cb_byte(idxs[i]));
            for (uint32_t d = 0; d < 8; d++) CHECK(close_to(got[i * 8 + d], want));
        }
    }
    // A per-row gain override replaces the payload's 1.0.
    {
        std::vector<uint8_t> stream = {0x00, 0x00, 0x00, 0x00};
        std::vector<uint8_t> blob = v3_blob(4096, 1, 16, dsv4::kV3FlagE4M3, stream);
        VqMatrix m;
        const char* err = nullptr;
        CHECK(VqMatrix::open(blob.data(), blob.size(), 0, 0, 1, 16, &m, &err));
        const float gov[1] = {4.0f};
        std::vector<float> got(16);
        m.dequant_f32(got.data(), gov);
        CHECK(close_to(got[0], dsv4::e4m3fn_to_f32(cb_byte(0)) * 4.0f));
    }
}

void test_bad_headers() {
    std::vector<uint8_t> good = blob_header(3, 1, {40, 0, 0});
    CHECK(dsv4::blob_ok(good.data(), good.size()));
    good[0] = 'X';
    CHECK(!dsv4::blob_ok(good.data(), good.size()));
    std::vector<uint8_t> bad_ver = blob_header(4, 1, {40, 0, 0});
    CHECK(!dsv4::blob_ok(bad_ver.data(), bad_ver.size()));
    std::vector<uint8_t> bad_nexp = blob_header(3, 0, {});
    CHECK(!dsv4::blob_ok(bad_nexp.data(), bad_nexp.size()));
    std::vector<uint8_t> short_blob(15, 0);
    CHECK(!dsv4::blob_ok(short_blob.data(), short_blob.size()));
}

std::vector<uint8_t> read_file(const std::string& path, bool* ok) {
    std::vector<uint8_t> data;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        *ok = false;
        return data;
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    data.resize(n > 0 ? static_cast<size_t>(n) : 0);
    *ok = std::fread(data.data(), 1, data.size(), f) == data.size();
    std::fclose(f);
    return data;
}

void test_fixture(const std::string& dir, const char* name) {
    bool ok = false;
    const std::vector<uint8_t> blob = read_file(dir + "/" + name + ".blob", &ok);
    CHECK(ok);
    const std::vector<uint8_t> ref = read_file(dir + "/" + name + ".ref.f32", &ok);
    CHECK(ok);
    if (!ok) return;
    VqMatrix m;
    const char* err = nullptr;
    CHECK(VqMatrix::open(blob.data(), blob.size(), 0, 0, 16, 64, &m, &err));
    std::vector<float> got(16 * 64);
    m.dequant_f32(got.data());
    CHECK(got.size() * 4 == ref.size());
    size_t bad = 0;
    for (size_t i = 0; i < got.size() && (i + 1) * 4 <= ref.size(); i++) {
        float want;
        std::memcpy(&want, ref.data() + i * 4, 4);
        uint32_t a, b;
        std::memcpy(&a, &got[i], 4);
        std::memcpy(&b, &want, 4);
        if (a != b) bad++;
    }
    CHECK(bad == 0);
    std::printf("fixture %s: %zu values, %zu mismatches\n", name, got.size(), bad);
}

}  // namespace

int main(int argc, char** argv) {
    test_v2_synthetic();
    test_v3_synthetic();
    test_bad_headers();
    if (argc > 1) {
        test_fixture(argv[1], "v2_nc256");
        test_fixture(argv[1], "v2_nc4096");
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
