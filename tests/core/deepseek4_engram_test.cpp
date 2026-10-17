// tests/core/deepseek4_engram_test.cpp - P4's host gate: the engram hash, the O_DIRECT row reader, the fp8 row
// dequant and the gate's f64 emulation.
//
//   deepseek4_engram_test [reference-dir] [engram-shard-dir]
//
// reference-dir (default "reference") is the reference runtime's fixture tree:
//   ds4-dfm-rs/crates/ds4-core/tests/fixtures/engram/   the hash gates
//   ds4-dfm-rs/tests/fixtures/ds41/engram/              the dequant and gate gates
// It is research material, not in git; a missing fixture prints SKIP and the rest still runs.
//
// engram-shard-dir, when given, is the directory that holds the artifact's sidecar shards (the files the
// metadata names model-00047-of-00048.safetensors and model-00048-of-00048.safetensors).  The reader then runs
// over the real 101.5 GB planes: the four rows measured when the artifact was bound, and every row id in the
// golden prompt, compared between the O_DIRECT and buffered paths and against an independent std::FILE read.
//
// The gates, strongest first:
//   hash       24 ROW lines from the engine's own v41_engram_hash (gen_engram_ref.c), every position x layer
//   mask       16 lines from the image-mask variant (gen_engram_mask_ref.c)
//   golden     the engine's captured L01/L14 rows for prompt p1, 8 positions x 24 ids, from the artifact's
//              real constants - this is what pins the flat LAYER-MAJOR indexing of multipliers/primes
//   GGUF       the four tables written into a v3 file and read back through load_engram_hash, then the same
//              golden rows: the loader's order, types and byte copy
//   dequant    rows.bin -> rows.ref.f32, BYTE FOR BYTE (every product is exact; P4 requires bit equality)
//   gate       engram.ref.f32, the reference's f64 emulation, BYTE FOR BYTE against this f64 emulation
//   shard      a synthetic two-plane file through all four Io modes and refused spans; real shards when given
//
// The gate arithmetic is f64 on both sides here (the CUDA test holds the device's f32/expf tolerance).  Build
// with -ffp-contract=off: a fused multiply-add changes the f64 sums and this comparison is exact.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "strata/artifact/deepseek4_engram.hpp"

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

using strata::deepseek4::EngramHash;
using strata::deepseek4::EngramShard;

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

template <typename T>
std::vector<T> read_array(const std::string& path) {
    std::vector<uint8_t> bytes;
    if (!read_file(path, &bytes) || bytes.size() % sizeof(T) != 0) return {};
    std::vector<T> v(bytes.size() / sizeof(T));
    std::memcpy(v.data(), bytes.data(), bytes.size());
    return v;
}

// One line's tokens, starting at `p` (which points into `line`); `n` values, parsed as `base`.
// Returns the position after the last token, or nullptr on a malformed line.
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
uint32_t parse_u32(const char* p, char** end, int) { return static_cast<uint32_t>(std::strtoul(p, end, 10)); }
uint32_t parse_hex32(const char* p, char** end, int) { return static_cast<uint32_t>(std::strtoul(p, end, 16)); }

// A value on the bf16 grid, read back as f32 (for the gate fixture's h rows, already quantized).
float hex_to_f32(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

bool fixture_dir(const std::string& refdir, const char* rel, std::string* out) {
    *out = refdir + "/" + rel;
    std::FILE* f = std::fopen((*out + "/engram_ref.txt").c_str(), "rb");
    if (f != nullptr) {
        std::fclose(f);
        return true;
    }
    // The ds41 fixture directory has no engram_ref.txt; probe its own sentinel.
    std::FILE* g = std::fopen((*out + "/engram.cases.txt").c_str(), "rb");
    if (g != nullptr) {
        std::fclose(g);
        return true;
    }
    // The hash directory is the one with the .bin constants; this branch is reached for it only when
    // engram_ref.txt is absent, which means the directory is not there at all.
    return false;
}

// ---- 1. The engine's own hash fixture (gen_engram_ref.c output) -----------------------------------------

void check_hash_fixture(const std::string& dir) {
    std::string text;
    if (!read_text(dir + "/engram_ref.txt", &text)) {
        std::printf("SKIP hash fixture (no %s/engram_ref.txt)\n", dir.c_str());
        return;
    }
    EngramHash h;
    std::vector<int32_t> hist;
    struct Row {
        uint32_t ei;
        int64_t p;
        std::vector<int64_t> want;
    };
    std::vector<Row> rows;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        const char* p = line.c_str();
        if (std::strncmp(p, "NGRAM ", 6) == 0) {
            h.max_ngram = parse_u32(p + 6, nullptr, 10);
        } else if (std::strncmp(p, "HEADS ", 6) == 0) {
            h.heads = parse_u32(p + 6, nullptr, 10);
        } else if (std::strncmp(p, "PAD ", 4) == 0) {
            h.pad = static_cast<int32_t>(parse_i64(p + 4, nullptr, 10));
        } else if (std::strncmp(p, "NVOCAB ", 7) == 0) {
            h.n_vocab = parse_u32(p + 7, nullptr, 10);
        } else if (std::strncmp(p, "TMAPI32 ", 8) == 0) {
            const char* q = skip_word(p + 8);   // count
            h.token_map.resize(64);
            CHECK(parse_values(q, h.token_map.size(), 10, h.token_map.data(), parse_i64) != nullptr);
        } else if (std::strncmp(p, "MULT ", 5) == 0) {
            const char* q = skip_word(p + 5);
            h.multipliers.resize(8);
            CHECK(parse_values(q, h.multipliers.size(), 10, h.multipliers.data(), parse_i64) != nullptr);
        } else if (std::strncmp(p, "PRIM ", 5) == 0) {
            const char* q = skip_word(p + 5);
            h.primes.resize(48);
            CHECK(parse_values(q, h.primes.size(), 10, h.primes.data(), parse_i64) != nullptr);
        } else if (std::strncmp(p, "OFFS ", 5) == 0) {
            const char* q = skip_word(p + 5);
            h.offsets.resize(48);
            CHECK(parse_values(q, h.offsets.size(), 10, h.offsets.data(), parse_i64) != nullptr);
        } else if (std::strncmp(p, "HIST ", 5) == 0) {
            const char* q = skip_word(p + 5);
            hist.resize(12);
            CHECK(parse_values(q, hist.size(), 10, hist.data(), parse_i64) != nullptr);
        } else if (std::strncmp(p, "ROW ", 4) == 0) {
            const char* q = p + 4;
            char* end = nullptr;
            Row r;
            r.ei = static_cast<uint32_t>(std::strtoul(q, &end, 10));
            r.p = std::strtoll(end, &end, 10);
            r.want.resize(24);
            CHECK(parse_values(end, r.want.size(), 10, r.want.data(), parse_i64) != nullptr);
            rows.push_back(std::move(r));
        }
    }
    CHECK(h.max_ngram == 4 && h.heads == 8 && h.pad == 2 && h.n_vocab == 64);
    CHECK(h.cols() == 24);
    std::string err;
    CHECK(h.valid(2, err));
    CHECK(rows.size() == 24);
    std::vector<int64_t> got(h.cols());
    size_t bad_rows = 0, bad_vals = 0;
    for (const Row& r : rows) {
        h.rows(hist.data(), hist.size(), r.p, r.ei, got.data());
        if (got != r.want) {
            bad_rows++;
            for (size_t i = 0; i < got.size(); i++) bad_vals += got[i] != r.want[i];
        }
    }
    CHECK(bad_rows == 0);
    std::printf("hash fixture: %zu rows (ei x p), %zu rows and %zu values differ\n", rows.size(), bad_rows,
                bad_vals);
    g_sections++;
}

// ---- 2. The image-mask variant (gen_engram_mask_ref.c output) -------------------------------------------

void check_mask_fixture(const std::string& dir) {
    std::string text;
    if (!read_text(dir + "/engram_mask_ref.txt", &text)) {
        std::printf("SKIP mask fixture (no %s/engram_mask_ref.txt)\n", dir.c_str());
        return;
    }
    // The generator's constants, echoed in gen_engram_mask_ref.c.
    constexpr int32_t kHist[8] = {5, 6, 1, 3, 4, 0, 2, 7};
    constexpr uint8_t kMask[8] = {0, 0, 1, 1, 0, 0, 0, 0};
    EngramHash h;
    h.max_ngram = 4;
    h.heads = 2;
    h.pad = 2;
    h.n_vocab = 8;
    h.token_map.resize(8);
    h.multipliers.resize(2 * 4);
    h.primes.resize(2 * (4 - 1) * 2);
    h.offsets.resize(h.primes.size());
    for (uint32_t j = 0; j < 8; j++) h.token_map[j] = static_cast<int32_t>(j * 7 + 3);
    for (size_t j = 0; j < h.multipliers.size(); j++) h.multipliers[j] = static_cast<int64_t>(j * 13 + 5);
    for (size_t j = 0; j < h.primes.size(); j++) {
        h.primes[j] = 101 + static_cast<int64_t>(j) * 17;
        h.offsets[j] = static_cast<int64_t>(j) * 1000;
    }
    std::string err;
    CHECK(h.valid(2, err));

    std::istringstream in(text);
    std::string line;
    std::vector<int64_t> got(h.cols());
    size_t lines = 0, bad = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const char* p = line.c_str();
        char* end = nullptr;
        const uint32_t ei = static_cast<uint32_t>(std::strtoul(p, &end, 10));
        const int64_t pos = std::strtoll(end, &end, 10);
        int64_t want[6];
        CHECK(parse_values(end, 6, 10, want, parse_i64) != nullptr);
        h.rows_masked(kHist, 8, kMask, 8, pos, ei, got.data());
        for (size_t i = 0; i < 6; i++) {
            if (got[i] != want[i]) {
                if (bad < 4)
                    std::fprintf(stderr, "  mask ei=%u p=%lld col %zu: got %lld want %lld\n", ei,
                                 static_cast<long long>(pos), i, static_cast<long long>(got[i]),
                                 static_cast<long long>(want[i]));
                bad++;
            }
        }
        lines++;
    }
    CHECK(lines == 16);
    CHECK(bad == 0);
    std::printf("mask fixture: %zu lines, %zu values differ\n", lines, bad);
    g_sections++;
}

// ---- 3. The artifact's real constants against the engine's captured prompt rows -------------------------

bool load_real_hash(const std::string& dir, EngramHash* h) {
    auto tmap = read_array<int32_t>(dir + "/token_map.i32.bin");
    auto mult = read_array<int64_t>(dir + "/multipliers.i64.bin");
    auto prim = read_array<int64_t>(dir + "/primes.i64.bin");
    auto offs = read_array<int64_t>(dir + "/offsets.i64.bin");
    if (tmap.size() != 129280 || mult.size() != 8 || prim.size() != 48 || offs.size() != 48) return false;
    h->token_map = std::move(tmap);
    h->multipliers = std::move(mult);
    h->primes = std::move(prim);
    h->offsets = std::move(offs);
    h->max_ngram = 4;
    h->heads = 8;
    h->pad = 2;
    h->n_vocab = 129280;
    return true;
}

void check_golden_rows(const std::string& dir, const EngramHash& h) {
    std::string ids_text, l01, l14;
    if (!read_text(dir + "/p1.ids.txt", &ids_text) || !read_text(dir + "/p1.erows_L01.txt", &l01) ||
        !read_text(dir + "/p1.erows_L14.txt", &l14)) {
        std::printf("SKIP golden rows (fixtures absent)\n");
        return;
    }
    std::istringstream idin(ids_text);
    std::vector<int32_t> ids;
    int64_t v;
    while (idin >> v) ids.push_back(static_cast<int32_t>(v));
    CHECK(ids.size() == 8);

    std::vector<int64_t> got(h.cols());
    size_t positions = 0, bad = 0;
    const char* names[2] = {"L01", "L14"};
    const std::string* texts[2] = {&l01, &l14};
    for (uint32_t ei = 0; ei < 2; ei++) {
        std::istringstream in(*texts[ei]);
        std::string line;
        int64_t p = 0;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            int64_t want[24];
            CHECK(parse_values(line.c_str(), 24, 10, want, parse_i64) != nullptr);
            h.rows(ids.data(), ids.size(), p, ei, got.data());
            for (size_t i = 0; i < got.size(); i++) {
                if (got[i] != want[i]) {
                    if (bad < 4)
                        std::fprintf(stderr, "  golden %s p=%lld col %zu: got %lld want %lld\n", names[ei],
                                     static_cast<long long>(p), i, static_cast<long long>(got[i]),
                                     static_cast<long long>(want[i]));
                    bad++;
                }
            }
            positions++;
            p++;
        }
        CHECK(p == static_cast<int64_t>(ids.size()));
    }
    CHECK(bad == 0);
    std::printf("golden rows: %zu positions x 2 layers (real artifact constants), %zu values differ\n", positions,
                bad);
    g_sections++;
}

// ---- 4. load_engram_hash over a minimal v3 GGUF built from the same bytes -------------------------------

void put32(std::vector<uint8_t>& b, uint32_t v) { b.insert(b.end(), reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + 4); }
void put64(std::vector<uint8_t>& b, uint64_t v) { b.insert(b.end(), reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + 8); }
void puts_str(std::vector<uint8_t>& b, const std::string& s) {
    put64(b, s.size());
    b.insert(b.end(), s.data(), s.data() + s.size());
}

void check_gguf_loader(const std::string& dir, const EngramHash& want) {
    if (want.token_map.empty()) {
        std::printf("SKIP GGUF loader (no real hash tables)\n");
        return;
    }
    // The four tensors with the artifact's own shapes: token_map [129280] I32; multipliers [4, 2] I64;
    // primes and offsets [(ngram-1)*heads, layers] = [24, 2] I64.  Payloads are the golden fixture bytes.
    struct T {
        const char* name;
        std::vector<uint64_t> shape;
        uint32_t type;
        std::vector<uint8_t> bytes;
    };
    std::vector<T> ts;
    auto bytes_of = [](const std::vector<int32_t>& v) {
        std::vector<uint8_t> b(v.size() * 4);
        std::memcpy(b.data(), v.data(), b.size());
        return b;
    };
    auto bytes_of64 = [](const std::vector<int64_t>& v) {
        std::vector<uint8_t> b(v.size() * 8);
        std::memcpy(b.data(), v.data(), b.size());
        return b;
    };
    ts.push_back({"engram.token_map", {129280}, 26, bytes_of(want.token_map)});
    ts.push_back({"engram.multipliers", {4, 2}, 27, bytes_of64(want.multipliers)});
    ts.push_back({"engram.primes", {24, 2}, 27, bytes_of64(want.primes)});
    ts.push_back({"engram.offsets", {24, 2}, 27, bytes_of64(want.offsets)});

    std::vector<uint8_t> b;
    put32(b, 0x46554747u);
    put32(b, 3);
    put64(b, ts.size());
    put64(b, 0);   // no metadata: load_engram_hash reads tensors only
    uint64_t at = 0;
    std::vector<uint64_t> offs;
    for (const T& t : ts) {
        puts_str(b, t.name);
        put32(b, static_cast<uint32_t>(t.shape.size()));
        for (uint64_t d : t.shape) put64(b, d);
        put32(b, t.type);
        put64(b, at);
        offs.push_back(at);
        at = (at + t.bytes.size() + 31) / 32 * 32;
    }
    const uint64_t data_start = (b.size() + 31) / 32 * 32;
    b.resize(static_cast<size_t>(data_start), 0);
    for (size_t i = 0; i < ts.size(); i++) {
        b.resize(static_cast<size_t>(data_start + offs[i]), 0);
        b.insert(b.end(), ts[i].bytes.begin(), ts[i].bytes.end());
    }
    const std::string path = "deepseek4_engram_hash_test.gguf";
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
        CHECK(static_cast<bool>(f));
    }

    strata::deepseek4::Geometry geo;
    geo.vocab_size = 129280;
    geo.engram_max_ngram_size = 4;
    geo.engram_head_count = 8;
    geo.engram_head_dim = 256;
    geo.engram_vocab_size = 16000000;   // the row-slot space, not the token_map length
    geo.engram_compressed_vocab_size = 99092;
    geo.engram_pad_id_compressed = 2;
    geo.engram_layer_ids = {1, 14};

    try {
        strata::GgufFile g(path);
        EngramHash got;
        const std::string err = strata::deepseek4::load_engram_hash(g, geo, got);
        CHECK(err.empty());
        if (!err.empty()) std::fprintf(stderr, "  load_engram_hash: %s\n", err.c_str());
        CHECK(got.max_ngram == want.max_ngram && got.heads == want.heads && got.pad == want.pad &&
              got.n_vocab == want.n_vocab);
        CHECK(got.token_map == want.token_map);
        CHECK(got.multipliers == want.multipliers);
        CHECK(got.primes == want.primes);
        CHECK(got.offsets == want.offsets);

        // The loaded hash must reproduce the engine's golden rows, so the order is proven end to end.
        std::string ids_text;
        if (read_text(dir + "/p1.ids.txt", &ids_text)) {
            std::istringstream idin(ids_text);
            std::vector<int32_t> ids;
            int64_t v;
            while (idin >> v) ids.push_back(static_cast<int32_t>(v));
            std::string l14;
            if (read_text(dir + "/p1.erows_L14.txt", &l14) && ids.size() == 8) {
                std::istringstream in(l14);
                std::string line;
                std::vector<int64_t> got_rows(got.cols());
                size_t bad = 0;
                int64_t p = 0;
                while (std::getline(in, line)) {
                    if (line.empty()) continue;
                    int64_t want_rows[24];
                    CHECK(parse_values(line.c_str(), 24, 10, want_rows, parse_i64) != nullptr);
                    got.rows(ids.data(), ids.size(), p, 1, got_rows.data());
                    for (size_t i = 0; i < got_rows.size(); i++) bad += got_rows[i] != want_rows[i];
                    p++;
                }
                CHECK(bad == 0);
                std::printf("GGUF loader: tables read from a v3 file match the fixture, %zu golden L14 values differ\n",
                            bad);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  GgufFile(%s): %s\n", path.c_str(), e.what());
        CHECK(false);
    }
    std::remove(path.c_str());
    g_sections++;
}

// ---- 5. Row dequant: rows.bin -> rows.ref.f32, bit-exact ------------------------------------------------

void check_dequant(const std::string& dir) {
    std::vector<uint8_t> raw;
    std::vector<uint8_t> ref_bytes;
    if (!read_file(dir + "/rows.bin", &raw) || !read_file(dir + "/rows.ref.f32", &ref_bytes)) {
        std::printf("SKIP dequant (no rows.bin / rows.ref.f32)\n");
        return;
    }
    CHECK(raw.size() == 5 * 264);
    CHECK(ref_bytes.size() == 5 * 256 * 4);
    std::vector<float> got(5 * 256);
    CHECK(strata::deepseek4::engram_rows_f32(raw.data(), 5, 256, got.data()));
    CHECK(got.size() == ref_bytes.size() / 4);
    std::vector<float> ref(got.size());
    std::memcpy(ref.data(), ref_bytes.data(), ref_bytes.size());
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
        uint32_t a, b2;
        std::memcpy(&a, &got[i], 4);
        std::memcpy(&b2, &ref[i], 4);
        if (a != b2) {
            if (bad < 4) std::fprintf(stderr, "  dequant %zu: got %08x want %08x\n", i, a, b2);
            bad++;
        }
    }
    CHECK(bad == 0);
    // E4M3 extremes and the UE8M0 zero exponent, against the reference's conversion (ds41_engram_ref.rs).
    CHECK(strata::deepseek4::engram_e4m3_to_f32(0x00) == 0.0f);
    CHECK(std::signbit(strata::deepseek4::engram_e4m3_to_f32(0x80)));
    CHECK(strata::deepseek4::engram_e4m3_to_f32(0x01) == std::ldexp(1.0f, -9));       // subnormal min
    CHECK(strata::deepseek4::engram_e4m3_to_f32(0x08) == std::ldexp(1.0f, -6));       // smallest normal
    CHECK(strata::deepseek4::engram_e4m3_to_f32(0x78) == 256.0f);                     // 1.0 * 2^8
    CHECK(strata::deepseek4::engram_e4m3_to_f32(0x70) == 128.0f);                     // 1.0 * 2^7
    CHECK(strata::deepseek4::engram_e4m3_to_f32(0x38) == 1.0f);
    CHECK(strata::deepseek4::engram_e4m3_to_f32(0xf0) == -128.0f);
    CHECK(std::isnan(strata::deepseek4::engram_e4m3_to_f32(0x7f)));
    CHECK(std::isnan(strata::deepseek4::engram_e4m3_to_f32(0xff)));
    CHECK(strata::deepseek4::engram_e8m0_to_f32(0) == std::ldexp(1.0f, -127));
    CHECK(strata::deepseek4::engram_e8m0_to_f32(127) == 1.0f);
    CHECK(strata::deepseek4::engram_e8m0_to_f32(120) == std::ldexp(1.0f, -7));
    std::printf("dequant: %zu values, %zu bit-diffs; conversions checked at the E4M3/UE8M0 edges\n", got.size(), bad);
    g_sections++;
}

// ---- 6. The gate's f64 emulation against the reference's, bit-exact -------------------------------------

struct GateCase {
    std::string kind;
    uint32_t n_tok = 0;
    std::vector<float> h, kv;
};

void check_gate(const std::string& dir) {
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
    CHECK(q_off + static_cast<uint64_t>(hc) * e * 4 + static_cast<uint64_t>(hc) * e * 4 <= img.size());
    CHECK(cases.size() == 3);
    CHECK(ref_bytes.size() == (size_t)3 * 2 * hc * e * 4);

    const float* qw = reinterpret_cast<const float*>(img.data() + q_off);
    const float* kw = reinterpret_cast<const float*>(img.data() + k_off);
    const float* ref = reinterpret_cast<const float*>(ref_bytes.data());
    size_t ref_off = 0;
    size_t total_bad = 0;
    for (const GateCase& c : cases) {
        const size_t n = static_cast<size_t>(c.n_tok) * hc * e;
        std::vector<float> got(n);
        strata::deepseek4::engram_gate_emulate(c.h.data(), c.kv.data(), qw, kw, e, hc, c.n_tok, eps, got.data());
        size_t bad = 0;
        float worst = 0.0f;
        for (size_t i = 0; i < n; i++) {
            uint32_t a, b2;
            std::memcpy(&a, &got[i], 4);
            std::memcpy(&b2, &ref[ref_off + i], 4);
            if (a != b2) {
                if (bad < 3)
                    std::fprintf(stderr, "  gate %s value %zu: got %a want %a\n", c.kind.c_str(), i, got[i],
                                 ref[ref_off + i]);
                bad++;
                const float d = std::fabs(got[i] - ref[ref_off + i]);
                if (d > worst) worst = d;
            }
        }
        CHECK(bad == 0);
        total_bad += bad;
        std::printf("gate case %-6s n=%u: %zu values, %zu bit-diffs (worst |d| %.3e)\n", c.kind.c_str(),
                    c.n_tok, n, bad, worst);
        ref_off += n;
    }
    CHECK(ref_off == ref_bytes.size() / 4);
    std::printf("gate: %zu cases, %zu bit-diffs against the reference f64 emulation\n", cases.size(), total_bad);
    g_sections++;
}

// ---- 7a. The synthetic two-plane shard, through every Io mode -------------------------------------------

void fill_synthetic(std::vector<uint8_t>* bytes) {
    constexpr uint64_t kRows = 4;
    constexpr uint32_t kHd = 256;
    constexpr uint64_t kNsc = 8;
    constexpr uint64_t kWOff = 4096;
    const uint64_t s_off = kWOff + kRows * kHd;
    bytes->assign(static_cast<size_t>(s_off + kRows * kNsc) + 4096, 0);
    for (uint64_t r = 0; r < kRows; r++) {
        for (uint64_t i = 0; i < kHd; i++) (*bytes)[static_cast<size_t>(kWOff + r * kHd + i)] = static_cast<uint8_t>(r * 37 + i);
        for (uint64_t i = 0; i < kNsc; i++)
            (*bytes)[static_cast<size_t>(s_off + r * kNsc + i)] = static_cast<uint8_t>(0x80 | (r + i));
    }
}

bool write_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(f);
}

// Reads rows 0..3 and the last scale byte through `io`, and cross-checks them against `want`.
size_t check_rows_io(const std::string& path, EngramShard::Io io, const std::vector<uint8_t>& want,
                     EngramShard* shard_out, const char* label) {
    constexpr uint32_t kHd = 256;
    constexpr uint64_t kWOff = 4096;
    constexpr uint64_t kSOff = kWOff + 4 * kHd;
    EngramShard shard;
    std::string err;
    if (!shard.open(path, err, io)) {
        std::printf("  %s: refused (%s)\n", label, err.c_str());
        return 1;
    }
    size_t bad = 0;
    if (!shard.check_span(kWOff, kSOff, 4, kHd, err)) {
        std::fprintf(stderr, "  %s: check_span: %s\n", label, err.c_str());
        bad++;
    }
    std::string span_err;
    if (shard.check_span(kWOff, kSOff, 4 + 32, kHd, span_err)) {
        std::fprintf(stderr, "  %s: a span past the file was accepted\n", label);
        bad++;
    }
    for (uint64_t r = 0; r < 4; r++) {
        uint8_t weights[kHd], scale[8];
        if (!shard.row(kWOff, kSOff, r, kHd, weights, scale, err)) {
            std::fprintf(stderr, "  %s: row %llu: %s\n", label, static_cast<unsigned long long>(r), err.c_str());
            bad++;
            continue;
        }
        for (uint32_t i = 0; i < kHd; i++) bad += weights[i] != want[static_cast<size_t>(kWOff + r * kHd + i)];
        for (uint32_t i = 0; i < 8; i++) bad += scale[i] != want[static_cast<size_t>(kSOff + r * 8 + i)];
    }
    // The batched entry point must agree with the per-row one.
    const int64_t ids[3] = {3, 0, 2};
    std::vector<uint8_t> batched(3 * EngramShard::row_stride(kHd));
    if (!shard.rows(kWOff, kSOff, ids, 3, kHd, batched.data(), err)) {
        std::fprintf(stderr, "  %s: rows: %s\n", label, err.c_str());
        bad++;
    } else {
        for (size_t i = 0; i < batched.size(); i++) {
            const uint64_t r = static_cast<uint64_t>(ids[i / (kHd + 8)]);
            const uint32_t j = static_cast<uint32_t>(i % (kHd + 8));
            const uint8_t want_byte = j < kHd ? want[static_cast<size_t>(kWOff + r * kHd + j)]
                                              : want[static_cast<size_t>(kSOff + r * 8 + (j - kHd))];
            bad += batched[i] != want_byte;
        }
    }
    std::printf("  %s: %s%s, %zu mismatches\n", label, shard.is_direct() ? "O_DIRECT" : "buffered",
                shard.is_aligned() ? " + aligned bounce" : "", bad);
    if (shard_out != nullptr) *shard_out = std::move(shard);
    return bad;
}

void check_synthetic_shard() {
    std::vector<uint8_t> bytes;
    fill_synthetic(&bytes);
    const std::string disk = "deepseek4_engram_shard_test.bin";
    CHECK(write_file(disk, bytes));

    size_t bad = 0;
    // The synthetic rows, cross-checked inside check_rows_io against the file bytes.
    bad += check_rows_io(disk, EngramShard::Io::kAuto, bytes, nullptr, "auto");
    bad += check_rows_io(disk, EngramShard::Io::kBuffered, bytes, nullptr, "buffered");
    bad += check_rows_io(disk, EngramShard::Io::kAligned, bytes, nullptr, "aligned");
    // kDirect only runs where the filesystem takes O_DIRECT; elsewhere its refusal is the fallback's premise.
    EngramShard direct;
    std::string err;
    if (direct.open(disk, err, EngramShard::Io::kDirect)) {
        direct.close();
        bad += check_rows_io(disk, EngramShard::Io::kDirect, bytes, nullptr, "direct");
    } else {
        std::printf("  direct: refused here (%s); the fallback is what kAuto takes\n", err.c_str());
    }

    // Every mode that opened must return the same bytes; the comparisons above already show that, but state
    // it once over the full row form as well.
    EngramShard a, b2;
    if (a.open(disk, err, EngramShard::Io::kAligned) && b2.open(disk, err, EngramShard::Io::kBuffered)) {
        uint8_t ra[264], rb[264];
        for (uint64_t r = 0; r < 4; r++) {
            CHECK(a.row(4096, 5120, r, 256, ra, ra + 256, err));
            CHECK(b2.row(4096, 5120, r, 256, rb, rb + 256, err));
            bad += std::memcmp(ra, rb, 264) != 0;
        }
    }
    CHECK(bad == 0);
    std::remove(disk.c_str());
    std::printf("synthetic shard: both planes, four Io modes, refused span\n");
    g_sections++;
}

// ---- 7b. The real artifact shards, when the directory is given ------------------------------------------

void check_real_shard(const std::string& dir, const std::string& golden_dir) {
    // The geometry the metadata carries for the artifact (reference/gguf-header.txt): table offsets are
    // ABSOLUTE file offsets (weight_offset + rows*256 == scale_offset).
    struct Table {
        const char* file;
        uint64_t weight_off, scale_off, rows;
        const uint8_t row0_weight[16], row0_scale[8];
        const uint8_t last_weight[16], last_scale[8];
        uint64_t row0, last_row;
    };
    const Table tables[2] = {
        {"model-00047-of-00048.safetensors", 664, 98305579672ull, 384006168ull,
         {0xe2, 0x6a, 0xed, 0x51, 0xbc, 0xd8, 0xf5, 0xe9, 0xe1, 0xf6, 0xf2, 0x71, 0x5a, 0xf1, 0xcf, 0xe2},
         {0x78, 0x78, 0x77, 0x78, 0x78, 0x78, 0x78, 0x78},
         {0xd6, 0xf2, 0x6f, 0x72, 0x62, 0x70, 0xe1, 0xe9, 0x68, 0x76, 0x74, 0x77, 0x6d, 0xec, 0x61, 0xf0},
         {0x78, 0x77, 0x78, 0x77, 0x78, 0x78, 0x77, 0x78}, 0, 384006167ull},
        {"model-00048-of-00048.safetensors", 672, 98308271264ull, 384016682ull,
         {0x69, 0xb6, 0xf0, 0xe1, 0x68, 0x64, 0x73, 0xe1, 0xeb, 0xf8, 0x76, 0xf3, 0xd7, 0x6f, 0x6b, 0x70},
         {0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78},
         {0x6d, 0x4d, 0x51, 0xd0, 0x64, 0x6a, 0x68, 0xee, 0xd9, 0xd9, 0x6b, 0x70, 0x6c, 0xf1, 0x79, 0xe2},
         {0x78, 0x77, 0x77, 0x78, 0x78, 0x78, 0x78, 0x78}, 0, 384016681ull},
    };

    size_t bad = 0;
    for (size_t i = 0; i < 2; i++) {
        const Table& t = tables[i];
        const std::string path = dir + "/" + t.file;
        EngramShard auto_shard, buffered;
        std::string err;
        if (!auto_shard.open(path, err)) {
            std::printf("SKIP real shard %s: %s\n", t.file, err.c_str());
            continue;
        }
        if (!auto_shard.check_span(t.weight_off, t.scale_off, t.rows, 256, err)) {
            std::fprintf(stderr, "  %s: check_span: %s\n", t.file, err.c_str());
            bad++;
        }
        if (!buffered.open(path, err, EngramShard::Io::kBuffered)) {
            std::fprintf(stderr, "  %s: buffered open: %s\n", t.file, err.c_str());
            bad++;
            continue;
        }
        std::printf("real shard %s: %s (size %llu)\n", t.file, auto_shard.is_direct() ? "O_DIRECT" : "buffered (fallback)",
                    static_cast<unsigned long long>(auto_shard.size()));

        // Row 0 and the last row: the two planes' edges, against the bytes measured at binding.
        for (int edge = 0; edge < 2; edge++) {
            const uint64_t r = edge == 0 ? t.row0 : t.last_row;
            const uint8_t* prefix = edge == 0 ? t.row0_weight : t.last_weight;
            const uint8_t* scale_want = edge == 0 ? t.row0_scale : t.last_scale;
            uint8_t wa[264], wb[264];
            if (!auto_shard.row(t.weight_off, t.scale_off, r, 256, wa, wa + 256, err) ||
                !buffered.row(t.weight_off, t.scale_off, r, 256, wb, wb + 256, err)) {
                std::fprintf(stderr, "  %s row %llu: %s\n", t.file, static_cast<unsigned long long>(r), err.c_str());
                bad++;
                continue;
            }
            if (std::memcmp(wa, prefix, 16) != 0) {
                std::fprintf(stderr, "  %s row %llu: weight prefix differs\n", t.file,
                             static_cast<unsigned long long>(r));
                bad++;
            }
            if (std::memcmp(wa + 256, scale_want, 8) != 0) {
                std::fprintf(stderr, "  %s row %llu: scales differ\n", t.file, static_cast<unsigned long long>(r));
                bad++;
            }
            bad += std::memcmp(wa, wb, 264) != 0;
        }

        // Every golden prompt row through both paths, and through an independent std::FILE read.
        std::string ids_text;
        const char* golden = i == 0 ? "p1.erows_L01.txt" : "p1.erows_L14.txt";
        if (golden_dir.empty() || !read_text(golden_dir + "/" + golden, &ids_text)) {
            std::printf("  (no %s; golden rows skipped)\n", golden);
            continue;
        }
        std::istringstream ids_in(ids_text);
        std::string idline;
        size_t n_ids = 0;
        std::ifstream plain(path, std::ios::binary);
        while (std::getline(ids_in, idline)) {
            if (idline.empty()) continue;
            int64_t ids[24];
            if (parse_values(idline.c_str(), 24, 10, ids, parse_i64) == nullptr) {
                bad++;
                break;
            }
            for (size_t c = 0; c < 24; c++) {
                uint8_t wa[264], wb[264], wc[264];
                if (!auto_shard.row(t.weight_off, t.scale_off, static_cast<uint64_t>(ids[c]), 256, wa, wa + 256, err) ||
                    !buffered.row(t.weight_off, t.scale_off, static_cast<uint64_t>(ids[c]), 256, wb, wb + 256, err)) {
                    std::fprintf(stderr, "  %s golden id %lld: %s\n", t.file, static_cast<long long>(ids[c]),
                                 err.c_str());
                    bad++;
                    continue;
                }
                plain.seekg(static_cast<std::streamoff>(t.weight_off + static_cast<uint64_t>(ids[c]) * 256));
                plain.read(reinterpret_cast<char*>(wc), 256);
                plain.seekg(static_cast<std::streamoff>(t.scale_off + static_cast<uint64_t>(ids[c]) * 8));
                plain.read(reinterpret_cast<char*>(wc + 256), 8);
                if (!plain) {
                    bad++;
                    continue;
                }
                bad += std::memcmp(wa, wb, 264) != 0;
                bad += std::memcmp(wa, wc, 264) != 0;
                n_ids++;
            }
        }
        std::printf("  %s: %zu golden rows, O_DIRECT vs buffered vs FILE\n", golden, n_ids);
    }
    CHECK(bad == 0);
    std::printf("real shards: %zu mismatches\n", bad);
    g_sections++;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string refdir = argc > 1 ? argv[1] : "reference";
    const std::string sharddir = argc > 2 ? argv[2] : "";
    std::string hashdir, gatedir;
    const bool have_hash = fixture_dir(refdir, "ds4-dfm-rs/crates/ds4-core/tests/fixtures/engram", &hashdir);
    const bool have_gate = fixture_dir(refdir, "ds4-dfm-rs/tests/fixtures/ds41/engram", &gatedir);

    if (have_hash) {
        check_hash_fixture(hashdir);
        check_mask_fixture(hashdir);
        EngramHash real;
        if (load_real_hash(hashdir, &real)) {
            check_golden_rows(hashdir, real);
            check_gguf_loader(hashdir, real);
        } else {
            std::printf("SKIP golden rows / GGUF loader (real hash .bin fixtures absent)\n");
        }
    } else {
        std::printf("SKIP hash gates (no %s/ds4-dfm-rs/crates/ds4-core/tests/fixtures/engram)\n", refdir.c_str());
    }
    if (have_gate) {
        check_dequant(gatedir);
        check_gate(gatedir);
    } else {
        std::printf("SKIP dequant/gate gates (no %s/ds4-dfm-rs/tests/fixtures/ds41/engram)\n", refdir.c_str());
    }
    check_synthetic_shard();
    if (!sharddir.empty()) check_real_shard(sharddir, have_hash ? hashdir : std::string());

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    if (g_sections == 0) {
        std::printf("SKIP: no fixtures under %s\n", refdir.c_str());
        return 77;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
