// tests/core/deepseek4_test.cpp - the DeepSeek-V4.1-Flash loader's family half, on synthetic files and blobs: no
// model, no GPU, no download.
//
//   * the family maps from "deepseek4", is recognised and refused by name (no engine path yet), and the Qwen
//     families are unchanged;
//   * the family type table gives 42 = vqblob (1/1) and 44 = fp8_32x32 (1,024/1,025), where the shared table says
//     Q2_0 (64/18) and has no 44 - the collision the family-scoped table exists to resolve;
//   * read_geometry reads a complete metadata set and names the key or invariant that is wrong for a missing key,
//     a short compress_ratios, a broken engram row layout and a tower target outside the blocks;
//   * check_blob accepts a synthetic v2 container and v3 containers with and without the 13-bit plane, and refuses
//     a wrong layer, a wrong expert count, an absent slot, a bad payload magic and a truncated index stream, in
//     agreement with tools/dsv4/vq_decode.hpp's VqMatrix::open;
//   * check_inventory passes a synthetic model built from expected_tensors, and names a missing tensor, an
//     unexpected one, a wrong type, a wrong shape and a payload that runs past EOF.
#include "gguf_fixture.hpp"

#include "strata/artifact/deepseek4.hpp"

#include "../../tools/dsv4/vq_decode.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ds = strata::deepseek4;

namespace {

int g_fail = 0;
int g_checks = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
void check_has(bool ok, const std::string& haystack, const std::string& needle, const std::string& what) {
    check(ok && haystack.find(needle) != std::string::npos, what + " (saw: " + haystack + ")");
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("strata-deepseek4-test-" +
                                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

using fixture::Kv;
using fixture::Tensor;

void set_kv(std::vector<Kv>& kv, const Kv& v) {
    for (Kv& k : kv)
        if (k.key == v.key) {
            k = v;
            return;
        }
    kv.push_back(v);
}
void erase_kv(std::vector<Kv>& kv, const std::string& key) {
    for (size_t i = 0; i < kv.size(); ++i)
        if (kv[i].key == key) {
            kv.erase(kv.begin() + (long) i);
            return;
        }
}

// A small but structurally complete metadata set: two blocks, one MTP tower, one engram layer.  Widths are
// multiples of 256 where a Q4_K row lives and of 32 where an fp8_32x32 tile lives, as the real artifact's are.
const uint64_t kE = 256, kV = 512, kQ = 256, kHeads = 8, kKV = 64, kOL = 64, kNG = 4, kFF = 256;
const uint64_t kBlocks = 2, kTowers = 1, kExperts = 4, kMtpExperts = 2;

std::vector<Kv> deepseek_kv() {
    return {
        fixture::str("general.architecture", "deepseek4"),
        fixture::str("deepseek4.variant", "v41_flash"),
        fixture::str("deepseek4.recipe", "test"),
        fixture::u32("deepseek4.block_count", kBlocks),
        fixture::u32("deepseek4.context_length", 65536),
        fixture::u32("deepseek4.embedding_length", kE),
        fixture::u32("deepseek4.vocab_size", kV),
        fixture::u32("deepseek4.attention.head_count", kHeads),
        fixture::u32("deepseek4.attention.head_count_kv", 1),
        fixture::u32("deepseek4.attention.key_length", kKV),
        fixture::u32("deepseek4.attention.value_length", kKV),
        fixture::u32("deepseek4.attention.q_lora_rank", kQ),
        fixture::u32("deepseek4.attention.output_lora_rank", kOL),
        fixture::u32("deepseek4.attention.output_group_count", kNG),
        fixture::u32("deepseek4.attention.sliding_window", 16),
        fixture::f32("deepseek4.attention.layer_norm_rms_epsilon", 1e-20f),
        fixture::u32("deepseek4.attention.compress_rope_freq_base", 160000),
        fixture::u32("deepseek4.rope.dimension_count", 16),
        fixture::u32("deepseek4.expert_count", kExperts),
        fixture::u32("deepseek4.expert_used_count", 2),
        fixture::u32("deepseek4.expert_feed_forward_length", kFF),
        fixture::u32("deepseek4.expert_shared_count", 1),
        fixture::f32("deepseek4.expert_weights_scale", 1.5f),
        fixture::u32("deepseek4.expert_weights_norm", 1),
        fixture::str("deepseek4.expert_gating_func", "sqrtsoftplus"),
        fixture::u32("deepseek4.hyper_connection.count", 2),
        fixture::u32("deepseek4.hyper_connection.sinkhorn_iterations", 2),
        fixture::f32("deepseek4.hyper_connection.epsilon", 1e-6f),
        fixture::u32("deepseek4.attention.indexer.head_count", 2),
        fixture::u32("deepseek4.attention.indexer.key_length", 32),
        fixture::u32("deepseek4.attention.indexer.top_k", 4),
        fixture::u32("deepseek4.attention.candidate.source_layer", 1),
        fixture::u32("deepseek4.attention.candidate.topk_blocks", 8),
        fixture::u32("deepseek4.attention.candidate.block_size", 4),
        fixture::u32("deepseek4.rope.freq_base", 10000),
        fixture::u32("deepseek4.rope.scaling.factor", 16),
        fixture::u32("deepseek4.rope.scaling.yarn_beta_fast", 32),
        fixture::u32("deepseek4.rope.scaling.yarn_beta_slow", 1),
        fixture::u32("deepseek4.rope.scaling.original_context_length", 65536),
        fixture::u32("deepseek4.mtp.tower_count", kTowers),
        fixture::u32("deepseek4.mtp.expert_count", kMtpExperts),
        fixture::u32("deepseek4.mtp.expert_used_count", 1),
        fixture::u32("deepseek4.mtp.block_size", 2),
        fixture::u32("deepseek4.mtp.noise_token_id", kV - 1),
        fixture::u32("deepseek4.mtp.markov_rank", 8),
        fixture::u32s("deepseek4.mtp.target_layers", {1}),
        fixture::u32("deepseek4.engram.max_ngram_size", 4),
        fixture::u32("deepseek4.engram.head_count", 2),
        fixture::u32("deepseek4.engram.head_dim", 8),
        fixture::u32("deepseek4.engram.vocab_size", 1000),
        fixture::u32("deepseek4.engram.compressed_vocab_size", 100),
        fixture::u32("deepseek4.engram.pad_id_compressed", 2),
        fixture::u32s("deepseek4.engram.layer_ids", {1}),
        fixture::u32s("deepseek4.engram.num_embeddings", {16}),
        fixture::str("deepseek4.engram.0.table_path", "/tmp/engram.safetensors"),
        fixture::u32("deepseek4.engram.0.weight_offset", 0),
        fixture::u32("deepseek4.engram.0.scale_offset", 16 * 256),
        fixture::u32s("deepseek4.attention.compress_ratios", {0, 2, 0}),
        fixture::u32s("deepseek4.swiglu_clamp_exp", {10, 10}),
        fixture::u32s("deepseek4.attention.kv_source_layers", {0}),
        fixture::u32s("deepseek4.attention.index_source_layers", {0, 1}),
    };
}

ds::Geometry read_small(const fs::path& dir, std::string* err) {
    const fs::path p = dir / "meta.gguf";
    fixture::write(p, deepseek_kv(), {});
    strata::GgufFile g(p.string());
    ds::Geometry geo;
    *err = ds::read_geometry(g, geo);
    return geo;
}

void put16(std::vector<uint8_t>& b, size_t off, uint16_t v) { std::memcpy(b.data() + off, &v, 2); }
void put32(std::vector<uint8_t>& b, size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); }
void put64(std::vector<uint8_t>& b, size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); }

std::vector<uint8_t> make_v3_blob(uint32_t layer, uint32_t experts, const ds::Geometry& g, uint32_t nc) {
    const uint32_t E = (uint32_t) g.embedding_length, FF = (uint32_t) g.expert_feed_forward_length;
    const bool plane = ds::index_bits(nc) > 12;
    const uint64_t table_end = 16 + (uint64_t) experts * 24;
    const uint64_t cb_off = table_end;
    uint64_t at = cb_off + (uint64_t) nc * 8;
    std::vector<uint64_t> slots((size_t) experts * 3);
    for (uint32_t e = 0; e < experts; ++e)
        for (uint32_t w = 0; w < 3; ++w) {
            const uint32_t rows = w == 2 ? E : FF, cols = w == 2 ? FF : E;
            const uint32_t nidx = cols / 8;
            const uint64_t mrow = ((uint64_t) nidx * 12 + 7) / 8;
            const uint64_t prow = ((uint64_t) nidx + 7) / 8;
            // The decoder reads a 3-byte window at the last index, which reaches one byte past rows*mrow; the
            // published artifact pads each payload, and the checker uses the same formula.
            const uint64_t main_len = (uint64_t) rows * mrow;
            const uint64_t last_window = (uint64_t) (rows - 1) * mrow + (((uint64_t) nidx - 1) * 12 >> 3) + 3;
            const uint64_t stream = std::max(main_len, last_window);
            slots[(size_t) e * 3 + w] = at;
            at += 32 + (uint64_t) rows * 2 + stream + (plane ? (uint64_t) rows * prow : 0);
        }
    std::vector<uint8_t> b((size_t) at, 0);
    put32(b, 0, ds::kBlobMagic);
    put32(b, 4, 3);
    put32(b, 8, layer);
    put32(b, 12, experts);
    for (size_t i = 0; i < slots.size(); ++i) put64(b, 16 + i * 8, slots[i]);
    for (uint32_t e = 0; e < experts; ++e)
        for (uint32_t w = 0; w < 3; ++w) {
            const uint64_t off = slots[(size_t) e * 3 + w];
            put32(b, (size_t) off, ds::kMatV3Magic);
            put16(b, (size_t) off + 4, 8);
            put16(b, (size_t) off + 6, (uint16_t) nc);
            put32(b, (size_t) off + 8, w == 2 ? E : FF);
            put32(b, (size_t) off + 12, w == 2 ? FF : E);
            put32(b, (size_t) off + 16, ds::kV3FlagE4M3 | (plane ? ds::kV3FlagPlane : 0));
            put32(b, (size_t) off + 20, 12);
            put64(b, (size_t) off + 24, cb_off);
        }
    return b;
}

std::vector<uint8_t> make_v2_blob(uint32_t layer, uint32_t experts, const ds::Geometry& g, uint32_t nc) {
    const uint32_t E = (uint32_t) g.embedding_length, FF = (uint32_t) g.expert_feed_forward_length;
    uint64_t at = 16 + (uint64_t) experts * 24;
    std::vector<uint64_t> slots((size_t) experts * 3);
    for (uint32_t e = 0; e < experts; ++e)
        for (uint32_t w = 0; w < 3; ++w) {
            const uint32_t rows = w == 2 ? E : FF, cols = w == 2 ? FF : E;
            const uint32_t nidx = cols / 8;
            const uint64_t total = (uint64_t) rows * nidx;
            slots[(size_t) e * 3 + w] = at;
            at += 16 + (uint64_t) nc * 16 + (uint64_t) rows * 2 + total;   // nc=256 keeps this an 8-bit stream
        }
    std::vector<uint8_t> b((size_t) at, 0);
    put32(b, 0, ds::kBlobMagic);
    put32(b, 4, 2);
    put32(b, 8, layer);
    put32(b, 12, experts);
    for (size_t i = 0; i < slots.size(); ++i) put64(b, 16 + i * 8, slots[i]);
    for (uint32_t e = 0; e < experts; ++e)
        for (uint32_t w = 0; w < 3; ++w) {
            const uint64_t off = slots[(size_t) e * 3 + w];
            put32(b, (size_t) off, ds::kMatV2Magic);
            put16(b, (size_t) off + 4, 8);
            put16(b, (size_t) off + 6, (uint16_t) nc);
            put32(b, (size_t) off + 8, w == 2 ? E : FF);
            put32(b, (size_t) off + 12, w == 2 ? FF : E);
        }
    return b;
}

void type_table_tests() {
    std::printf("type table\n");
    int se = 0, sb = 0;
    check(strata::block_geometry(42, se, sb) && se == 64 && sb == 18, "the shared table reads type 42 as Q2_0 (64/18)");
    uint64_t de = 0, db = 0;
    check(ds::block_geometry(42, de, db) && de == 1 && db == 1, "the deepseek4 table reads type 42 as vqblob (1/1)");
    check(ds::block_geometry(44, de, db) && de == 1024 && db == 1025, "type 44 is fp8_32x32 (1,024/1,025)");
    check(ds::block_geometry(40, de, db) && de == 256 && db == 34, "type 40 is go1b (256/34), not NVFP4");
    check(ds::block_geometry(26, de, db) && de == 1 && db == 4, "type 26 is I32 (1/4)");
    check(ds::block_geometry(27, de, db) && de == 1 && db == 8, "type 27 is I64 (1/8)");
    check(std::string(ds::type_name(42)) == "vqblob" && std::string(strata::ggml_type_name(42)) == "Q2_0",
          "type 42's name is family-scoped too");

    strata::TensorInfo t;
    t.type = ds::kTypeVqBlob;
    t.shape = {2767968272ull};
    check(ds::payload_bytes(t) == 2767968272ull, "a vqblob's payload is its declared byte length");
    t.type = ds::kTypeFp8_32x32;
    t.shape = {1280, 32768};
    check(ds::payload_bytes(t) == 41984000ull, "fp8_32x32 [1280, 32768] is 41,984,000 bytes (a whole-matrix tile)");
    t.type = 12;   // Q4_K
    t.shape = {256, 10};
    check(ds::payload_bytes(t) == 1440ull, "Q4_K still uses the shared 256/144 block");
    t.type = 26;
    t.shape = {129280};
    check(ds::payload_bytes(t) == 517120ull, "I32 token map is 4 bytes per entry");
}

void family_tests() {
    std::printf("family\n");
    check(strata::model_family_from_arch("deepseek4") == strata::ModelFamily::Deepseek4, "deepseek4 maps to the family");
    check(std::string(strata::model_family_arch(strata::ModelFamily::Deepseek4)) == "deepseek4", "and back");
    check(std::string(strata::model_family_title(strata::ModelFamily::Deepseek4)).find("DeepSeek-V4.1-Flash") !=
              std::string::npos,
          "its title names the model");
    check(!strata::model_family_runs(strata::ModelFamily::Deepseek4), "no engine path runs it");
    const std::string msg = strata::model_family_not_supported(strata::ModelFamily::Deepseek4);
    check(msg.find("recognised") != std::string::npos && msg.find("no engine path") != std::string::npos,
          "its refusal says recognised but not runnable");
    check(strata::model_family_from_arch("qwen4exp") == strata::ModelFamily::Qwen4Exp &&
              strata::model_family_runs(strata::ModelFamily::Qwen4Exp),
          "qwen4exp is unchanged");
}

void geometry_tests(const fs::path& dir) {
    std::printf("geometry\n");
    std::string err;
    const ds::Geometry geo = read_small(dir, &err);
    check(err.empty(), "a complete metadata set reads");
    if (!err.empty()) {
        std::printf("    %s\n", err.c_str());
        return;
    }
    check(geo.block_count == kBlocks && geo.embedding_length == kE && geo.expert_count == kExperts,
          "the scalar keys land");
    check(geo.compress_ratios.size() == kBlocks + kTowers && geo.swiglu_clamp_exp.size() == kBlocks,
          "the arrays land with their lengths");
    check(geo.engram_tables.size() == 1 && geo.engram_tables[0].rows == 16 &&
              geo.engram_tables[0].scale_offset == 4096,
          "the engram table's offsets satisfy the 264-byte row layout");

    {
        strata::GgufFile g((dir / "meta.gguf").string());
        const std::string arch = strata::check_architecture(g);
        check_has(!arch.empty(), arch, "recognised", "the architecture guard refuses it by name");
        const std::string eng = strata::check_engine_architecture(g);
        check_has(!eng.empty(), eng, "no engine path", "the engine guard says why");
    }

    auto expect_error = [&](std::vector<Kv> kv, const std::string& needle, const std::string& what) {
        const fs::path p = dir / ("bad-" + std::to_string(g_fail) + ".gguf");
        fixture::write(p, kv, {});
        strata::GgufFile g(p.string());
        ds::Geometry bad;
        const std::string e = ds::read_geometry(g, bad);
        check_has(!e.empty(), e, needle, what);
    };

    std::vector<Kv> kv = deepseek_kv();
    erase_kv(kv, "deepseek4.block_count");
    expect_error(kv, "block_count", "a missing key is named");

    kv = deepseek_kv();
    set_kv(kv, fixture::u32s("deepseek4.attention.compress_ratios", {0, 2}));
    expect_error(kv, "compress_ratios", "a compress_ratios length that does not cover the towers is refused");

    kv = deepseek_kv();
    set_kv(kv, fixture::u32("deepseek4.engram.0.scale_offset", 4095));
    expect_error(kv, "scale_offset", "a broken engram row layout is refused");

    kv = deepseek_kv();
    set_kv(kv, fixture::u32s("deepseek4.mtp.target_layers", {5}));
    expect_error(kv, "target_layers", "a tower target outside the blocks is refused");
}

void blob_tests(const fs::path& dir) {
    std::printf("expert blobs\n");
    std::string err;
    const ds::Geometry geo = read_small(dir, &err);
    if (!err.empty()) {
        check(false, "geometry needed for the blob tests: " + err);
        return;
    }
    const uint32_t E = (uint32_t) geo.embedding_length, FF = (uint32_t) geo.expert_feed_forward_length;
    const uint32_t nidx_gate = E / 8;

    {
        const std::vector<uint8_t> b = make_v3_blob(1, 4, geo, 4096);
        ds::BlobInfo info;
        std::string e;
        const bool ok = ds::check_blob(b.data(), b.size(), 1, true, geo, &info, e);
        check(ok, "a v3 12-bit container passes" + (e.empty() ? "" : ": " + e));
        check(info.version == 3 && info.layer == 1 && info.experts == 4 && info.nc == 4096 && info.nbit == 12 &&
                  info.complete && info.payloads_checked == 12,
              "its geometry is reported: 4 experts, nc 4096, 12 payloads, complete");

        dsv4::VqMatrix m;
        const char* derr = nullptr;
        check(dsv4::VqMatrix::open(b.data(), b.size(), 0, 0, FF, E, &m, &derr),
              "the decoder agrees the first gate matrix is well formed");
        check(m.nidx_row() == nidx_gate && m.rows == FF && m.cols == E, "and reads the same rows/cols");
    }
    {
        const std::vector<uint8_t> b = make_v3_blob(0, 4, geo, 8192);
        ds::BlobInfo info;
        std::string e;
        check(ds::check_blob(b.data(), b.size(), 0, true, geo, &info, e),
              "a v3 13-bit (plane) container passes" + (e.empty() ? "" : ": " + e));
        check(info.nbit == 13 && info.complete, "nc 8192 is a 13-bit index with a plane");
    }
    {
        const std::vector<uint8_t> b = make_v2_blob(1, 4, geo, 256);
        ds::BlobInfo info;
        std::string e;
        check(ds::check_blob(b.data(), b.size(), 1, true, geo, &info, e),
              "a v2 8-bit container passes" + (e.empty() ? "" : ": " + e));
        check(info.version == 2 && info.nbit == 8, "v2 and its 8-bit width are reported");
    }
    {
        const std::vector<uint8_t> b = make_v3_blob(1, 4, geo, 4096);
        ds::BlobInfo info;
        std::string e;
        check(!ds::check_blob(b.data(), b.size(), 0, true, geo, &info, e) && e.find("layer") != std::string::npos,
              "a container for another layer is refused by name: " + e);
    }
    {
        const std::vector<uint8_t> b = make_v3_blob(1, 3, geo, 4096);
        ds::BlobInfo info;
        std::string e;
        check(!ds::check_blob(b.data(), b.size(), 1, true, geo, &info, e) && e.find("experts") != std::string::npos,
              "a wrong expert count is refused: " + e);
    }
    {
        const std::vector<uint8_t> b = make_v3_blob(0, 2, geo, 4096);
        ds::BlobInfo info;
        std::string e;
        check(ds::check_blob(b.data(), b.size(), 999, false, geo, &info, e),
              "a tower blob checks against mtp.expert_count and ignores the layer field" +
                  (e.empty() ? "" : ": " + e));
    }
    {
        std::vector<uint8_t> b = make_v3_blob(1, 4, geo, 4096);
        put64(b, 16 + (0 * 3 + 0) * 8, 0);
        ds::BlobInfo info;
        std::string e;
        check(!ds::check_blob(b.data(), b.size(), 1, true, geo, &info, e) && e.find("absent") != std::string::npos,
              "an absent slot is refused: " + e);
    }
    {
        std::vector<uint8_t> b = make_v3_blob(1, 4, geo, 4096);
        const uint64_t off = 0;   // first slot's offset is at byte 16; read it
        uint64_t slot;
        std::memcpy(&slot, b.data() + 16, 8);
        (void) off;
        put32(b, (size_t) slot, 0x12345678u);
        ds::BlobInfo info;
        std::string e;
        check(!ds::check_blob(b.data(), b.size(), 1, true, geo, &info, e) && e.find("magic") != std::string::npos,
              "a bad payload magic is refused: " + e);
    }
    {
        // Cut the last payload inside its index stream but keep its full header and gains.
        std::vector<uint8_t> b = make_v3_blob(1, 4, geo, 4096);
        const uint64_t last = b.size() - 1;   // the final payload ends at the container end
        uint64_t last_off = 0;
        for (uint64_t i = 0; i < 4 * 3; ++i) {
            uint64_t v;
            std::memcpy(&v, b.data() + 16 + i * 8, 8);
            if (v < last && v > last_off) last_off = v;
        }
        const uint64_t cut = last_off + 32 + (uint64_t) E * 2 + 8;   // header + gains + 8 stream bytes
        ds::BlobInfo info;
        std::string e;
        check(!ds::check_blob(b.data(), (size_t) cut, 1, true, geo, &info, e) &&
                  e.find("truncated") != std::string::npos,
              "a truncated index stream is refused: " + e);
        // The tensor's declared length ends before a payload header: that is malformed, not a partial check.
        check(!ds::check_blob(b.data(), (size_t) (last_off + 8), 1, true, geo, &info, e),
              "a declared length that ends before a payload is refused");
    }
}

std::vector<Tensor> build_model(const ds::Geometry& geo) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    for (const auto& kv : ds::expected_tensors(geo)) {
        strata::TensorInfo info;
        info.type = kv.second.type;
        info.shape = kv.second.shape;
        if (info.shape.empty()) {
            switch (info.type) {
            case ds::kTypeVqBlob:
                info.shape = {64};
                break;
            case ds::kTypeFp8_32x32:
                info.shape = {32, 32};
                break;
            case 26:
                info.shape = {4};
                break;
            case 27:
                info.shape = {4};
                break;
            default:
                info.shape = {32};
                break;
            }
        }
        Tensor t;
        t.name = kv.first;
        t.shape = info.shape;
        t.type = info.type;
        t.seed = seed++;
        t.bytes = ds::payload_bytes(info);
        ts.push_back(std::move(t));
    }
    return ts;
}

void inventory_tests(const fs::path& dir) {
    std::printf("inventory\n");
    std::string err;
    const ds::Geometry geo = read_small(dir, &err);
    if (!err.empty()) {
        check(false, "geometry needed for the inventory tests: " + err);
        return;
    }
    const std::vector<Tensor> model = build_model(geo);
    const std::map<std::string, ds::TensorSpec> specs = ds::expected_tensors(geo);
    check(model.size() == specs.size(), "the synthetic model carries one tensor per expected name (" +
                                            std::to_string(model.size()) + ")");

    const fs::path p = dir / "model.gguf";
    fixture::write(p, deepseek_kv(), model);
    strata::GgufFile g(p.string());
    ds::Inventory inv = ds::check_inventory(g, geo, false);
    check(inv.errors.empty() && inv.tensors == model.size(), "the complete synthetic model passes the inventory");
    if (!inv.errors.empty())
        for (const std::string& e : inv.errors) std::printf("    %s\n", e.c_str());

    auto expect_inventory_error = [&](std::vector<Tensor> ts, bool allow_partial, const std::string& needle,
                                      const std::string& what) {
        const fs::path q = dir / ("inv-" + std::to_string(g_fail) + ".gguf");
        fixture::write(q, deepseek_kv(), ts);
        strata::GgufFile f(q.string());
        const ds::Inventory i = ds::check_inventory(f, geo, allow_partial);
        std::string all;
        for (const std::string& e : i.errors) all += e + "\n";
        check_has(!i.errors.empty(), all, needle, what);
    };

    {
        std::vector<Tensor> ts = model;
        ts.erase(ts.begin() + 5);
        expect_inventory_error(ts, false, "missing tensor", "a missing tensor is named");
    }
    {
        std::vector<Tensor> ts = model;
        Tensor extra;
        extra.name = "blk.0.not_a_tensor.weight";
        extra.shape = {32};
        extra.type = 0;
        extra.seed = 1;
        extra.bytes = 128;
        ts.push_back(extra);
        expect_inventory_error(ts, false, "unexpected tensor", "an unexpected tensor is named");
    }
    {
        std::vector<Tensor> ts = model;
        for (Tensor& t : ts)
            if (t.name == "token_embd.weight") t.type = 0;
        expect_inventory_error(ts, false, "is type F32", "a wrong tensor type is named");
    }
    {
        std::vector<Tensor> ts = model;
        for (Tensor& t : ts)
            if (t.name == "token_embd.weight") t.shape = {kE, kV + 32};
        expect_inventory_error(ts, false, "shape", "a wrong tensor shape is named");
    }
    {
        const fs::path q = dir / "truncated.gguf";
        fixture::write(q, deepseek_kv(), model, 100);
        strata::GgufFile f(q.string());
        const ds::Inventory strict = ds::check_inventory(f, geo, false);
        std::string all;
        for (const std::string& e : strict.errors) all += e + "\n";
        check_has(!strict.errors.empty(), all, "runs past the end", "a truncated file is refused");
        const ds::Inventory partial = ds::check_inventory(f, geo, true);
        check(partial.errors.empty() && partial.beyond_eof == 1, "--partial explains the one cut payload");
    }
}

}  // namespace

int main() {
    TempDir dir;
    family_tests();
    type_table_tests();
    geometry_tests(dir.path);
    blob_tests(dir.path);
    inventory_tests(dir.path);
    std::printf("\n%s: %d checks, %d failure%s\n", g_fail == 0 ? "PASS" : "FAIL", g_checks, g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
