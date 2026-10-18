// include/strata/artifact/deepseek4.hpp - the DeepSeek-V4.1-Flash (`deepseek4`) family's tensor types, metadata
// geometry, expert-blob container checks and tensor inventory.
//
// This is the loader's family-scoped half of the port (docs/DEEPSEEK4.md, P1).  It is deliberately separate from
// gguf_reader.hpp's type table: GGUF type 42 is Q2_0 (64 values / 18 bytes) to Strata's shipped models and a
// `vqblob` (1 byte / 1 byte, a DQVL container) here, and type 40/41 are NVFP4/Q1_0 there and the DeepSeek go1b/go2b
// here.  Every function in this file therefore takes the family as a premise; the shared table is not changed, so
// the Qwen loading path stays byte-identical.
//
// Nothing here decodes weights.  The container and payload headers are checked so a malformed or truncated file
// is refused with a named error before any kernel sees it; the index streams themselves are decoded by
// tools/dsv4/vq_decode.hpp (bit-exact against the reference's fixtures).
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace strata::deepseek4 {

// ---- The family's GGML type ids (measured from the artifact; see docs/DEEPSEEK4.md §2).
inline constexpr uint32_t kTypeGo1b = 40;
inline constexpr uint32_t kTypeGo2b = 41;
inline constexpr uint32_t kTypeVqBlob = 42;
inline constexpr uint32_t kTypeFp4x32 = 43;
inline constexpr uint32_t kTypeFp8_32x32 = 44;

/// Type 42 is a DQVL container here, not Q2_0.  Falls back to the shared table for the types the two families agree
/// on.
inline const char* type_name(uint32_t t) {
    switch (t) {
    case kTypeGo1b:
        return "go1b";
    case kTypeGo2b:
        return "go2b";
    case kTypeVqBlob:
        return "vqblob";
    case kTypeFp4x32:
        return "fp4x32";
    case kTypeFp8_32x32:
        return "fp8_32x32";
    case 26:
        return "I32";
    case 27:
        return "I64";
    default:
        return ggml_type_name(t);
    }
}

/// (elements per block, bytes per block) under the DeepSeek id table.  The five overridden ids come from the
/// reference runtime's `tensors.rs`, and 42/44 were confirmed against every tensor's file gap in the artifact.
inline bool block_geometry(uint32_t t, uint64_t& elems, uint64_t& bytes) {
    switch (t) {
    case kTypeGo1b:   // one 2-bit sign/scale stream per 256 values
        elems = 256;
        bytes = 34;
        return true;
    case kTypeGo2b:
        elems = 256;
        bytes = 68;
        return true;
    case kTypeVqBlob:   // a byte-addressed container; the GGUF dimension is its byte length
        elems = 1;
        bytes = 1;
        return true;
    case kTypeFp4x32:   // 32 E2M1 values + one E8M0 scale
        elems = 32;
        bytes = 17;
        return true;
    case kTypeFp8_32x32:   // 32x32 E4M3 block + one UE8M0 scale: 1,025 bytes per 1,024 values
        elems = 1024;
        bytes = 1025;
        return true;
    case 26:   // the engram hash tables are raw scalars
        elems = 1;
        bytes = 4;
        return true;
    case 27:
        elems = 1;
        bytes = 8;
        return true;
    default:
        int e = 0, b = 0;
        if (!strata::block_geometry(t, e, b)) return false;
        elems = static_cast<uint64_t>(e);
        bytes = static_cast<uint64_t>(b);
        return true;
    }
}

/// A tensor's payload length under the family's table; 0 when the type or shape is not whole blocks.  fp8_32x32
/// tiles the whole 2-D matrix with 32x32 blocks rather than encoding each row on its own, so it is the one type
/// here whose dimension 0 need not be a multiple of the block's 1,024 values (it must be a multiple of 32, as must
/// dimension 1).
inline uint64_t payload_bytes(const TensorInfo& t) {
    uint64_t be = 0, bb = 0;
    if (t.shape.empty() || !block_geometry(t.type, be, bb)) return 0;
    uint64_t elements = 1;
    for (uint64_t d : t.shape) {
        if (d == 0 || elements > (std::numeric_limits<uint64_t>::max)() / d) return 0;
        elements *= d;
    }
    if (t.type == kTypeFp8_32x32) {
        if (t.shape.size() != 2 || t.shape[0] % 32 != 0 || t.shape[1] % 32 != 0) return 0;
    } else if (t.shape[0] % be) {
        return 0;
    }
    if (elements % be) return 0;
    const uint64_t blocks = elements / be;
    if (blocks > (std::numeric_limits<uint64_t>::max)() / bb) return 0;
    return blocks * bb;
}

// ---- Container magic numbers (the same values tools/dsv4/vq_decode.hpp uses).
inline constexpr uint32_t kBlobMagic = 0x4C565144u;    // 'DQVL'
inline constexpr uint32_t kMatV2Magic = 0x51565144u;   // 'DQVQ'
inline constexpr uint32_t kMatV3Magic = 0x33565144u;   // 'DQV3'
inline constexpr uint32_t kBlobVersionMin = 2;
inline constexpr uint32_t kBlobVersionMax = 3;
inline constexpr uint32_t kBlobHeader = 16;            // magic, version, layer, experts
inline constexpr uint32_t kV2Header = 16;              // magic, dim(u16), nc(u16), rows, cols
inline constexpr uint32_t kV3Header = 32;              // + flags, main bits, codebook offset (u64)
inline constexpr uint32_t kVqDim = 8;
inline constexpr uint32_t kV3MainBits = 12;
inline constexpr uint32_t kV3FlagE4M3 = 1u;
inline constexpr uint32_t kV3FlagPlane = 2u;

inline uint16_t rd_u16(const uint8_t* p, uint64_t off) {
    uint16_t v;
    std::memcpy(&v, p + off, sizeof(v));
    return v;
}
inline uint32_t rd_u32(const uint8_t* p, uint64_t off) {
    uint32_t v;
    std::memcpy(&v, p + off, sizeof(v));
    return v;
}
inline uint64_t rd_u64(const uint8_t* p, uint64_t off) {
    uint64_t v;
    std::memcpy(&v, p + off, sizeof(v));
    return v;
}

/// Index width for a codebook of `nc` entries; matches tools/dsv4/vq_decode.hpp's index_bits.
inline uint32_t index_bits(uint32_t nc) {
    if (nc <= 256) return 8;
    if (nc <= 512) return 9;
    if (nc <= 1024) return 10;
    if (nc <= 2048) return 11;
    if (nc <= 4096) return 12;
    return 13;
}

// ---- Metadata readers.  A missing or malformed key is an error naming the key, never a default beside live data.
inline bool get_u64(const GgufFile& g, const std::string& key, uint64_t& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (v == nullptr) {
        err = "missing metadata '" + key + "'";
        return false;
    }
    if (!v->is_num()) {
        err = "metadata '" + key + "' is not a number";
        return false;
    }
    if (v->type == MetaType::I8 || v->type == MetaType::I16 || v->type == MetaType::I32 || v->type == MetaType::I64) {
        const int64_t s = static_cast<int64_t>(v->u);
        if (s < 0) {
            err = "metadata '" + key + "' is negative";
            return false;
        }
        out = static_cast<uint64_t>(s);
        return true;
    }
    if (v->type == MetaType::F32 || v->type == MetaType::F64) {
        const double d = v->f;
        if (!(d >= 0) || std::floor(d) != d) {
            err = "metadata '" + key + "' is not a whole number";
            return false;
        }
        out = static_cast<uint64_t>(d);
        return true;
    }
    out = v->u;
    return true;
}

inline bool get_f64(const GgufFile& g, const std::string& key, double& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (v == nullptr) {
        err = "missing metadata '" + key + "'";
        return false;
    }
    if (!v->is_num()) {
        err = "metadata '" + key + "' is not a number";
        return false;
    }
    if (v->type == MetaType::I8 || v->type == MetaType::I16 || v->type == MetaType::I32 || v->type == MetaType::I64)
        out = static_cast<double>(static_cast<int64_t>(v->u));
    else if (v->type == MetaType::F32 || v->type == MetaType::F64)
        out = v->f;
    else
        out = static_cast<double>(v->u);
    return true;
}

inline bool get_bool(const GgufFile& g, const std::string& key, bool& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (v == nullptr) {
        err = "missing metadata '" + key + "'";
        return false;
    }
    if (v->type == MetaType::BOOL) {
        out = v->u != 0;
        return true;
    }
    if (v->is_num() && (v->num() == 0.0 || v->num() == 1.0)) {
        out = v->num() != 0.0;
        return true;
    }
    err = "metadata '" + key + "' is not a bool";
    return false;
}

inline bool get_str(const GgufFile& g, const std::string& key, std::string& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (v == nullptr) {
        err = "missing metadata '" + key + "'";
        return false;
    }
    if (v->type != MetaType::STRING) {
        err = "metadata '" + key + "' is not a string";
        return false;
    }
    out = v->s;
    return true;
}

/// An array of whole, non-negative numbers.  The header reader keeps only the first 64 items, so an array longer
/// than that cannot be validated and is refused rather than half-read.
inline bool get_num_array(const GgufFile& g, const std::string& key, std::vector<uint64_t>& out, std::string& err,
                          uint64_t max_value = (std::numeric_limits<uint64_t>::max)()) {
    const MetaValue* v = g.get(key);
    if (v == nullptr) {
        err = "missing metadata '" + key + "'";
        return false;
    }
    if (v->type != MetaType::ARRAY) {
        err = "metadata '" + key + "' is not an array";
        return false;
    }
    if (v->items.size() != v->count) {
        err = "metadata '" + key + "' has " + std::to_string(v->count) +
              " items; only the first 64 can be checked";
        return false;
    }
    out.clear();
    out.reserve(v->items.size());
    for (const MetaValue& it : v->items) {
        if (!it.is_num() || it.num() < 0 || std::floor(it.num()) != it.num() || it.num() > (double)max_value) {
            err = "metadata '" + key + "' has a non-whole or out-of-range item";
            return false;
        }
        out.push_back(static_cast<uint64_t>(it.num()));
    }
    return true;
}

// ---- Geometry.  Every key this loader reads; all are required.
struct EngramTable {
    uint64_t layer = 0;
    uint64_t rows = 0;
    std::string path;
    uint64_t weight_offset = 0;
    uint64_t scale_offset = 0;
};

struct Geometry {
    std::string variant;
    std::string recipe;
    uint64_t block_count = 0;
    uint64_t context_length = 0;
    uint64_t embedding_length = 0;
    uint64_t vocab_size = 0;

    uint64_t head_count = 0;
    uint64_t head_count_kv = 0;
    uint64_t key_length = 0;
    uint64_t value_length = 0;
    uint64_t q_lora_rank = 0;
    uint64_t output_lora_rank = 0;
    uint64_t output_group_count = 0;
    uint64_t rope_dimension_count = 0;
    uint64_t sliding_window = 0;
    double rms_epsilon = 0;
    double compress_rope_freq_base = 0;

    uint64_t expert_count = 0;
    uint64_t expert_used_count = 0;
    uint64_t expert_feed_forward_length = 0;
    uint64_t expert_shared_count = 0;
    double expert_weights_scale = 0;
    bool expert_weights_norm = false;
    std::string expert_gating_func;

    uint64_t hyper_connection_count = 0;
    uint64_t sinkhorn_iterations = 0;
    double hyper_connection_epsilon = 0;

    uint64_t indexer_head_count = 0;
    uint64_t indexer_key_length = 0;
    uint64_t indexer_top_k = 0;
    uint64_t candidate_source_layer = 0;
    uint64_t candidate_topk_blocks = 0;
    uint64_t candidate_block_size = 0;

    double rope_freq_base = 0;
    double rope_scaling_factor = 0;
    double yarn_beta_fast = 0;
    double yarn_beta_slow = 0;
    uint64_t rope_original_context_length = 0;

    uint64_t mtp_tower_count = 0;
    uint64_t mtp_expert_count = 0;
    uint64_t mtp_expert_used_count = 0;
    uint64_t mtp_block_size = 0;
    uint64_t mtp_noise_token_id = 0;
    uint64_t mtp_markov_rank = 0;

    uint64_t engram_max_ngram_size = 0;
    uint64_t engram_head_count = 0;
    uint64_t engram_head_dim = 0;
    uint64_t engram_vocab_size = 0;
    uint64_t engram_compressed_vocab_size = 0;
    uint64_t engram_pad_id_compressed = 0;
    std::vector<uint64_t> compress_ratios;        // block_count + tower_count entries
    std::vector<uint64_t> swiglu_clamp_exp;       // block_count entries
    std::vector<uint64_t> kv_source_layers;
    std::vector<uint64_t> index_source_layers;
    std::vector<uint64_t> mtp_target_layers;      // tower_count entries
    std::vector<uint64_t> engram_layer_ids;
    std::vector<uint64_t> engram_num_embeddings;
    std::vector<EngramTable> engram_tables;
};

namespace detail {

inline bool contains(const std::vector<uint64_t>& v, uint64_t x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

}  // namespace detail

/// Reads and validates the family's metadata.  Returns "" on success, or an error naming the key or invariant.
inline std::string read_geometry(const GgufFile& g, Geometry& out) {
    out = Geometry{};
    if (model_family(g) != ModelFamily::Deepseek4) return "deepseek4: the file's architecture is not 'deepseek4'";
    std::string err;
    auto num = [&](const char* key, uint64_t& dst) { return get_u64(g, std::string("deepseek4.") + key, dst, err); };
    auto flt = [&](const char* key, double& dst) { return get_f64(g, std::string("deepseek4.") + key, dst, err); };
    auto arr = [&](const char* key, std::vector<uint64_t>& dst) {
        return get_num_array(g, std::string("deepseek4.") + key, dst, err, 0xFFFFFFFFu);
    };

    if (!get_str(g, "deepseek4.variant", out.variant, err) || !get_str(g, "deepseek4.recipe", out.recipe, err) ||
        !num("block_count", out.block_count) || !num("context_length", out.context_length) ||
        !num("embedding_length", out.embedding_length) || !num("vocab_size", out.vocab_size) ||
        !num("attention.head_count", out.head_count) || !num("attention.head_count_kv", out.head_count_kv) ||
        !num("attention.key_length", out.key_length) || !num("attention.value_length", out.value_length) ||
        !num("attention.q_lora_rank", out.q_lora_rank) ||
        !num("attention.output_lora_rank", out.output_lora_rank) ||
        !num("attention.output_group_count", out.output_group_count) ||
        !num("rope.dimension_count", out.rope_dimension_count) || !num("attention.sliding_window", out.sliding_window) ||
        !flt("attention.layer_norm_rms_epsilon", out.rms_epsilon) ||
        !flt("attention.compress_rope_freq_base", out.compress_rope_freq_base) ||
        !num("expert_count", out.expert_count) || !num("expert_used_count", out.expert_used_count) ||
        !num("expert_feed_forward_length", out.expert_feed_forward_length) ||
        !num("expert_shared_count", out.expert_shared_count) || !flt("expert_weights_scale", out.expert_weights_scale) ||
        !get_bool(g, "deepseek4.expert_weights_norm", out.expert_weights_norm, err) ||
        !get_str(g, "deepseek4.expert_gating_func", out.expert_gating_func, err) ||
        !num("hyper_connection.count", out.hyper_connection_count) ||
        !num("hyper_connection.sinkhorn_iterations", out.sinkhorn_iterations) ||
        !flt("hyper_connection.epsilon", out.hyper_connection_epsilon) ||
        !num("attention.indexer.head_count", out.indexer_head_count) ||
        !num("attention.indexer.key_length", out.indexer_key_length) ||
        !num("attention.indexer.top_k", out.indexer_top_k) ||
        !num("attention.candidate.source_layer", out.candidate_source_layer) ||
        !num("attention.candidate.topk_blocks", out.candidate_topk_blocks) ||
        !num("attention.candidate.block_size", out.candidate_block_size) || !flt("rope.freq_base", out.rope_freq_base) ||
        !flt("rope.scaling.factor", out.rope_scaling_factor) ||
        !flt("rope.scaling.yarn_beta_fast", out.yarn_beta_fast) ||
        !flt("rope.scaling.yarn_beta_slow", out.yarn_beta_slow) ||
        !num("rope.scaling.original_context_length", out.rope_original_context_length) ||
        !num("mtp.tower_count", out.mtp_tower_count) || !num("mtp.expert_count", out.mtp_expert_count) ||
        !num("mtp.expert_used_count", out.mtp_expert_used_count) || !num("mtp.block_size", out.mtp_block_size) ||
        !num("mtp.noise_token_id", out.mtp_noise_token_id) || !num("mtp.markov_rank", out.mtp_markov_rank) ||
        !num("engram.max_ngram_size", out.engram_max_ngram_size) || !num("engram.head_count", out.engram_head_count) ||
        !num("engram.head_dim", out.engram_head_dim) || !num("engram.vocab_size", out.engram_vocab_size) ||
        !num("engram.compressed_vocab_size", out.engram_compressed_vocab_size) ||
        !num("engram.pad_id_compressed", out.engram_pad_id_compressed) ||
        !arr("attention.compress_ratios", out.compress_ratios) || !arr("swiglu_clamp_exp", out.swiglu_clamp_exp) ||
        !arr("attention.kv_source_layers", out.kv_source_layers) ||
        !arr("attention.index_source_layers", out.index_source_layers) ||
        !arr("mtp.target_layers", out.mtp_target_layers) || !arr("engram.layer_ids", out.engram_layer_ids) ||
        !arr("engram.num_embeddings", out.engram_num_embeddings))
        return "deepseek4: " + err;

    auto bad = [](const std::string& what) { return "deepseek4: " + what; };
    if (out.block_count == 0 || out.block_count > 1024) return bad("block_count is not 1..1024");
    if (out.context_length == 0) return bad("context_length is 0");
    if (out.embedding_length == 0) return bad("embedding_length is 0");
    if (out.vocab_size == 0) return bad("vocab_size is 0");
    if (out.head_count == 0 || out.head_count_kv == 0 || out.head_count_kv > out.head_count)
        return bad("head_count / head_count_kv are inconsistent");
    if (out.key_length == 0 || out.value_length == 0) return bad("key_length or value_length is 0");
    if (out.q_lora_rank == 0 || out.output_lora_rank == 0 || out.output_group_count == 0)
        return bad("q_lora_rank / output_lora_rank / output_group_count is 0");
    if (out.rope_dimension_count == 0) return bad("rope.dimension_count is 0");
    if (out.sliding_window == 0) return bad("attention.sliding_window is 0");
    if (!(out.rms_epsilon > 0) || !(out.compress_rope_freq_base > 0)) return bad("an epsilon or RoPE base is not positive");
    if (out.expert_count == 0 || out.expert_count > 65535) return bad("expert_count is not 1..65535");
    if (out.expert_used_count == 0 || out.expert_used_count > out.expert_count)
        return bad("expert_used_count is 0 or above expert_count");
    if (out.expert_feed_forward_length == 0) return bad("expert_feed_forward_length is 0");
    if (!(out.expert_weights_scale > 0)) return bad("expert_weights_scale is not positive");
    if (out.expert_gating_func.empty()) return bad("expert_gating_func is empty");
    if (out.hyper_connection_count == 0 || out.sinkhorn_iterations == 0)
        return bad("hyper_connection.count or sinkhorn_iterations is 0");
    if (!(out.hyper_connection_epsilon > 0) || out.hyper_connection_epsilon >= 1)
        return bad("hyper_connection.epsilon is not in (0, 1)");
    if (out.indexer_head_count == 0 || out.indexer_key_length == 0 || out.indexer_top_k == 0)
        return bad("an indexer count is 0");
    if (out.candidate_source_layer >= out.block_count) return bad("candidate.source_layer is not a block");
    if (out.candidate_topk_blocks == 0 || out.candidate_block_size == 0)
        return bad("candidate.topk_blocks or candidate.block_size is 0");
    if (!(out.rope_freq_base > 0) || !(out.rope_scaling_factor > 0) || !(out.yarn_beta_fast > 0) ||
        !(out.yarn_beta_slow > 0) || out.rope_original_context_length == 0)
        return bad("a RoPE scaling value is not positive");
    if (out.mtp_tower_count > 64) return bad("mtp.tower_count is above 64");
    if (out.mtp_tower_count > 0) {
        if (out.mtp_expert_count == 0 || out.mtp_expert_count > 65535) return bad("mtp.expert_count is not 1..65535");
        if (out.mtp_expert_used_count == 0 || out.mtp_expert_used_count > out.mtp_expert_count)
            return bad("mtp.expert_used_count is 0 or above mtp.expert_count");
        if (out.mtp_block_size == 0 || out.mtp_markov_rank == 0) return bad("mtp.block_size or markov_rank is 0");
        if (out.mtp_noise_token_id >= out.vocab_size) return bad("mtp.noise_token_id is outside the vocabulary");
        if (out.mtp_target_layers.size() != out.mtp_tower_count)
            return bad("mtp.target_layers has " + std::to_string(out.mtp_target_layers.size()) + " entries, but " +
                       "mtp.tower_count is " + std::to_string(out.mtp_tower_count));
        for (uint64_t t : out.mtp_target_layers)
            if (t >= out.block_count) return bad("an mtp.target_layers entry is not a block");
    }
    if (out.engram_max_ngram_size == 0 || out.engram_head_count == 0 || out.engram_head_dim == 0 ||
        out.engram_vocab_size == 0 || out.engram_compressed_vocab_size == 0)
        return bad("an engram size is 0");
    if (out.engram_pad_id_compressed >= out.engram_compressed_vocab_size)
        return bad("engram.pad_id_compressed is outside the compressed vocabulary");
    if (out.engram_layer_ids.size() != out.engram_num_embeddings.size())
        return bad("engram.layer_ids and engram.num_embeddings differ in length");
    if (out.compress_ratios.size() != out.block_count + out.mtp_tower_count)
        return bad("attention.compress_ratios has " + std::to_string(out.compress_ratios.size()) +
                   " entries; block_count + mtp.tower_count is " +
                   std::to_string(out.block_count + out.mtp_tower_count));
    for (uint64_t r : out.compress_ratios)
        if (r > 8) return bad("a compress ratio is above 8");
    if (out.swiglu_clamp_exp.size() != out.block_count)
        return bad("swiglu_clamp_exp has " + std::to_string(out.swiglu_clamp_exp.size()) + " entries; block_count is " +
                   std::to_string(out.block_count));
    for (uint64_t e : out.swiglu_clamp_exp)
        if (e == 0 || e > 1000) return bad("a swiglu_clamp_exp entry is outside 1..1000");
    if (out.kv_source_layers.empty() || out.index_source_layers.empty()) return bad("a source-layer list is empty");
    for (uint64_t l : out.kv_source_layers)
        if (l >= out.block_count) return bad("a kv_source_layers entry is not a block");
    for (uint64_t l : out.index_source_layers)
        if (l >= out.block_count) return bad("an index_source_layers entry is not a block");

    // Each engram layer's sidecar table: rows, byte offsets, and the row layout the reader implements (256 weight
    // bytes + 8 scale bytes = 264).  The offsets must satisfy weight_offset + rows*256 == scale_offset, which is
    // what makes the row addressing correct.
    out.engram_tables.clear();
    for (size_t i = 0; i < out.engram_layer_ids.size(); ++i) {
        const uint64_t layer = out.engram_layer_ids[i];
        const uint64_t rows = out.engram_num_embeddings[i];
        if (layer >= out.block_count) return bad("an engram.layer_ids entry is not a block");
        if (rows == 0) return bad("an engram.num_embeddings entry is 0");
        EngramTable t;
        t.layer = layer;
        t.rows = rows;
        const std::string p = "deepseek4.engram." + std::to_string(i) + ".";
        if (!get_str(g, p + "table_path", t.path, err) || !get_u64(g, p + "weight_offset", t.weight_offset, err) ||
            !get_u64(g, p + "scale_offset", t.scale_offset, err))
            return "deepseek4: " + err;
        if (t.path.empty()) return bad("engram table " + std::to_string(i) + " has an empty path");
        const uint64_t weight_bytes = rows * 256;
        if (weight_bytes / 256 != rows) return bad("engram table " + std::to_string(i) + " row count overflows");
        if (t.weight_offset + weight_bytes != t.scale_offset)
            return bad("engram table " + std::to_string(i) +
                       ": weight_offset + rows*256 does not equal scale_offset (row layout is not 264 bytes)");
        out.engram_tables.push_back(std::move(t));
    }
    return {};
}

// ---- The expert blob (GGUF type 42).
struct BlobInfo {
    uint32_t version = 0;
    uint32_t layer = 0;
    uint32_t experts = 0;
    uint32_t nc = 0;        ///< codebook size; uniform across every payload in the container
    uint32_t nbit = 0;
    uint64_t version3_payloads = 0;
    uint64_t payloads_checked = 0;
    bool complete = true;   ///< every slot's payload was inside the buffer (a prefix is an error, not a partial check)
};

/// Checks a whole DQVL container: header, the expert x {gate, up, down} slot table, and every payload header and
/// stream bounds.  `expect_main` selects the expected expert count (main layers vs MTP towers) and, when true,
/// that the container's layer field is `expect_layer`.  `size` is the tensor's declared length, so a buffer that
/// ends inside a payload is refused.  Each payload is bounded by the next slot's offset, and the shared v3 codebook
/// must sit between the slot table and the first payload.
inline bool check_blob(const uint8_t* d, uint64_t size, uint32_t expect_layer, bool expect_main, const Geometry& g,
                       BlobInfo* info, std::string& err) {
    *info = BlobInfo{};
    auto fail = [&](const std::string& what) {
        err = what;
        return false;
    };
    if (size < kBlobHeader) return fail("container shorter than its 16-byte header");
    if (rd_u32(d, 0) != kBlobMagic) return fail("not a DQVL container");
    const uint32_t ver = rd_u32(d, 4);
    if (ver < kBlobVersionMin || ver > kBlobVersionMax)
        return fail("container version " + std::to_string(ver) + " is not 2 or 3");
    const uint32_t layer = rd_u32(d, 8);
    const uint32_t experts = rd_u32(d, 12);
    const uint64_t expected_experts = expect_main ? g.expert_count : g.mtp_expert_count;
    if (expect_main && layer != expect_layer)
        return fail("container is for layer " + std::to_string(layer) + ", expected " + std::to_string(expect_layer));
    if (experts != expected_experts)
        return fail("container holds " + std::to_string(experts) + " experts, expected " +
                    std::to_string(expected_experts));
    if (experts == 0 || experts > 65535) return fail("container expert count is 0 or above 65,535");
    const uint64_t table_end = kBlobHeader + static_cast<uint64_t>(experts) * 3u * 8u;
    if (table_end > size) return fail("slot table runs past the container");

    const char* which_name[3] = {"gate", "up", "down"};
    const uint32_t E = static_cast<uint32_t>(g.embedding_length);
    const uint32_t FF = static_cast<uint32_t>(g.expert_feed_forward_length);

    // The slot offsets in file order: a payload runs from its offset to the next slot's, and the shared v3 codebook
    // sits between the slot table and the first payload.  Bounding each payload by its neighbour (not by the
    // container end) catches a payload that overruns the next one.
    std::vector<uint64_t> offsets;
    offsets.reserve(static_cast<size_t>(experts) * 3u);
    for (uint32_t e = 0; e < experts; ++e) {
        for (uint32_t which = 0; which < 3; ++which) {
            const uint64_t off = rd_u64(d, kBlobHeader + (static_cast<uint64_t>(e) * 3u + which) * 8u);
            if (off == 0) return fail("expert " + std::to_string(e) + " " + which_name[which] + " slot is absent");
            if (off < table_end)
                return fail("expert " + std::to_string(e) + " " + which_name[which] +
                            " slot points into the slot table");
            if (off > size)
                return fail("expert " + std::to_string(e) + " " + which_name[which] +
                            " slot points past the container");
            offsets.push_back(off);
        }
    }
    std::sort(offsets.begin(), offsets.end());
    for (size_t i = 1; i < offsets.size(); ++i)
        if (offsets[i] == offsets[i - 1]) return fail("two payload slots share offset " + std::to_string(offsets[i]));
    const uint64_t first_payload = offsets.front();

    bool have_nc = false, have_cb = false;
    uint32_t first_nc = 0, first_ver = 0;
    uint64_t first_cb = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        for (uint32_t which = 0; which < 3; ++which) {
            const uint64_t off = rd_u64(d, kBlobHeader + (static_cast<uint64_t>(e) * 3u + which) * 8u);
            const auto next = std::upper_bound(offsets.begin(), offsets.end(), off);
            const uint64_t end = next == offsets.end() ? size : *next;
            const std::string where = "expert " + std::to_string(e) + " " + which_name[which];
            const uint32_t expected_rows = which == 2 ? E : FF;
            const uint32_t expected_cols = which == 2 ? FF : E;
            if (end - off < kV2Header) return fail(where + " payload is shorter than its header");
            const uint8_t* p = d + off;
            const uint32_t magic = rd_u32(p, 0);
            const bool v3 = magic == kMatV3Magic;
            if (!v3 && magic != kMatV2Magic) return fail(where + " payload magic is neither DQVQ nor DQV3");
            if (v3 && end - off < kV3Header) return fail(where + " v3 payload is shorter than its header");
            if (ver == 3 && !v3)
                return fail("v3 container holds a v2 payload (expert " + std::to_string(e) + ", " +
                            which_name[which] + ")");
            if (ver == 2 && v3)
                return fail("v2 container holds a v3 payload (expert " + std::to_string(e) + ", " +
                            which_name[which] + ")");
            const uint32_t dim = rd_u16(p, 4);
            const uint32_t nc = rd_u16(p, 6);
            const uint32_t rows = rd_u32(p, 8);
            const uint32_t cols = rd_u32(p, 12);
            if (dim != kVqDim) return fail("payload dim is not 8");
            if (cols == 0 || cols % kVqDim != 0) return fail("payload cols is not a multiple of 8");
            if (rows != expected_rows || cols != expected_cols)
                return fail(where + " is " + std::to_string(rows) + "x" + std::to_string(cols) + ", expected " +
                            std::to_string(expected_rows) + "x" + std::to_string(expected_cols));
            if (nc == 0) return fail("payload codebook size is 0");
            if (have_nc && (nc != first_nc || ver != first_ver))
                return fail("container mixes payload versions or codebook sizes");
            have_nc = true;
            first_nc = nc;
            first_ver = ver;
            const uint32_t nbit = index_bits(nc);
            const uint64_t nidx = cols / kVqDim;
            if (v3) {
                const uint32_t flags = rd_u32(p, 16);
                const uint32_t mnb = rd_u32(p, 20);
                const uint64_t cb_off = rd_u64(p, 24);
                if ((flags & kV3FlagE4M3) == 0) return fail("v3 payload lacks the E4M3 codebook flag");
                if (mnb != kV3MainBits) return fail("v3 payload main width is not 12 bits");
                const bool plane = (flags & kV3FlagPlane) != 0;
                if (plane != (nbit > kV3MainBits))
                    return fail("v3 plane flag disagrees with the codebook size");
                if (have_cb && cb_off != first_cb) return fail("v3 payloads disagree on the codebook offset");
                have_cb = true;
                first_cb = cb_off;
                const uint64_t cb_len = static_cast<uint64_t>(nc) * kVqDim;
                if (cb_off < table_end) return fail("the v3 codebook overlaps the slot table");
                if (cb_off > first_payload || cb_len > first_payload - cb_off)
                    return fail("the v3 codebook overlaps the first payload");
                const uint64_t gains_off = off + kV3Header;
                const uint64_t gains_len = static_cast<uint64_t>(rows) * 2;
                if (gains_len > end - gains_off) return fail(where + " gains run past its payload");
                const uint64_t stream_len = end - (gains_off + gains_len);
                const uint64_t mrow = (nidx * kV3MainBits + 7) / 8;
                const uint64_t main_len = static_cast<uint64_t>(rows) * mrow;
                const uint64_t last_window = static_cast<uint64_t>(rows - 1) * mrow +
                                             (((nidx - 1) * kV3MainBits) >> 3) + 3;
                const uint64_t need = std::max(main_len, last_window);
                if (stream_len < need) return fail(where + ": v3 index stream truncated");
                if (plane) {
                    const uint64_t prow = (nidx + 7) / 8;
                    if (stream_len < main_len + static_cast<uint64_t>(rows) * prow)
                        return fail(where + ": v3 index plane truncated");
                }
            } else {
                const uint64_t cb_len = static_cast<uint64_t>(nc) * kVqDim * 2;
                const uint64_t gains_off = off + kV2Header;
                const uint64_t gains_len = static_cast<uint64_t>(rows) * 2;
                if (cb_len > end - gains_off || gains_len > end - gains_off - cb_len)
                    return fail(where + " payload is shorter than its codebook and gains");
                const uint64_t stream_len = end - (gains_off + cb_len + gains_len);
                const uint64_t total = static_cast<uint64_t>(rows) * nidx;
                if (nbit != 8) {
                    const uint64_t need = ((total - 1) * nbit >> 3) + 3;
                    if (stream_len < need) return fail(where + ": v2 index stream truncated");
                } else if (stream_len < total) {
                    return fail(where + ": v2 index stream truncated");
                }
            }
            info->payloads_checked++;
            if (v3) info->version3_payloads++;
        }
    }
    info->version = ver;
    info->layer = layer;
    info->experts = experts;
    info->nc = first_nc;
    info->nbit = index_bits(first_nc);
    info->complete = true;
    err.clear();
    return true;
}

// ---- Tensor inventory.  The expected name set is generated from the geometry; one entry per tensor of this
// published artifact's recipe (type 42 expert blobs, fp8_32x32 towers, Q4_K skeleton).
struct TensorSpec {
    uint32_t type = 0;
    std::vector<uint64_t> shape;   ///< empty = shape not checked
    int ndim = -1;                 ///< when shape is empty: required rank, or -1 for any
};

namespace detail {

inline void want(std::map<std::string, TensorSpec>& s, const std::string& name, uint32_t type) {
    s.emplace(name, TensorSpec{type, {}, -1});
}
inline void want_shape(std::map<std::string, TensorSpec>& s, const std::string& name, uint32_t type,
                       std::vector<uint64_t> shape) {
    s.emplace(name, TensorSpec{type, std::move(shape), -1});
}
inline void want_rank1(std::map<std::string, TensorSpec>& s, const std::string& name, uint32_t type) {
    s.emplace(name, TensorSpec{type, {}, 1});
}

// The 22 tensors every block and every MTP tower carries (the blob included).
inline void base_block(std::map<std::string, TensorSpec>& s, const std::string& p, uint64_t E, uint64_t FF,
                       uint64_t Q, uint64_t H, uint64_t KL, uint64_t OL, uint64_t NG, uint64_t VL, uint32_t skel,
                       uint64_t experts) {
    want(s, p + "hc_attn_fn.weight", 0);
    want(s, p + "hc_attn_scale.weight", 0);
    want(s, p + "hc_attn_base.weight", 0);
    want_shape(s, p + "attn_norm.weight", 0, {E});
    want_shape(s, p + "attn_q_a.weight", skel, {E, Q});
    want_shape(s, p + "attn_q_a_norm.weight", 0, {Q});
    want_shape(s, p + "attn_q_b.weight", skel, {Q, H * KL});
    want_shape(s, p + "attn_kv.weight", skel, {E, KL});
    want_shape(s, p + "attn_kv_a_norm.weight", 0, {KL});
    want_shape(s, p + "attn_sinks.weight", 0, {H});
    if (VL > 0 && NG > 0 && OL > 0) {
        want_shape(s, p + "attn_output_a.weight", skel, {VL * NG, OL * NG});
        want_shape(s, p + "attn_output_b.weight", skel, {OL * NG, E});
    } else {
        want(s, p + "attn_output_a.weight", skel);
        want(s, p + "attn_output_b.weight", skel);
    }
    want(s, p + "hc_ffn_fn.weight", 0);
    want(s, p + "hc_ffn_scale.weight", 0);
    want(s, p + "hc_ffn_base.weight", 0);
    want_shape(s, p + "ffn_norm.weight", 0, {E});
    want_shape(s, p + "ffn_gate_inp.weight", 30, {E, experts});
    want_shape(s, p + "exp_probs_b.bias", 0, {experts});
    want_rank1(s, p + "ffn_exps_vq.blob", kTypeVqBlob);
    want_shape(s, p + "ffn_gate_shexp.weight", skel, {E, FF});
    want_shape(s, p + "ffn_up_shexp.weight", skel, {E, FF});
    want_shape(s, p + "ffn_down_shexp.weight", skel, {FF, E});
}

}  // namespace detail

/// Every tensor this family's published artifact carries, named from the geometry.
inline std::map<std::string, TensorSpec> expected_tensors(const Geometry& g) {
    using detail::want;
    using detail::want_rank1;
    using detail::want_shape;
    std::map<std::string, TensorSpec> s;
    const uint64_t E = g.embedding_length, V = g.vocab_size, Q = g.q_lora_rank;
    const uint64_t H = g.head_count, KL = g.key_length, VL = g.value_length;
    const uint64_t OL = g.output_lora_rank, NG = g.output_group_count, FF = g.expert_feed_forward_length;
    const bool kv_ok = KL == VL;

    for (uint64_t L = 0; L < g.block_count; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        detail::base_block(s, p, E, FF, Q, H, KL, OL, NG, kv_ok ? VL : 0, 12, g.expert_count);
        const uint64_t ratio = L < g.compress_ratios.size() ? g.compress_ratios[L] : 0;
        if (detail::contains(g.kv_source_layers, L)) {
            want_shape(s, p + "attn_compressor_kv.weight", 30, {E, KL});
            want_shape(s, p + "attn_compressor_norm.weight", 0, {KL});
            if (ratio == 2) want(s, p + "attn_compressor_gate.weight", 30);
        }
        const bool index_source = detail::contains(g.index_source_layers, L);
        if (index_source) {
            want_shape(s, p + "indexer.proj.weight", 30, {E, g.indexer_head_count});
            want_shape(s, p + "indexer.attn_q_b.weight", 12, {Q, g.indexer_head_count * g.indexer_key_length});
            if (detail::contains(g.kv_source_layers, L)) {
                want_shape(s, p + "indexer.wk.weight", 30, {KL, g.indexer_key_length});
                want_shape(s, p + "indexer.k_norm.weight", 0, {g.indexer_key_length});
            }
        }
        if (detail::contains(g.engram_layer_ids, L)) {
            want_shape(s, p + "engram_k.weight", 0, {E, g.engram_max_ngram_size});
            want_shape(s, p + "engram_q.weight", 0, {E, g.engram_max_ngram_size});
            want(s, p + "engram_wkv.weight", kTypeFp8_32x32);
        }
    }

    want_shape(s, "token_embd.weight", 12, {E, V});
    want_shape(s, "output.weight", 12, {E, V});
    want_shape(s, "output_norm.weight", 0, {E});

    if (!g.engram_layer_ids.empty()) {
        want(s, "engram.token_map", 26);
        want(s, "engram.multipliers", 27);
        want(s, "engram.primes", 27);
        want(s, "engram.offsets", 27);
    }

    if (g.mtp_tower_count > 0) {
        want_shape(s, "mtp.main_norm.weight", 0, {E});
        want_shape(s, "mtp.main_proj.weight", kTypeFp8_32x32, {g.mtp_tower_count * E, E});
        want_shape(s, "mtp.markov_embd.weight", 30, {g.mtp_markov_rank, V});
        want_shape(s, "mtp.markov_head.weight", 30, {g.mtp_markov_rank, V});
        want_shape(s, "mtp.confidence.weight", 30, {E + g.mtp_markov_rank, 1});
        want_shape(s, "mtp.out_norm.weight", 0, {E});
        for (uint64_t T = 0; T < g.mtp_tower_count; ++T) {
            const std::string p = "mtp." + std::to_string(T) + ".";
            detail::base_block(s, p, E, FF, Q, H, KL, OL, NG, kv_ok ? VL : 0, kTypeFp8_32x32, g.mtp_expert_count);
        }
    }
    return s;
}

struct Inventory {
    uint64_t tensors = 0;
    uint64_t bytes = 0;
    std::map<uint32_t, uint64_t> type_counts;
    std::vector<std::string> errors;   ///< each names the tensor or metadata key it is about
    uint64_t beyond_eof = 0;           ///< tensors whose payload extends past the file (a prefix download)
};

/// Checks every tensor's name, type, shape and payload length against `expected_tensors`, and its offset against
/// the file.  A payload past EOF is counted in `beyond_eof`; `allow_partial` decides whether that is an error.
/// `assembled_size`, when nonzero, replaces `g.file_size()`: a download split into byte-identical parts is
/// checked against the length it will have once assembled, while `g` still reads the header from part 1.
inline Inventory check_inventory(const GgufFile& g, const Geometry& geo, bool allow_partial,
                                 uint64_t assembled_size = 0) {
    Inventory inv;
    const auto spec = expected_tensors(geo);
    struct Span {
        uint64_t offset, bytes;
        const std::string* name;
        bool inside;
    };
    std::vector<Span> spans;
    const uint64_t file_size = assembled_size != 0 ? assembled_size : g.file_size();
    const uint64_t payload_end = file_size > g.data_start() ? file_size - g.data_start() : 0;

    for (const TensorInfo& t : g.tensors()) {
        inv.tensors++;
        inv.type_counts[t.type]++;
        const auto it = spec.find(t.name);
        if (it == spec.end()) {
            inv.errors.push_back("unexpected tensor '" + t.name + "' (not in the deepseek4 inventory)");
        } else {
            const TensorSpec& sp = it->second;
            if (t.type != sp.type)
                inv.errors.push_back("tensor '" + t.name + "' is type " + type_name(t.type) + " (" +
                                     std::to_string(t.type) + "), expected " + type_name(sp.type) + " (" +
                                     std::to_string(sp.type) + ")");
            if (!sp.shape.empty() && t.shape != sp.shape) {
                std::string got, want;
                for (uint64_t d : t.shape) got += (got.empty() ? "" : "x") + std::to_string(d);
                for (uint64_t d : sp.shape) want += (want.empty() ? "" : "x") + std::to_string(d);
                inv.errors.push_back("tensor '" + t.name + "' has shape [" + got + "], expected [" + want + "]");
            }
            if (sp.shape.empty() && sp.ndim >= 0 && t.shape.size() != (size_t) sp.ndim)
                inv.errors.push_back("tensor '" + t.name + "' has rank " + std::to_string(t.shape.size()) +
                                     ", expected " + std::to_string(sp.ndim));
        }
        const uint64_t bytes = payload_bytes(t);
        if (bytes == 0) {
            inv.errors.push_back("tensor '" + t.name + "' has no whole-block payload size (type " +
                                 type_name(t.type) + ", shape dim0 " +
                                 (t.shape.empty() ? std::string("none") : std::to_string(t.shape[0])) + ")");
            continue;
        }
        const bool inside = t.offset <= payload_end && bytes <= payload_end - t.offset;
        spans.push_back(Span{t.offset, bytes, &t.name, inside});
        if (inside) {
            inv.bytes += bytes;
        } else {
            inv.beyond_eof++;
            if (!allow_partial)
                inv.errors.push_back("tensor '" + t.name + "' runs past the end of the file (needs " +
                                     std::to_string(bytes) + " bytes at offset " + std::to_string(t.offset) + ", " +
                                     std::to_string(payload_end) + " available)");
        }
    }

    for (const auto& kv : spec)
        if (g.find(kv.first) == nullptr) inv.errors.push_back("missing tensor '" + kv.first + "'");

    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) { return a.offset < b.offset; });
    for (size_t i = 1; i < spans.size(); ++i) {
        if (!spans[i - 1].inside) continue;
        if (spans[i - 1].offset + spans[i - 1].bytes > spans[i].offset)
            inv.errors.push_back("tensor '" + *spans[i - 1].name + "' overlaps '" + *spans[i].name + "'");
    }
    return inv;
}

}  // namespace strata::deepseek4
