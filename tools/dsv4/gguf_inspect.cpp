// tools/dsv4/gguf_inspect.cpp - check a DeepSeek-V4.1-Flash GGUF against the family loader's inventory.
//
// Prints the metadata geometry, the tensor-type histogram, every inventory error by tensor name, and (when the
// payload is in the file) every expert blob's container, slot table and payload headers.  It reads the header only
// unless blobs are present, so it can also check a prefix of a download.
//
// Build:
//   g++ -O2 -std=c++17 -Iinclude tools/dsv4/gguf_inspect.cpp -o /tmp/dsv4_gguf_inspect
// Run:
//   /tmp/dsv4_gguf_inspect <model.gguf> [--partial] [--no-blobs]
//
// --partial  allow tensor payloads to extend past the end of the file (a header prefix or a download in progress)
// --no-blobs skip the per-blob container checks

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "strata/artifact/deepseek4.hpp"

namespace ds = strata::deepseek4;

namespace {

std::string list64(const std::vector<uint64_t>& v) {
    std::string s;
    for (uint64_t x : v) s += (s.empty() ? "" : ", ") + std::to_string(x);
    return s;
}

void print_geometry(const ds::Geometry& g) {
    std::printf("geometry\n");
    std::printf("  variant %s, recipe %s\n", g.variant.c_str(), g.recipe.c_str());
    std::printf("  blocks %llu, context %llu, width %llu, vocab %llu\n",
                (unsigned long long) g.block_count, (unsigned long long) g.context_length,
                (unsigned long long) g.embedding_length, (unsigned long long) g.vocab_size);
    std::printf("  attention: %llu heads, %llu kv, key %llu, value %llu, q_lora %llu, out_lora %llu x %llu, "
                "rope dim %llu, window %llu, rms eps %.3g\n",
                (unsigned long long) g.head_count, (unsigned long long) g.head_count_kv,
                (unsigned long long) g.key_length, (unsigned long long) g.value_length,
                (unsigned long long) g.q_lora_rank, (unsigned long long) g.output_lora_rank,
                (unsigned long long) g.output_group_count, (unsigned long long) g.rope_dimension_count,
                (unsigned long long) g.sliding_window, g.rms_epsilon);
    std::printf("  experts: %llu, top %llu, shared %llu, ffn %llu, scale %.3g, norm %s, %s\n",
                (unsigned long long) g.expert_count, (unsigned long long) g.expert_used_count,
                (unsigned long long) g.expert_shared_count, (unsigned long long) g.expert_feed_forward_length,
                g.expert_weights_scale, g.expert_weights_norm ? "on" : "off", g.expert_gating_func.c_str());
    std::printf("  hyper-connections: %llu routes, %llu Sinkhorn iterations, eps %.3g\n",
                (unsigned long long) g.hyper_connection_count, (unsigned long long) g.sinkhorn_iterations,
                g.hyper_connection_epsilon);
    std::printf("  indexer: %llu heads x %llu, top %llu; candidate layer %llu, %llu blocks of %llu\n",
                (unsigned long long) g.indexer_head_count, (unsigned long long) g.indexer_key_length,
                (unsigned long long) g.indexer_top_k, (unsigned long long) g.candidate_source_layer,
                (unsigned long long) g.candidate_topk_blocks, (unsigned long long) g.candidate_block_size);
    std::printf("  rope: base %.6g, factor %.6g, YaRN %.6g/%.6g, original context %llu, compress base %.6g\n",
                g.rope_freq_base, g.rope_scaling_factor, g.yarn_beta_fast, g.yarn_beta_slow,
                (unsigned long long) g.rope_original_context_length, g.compress_rope_freq_base);
    std::printf("  compress ratios [%s]\n", list64(g.compress_ratios).c_str());
    std::printf("  kv source layers [%s]\n", list64(g.kv_source_layers).c_str());
    std::printf("  index source layers [%s]\n", list64(g.index_source_layers).c_str());
    std::printf("  swiglu clamp exp [%s]\n", list64(g.swiglu_clamp_exp).c_str());
    if (g.mtp_tower_count > 0) {
        std::printf("  MTP: %llu towers, %llu experts top %llu, block %llu, markov rank %llu, noise token %llu, "
                    "target layers [%s]\n",
                    (unsigned long long) g.mtp_tower_count, (unsigned long long) g.mtp_expert_count,
                    (unsigned long long) g.mtp_expert_used_count, (unsigned long long) g.mtp_block_size,
                    (unsigned long long) g.mtp_markov_rank, (unsigned long long) g.mtp_noise_token_id,
                    list64(g.mtp_target_layers).c_str());
    } else {
        std::printf("  MTP: none\n");
    }
    std::printf("  engram: %llu-gram, %llu heads x %llu, vocab %llu compressed to %llu (pad %llu), layers [%s]\n",
                (unsigned long long) g.engram_max_ngram_size, (unsigned long long) g.engram_head_count,
                (unsigned long long) g.engram_head_dim, (unsigned long long) g.engram_vocab_size,
                (unsigned long long) g.engram_compressed_vocab_size,
                (unsigned long long) g.engram_pad_id_compressed, list64(g.engram_layer_ids).c_str());
    for (const auto& t : g.engram_tables)
        std::printf("    layer %llu: %llu rows, weights at %llu, scales at %llu, %s\n",
                    (unsigned long long) t.layer, (unsigned long long) t.rows,
                    (unsigned long long) t.weight_offset, (unsigned long long) t.scale_offset, t.path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = nullptr;
    bool partial = false, check_blobs = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--partial") == 0) {
            partial = true;
        } else if (std::strcmp(argv[i], "--no-blobs") == 0) {
            check_blobs = false;
        } else if (path == nullptr) {
            path = argv[i];
        } else {
            std::fprintf(stderr, "unexpected argument %s\n", argv[i]);
            return 2;
        }
    }
    if (path == nullptr) {
        std::fprintf(stderr, "usage: %s <model.gguf> [--partial] [--no-blobs]\n", argv[0]);
        return 2;
    }

    int failures = 0;
    try {
        strata::GgufFile g(path);
        std::printf("%s: GGUF v%u, %zu tensors, data starts at %llu, file %llu bytes\n", path, g.version(),
                    g.tensors().size(), (unsigned long long) g.data_start(),
                    (unsigned long long) g.file_size());
        if (strata::model_family(g) != strata::ModelFamily::Deepseek4) {
            std::fprintf(stderr, "not a deepseek4 file (general.architecture is not 'deepseek4')\n");
            return 1;
        }

        ds::Geometry geo;
        const std::string geom_err = ds::read_geometry(g, geo);
        if (!geom_err.empty()) {
            std::fprintf(stderr, "%s\n", geom_err.c_str());
            return 1;
        }
        print_geometry(geo);

        const ds::Inventory inv = ds::check_inventory(g, geo, partial);
        std::printf("inventory: %llu tensors, %llu payload bytes in the file", (unsigned long long) inv.tensors,
                    (unsigned long long) inv.bytes);
        if (inv.beyond_eof != 0) std::printf(", %llu beyond the end of the file", (unsigned long long) inv.beyond_eof);
        std::printf("\n  types:");
        for (const auto& kv : inv.type_counts)
            std::printf(" %s x%llu", ds::type_name(kv.first), (unsigned long long) kv.second);
        std::printf("\n");
        for (const std::string& e : inv.errors) std::printf("  ERROR %s\n", e.c_str());
        failures += (int) inv.errors.size();

        if (check_blobs) {
            const uint64_t payload_end = g.file_size() - g.data_start();
            std::printf("expert blobs\n");
            uint64_t checked = 0, skipped = 0;
            for (const strata::TensorInfo& t : g.tensors()) {
                if (t.type != ds::kTypeVqBlob) continue;
                uint32_t layer = 0;
                bool main = false;
                if (std::sscanf(t.name.c_str(), "blk.%u.ffn_exps_vq.blob", &layer) == 1)
                    main = true;
                else if (std::sscanf(t.name.c_str(), "mtp.%u.ffn_exps_vq.blob", &layer) == 1)
                    main = false;
                else {
                    std::printf("  ERROR %s: type-42 tensor whose name the loader cannot place\n", t.name.c_str());
                    failures++;
                    continue;
                }
                const uint64_t bytes = ds::payload_bytes(t);
                if (bytes == 0 || t.offset > payload_end || bytes > payload_end - t.offset) {
                    std::printf("  %s: payload not in this file, skipped\n", t.name.c_str());
                    skipped++;
                    continue;
                }
                ds::BlobInfo info;
                std::string err;
                if (!ds::check_blob(g.tensor_data(t), bytes, layer, main, geo, &info, err)) {
                    std::printf("  ERROR %s: %s\n", t.name.c_str(), err.c_str());
                    failures++;
                    continue;
                }
                std::printf("  %s: v%u layer %u, %u experts, nc %u (%u-bit), %llu payloads checked\n", t.name.c_str(),
                            info.version, info.layer, info.experts, info.nc, info.nbit,
                            (unsigned long long) info.payloads_checked);
                checked++;
            }
            std::printf("  %llu blobs checked, %llu skipped\n", (unsigned long long) checked,
                        (unsigned long long) skipped);
        }

        if (failures == 0)
            std::printf("OK\n");
        else
            std::printf("%d errors\n", failures);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
