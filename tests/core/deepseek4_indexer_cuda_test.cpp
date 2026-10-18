// tests/core/deepseek4_indexer_cuda_test.cpp - P3's sparse indexer gate.
//
//   deepseek4_indexer_cuda_test
//
// Three pieces from cuda/ds41_indexer.cuh:
//
//   score      score[i][g] = bf16(sum_h bf16(relu(bf16(q_h . k_g)) * w[i][h])), invisible groups -inf; four
//              groups per warp, with the per-lane partials and the xor reduction tree written out in the host
//              oracle, so only libm could differ.  The exact fixtures use small integers (every product and
//              partial sum is exact, so an FMA contraction cannot change a bit) and compare the whole row
//              bitwise against both the oracle and a hand-written expectation.
//   candidate  block score = in-block max, the block holding the newest position pinned +inf, topk_blocks
//              kept by a radix select; ties take the smaller block and the compact list is ascending.  The
//              hand fixtures cover the pin, the equal-key fill, the fewer-visible-than-kk fallback and
//              kk >= nb.
//   topk       min(index_topk, visible) keys in position order, ties take the smaller index, unreachable
//              slots -1; the compact candidate mapping and the graph position convention (ng and topk
//              recomputed from pos[0]) are checked too.
//
// The random fixtures compare the device kernels to the host oracle over float keys; selection is integer
// work over bit-cast keys, so those comparisons are bitwise too.  No model, no timing.  No GPU: SKIP, exits 77.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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

uint32_t g_rng = 0x9e3779b9u;
uint32_t urng() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
float urand() { return static_cast<float>(urng() & 0x00FFFFFFu) / 8388608.0f - 1.0f; }

float bf16r(float x) { return strata::deepseek4::engram_bf16r(x); }

void check_same_f32(const char* what, const std::vector<float>& got, const std::vector<float>& want) {
    CHECK(got.size() == want.size());
    size_t bad = 0;
    for (size_t i = 0; i < got.size() && i < want.size(); i++) {
        uint32_t a = 0, b = 0;
        std::memcpy(&a, &got[i], 4);
        std::memcpy(&b, &want[i], 4);
        if (a != b) {
            if (bad < 4)
                std::fprintf(stderr, "  %s value %zu: gpu %.9g (0x%08x) want %.9g (0x%08x)\n", what, i, got[i], a,
                             want[i], b);
            bad++;
        }
    }
    g_checks++;
    if (bad != 0) g_failures++;
    std::printf("  %-22s %zu values, %zu bits differ\n", what, got.size(), bad);
}

void check_same_i32(const char* what, const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
    CHECK(got.size() == want.size());
    size_t bad = 0;
    for (size_t i = 0; i < got.size() && i < want.size(); i++) {
        if (got[i] != want[i]) {
            if (bad < 6) std::fprintf(stderr, "  %s value %zu: gpu %d oracle %d\n", what, i, got[i], want[i]);
            bad++;
        }
    }
    g_checks++;
    if (bad != 0) g_failures++;
    std::printf("  %-22s %zu values, %zu differ\n", what, got.size(), bad);
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
void to_host_f(const float* d, std::vector<float>& v) {
    CHECK(cudaMemcpy(v.data(), d, v.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
}
int32_t* dev_i32_from(const std::vector<int32_t>& v) {
    int32_t* p = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&p), v.size() * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}
void to_host_i32(const int32_t* d, std::vector<int32_t>& v) {
    CHECK(cudaMemcpy(v.data(), d, v.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) == cudaSuccess);
}
uint8_t* dev_u8_from(const std::vector<uint8_t>& v) {
    uint8_t* p = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&p), v.size()) == cudaSuccess);
    CHECK(cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}
void to_host_u8(const uint8_t* d, std::vector<uint8_t>& v) {
    CHECK(cudaMemcpy(v.data(), d, v.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
}
void* dev_raw(size_t bytes) {
    void* p = nullptr;
    CHECK(cudaMalloc(&p, bytes) == cudaSuccess);
    return p;
}

// ---- indexer score ----

// Runs the exact fixtures: packs the keys on host and device (the packed bytes must agree), then compares the
// device score against the host oracle and, when given, the hand expectation.
void score_case(const char* name, uint32_t n_tok, uint32_t n_head, uint32_t dk, uint32_t ng, uint32_t ratio,
                uint32_t pos0, const std::vector<float>& q, const std::vector<float>& keys,
                const std::vector<float>& w, const std::vector<int32_t>* cand, uint32_t cand_bs, uint32_t cand_cap,
                int use_pos, int32_t posv, const std::vector<float>* expect) {
    const uint32_t n_key = ng;
    std::vector<uint8_t> hk((size_t) n_key * strata::kernels::kDeepseek4IdxkBytes, 0);
    strata::kernels::deepseek4_idxk_pack_host(hk.data(), 0, keys.data(), n_key);
    float* d_keys = dev_f_from(keys);
    uint8_t* d_kk = dev_u8_from(hk);
    CHECK(strata::kernels::deepseek4_idxk_pack(d_kk, 0, d_keys, n_key, nullptr));
    std::vector<uint8_t> packed_back(hk.size());
    to_host_u8(d_kk, packed_back);
    CHECK(std::memcmp(packed_back.data(), hk.data(), hk.size()) == 0);

    uint32_t ns = ng;
    if (cand) ns = strata::kernels::deepseek4_cand_ns(ng, cand_bs, cand_cap);
    if (use_pos) ns = ((uint32_t) posv + n_tok) / ratio;
    int32_t pos = posv;

    std::vector<float> want((size_t) n_tok * ns, 123.0f), got(want);
    strata::kernels::deepseek4_indexer_score_host(want.data(), q.data(), hk.data(), w.data(),
                                                  cand ? cand->data() : nullptr, pos0, ng, n_tok, n_head, dk, ratio,
                                                  cand_bs, cand_cap, use_pos ? &pos : nullptr);
    float* d_score = dev_f_from(got);
    float* d_q = dev_f_from(q);
    float* d_w = dev_f_from(w);
    int32_t* d_cand = cand ? dev_i32_from(*cand) : nullptr;
    int32_t* d_pos = nullptr;
    if (use_pos) {
        std::vector<int32_t> pv = {pos};
        d_pos = dev_i32_from(pv);
    }
    CHECK(strata::kernels::deepseek4_indexer_score(d_score, d_q, d_kk, d_w, d_cand, pos0, ng, n_tok, n_head, dk,
                                                   ratio, cand_bs, cand_cap, d_pos, nullptr));
    to_host_f(d_score, got);
    check_same_f32(name, got, want);
    if (expect) check_same_f32("  hand", got, *expect);

    cudaFree(d_keys);
    cudaFree(d_kk);
    cudaFree(d_score);
    cudaFree(d_q);
    cudaFree(d_w);
    if (d_cand) cudaFree(d_cand);
    if (d_pos) cudaFree(d_pos);
}

// The hand expectation: integer-valued inputs, so the dot is an exact integer and both sides round the same.
// The key matrix's row stride is the packed cache's 128 dims; only the first `dk` are dotted.
std::vector<float> score_hand(uint32_t n_tok, uint32_t n_head, uint32_t dk, uint32_t ng, uint32_t ratio,
                              uint32_t pos0, const std::vector<float>& q, const std::vector<float>& keys,
                              const std::vector<float>& w) {
    const uint32_t kr = strata::kernels::kDeepseek4IdxkDim;
    std::vector<float> hand((size_t) n_tok * ng, 123.0f);
    for (uint32_t i = 0; i < n_tok; i++) {
        const uint32_t vis = (pos0 + i + 1u) / ratio;
        for (uint32_t g = 0; g < ng; g++) {
            if (g >= vis) {
                hand[(size_t) i * ng + g] = -INFINITY;
                continue;
            }
            float acc = 0.0f;
            for (uint32_t h = 0; h < n_head; h++) {
                int64_t dot = 0;
                for (uint32_t d = 0; d < dk; d++)
                    dot += (int64_t)(int32_t) q[((size_t) i * n_head + h) * dk + d] *
                           (int32_t) keys[(size_t) g * kr + d];
                float dd = bf16r((float) dot);
                dd = fmaxf(dd, 0.0f);
                acc += bf16r(dd * w[(size_t) i * n_head + h]);
            }
            hand[(size_t) i * ng + g] = bf16r(acc);
        }
    }
    return hand;
}

// The compact row's hand expectation: dead items are -inf, live items score their mapped group.
std::vector<float> score_hand_cand(uint32_t n_tok, uint32_t n_head, uint32_t dk, uint32_t ng, uint32_t ratio,
                                   uint32_t pos0, uint32_t cand_bs, uint32_t cand_cap,
                                   const std::vector<int32_t>& cand, const std::vector<float>& q,
                                   const std::vector<float>& keys, const std::vector<float>& w) {
    const uint32_t ns = strata::kernels::deepseek4_cand_ns(ng, cand_bs, cand_cap);
    const uint32_t kr = strata::kernels::kDeepseek4IdxkDim;
    std::vector<float> hand((size_t) n_tok * ns, 123.0f);
    for (uint32_t i = 0; i < n_tok; i++) {
        const uint32_t vis = (pos0 + i + 1u) / ratio;
        const int32_t* cl = cand.data() + (size_t) i * (1u + cand_cap);
        for (uint32_t c = 0; c < ns; c++) {
            const uint32_t b = c / cand_bs;
            const uint32_t g = b < (uint32_t) cl[0] ? (uint32_t) cl[1u + b] * cand_bs + c % cand_bs : ng;
            if (g >= ng || g >= vis) {
                hand[(size_t) i * ns + c] = -INFINITY;
                continue;
            }
            float acc = 0.0f;
            for (uint32_t h = 0; h < n_head; h++) {
                int64_t dot = 0;
                for (uint32_t d = 0; d < dk; d++)
                    dot += (int64_t)(int32_t) q[((size_t) i * n_head + h) * dk + d] *
                           (int32_t) keys[(size_t) g * kr + d];
                float dd = bf16r((float) dot);
                dd = fmaxf(dd, 0.0f);
                acc += bf16r(dd * w[(size_t) i * n_head + h]);
            }
            hand[(size_t) i * ns + c] = bf16r(acc);
        }
    }
    return hand;
}

void score_cases() {
    {   // per = 1, two heads; row 0 sees one group, row 1 sees two; relu clips the negative dots
        const uint32_t n_tok = 2, n_head = 2, dk = 32, ng = 3, ratio = 1, pos0 = 0;
        const uint32_t kr = strata::kernels::kDeepseek4IdxkDim;
        std::vector<float> q((size_t) n_tok * n_head * dk), keys((size_t) ng * kr, 0.0f),
            w((size_t) n_tok * n_head);
        for (uint32_t i = 0; i < n_tok; i++)
            for (uint32_t h = 0; h < n_head; h++) {
                w[(size_t) i * n_head + h] = h == 0 ? 1.0f : 0.5f;
                for (uint32_t d = 0; d < dk; d++)
                    q[((size_t) i * n_head + h) * dk + d] = (float) ((int32_t) ((i + h + d) % 5u) - 2);
            }
        for (uint32_t g = 0; g < ng; g++)
            for (uint32_t d = 0; d < dk; d++) keys[(size_t) g * kr + d] = (float) ((g * 2u + d) % 5u);
        std::vector<float> hand = score_hand(n_tok, n_head, dk, ng, ratio, pos0, q, keys, w);
        score_case("score dk32", n_tok, n_head, dk, ng, ratio, pos0, q, keys, w, nullptr, 0, 0, false, 0, &hand);
    }
    {   // per = 4 (dk 128); row 0 sees one group, row 1 two
        const uint32_t n_tok = 2, n_head = 2, dk = 128, ng = 3, ratio = 2, pos0 = 2;
        const uint32_t kr = strata::kernels::kDeepseek4IdxkDim;
        std::vector<float> q((size_t) n_tok * n_head * dk), keys((size_t) ng * kr, 0.0f),
            w((size_t) n_tok * n_head);
        for (uint32_t i = 0; i < n_tok; i++)
            for (uint32_t h = 0; h < n_head; h++) {
                w[(size_t) i * n_head + h] = h == 0 ? 1.0f : 0.75f;
                for (uint32_t d = 0; d < dk; d++)
                    q[((size_t) i * n_head + h) * dk + d] = (float) ((int32_t) ((i + h + d) % 3u) - 1);
            }
        for (uint32_t g = 0; g < ng; g++)
            for (uint32_t d = 0; d < dk; d++) keys[(size_t) g * kr + d] = (float) ((g * 3u + d) % 5u);
        std::vector<float> hand = score_hand(n_tok, n_head, dk, ng, ratio, pos0, q, keys, w);
        score_case("score dk128", n_tok, n_head, dk, ng, ratio, pos0, q, keys, w, nullptr, 0, 0, false, 0, &hand);
    }
    {   // the compact C2 row: row 0's list is empty, row 1 lists block 2, row 2 lists blocks 0 and 3
        const uint32_t n_tok = 3, n_head = 1, dk = 32, ng = 8, ratio = 1, pos0 = 100, cand_bs = 2, cand_cap = 3;
        const uint32_t kr = strata::kernels::kDeepseek4IdxkDim;
        std::vector<int32_t> cand = {0, 0, 0, 0,  1, 2, 0, 0,  2, 0, 3, 0};
        std::vector<float> q((size_t) n_tok * n_head * dk), keys((size_t) ng * kr, 0.0f),
            w((size_t) n_tok * n_head, 1.0f);
        for (uint32_t i = 0; i < n_tok; i++)
            for (uint32_t d = 0; d < dk; d++) q[(size_t) i * dk + d] = (float) ((int32_t) ((i + d) % 5u) - 2);
        for (uint32_t g = 0; g < ng; g++)
            for (uint32_t d = 0; d < dk; d++) keys[(size_t) g * kr + d] = (float) ((g * 3u + d) % 5u);
        std::vector<float> hand = score_hand_cand(n_tok, n_head, dk, ng, ratio, pos0, cand_bs, cand_cap, cand, q,
                                                  keys, w);
        score_case("score compact", n_tok, n_head, dk, ng, ratio, pos0, q, keys, w, &cand, cand_bs, cand_cap, false,
                   0, &hand);
    }
    {   // the graph position recomputes ng from pos[0]: 4 groups become 3
        const uint32_t n_tok = 1, n_head = 1, dk = 32, ng = 4, ratio = 4, pos0 = 0;
        const uint32_t kr = strata::kernels::kDeepseek4IdxkDim;
        std::vector<float> q((size_t) dk), keys((size_t) ng * kr, 0.0f), w(1, 1.0f);
        for (uint32_t d = 0; d < dk; d++) q[d] = (float) ((int32_t) (d % 3u) - 1);
        for (uint32_t g = 0; g < ng; g++)
            for (uint32_t d = 0; d < dk; d++) keys[(size_t) g * kr + d] = (float) ((g + d) % 4u);
        score_case("score pos", n_tok, n_head, dk, ng, ratio, pos0, q, keys, w, nullptr, 0, 0, true, 11, nullptr);
    }
}

// ---- candidate blocks ----

void set_block(std::vector<float>& s, uint32_t ng, uint32_t row, uint32_t bs, uint32_t block, float v) {
    for (uint32_t j = block * bs; j < (block + 1u) * bs && j < ng; j++) s[(size_t) row * ng + j] = v;
}

void free_cand(float* d_score, int32_t* d_list, float* d_q, void* d_scr, int32_t* d_pos) {
    if (d_score) cudaFree(d_score);
    if (d_list) cudaFree(d_list);
    if (d_q) cudaFree(d_q);
    if (d_scr) cudaFree(d_scr);
    if (d_pos) cudaFree(d_pos);
}

void cand_case(const char* name, uint32_t n_tok, uint32_t pos0, uint32_t ng, uint32_t ratio, uint32_t topk_blocks,
               uint32_t bs, uint32_t nb_cap, const std::vector<float>& score, int use_pos, int32_t posv,
               const std::vector<int32_t>* expect) {
    const size_t row = 1u + topk_blocks;
    std::vector<int32_t> want((size_t) n_tok * row, 12345), got(want);
    int32_t pos = posv;
    strata::kernels::deepseek4_candidate_blocks_host(want.data(), score.data(), n_tok, pos0, ng, ratio,
                                                     topk_blocks, bs, use_pos ? &pos : nullptr);
    float* d_score = dev_f_from(score);
    int32_t* d_list = dev_i32_from(got);
    void* d_scr = dev_raw((size_t) n_tok * nb_cap * 5u);
    int32_t* d_pos = nullptr;
    if (use_pos) {
        std::vector<int32_t> pv = {pos};
        d_pos = dev_i32_from(pv);
    }
    CHECK(strata::kernels::deepseek4_candidate_blocks(d_list, d_score, static_cast<float*>(d_scr), n_tok, pos0,
                                                      ng, ratio, topk_blocks, bs, nb_cap, d_pos, nullptr));
    to_host_i32(d_list, got);
    check_same_i32(name, got, want);
    if (expect) check_same_i32("  hand", got, *expect);
    free_cand(d_score, d_list, nullptr, d_scr, d_pos);
}

void candidate_cases() {
    {   // the newest position's block is pinned +inf; the second slot goes to the top block among the rest
        const uint32_t n_tok = 2, ng = 16, ratio = 1, pos0 = 0, bs = 4, topk_blocks = 2;
        std::vector<float> s((size_t) n_tok * ng, -INFINITY);
        for (uint32_t i = 0; i < n_tok; i++) {
            set_block(s, ng, i, bs, 1, 2.0f);
            set_block(s, ng, i, bs, 2, 3.0f);
        }
        std::vector<int32_t> hand = {2, 0, 2, 2, 0, 2};
        cand_case("cand pin", n_tok, pos0, ng, ratio, topk_blocks, bs, 5, s, false, 0, &hand);
    }
    {   // all four blocks tie: the threshold fill takes the smaller blocks in ascending order
        const uint32_t n_tok = 1, ng = 16, ratio = 8, pos0 = 0, bs = 4, topk_blocks = 2;
        std::vector<float> s((size_t) n_tok * ng, 1.0f);
        std::vector<int32_t> hand = {2, 0, 1};
        cand_case("cand tie", n_tok, pos0, ng, ratio, topk_blocks, bs, 4, s, false, 0, &hand);
    }
    {   // fewer visible blocks than kk: the nonzero block plus the first zero-key blocks ascending
        const uint32_t n_tok = 1, ng = 16, ratio = 8, pos0 = 0, bs = 4, topk_blocks = 2;
        std::vector<float> s((size_t) n_tok * ng, -INFINITY);
        set_block(s, ng, 0, bs, 2, 1.5f);
        std::vector<int32_t> hand = {2, 0, 2};
        cand_case("cand short", n_tok, pos0, ng, ratio, topk_blocks, bs, 4, s, false, 0, &hand);
    }
    {   // kk >= nb: select every block that is not fully invisible
        const uint32_t n_tok = 1, ng = 8, ratio = 8, pos0 = 0, bs = 4, topk_blocks = 4;
        std::vector<float> s((size_t) n_tok * ng, -INFINITY);
        set_block(s, ng, 0, bs, 0, 1.0f);
        std::vector<int32_t> hand = {1, 0, 12345, 12345, 12345};
        cand_case("cand all", n_tok, pos0, ng, ratio, topk_blocks, bs, 4, s, false, 0, &hand);
    }
    {   // the graph position shrinks ng from 16 to 4
        const uint32_t n_tok = 1, ng = 16, ratio = 4, pos0 = 0, bs = 4, topk_blocks = 2;
        std::vector<float> s = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int32_t> hand = {1, 0, 12345};
        cand_case("cand pos", n_tok, pos0, ng, ratio, topk_blocks, bs, 4, s, true, 15, &hand);
    }
    {   // pos recomputes ng to 0: the kernel still writes an empty list header per row
        const uint32_t n_tok = 1, ng = 4, ratio = 8, pos0 = 0, bs = 4, topk_blocks = 2;
        std::vector<float> s = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int32_t> hand = {0, 12345, 12345};
        cand_case("cand pos empty", n_tok, pos0, ng, ratio, topk_blocks, bs, 4, s, true, 0, &hand);
    }
    {   // random blocks, pinned and radix-selected; bitwise against the oracle
        const uint32_t n_tok = 3, ng = 50, ratio = 3, pos0 = 7, bs = 4, topk_blocks = 5;
        std::vector<float> s((size_t) n_tok * ng);
        for (size_t i = 0; i < s.size(); i++) s[i] = (i % 11u == 0u) ? -INFINITY : 2.0f * urand();
        cand_case("cand random", n_tok, pos0, ng, ratio, topk_blocks, bs, 15, s, false, 0, nullptr);
    }
}

// ---- indexer topk ----

void topk_case(const char* name, uint32_t n_tok, uint32_t ng, uint32_t topk, uint32_t ratio,
               const std::vector<float>& score, const std::vector<int32_t>* cand, uint32_t cand_bs,
               uint32_t cand_cap, int use_pos, int32_t posv, const std::vector<int32_t>* expect) {
    uint32_t used_ng = ng, used_topk = topk;
    if (use_pos) {
        used_ng = ((uint32_t) posv + n_tok) / ratio;
        if (used_ng < used_topk) used_topk = used_ng;
    }
    std::vector<int32_t> want((size_t) n_tok * used_topk, 12345), got(want);
    int32_t pos = posv;
    strata::kernels::deepseek4_indexer_topk_host(want.data(), score.data(), n_tok, ng, topk, ratio,
                                                 cand ? cand->data() : nullptr, cand_bs, cand_cap,
                                                 use_pos ? &pos : nullptr);
    float* d_score = dev_f_from(score);
    int32_t* d_idx = dev_i32_from(got);
    int32_t* d_cand = cand ? dev_i32_from(*cand) : nullptr;
    int32_t* d_pos = nullptr;
    if (use_pos) {
        std::vector<int32_t> pv = {pos};
        d_pos = dev_i32_from(pv);
    }
    CHECK(strata::kernels::deepseek4_indexer_topk(d_idx, d_score, n_tok, ng, topk, ratio, d_cand, cand_bs,
                                                  cand_cap, d_pos, nullptr));
    to_host_i32(d_idx, got);
    check_same_i32(name, got, want);
    if (expect) check_same_i32("  hand", got, *expect);
    cudaFree(d_score);
    cudaFree(d_idx);
    if (d_cand) cudaFree(d_cand);
    if (d_pos) cudaFree(d_pos);
}

void topk_cases() {
    {
        std::vector<float> s = {0.5f, -INFINITY, 3.0f, 1.0f, 2.0f, -1.0f, 5.0f, -INFINITY, 4.0f, 2.0f};
        std::vector<int32_t> hand = {2, 4, 6, 8};
        topk_case("topk basic", 1, 10, 4, 1, s, nullptr, 0, 0, false, 0, &hand);
    }
    {
        std::vector<float> s = {2.0f, 2.0f, 2.0f, 2.0f, -INFINITY};
        std::vector<int32_t> hand = {0, 1, 2};
        topk_case("topk ties", 1, 5, 3, 1, s, nullptr, 0, 0, false, 0, &hand);
    }
    {
        std::vector<float> s = {3.0f, 2.0f, 2.0f, 1.0f};
        std::vector<int32_t> hand = {0, 1};
        topk_case("topk tie edge", 1, 4, 2, 1, s, nullptr, 0, 0, false, 0, &hand);
    }
    {   // topk larger than the visible count: the tail is -1
        std::vector<float> s = {1.0f, -INFINITY, 2.0f, -INFINITY, 3.0f, -INFINITY};
        std::vector<int32_t> hand = {0, 2, 4, -1, -1, -1, -1, -1};
        topk_case("topk tail", 1, 6, 8, 1, s, nullptr, 0, 0, false, 0, &hand);
    }
    {
        std::vector<float> s = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
        std::vector<int32_t> hand = {-1, -1, -1};
        topk_case("topk empty", 1, 5, 3, 1, s, nullptr, 0, 0, false, 0, &hand);
    }
    {   // NaN is not visible (the key maps to 0, the same as -inf)
        std::vector<float> s = {NAN, 1.0f, NAN, 0.5f};
        std::vector<int32_t> hand = {1, 3, -1};
        topk_case("topk nan", 1, 4, 3, 1, s, nullptr, 0, 0, false, 0, &hand);
    }
    {   // the compact candidate row maps back through the block list
        const uint32_t ng = 8, cand_bs = 2, cand_cap = 4, topk = 3;
        std::vector<int32_t> cand = {3, 1, 2, 3, 0};
        std::vector<float> s = {0.1f, 0.2f, 0.3f, -INFINITY, 0.5f, -INFINITY, -INFINITY, -INFINITY};
        std::vector<int32_t> hand = {3, 4, 6};
        topk_case("topk cand", 1, ng, topk, 1, s, &cand, cand_bs, cand_cap, false, 0, &hand);
    }
    {   // cap * bs < ng: the compact row is 12 items and the map covers the listed blocks only
        const uint32_t ng = 20, cand_bs = 4, cand_cap = 3, topk = 2;
        std::vector<int32_t> cand = {2, 1, 3, 0};
        std::vector<float> s(12);
        for (uint32_t c = 0; c < 12; c++) s[c] = -(float) c;
        std::vector<int32_t> hand = {4, 5};
        topk_case("topk cand short", 1, ng, topk, 1, s, &cand, cand_bs, cand_cap, false, 0, &hand);
    }
    {   // the graph position recomputes ng (16 -> 4) and the stride is the recomputed topk
        std::vector<float> s = {-INFINITY, 5.0f, 2.0f, -INFINITY};
        std::vector<int32_t> hand = {1, 2, -1, -1};
        topk_case("topk pos", 1, 16, 4, 4, s, nullptr, 0, 0, true, 15, &hand);
    }
    {
        std::vector<float> s = {-INFINITY, -INFINITY, NAN, -INFINITY};
        std::vector<int32_t> hand = {-1, -1, -1, -1};
        topk_case("topk pos empty", 1, 16, 4, 4, s, nullptr, 0, 0, true, 15, &hand);
    }
    {
        const uint32_t n_tok = 3, ng = 40, topk = 7;
        std::vector<float> s((size_t) n_tok * ng);
        for (size_t i = 0; i < s.size(); i++) s[i] = (i % 13u == 0u) ? -INFINITY : 3.0f * urand();
        topk_case("topk random", n_tok, ng, topk, 1, s, nullptr, 0, 0, false, 0, nullptr);
    }
}

// ---- guards and no-ops ----

void guards() {
    const uint32_t dk = 32;
    const uint32_t kr = strata::kernels::kDeepseek4IdxkDim;
    std::vector<float> q(kr, 1.0f), keys(kr, 1.0f), w(1, 1.0f);
    std::vector<uint8_t> hk(strata::kernels::kDeepseek4IdxkBytes, 0);
    strata::kernels::deepseek4_idxk_pack_host(hk.data(), 0, keys.data(), 1);
    float* dq = dev_f_from(q);
    float* dw = dev_f_from(w);
    uint8_t* dk_dev = dev_u8_from(hk);
    float* dscore = dev_f(8);
    int32_t* dlist = dev_i32_from(std::vector<int32_t>(8, 12345));
    int32_t* didx = dev_i32_from(std::vector<int32_t>(8, 12345));
    void* dscr = dev_raw((size_t) 8 * 8 * 5u);
    std::vector<int32_t> pos9(1, 1);
    int32_t* dpos = dev_i32_from(pos9);

    CHECK(!strata::kernels::deepseek4_indexer_score(nullptr, dq, dk_dev, dw, nullptr, 0, 1, 1, 1, dk, 1, 0, 0,
                                                    nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_indexer_score(dscore, nullptr, dk_dev, dw, nullptr, 0, 1, 1, 1, dk, 1, 0, 0,
                                                    nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_indexer_score(dscore, dq, nullptr, dw, nullptr, 0, 1, 1, 1, dk, 1, 0, 0,
                                                    nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_indexer_score(dscore, dq, dk_dev, nullptr, nullptr, 0, 1, 1, 1, dk, 1, 0, 0,
                                                    nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_indexer_score(dscore, dq, dk_dev, dw, nullptr, 0, 1, 1, 1, 160, 1, 0, 0,
                                                    nullptr, nullptr));   // 5 dims per lane
    CHECK(!strata::kernels::deepseek4_indexer_score(dscore, dq, dk_dev, dw, nullptr, 0, 1, 1, 1, 100, 1, 0, 0,
                                                    nullptr, nullptr));   // not a multiple of 32
    CHECK(!strata::kernels::deepseek4_indexer_score(dscore, dq, dk_dev, dw, nullptr, 0, 1, 1, 1, dk, 0, 0, 0,
                                                    nullptr, nullptr));   // ratio 0
    CHECK(!strata::kernels::deepseek4_indexer_score(dscore, dq, dk_dev, dw, nullptr, 0, 1, 9, 1, dk, 1, 0, 0, dpos,
                                                    nullptr));            // graph route n > 8
    CHECK(!strata::kernels::deepseek4_indexer_score(dscore, dq, dk_dev, dw, dlist, 0, 1, 1, 1, dk, 1, 0, 0, nullptr,
                                                    nullptr));            // list without bs/cap
    CHECK(strata::kernels::deepseek4_indexer_score(dscore, dq, dk_dev, dw, dlist, 0, 0, 1, 1, dk, 1, 0, 0, nullptr,
                                                   nullptr));             // ng == 0 before the list check
    CHECK(strata::kernels::deepseek4_indexer_score(dscore, dq, dk_dev, dw, nullptr, 0, 1, 1, 1, 96, 1, 0, 0, nullptr,
                                                   nullptr));             // per = 3 is supported

    CHECK(!strata::kernels::deepseek4_candidate_blocks(nullptr, dscore, static_cast<float*>(dscr), 1, 0, 8, 1, 2, 4,
                                                       8, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_candidate_blocks(dlist, nullptr, static_cast<float*>(dscr), 1, 0, 8, 1, 2, 4,
                                                       8, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_candidate_blocks(dlist, dscore, nullptr, 1, 0, 8, 1, 2, 4, 8, nullptr,
                                                       nullptr));
    CHECK(!strata::kernels::deepseek4_candidate_blocks(dlist, dscore, static_cast<float*>(dscr), 1, 0, 8, 1, 2, 0,
                                                       8, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_candidate_blocks(dlist, dscore, static_cast<float*>(dscr), 1, 0, 8, 0, 2, 4,
                                                       8, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_candidate_blocks(dlist, dscore, static_cast<float*>(dscr), 1, 0, 8, 1, 0, 4,
                                                       8, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_candidate_blocks(dlist, dscore, static_cast<float*>(dscr), 9, 0, 8, 1, 2, 4,
                                                       8, dpos, nullptr));   // graph route n > 8
    CHECK(!strata::kernels::deepseek4_candidate_blocks(dlist, dscore, static_cast<float*>(dscr), 1, 0, 8, 1, 2, 4,
                                                       1, nullptr, nullptr)); // scratch rows too small
    CHECK(strata::kernels::deepseek4_candidate_blocks(dlist, dscore, static_cast<float*>(dscr), 1, 0, 0, 1, 2, 4, 8,
                                                      nullptr, nullptr));   // ng == 0 succeeds

    CHECK(!strata::kernels::deepseek4_indexer_topk(nullptr, dscore, 1, 8, 4, 1, nullptr, 0, 0, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_indexer_topk(didx, nullptr, 1, 8, 4, 1, nullptr, 0, 0, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_indexer_topk(didx, dscore, 1, 8, 0, 1, nullptr, 0, 0, nullptr, nullptr));
    CHECK(!strata::kernels::deepseek4_indexer_topk(didx, dscore, 9, 8, 4, 1, nullptr, 0, 0, dpos, nullptr));
    CHECK(!strata::kernels::deepseek4_indexer_topk(didx, dscore, 1, 8, 4, 0, nullptr, 0, 0, dpos, nullptr));
    CHECK(!strata::kernels::deepseek4_indexer_topk(didx, dscore, 1, 8, 4, 1, dlist, 0, 0, nullptr, nullptr));
    CHECK(strata::kernels::deepseek4_indexer_topk(didx, dscore, 1, 0, 4, 1, nullptr, 0, 0, nullptr, nullptr));

    // ng == 0 succeeds and leaves the output alone, device and host alike.
    std::vector<int32_t> before(8, 12345), after(before);
    CHECK(strata::kernels::deepseek4_indexer_topk(didx, dscore, 1, 0, 4, 1, nullptr, 0, 0, nullptr, nullptr));
    to_host_i32(didx, after);
    CHECK(std::memcmp(after.data(), before.data(), before.size() * sizeof(int32_t)) == 0);

    std::vector<int32_t> hbefore(8, 12345), hafter(hbefore);
    strata::kernels::deepseek4_indexer_score_host(nullptr, q.data(), hk.data(), w.data(), nullptr, 0, 1, 1, 1, dk,
                                                  1, 0, 0, nullptr);
    strata::kernels::deepseek4_indexer_score_host(reinterpret_cast<float*>(hbefore.data()), q.data(), hk.data(),
                                                  w.data(), nullptr, 0, 1, 1, 1, 160, 1, 0, 0, nullptr);
    CHECK(std::memcmp(hafter.data(), hbefore.data(), hbefore.size() * sizeof(int32_t)) == 0);
    strata::kernels::deepseek4_candidate_blocks_host(hafter.data(),
                                                     reinterpret_cast<const float*>(hbefore.data()), 1, 0, 8, 1, 2,
                                                     0, nullptr);
    CHECK(std::memcmp(hafter.data(), hbefore.data(), hbefore.size() * sizeof(int32_t)) == 0);
    strata::kernels::deepseek4_indexer_topk_host(hafter.data(),
                                                 reinterpret_cast<const float*>(hbefore.data()), 1, 8, 4, 1,
                                                 hbefore.data(), 0, 0, nullptr);
    CHECK(std::memcmp(hafter.data(), hbefore.data(), hbefore.size() * sizeof(int32_t)) == 0);

    cudaFree(dq);
    cudaFree(dw);
    cudaFree(dk_dev);
    cudaFree(dscore);
    cudaFree(dlist);
    cudaFree(didx);
    cudaFree(dscr);
    cudaFree(dpos);
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }

    score_cases();
    candidate_cases();
    topk_cases();
    guards();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
