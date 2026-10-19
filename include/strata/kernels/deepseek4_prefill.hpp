// include/strata/kernels/deepseek4_prefill.hpp - the routed VQ experts' batched prefill arm.
//
// The per-pair kernels in deepseek4_moe.hpp are the reference's correctness arms: one warp per (token, expert)
// pair, a packed-stream dot per weight row.  For a prompt block they cost the same per token as decode, which
// is the whole gap to the reference engine's prefill.  This module is the reference's performance arm
// (reference/ds4-dfm-rs/cuda/ds41_vq_prefill.cuh + ds41_vq_prefill_mma.cuh, MIT, rewritten here):
//
//     sort    host stable counting sort of the block's (token, pick) pairs by expert;  off[e] starts e's run,
//             perm[p] is the pair at sorted position p, inv[i] is that pair's position (-1 when skipped)
//     gather  xs[p] = bf16(x[token(p)])                          sorted activations, one row per pair
//     gate    g16[p] = bf16(W1[e] * xs[p] * gain)                bf16 tensor cores, 128 rows x 128 tokens
//     up      h16[p] = swiglu(g16[p], bf16(W3[e] * xs[p] * gain), clamp)
//     down    ys[p]  = bf16(W2[e] * h16[p] * gain)
//     reduce  out[t] = sum_k w[t][k] * ys[inv[t*topk+k]]         ascending k, the reference's order
//
// WHY THE NUMBERS MATCH THE PER-PAIR ARM.  The v3 codebook is E4M3, whose 3 mantissa bits fit bf16 exactly, so
// E4M3 -> bf16 is a lossless conversion both arms already perform.  Activations and the mid vector are already
// bf16 on this path.  Every product is then exact in f32 and the tensor core accumulates in f32.  What is left
// is the K accumulation ORDER: the per-pair arm sums 32 lanes of 8-value codewords and an xor tree; this arm
// sums k16 groups.  The reference measured the difference at 0.5% of outputs one bf16 ulp off and relative L2
// 1.7e-4, with the same exit rounding point (sum -> row gain -> bf16r).  Each k16 starts from zero and is added
// back with a plain FADD: accumulating a long K inside the tensor core truncates on alignment and measured
// +0.68% PPL on the engine (2026-09-24).
//
// WHAT THIS ARM DOES NOT TAKE.  v2 payloads (their per-payload f16 codebook would lose mantissa bits to bf16),
// column counts that are not a multiple of 64, and codebooks that are not 12 or 13 bits.  The entry returns
// false before launching anything for those, and the caller keeps the per-pair kernels.
//
// THE SCRATCH IS THE CALLER'S.  One process drives several devices here (the P7 layer split), so nothing in
// this module is a static: the caller allocates one Deepseek4MoePrefillScratch per stage with the sizes
// deepseek4_moe_prefill_sizes reports and owns its lifetime.  No allocation happens in the entry.
#pragma once

#include <cstdint>

namespace strata::kernels {

// One work item: up to 128 tokens of one expert, starting at that expert's t0-th sorted pair.  A field named
// in the driver's copy of this struct, so the layout is fixed.
struct Deepseek4MoePrefillItem {
    int32_t e = 0;
    int32_t t0 = 0;
    int32_t nt = 0;
    int32_t pad = 0;
};

// Device buffers, filled by the caller from deepseek4_moe_prefill_sizes.  pairs_cap bounds n_tok * topk and
// experts_cap bounds n_expert (the plan arrays); mid_cap bounds mid_dim (g16/h16 rows); cb_cap is in bf16
// elements (nc * 8).
struct Deepseek4MoePrefillScratch {
    uint16_t* xs = nullptr;
    uint16_t* g16 = nullptr;
    uint16_t* h16 = nullptr;
    uint16_t* cb16 = nullptr;
    int32_t* perm = nullptr;
    int32_t* inv = nullptr;
    uint32_t* off = nullptr;
    Deepseek4MoePrefillItem* items = nullptr;
    uint32_t pairs_cap = 0;
    uint32_t experts_cap = 0;
    uint32_t mid_cap = 0;
    uint32_t cb_cap = 0;
};

struct Deepseek4MoePrefillSizes {
    uint64_t xs = 0;
    uint64_t g16 = 0;
    uint64_t h16 = 0;
    uint64_t cb16 = 0;
    uint64_t perm = 0;
    uint64_t inv = 0;
    uint64_t off = 0;
    uint64_t items = 0;
};

// Bytes per scratch buffer for a block of pairs_cap pairs over n_expert experts, in_dim -> mid_dim, and a
// codebook of nc words.  False when a dimension has no compiled arm (the caller then leaves the arm off).
bool deepseek4_moe_prefill_sizes(uint32_t pairs_cap, uint32_t n_expert, uint32_t in_dim, uint32_t mid_dim,
                                 uint32_t nc, Deepseek4MoePrefillSizes* out);

// out[t][o] = sum_k w[t][k] * ys[inv[t*topk + k]][o], where ys holds the sorted down rows: the per-pair
// reduce's arithmetic and k order, indexed through the sort.  sel_host is the block's selection read back
// from the device (n_tok * topk ints); a negative entry is skipped exactly as the per-pair arm's zero partial
// is skipped by its own reduce.
//
// d_ys is caller scratch of at least n_tok * topk * out_dim floats (the driver's partial buffer is sized for
// it).  The entry launches on `stream` and does not synchronize.  Returns false without launching when the
// payload version or the geometry has no arm; a false return after a launch (an allocation or launch error)
// leaves d_out unspecified and the caller must recompute the layer with the per-pair entry.
bool deepseek4_routed_moe_prefill(float* d_out, float* d_ys, const void* d_blob, uint64_t blob_size,
                                  uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim, const int32_t* sel_host,
                                  const float* d_w, uint32_t topk, float clamp, const float* d_x, uint32_t n_tok,
                                  uint32_t n_expert, uint32_t nc, uint32_t blob_v3,
                                  Deepseek4MoePrefillScratch* scratch, void* stream);

// ---- The split entries, for a layer whose pair space is shared with the host (the --cpu-moe arm). ----------
//
// The same plan and three stages as the entry above, but instead of a reduce the function returns once the down
// rows are in d_ys.  A selection entry that is negative (the driver marked the pair --cpu-moe) still receives a
// slot in d_ys: the sorted GPU rows come first, then one slot per negative pair in pair order, and scratch->inv
// names the slot.  No item is built for a host slot, so it stays untouched until the caller fills it with
// deepseek4_routed_moe_prefill_fill_cpu; the reduce below then sums every pair, sorted and host-filled alike,
// which makes the result the same expression the per-pair arm's tail computes.
//
// The caller can start its host workers before this entry and wait for them after it: the entry does not sync,
// so the host's share overlaps the three stage launches.  It returns false before launching for the same
// geometry/version reasons as the full entry.
bool deepseek4_routed_moe_prefill_partial(float* d_ys, const void* d_blob, uint64_t blob_size, uint32_t in_dim,
                                          uint32_t mid_dim, uint32_t out_dim, const int32_t* sel_host, uint32_t topk,
                                          float clamp, const float* d_x, uint32_t n_tok, uint32_t n_expert,
                                          uint32_t nc, uint32_t blob_v3, Deepseek4MoePrefillScratch* scratch,
                                          void* stream);

// Fill the host's slots: ys[inv[pair_of[r]]] = the bf16 row r of `rows`, which is a device-visible pointer to
// pinned host memory (cudaHostAllocMapped).  n_rows host pairs were computed, one row each, and pair_of is their
// original (t*topk+k) indices.  Launches on `stream`, no sync; nothing is written for r with no slot.
bool deepseek4_routed_moe_prefill_fill_cpu(float* d_ys, const int32_t* d_inv, const int32_t* d_pair_of,
                                           const uint16_t* d_rows, uint32_t n_rows, uint32_t out_dim,
                                           void* stream);

// The reduce the full entry ends with: out[t] = sum_k w[t][k] * ys[inv[t*topk+k]].  A negative inv entry is
// skipped, so a plan without host slots behaves exactly as before; the split path's plan gives every pair a
// slot.  Launches on `stream`, no sync.
bool deepseek4_routed_moe_prefill_reduce(float* d_out, const float* d_ys, const int32_t* d_inv, const float* d_w,
                                         uint32_t topk, uint32_t n_tok, uint32_t out_dim, void* stream);

}  // namespace strata::kernels
