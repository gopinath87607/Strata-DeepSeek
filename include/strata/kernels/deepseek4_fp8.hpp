// include/strata/kernels/deepseek4_fp8.hpp - P5: the DSpark towers' fp8 skeleton matmuls and the draft's
// small helpers (main-hidden capture, ring gather, markov gather, argmax).
//
// The MTP towers carry their dense matrices as fp8_32x32 (type 44): E4M3 values tiled 32x32 with one UE8M0
// scale per tile, the same format the engram's wkv uses (deepseek4_engram.hpp), so the GEMV is that module's
// kernel and the notes there cover the layout and the activation knife.  Two entries live here because the
// trunk needs them and the engram does not:
//
//   matmul_fp8          y[t][o] = sum_i W[o][i] * x[t][i], W [out_dim][in_dim] E4M3 + tile scales, the
//                       activations read on the bf16 grid for n_tok > 1 (the engine's knife) and the output
//                       rounded to bf16 when round_out (every tower projection but the head).
//   matmul_fp8_grouped  the tower's attn_output_a, the reference's v41_tproj_grouped: the weight is n_groups
//                       back-to-back [rank][group_dim] matrices (the same interpretation the q4_K grouped entry
//                       gives its tensor), out[t][g][o] = sum_c W[g][o][c] * x[t][g][c].
//
// The remaining four are the draft round's plumbing, each the reference kernel from cuda/ds41_draft.cuh:
//
//   hc_mean     the main-hidden capture: mean over the HC routes of a target layer's input, written into the
//               ring cell pos % cap at slot `slot` (the concatenated per-target-layer main hidden).
//   ring_rows   the contiguous run [first, first+rows) out of that ring, in order, for main_proj's batch.
//   row_gather  one row of the markov embedding table, selected by a device token id.
//   argmax      the greedy pick over one logits row (strict >, ties to the lower id).
//
// A null `stream` launches on the default stream and synchronizes, the same convention as the other P3 entry
// points.  fp8 weights and the bf16 activation copy must be 16-byte aligned (cudaMalloc).  Like the rest of the
// DeepSeek module these are CUDA-only; P8 owns the HIP/Metal twins.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// y = W x for an fp8_32x32 skeleton matrix.  `w` holds in_dim*out_dim E4M3 bytes, then
/// ceil(out_dim/32)*ceil(in_dim/32) E8M0 scales (tile (r/32, c/32)); `x` is [n_tok][in_dim] f32, `y` is
/// [n_tok][out_dim] f32.  `x_bf16` is the [n_tok][in_dim] copy `deepseek4_engram_x_to_bf16` wrote, required
/// when n_tok > 1.  Returns false, launching nothing, when a pointer is null (x_bf16 only when n_tok > 1),
/// n_tok is 0, out_dim is 0, or in_dim is not a positive multiple of 32.
bool deepseek4_matmul_fp8(float* d_out, const uint8_t* d_w, const float* d_x, const uint16_t* d_x_bf16,
                          uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, bool round_out, void* stream);

/// Grouped fp8: out[t][g*rank + o] = sum_c w_g[o][c] * x[t][g*group_dim + c].  The weight is n_groups row-major
/// [rank][group_dim] E4M3 matrices stored back to back, the scale plane tiled over the whole tensor with
/// sbc = group_dim/32 row-block columns (rank a multiple of 32 keeps each group's first row block aligned).
/// `x_bf16` has the same contract as above.  Returns false on the same guards plus rank % 32 != 0 or
/// group_dim % 32 != 0.
bool deepseek4_matmul_fp8_grouped(float* d_out, const uint8_t* d_w, const float* d_x, const uint16_t* d_x_bf16,
                                  uint32_t n_groups, uint32_t group_dim, uint32_t rank, uint32_t n_tok,
                                  bool round_out, void* stream);

/// The main-hidden ring write (cuda/ds41_draft.cuh:167-178, the eager arm): for rows [src_row0, src_row0+n_rows)
/// of `hc` [n_tok][n_hc][n_embd], out[(dst_pos0 + t) % cap][slot][d] = sum_c hc[t][c][d] / n_hc.  The slot
/// stride is n_slot * n_embd, so `out` is [cap][n_slot][n_embd] f32.  Returns false when n_rows > cap or a
/// pointer/shape is empty.
bool deepseek4_hc_mean(float* d_out, const float* d_hc, uint32_t n_embd, uint32_t n_hc, uint32_t n_rows,
                       uint32_t src_row0, uint32_t slot, uint32_t n_slot, uint32_t dst_pos0, uint32_t cap,
                       void* stream);

/// out[t][e] = ring[(first + t) % cap][e] for t in [0, rows), e in [0, row_elems).  `ring` is [cap][row_elems].
/// Returns false when rows == 0, row_elems == 0, cap == 0, or a pointer is null.
bool deepseek4_ring_rows(float* d_out, const float* d_ring, uint32_t row_elems, uint32_t cap, uint32_t first,
                         uint32_t rows, void* stream);

/// out[dst_index * row_elems + e] = table[ids[src_index]][e], f32 output, bf16 table (the markov embedding).
/// An id outside [0, n_rows) writes a zero row (a valid model never produces one; the same determinism choice
/// as the embedding).  Returns false on a null pointer or an empty shape.
bool deepseek4_row_gather_bf16(float* d_out, const uint16_t* d_table, uint32_t n_rows, uint32_t row_elems,
                               const int32_t* d_ids, uint32_t src_index, uint32_t dst_index, void* stream);

/// The greedy argmax over n f32 values: strict >, so ties take the lower index (the driver's argmax_logits
/// rule).  `d_out` receives one int32.  Returns false on a null pointer or n == 0.
bool deepseek4_argmax(int32_t* d_out, const float* d_x, uint32_t n, void* stream);

/// out[i] += add[i], the reference's ds4_gpu_v41_row_add_tensor on one logits row: the markov head's bias.
/// `d_out_row` points at the row itself (the caller offsets by row * n).  Returns false on a null pointer or n == 0.
bool deepseek4_row_add(float* d_out_row, const float* d_add, uint32_t n, void* stream);

}  // namespace strata::kernels
