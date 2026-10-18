// include/strata/kernels/deepseek4_dense_tc.hpp - the bf16 tensor-core tiles behind --tc-gemm (opt-in).
//
// The q4_K, grouped wo_a and bf16 matmuls of deepseek4_dense.hpp each gain a tensor-core variant here; the
// wrappers in deepseek4_dense.cu try them first when deepseek4_dense_set_tc(true) is in force and fall back
// to the SIMT tiles when a launch fails.  The tiles live in their own library (deepseek4_dense_tc.cu) because
// mma.sync.bf16 needs sm_80 or newer and the rest of the tree compiles at a lower floor; they run on the
// sm_86/sm_120 cards the port is measured on.  A failed launch is reported and returns false.
#pragma once

#include <cstdint>

namespace strata {
namespace kernels {

bool deepseek4_matmul_tc_bf16(float* d_out, const uint16_t* d_w, const float* d_x, uint32_t in_dim,
                              uint32_t out_dim, uint32_t n_tok, bool round_out, void* stream);
bool deepseek4_matmul_tc_q4k(float* d_out, const void* d_w, const float* d_x, uint32_t in_dim, uint32_t out_dim,
                             uint32_t n_tok, bool round_out, void* stream);
bool deepseek4_matmul_tc_q4k_grouped(float* d_out, const void* d_w, const float* d_x, uint32_t n_groups,
                                     uint32_t group_dim, uint32_t rank, uint32_t n_tok, bool round_out,
                                     void* stream);

}  // namespace kernels
}  // namespace strata
