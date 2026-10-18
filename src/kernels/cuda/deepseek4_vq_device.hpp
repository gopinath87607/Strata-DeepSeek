// src/kernels/cuda/deepseek4_vq_device.hpp - the VQ row dot in its one-warp form.
//
// Not a public header: it needs the warp shuffle, so it is compiled into the CUDA/HIP translation units only.
// It exists so the P2 GEMV and the P3 MoE kernels invoke the SAME arithmetic - the reference's rule is that
// nobody keeps a second copy of the lane map, the xor tree or the multiply-add expression (cuda/ds41_vq_row.cuh).
//
// The pieces themselves are in include/strata/kernels/deepseek4_vq.hpp as __host__ __device__ functions, so
// the MoE host oracle runs the identical per-lane accumulation and reduction.
#pragma once

#include "strata/kernels/deepseek4_vq.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace strata::kernels {

// One warp, one row.  Lane j takes groups i = j, j + 32, ... - the reference's lane<->index map - so each
// lane's per-group accumulation and the xor tree below are its single-token path.  `x` is cols/2 uint32.
template <int NBIT, int V3, int EXT>
__device__ __forceinline__ float deepseek4_vq_row_dot(const Deepseek4VqMatrix& m, uint32_t r, const uint32_t* x) {
    float acc = deepseek4_vq_lane_partial<NBIT, V3, EXT>(m, r, threadIdx.x & 31u, x);
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return deepseek4_vq_apply_gain(acc, m, r);
}

}  // namespace strata::kernels

#endif
