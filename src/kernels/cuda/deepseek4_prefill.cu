// src/kernels/cuda/deepseek4_prefill.cu - the routed VQ experts' batched prefill arm (see the header).
//
// One block is 16 warps (512 threads): 8 warps own 16 weight rows each (128 rows), the other 8 the same rows
// over the token half they own (the reference's VQM_TW split).  A round decodes 64 K columns: every thread
// takes 2 of its row's 8 codewords out of the packed stream, converts them through the bf16 codebook into its
// A-tile slot, and 512 threads copy the round's 128x64 activation tile; then the warps run ldmatrix and 4
// mma.m16n8k16 per k16 group.  Blocks are persistent and grid-stride over the work items, so the codebook
// (shared, 12-bit layers) is filled once per launch and the SM stays on one tile shape.
//
// The bitstream decode reads two aligned 32-bit words per thread per round.  The 24 bits a thread needs always
// lie inside that 64-bit window, but on the LAST row of a payload the second word can reach 4 bytes past the
// stream's end: the caller's container pads its allocation by 8 bytes (the driver's stage_blob), and the parser
// is the only bounds gate.  Nothing is read outside that allowance.
#include "strata/kernels/deepseek4_prefill.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "strata/kernels/deepseek4_dense.hpp"
#include "strata/kernels/deepseek4_vq.hpp"

namespace strata::kernels {

namespace {

constexpr uint32_t kPfRows = 128u;          // weight rows a block owns
constexpr uint32_t kPfTok = 128u;           // tokens per work item (16 n8 slices)
constexpr uint32_t kPfK = 64u;              // K columns per round (8 codewords a row)
constexpr uint32_t kPfThreads = 512u;
constexpr uint32_t kPfCw = kPfK / 8u;       // 8 codewords per row-round
constexpr uint32_t kPfTpr = kPfThreads / kPfRows;   // 4 threads a row
constexpr uint32_t kPfCpt = kPfCw / kPfTpr;         // 2 codewords a thread
constexpr uint32_t kPfSlices = kPfTok / 8u;         // 16 n8 slices an item
constexpr uint32_t kPfSlicesPerWarp = kPfSlices / 2u;   // 8, one token half per warp group
constexpr uint32_t kPfABytes = kPfRows * kPfK * 2u;
constexpr uint32_t kPfBBytes = kPfTok * kPfK * 2u;
constexpr uint32_t kPfTileBytes = kPfABytes + kPfBBytes;

static_assert(kPfCpt * kPfTpr == kPfCw, "decode split");
static_assert(kPfSlicesPerWarp * 2u == kPfSlices, "token halves");

// ---- device pieces ------------------------------------------------------------------------------------

__device__ __forceinline__ void pf_ldsm4(uint32_t* r, const uint8_t* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void pf_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A tile row is 64 bf16 = 8 chunks of 16 B; chunk ^ (row & 7) puts the 8 rows one ldmatrix reads into 8
// different bank groups.
__device__ __forceinline__ uint32_t pf_swz(uint32_t row, uint32_t chunk) {
    return row * 128u + ((chunk ^ (row & 7u)) << 4);
}

// The bf16 codebook's 8 values for codeword v, 16 B.  CB=1: a shared copy (12-bit layers, 64 KB); CB=0: the
// global table (13-bit layers, 128 KB exceeds shared).
template <int CB>
__device__ __forceinline__ uint4 pf_cw(const uint16_t* cbs, const uint16_t* cb16, uint32_t v) {
    const uint16_t* p = CB ? cbs : cb16;
    return *reinterpret_cast<const uint4*>(p + static_cast<size_t>(v) * 8u);
}

__device__ __forceinline__ uint16_t pf_bf16_hi(float v) {
    return static_cast<uint16_t>(__float_as_uint(deepseek4_dense_bf16r(v)) >> 16);
}
__device__ __forceinline__ float pf_bf16_lo(uint16_t h) {
    return __uint_as_float(static_cast<uint32_t>(h) << 16);
}

// ---- the gather and the codebook ----------------------------------------------------------------------

__global__ void pf_gather16_kernel(uint16_t* __restrict__ xs, const float* __restrict__ x,
                                   const int32_t* __restrict__ perm, uint32_t topk, uint32_t in_dim) {
    const uint32_t i = blockIdx.x;
    const uint32_t t = static_cast<uint32_t>(perm[i]) / topk;
    const float* src = x + static_cast<uint64_t>(t) * in_dim;
    uint16_t* dst = xs + static_cast<uint64_t>(i) * in_dim;
    for (uint32_t d = threadIdx.x; d < in_dim; d += blockDim.x) dst[d] = pf_bf16_hi(src[d]);
}

// The layer's one v3 codebook, E4M3 -> bf16 (exact: 3 mantissa bits).  Thread 0 opens the first item's gate
// payload to find it - the same pointer every payload of the layer carries.
__global__ void pf_cb16_kernel(uint16_t* __restrict__ cb16, const uint8_t* __restrict__ blob, uint64_t blob_size,
                               uint32_t expert, uint32_t rows, uint32_t cols, uint32_t nc) {
    __shared__ const uint8_t* cbp;
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const Deepseek4VqMatrix m = deepseek4_vq_parse(blob, blob_size, expert, 0u, rows, cols);
        cbp = m.codebook;
        ok = (m.ok && m.v3) ? 1 : 0;
    }
    __syncthreads();
    if (!ok) return;
    const uint32_t n = nc * 8u;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        cb16[i] = pf_bf16_hi(deepseek4_vq_e4m3_to_f32(__ldg(cbp + i)));
    }
}

// ---- the tile kernel ----------------------------------------------------------------------------------

// EXT: the 13th index bit rides a plane.  MODE 0 = gate (g16), 1 = up (reads g16, clamp + SwiGLU -> h16),
// 2 = down (ys).  CB: the codebook lives in shared (1) or global (0).  Every exit is sum -> row gain -> bf16r,
// and each k16 starts from zero and is added back with a plain FADD (the header's note).
template <int EXT, int MODE, int CB>
__global__ __launch_bounds__(kPfThreads, 1) void pf_mma_kernel(
        uint16_t* __restrict__ g16, uint16_t* __restrict__ h16, float* __restrict__ ys,
        const uint8_t* __restrict__ blob, uint64_t blob_size, const Deepseek4MoePrefillItem* __restrict__ items,
        uint32_t nitems, const uint16_t* __restrict__ act, const uint32_t* __restrict__ off, uint32_t m_rows,
        uint32_t k_cols, float clamp, uint32_t nc, const uint16_t* __restrict__ cb16) {
    extern __shared__ __align__(16) uint8_t pfsh[];
    uint16_t* cbs = reinterpret_cast<uint16_t*>(pfsh);
    uint8_t* As = pfsh + (CB ? nc * 16u : 0u);
    uint8_t* Bs = As + kPfABytes;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t wr = warp & 7u, th = warp >> 3u;
    const uint32_t ar = tid / kPfTpr, sub = tid % kPfTpr;
    if (CB) {
        for (uint32_t v = tid; v < nc; v += kPfThreads)
            *reinterpret_cast<uint4*>(cbs + static_cast<size_t>(v) * 8u) =
                *reinterpret_cast<const uint4*>(cb16 + static_cast<size_t>(v) * 8u);
        __syncthreads();
    }
    const uint32_t ntile = (m_rows + kPfRows - 1u) / kPfRows;
    const uint32_t nwork = nitems * ntile;
    const uint32_t nit = k_cols / kPfK;
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const Deepseek4MoePrefillItem it = items[w / ntile];
        const uint32_t r0 = (w % ntile) * kPfRows;
        const uint32_t nt = static_cast<uint32_t>(it.nt);
        const uint32_t base = off[it.e] + static_cast<uint32_t>(it.t0);
        const Deepseek4VqMatrix m =
            deepseek4_vq_parse(blob, blob_size, static_cast<uint32_t>(it.e), static_cast<uint32_t>(MODE), m_rows,
                               k_cols);
        if (!m.ok) {   // the host walked the slot table; a bad payload zeroes its rows (the per-pair arm does too)
            if (MODE == 2)
                for (uint32_t i = tid; i < kPfRows * nt; i += kPfThreads)
                    if (r0 + i % kPfRows < m_rows)
                        ys[static_cast<uint64_t>(base + i / kPfRows) * m_rows + r0 + i % kPfRows] = 0.f;
            continue;
        }
        const uint32_t grow = r0 + ar;
        const uint32_t mrow = (m.nidx_row * 12u) >> 3;
        const uint32_t erow = (m.nidx_row + 7u) >> 3;
        const bool rv = grow < m_rows;
        const uint8_t* rowp = m.stream + static_cast<size_t>(rv ? grow : 0u) * mrow;
        const uint8_t* ep = EXT ? m.plane + static_cast<size_t>(rv ? grow : 0u) * erow : nullptr;
        const uint32_t bt_max = (nt + 7u) & ~7u;   // slices past this token are never read
        float acc[kPfSlicesPerWarp][4];
#pragma unroll
        for (uint32_t f = 0; f < kPfSlicesPerWarp; f++) acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f;
        uint32_t w0 = 0, w1 = 0, eb = 0;
        uint4 bx[(kPfTok * kPfCw) / kPfThreads];
        auto load_round = [&](uint32_t i) {
            // This thread's codewords sit at bit offset i*96 + sub*24: 24 bits inside a 64-bit window.
            const uint32_t bitoff = i * 12u * kPfCw + sub * 12u * kPfCpt;
            const uint32_t a = (bitoff >> 5) << 2;
            if (rv) {
                w0 = __ldg(reinterpret_cast<const uint32_t*>(rowp + a));
                w1 = __ldg(reinterpret_cast<const uint32_t*>(rowp + a + 4u));
                if (EXT) eb = __ldg(ep + i);   // the plane: 8 codewords a round = 1 byte
            }
#pragma unroll
            for (uint32_t c = 0; c < (kPfTok * kPfCw) / kPfThreads; c++) {
                const uint32_t ch = tid + c * kPfThreads, bt = ch / kPfCw, bc = ch % kPfCw;
                if (bt < bt_max)
                    bx[c] = __ldg(reinterpret_cast<const uint4*>(act + static_cast<uint64_t>(base + bt) * k_cols +
                                                               static_cast<uint64_t>(i) * kPfK + bc * 8u));
            }
        };
        load_round(0);
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t sh = (i * 12u * kPfCw + sub * 12u * kPfCpt) & 31u;
            const uint64_t u = ((static_cast<uint64_t>(w1) << 32) | w0) >> sh;
#pragma unroll
            for (uint32_t q = 0; q < kPfCpt; q++) {
                uint32_t v = static_cast<uint32_t>(u >> (12u * q)) & 0xFFFu;
                if (EXT) v |= ((eb >> (sub * kPfCpt + q)) & 1u) << 12;
                *reinterpret_cast<uint4*>(As + pf_swz(ar, sub * kPfCpt + q)) = pf_cw<CB>(cbs, cb16, v);
            }
#pragma unroll
            for (uint32_t c = 0; c < (kPfTok * kPfCw) / kPfThreads; c++) {
                const uint32_t ch = tid + c * kPfThreads;
                if (ch / kPfCw < bt_max) *reinterpret_cast<uint4*>(Bs + pf_swz(ch / kPfCw, ch % kPfCw)) = bx[c];
            }
            __syncthreads();
            if (i + 1u < nit) load_round(i + 1u);   // next round in flight during this round's math
#pragma unroll
            for (uint32_t kk = 0; kk < kPfK / 16u; kk++) {
                uint32_t a[4];
                {
                    const uint32_t mt = lane >> 3u, row = wr * 16u + (lane & 7u) + (mt & 1u) * 8u;
                    pf_ldsm4(a, As + pf_swz(row, kk * 2u + (mt >> 1)));
                }
#pragma unroll
                for (uint32_t np = 0; np < kPfSlicesPerWarp / 2u; np++) {
                    const uint32_t f0 = th * kPfSlicesPerWarp + 2u * np;   // this warp's 2np-th slice
                    if (f0 * 8u >= nt) break;
                    uint32_t b[4];
                    const uint32_t mt = lane >> 3u, tok = (f0 + (mt >> 1)) * 8u + (lane & 7u);
                    pf_ldsm4(b, Bs + pf_swz(tok, kk * 2u + (mt & 1u)));
                    float t0[4] = {0.f, 0.f, 0.f, 0.f}, t1[4] = {0.f, 0.f, 0.f, 0.f};
                    pf_mma(t0, a, b[0], b[1]);
#pragma unroll
                    for (int c = 0; c < 4; c++) acc[2u * np][c] += t0[c];
                    if ((f0 + 1u) * 8u < nt) {
                        pf_mma(t1, a, b[2], b[3]);
#pragma unroll
                        for (int c = 0; c < 4; c++) acc[2u * np + 1u][c] += t1[c];
                    }
                }
            }
            __syncthreads();
        }
        // exit: c0,c1 = row lane/4 at tokens (lane&3)*2+{0,1}; c2,c3 = row +8.
        const uint32_t rr[2] = {r0 + wr * 16u + (lane >> 2u), r0 + wr * 16u + (lane >> 2u) + 8u};
#pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t r = rr[h];
            if (r >= m_rows) continue;
            const float g = deepseek4_vq_f16_to_f32(deepseek4_vq_rd_u16(m.gains + static_cast<size_t>(r) * 2u));
#pragma unroll
            for (uint32_t f = 0; f < kPfSlicesPerWarp; f++) {
                const uint32_t gf = th * kPfSlicesPerWarp + f;
                if (gf * 8u >= nt) break;
#pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t t = gf * 8u + (lane & 3u) * 2u + e;
                    if (t >= nt) continue;
                    const uint64_t o = static_cast<uint64_t>(base + t) * m_rows + r;
                    const float val = deepseek4_dense_bf16r(acc[f][h * 2u + e] * g);
                    if (MODE == 0) {
                        g16[o] = pf_bf16_hi(val);
                    } else if (MODE == 1) {
                        float gv = pf_bf16_lo(g16[o]);
                        float uv = val;
                        if (clamp > 0.f) {
                            gv = fminf(gv, clamp);
                            uv = fminf(fmaxf(uv, -clamp), clamp);
                        }
                        h16[o] = pf_bf16_hi((gv / (1.0f + expf(-gv))) * uv);
                    } else {
                        ys[o] = val;
                    }
                }
            }
        }
    }
}

// The per-pair reduce's arithmetic and k order, indexed through the sort.
__global__ void pf_reduce_kernel(float* __restrict__ out, const float* __restrict__ ys,
                                 const int32_t* __restrict__ inv, const float* __restrict__ w, uint32_t topk,
                                 uint32_t out_dim) {
    const uint32_t t = blockIdx.y, o = blockIdx.x * 256u + threadIdx.x;
    if (o >= out_dim) return;
    float a = 0.f;
    for (uint32_t k = 0; k < topk; k++) {
        const int32_t i = inv[t * topk + k];
        if (i >= 0) a += w[t * topk + k] * ys[static_cast<uint64_t>(i) * out_dim + o];
    }
    out[static_cast<uint64_t>(t) * out_dim + o] = a;
}

template <int EXT, int MODE, int CB>
bool pf_launch(uint16_t* g16, uint16_t* h16, float* ys, const uint8_t* blob, uint64_t blob_size,
               const Deepseek4MoePrefillScratch* s, const uint16_t* act, uint32_t nitems, uint32_t m_rows,
               uint32_t k_cols, uint32_t nc, float clamp, cudaStream_t stream) {
    const uint32_t shb = (CB ? nc * 16u : 0u) + kPfTileBytes;
    cudaError_t ce = cudaFuncSetAttribute(pf_mma_kernel<EXT, MODE, CB>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shb));
    if (ce != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    int nsm = 0;
    if (cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || nsm <= 0) {
        (void)cudaGetLastError();
        return false;
    }
    int occ = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, pf_mma_kernel<EXT, MODE, CB>,
                                                      static_cast<int>(kPfThreads), static_cast<int>(shb)) !=
            cudaSuccess ||
        occ < 1) {
        (void)cudaGetLastError();
        occ = 1;
    }
    const uint32_t ntile = (m_rows + kPfRows - 1u) / kPfRows;
    const uint64_t nwork = static_cast<uint64_t>(nitems) * ntile;
    const uint32_t grid = static_cast<uint32_t>(std::min<uint64_t>(nwork, static_cast<uint64_t>(nsm) * occ));
    pf_mma_kernel<EXT, MODE, CB><<<grid, kPfThreads, shb, stream>>>(
        g16, h16, ys, blob, blob_size, s->items, nitems, act, s->off, m_rows, k_cols, clamp, nc, s->cb16);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool deepseek4_moe_prefill_sizes(uint32_t pairs_cap, uint32_t n_expert, uint32_t in_dim, uint32_t mid_dim,
                                 uint32_t nc, Deepseek4MoePrefillSizes* out) {
    if (out == nullptr || pairs_cap == 0 || n_expert == 0 || in_dim == 0 || mid_dim == 0) return false;
    if (in_dim % kPfK || mid_dim % kPfK || in_dim % 16u || mid_dim % 16u) return false;
    if (nc != 4096u && nc != 8192u) return false;
    Deepseek4MoePrefillSizes s;
    s.xs = static_cast<uint64_t>(pairs_cap) * in_dim * 2u;
    s.g16 = static_cast<uint64_t>(pairs_cap) * mid_dim * 2u;
    s.h16 = s.g16;
    s.cb16 = static_cast<uint64_t>(nc) * 8u * 2u;
    s.perm = static_cast<uint64_t>(pairs_cap) * 4u;
    s.inv = s.perm;
    s.off = (static_cast<uint64_t>(n_expert) + 1u) * 4u;
    s.items = static_cast<uint64_t>(pairs_cap) * sizeof(Deepseek4MoePrefillItem);
    *out = s;
    return true;
}

bool deepseek4_routed_moe_prefill(float* d_out, float* d_ys, const void* d_blob, uint64_t blob_size,
                                  uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim, const int32_t* sel_host,
                                  const float* d_w, uint32_t topk, float clamp, const float* d_x, uint32_t n_tok,
                                  uint32_t n_expert, uint32_t nc, uint32_t blob_v3,
                                  Deepseek4MoePrefillScratch* s, void* stream_void) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_void);
    if (s == nullptr || d_out == nullptr || d_ys == nullptr || d_blob == nullptr || sel_host == nullptr ||
        d_w == nullptr || d_x == nullptr || s->items == nullptr || s->perm == nullptr || s->inv == nullptr ||
        s->off == nullptr || s->xs == nullptr || s->g16 == nullptr || s->h16 == nullptr || s->cb16 == nullptr) {
        return false;
    }
    if (blob_v3 == 0) return false;   // v2's f16 codebook loses mantissa bits to bf16: the per-pair arm stays
    if (n_tok == 0 || topk == 0 || n_tok > 65535u) return false;
    if (nc != 4096u && nc != 8192u) return false;
    // Every weight matrix's K is one of in_dim / mid_dim, and the tile round is 64 columns.
    if (in_dim % kPfK || mid_dim % kPfK || in_dim % 16u || mid_dim % 16u) return false;
    const uint64_t npairs = static_cast<uint64_t>(n_tok) * topk;
    if (npairs > s->pairs_cap || n_expert > s->experts_cap || mid_dim > s->mid_cap || nc * 8u > s->cb_cap) {
        return false;
    }

    // ---- the plan: stable counting sort by expert, items cut at 128 tokens (all host, ~microseconds) ----
    std::vector<uint32_t> cnt(n_expert, 0u), off(static_cast<size_t>(n_expert) + 1u, 0u);
    uint32_t nvalid = 0;
    for (uint64_t i = 0; i < npairs; i++) {
        const int32_t e = sel_host[i];
        if (e >= 0 && static_cast<uint32_t>(e) < n_expert) {
            ++cnt[e];
            ++nvalid;
        }
    }
    for (uint32_t e = 0; e < n_expert; e++) off[e + 1u] = off[e] + cnt[e];
    std::vector<int32_t> perm(npairs, -1), inv(npairs, -1);
    {
        std::vector<uint32_t> cur(off.begin(), off.end() - 1);
        for (uint64_t i = 0; i < npairs; i++) {
            const int32_t e = sel_host[i];
            if (e < 0 || static_cast<uint32_t>(e) >= n_expert) continue;
            const uint32_t p = cur[e]++;
            perm[p] = static_cast<int32_t>(i);
            inv[i] = static_cast<int32_t>(p);
        }
    }
    std::vector<Deepseek4MoePrefillItem> items;
    items.reserve(static_cast<size_t>(nvalid / kPfTok + n_expert));
    for (uint32_t e = 0; e < n_expert; e++) {
        for (uint32_t t0 = 0; t0 < cnt[e]; t0 += kPfTok) {
            Deepseek4MoePrefillItem it;
            it.e = static_cast<int32_t>(e);
            it.t0 = static_cast<int32_t>(t0);
            it.nt = static_cast<int32_t>(std::min<uint32_t>(kPfTok, cnt[e] - t0));
            items.push_back(it);
        }
    }
    const uint32_t nitems = static_cast<uint32_t>(items.size());

    if (cudaMemcpyAsync(s->perm, perm.data(), npairs * 4u, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(s->inv, inv.data(), npairs * 4u, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(s->off, off.data(), (static_cast<size_t>(n_expert) + 1u) * 4u, cudaMemcpyHostToDevice,
                        stream) != cudaSuccess ||
        (nitems != 0 &&
         cudaMemcpyAsync(s->items, items.data(), static_cast<size_t>(nitems) * sizeof(Deepseek4MoePrefillItem),
                         cudaMemcpyHostToDevice, stream) != cudaSuccess)) {
        (void)cudaGetLastError();
        return false;
    }

    if (nvalid != 0) {
        // The codebook goes through the first item's gate payload (the same one every payload of the layer
        // names); 12-bit layers then copy it into shared per block, 13-bit read the global table.
        pf_cb16_kernel<<<1, 256, 0, stream>>>(s->cb16, static_cast<const uint8_t*>(d_blob), blob_size,
                                              static_cast<uint32_t>(items[0].e), mid_dim, in_dim, nc);
        if (cudaGetLastError() != cudaSuccess) return false;
        pf_gather16_kernel<<<nvalid, 256, 0, stream>>>(s->xs, d_x, s->perm, topk, in_dim);
        if (cudaGetLastError() != cudaSuccess) return false;
        const bool ext = nc == 8192u;
        bool ok;
        if (ext) {
            ok = pf_launch<1, 0, 0>(s->g16, nullptr, nullptr, static_cast<const uint8_t*>(d_blob), blob_size, s,
                                    s->xs, nitems, mid_dim, in_dim, nc, clamp, stream) &&
                 pf_launch<1, 1, 0>(s->g16, s->h16, nullptr, static_cast<const uint8_t*>(d_blob), blob_size, s,
                                    s->xs, nitems, mid_dim, in_dim, nc, clamp, stream) &&
                 pf_launch<1, 2, 0>(nullptr, nullptr, d_ys, static_cast<const uint8_t*>(d_blob), blob_size, s,
                                    s->h16, nitems, out_dim, mid_dim, nc, clamp, stream);
        } else {
            ok = pf_launch<0, 0, 1>(s->g16, nullptr, nullptr, static_cast<const uint8_t*>(d_blob), blob_size, s,
                                    s->xs, nitems, mid_dim, in_dim, nc, clamp, stream) &&
                 pf_launch<0, 1, 1>(s->g16, s->h16, nullptr, static_cast<const uint8_t*>(d_blob), blob_size, s,
                                    s->xs, nitems, mid_dim, in_dim, nc, clamp, stream) &&
                 pf_launch<0, 2, 1>(nullptr, nullptr, d_ys, static_cast<const uint8_t*>(d_blob), blob_size, s,
                                    s->h16, nitems, out_dim, mid_dim, nc, clamp, stream);
        }
        if (!ok) return false;
    }
    const dim3 rgrid((out_dim + 255u) / 256u, n_tok);
    pf_reduce_kernel<<<rgrid, 256, 0, stream>>>(d_out, d_ys, s->inv, d_w, topk, out_dim);
    if (cudaGetLastError() != cudaSuccess) return false;
    return true;
}

}  // namespace strata::kernels
