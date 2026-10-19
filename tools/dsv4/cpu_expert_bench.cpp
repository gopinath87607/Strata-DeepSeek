// tools/dsv4/cpu_expert_bench.cpp - how fast can the CPU decode the routed VQ experts?
//
// Qwen/Strata keeps its experts in host RAM and computes them with a CPU worker pool. This asks the same
// question for DeepSeek-V4.1-Flash: if the 28 cores read the VQ payloads out of host RAM and run
// gate/up/down in AVX2, how many prefill tokens per second can they add to (or replace in) the GPU path?
//
// The benchmark builds a synthetic layer in the real DQVL/DQV3 layout and the real geometry (384 experts,
// in 5120, mid 2304, out 5120, one shared E4M3 codebook, 12- or 13-bit indices), checks the AVX2 row dot
// against the host decoder in vq_decode.hpp, then runs a full layer with T tokens per expert. T maps to a
// block of 384*T/topk tokens (topk 6), and the reported rate is the whole model: 40 layers at that rate.
//
// The kernel is the one a CPU expert pool would use: each expert's codeword is decoded once and its 8-value
// codebook word feeds the dot of up to 8 tokens (one accumulator per token), so the 12/13-bit unpacking and
// the codebook gather are amortised over the token tile. Weights stay packed; nothing is dequantised to a
// materialised f32 matrix.
//
//   g++ -O3 -mavx2 -mfma -std=c++17 -pthread tools/dsv4/cpu_expert_bench.cpp -o /tmp/dsv4_cpu_bench
//   numactl --interleave=0,1 /tmp/dsv4_cpu_bench                  # 28 threads, 13-bit, T = 8 16 32 64
//   numactl --interleave=0,1 /tmp/dsv4_cpu_bench 56 8192 16      # SMT, 13-bit, one 1024-token block
//
// Exit status is 0 when the AVX2 row dot agrees with the host decoder and the timed runs produce a finite
// non-zero sum.

#include "vq_decode.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#if !defined(__AVX2__) || !defined(__FMA__)
#error "build with -mavx2 -mfma"
#endif

#include <immintrin.h>

namespace {

using dsv4::VqMatrix;

constexpr uint32_t kExperts = 384;
constexpr uint32_t kInDim = 5120;
constexpr uint32_t kMidDim = 2304;
constexpr uint32_t kOutDim = 5120;
constexpr uint32_t kTopk = 6;
constexpr uint32_t kLayers = 40;
constexpr float kClamp = 10.0f;

// ---- small helpers -------------------------------------------------------------------------------------

inline uint64_t xs64(uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

inline uint32_t next_u32(uint64_t& s) { return static_cast<uint32_t>(xs64(s)); }

inline float bf16_round(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float o;
    std::memcpy(&o, &u, 4);
    return o;
}

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

inline double now_s() {    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// ---- the CPU-side view of one expert matrix ------------------------------------------------------------

struct CpuVq {
    const uint8_t* stream = nullptr;
    const uint8_t* plane = nullptr;
    const uint16_t* gains = nullptr;
    const uint16_t* cb = nullptr;  // nc * 8, expanded to bf16
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t nidx = 0;
    uint32_t mrow = 0;
    uint32_t prow = 0;
    bool has_plane = false;

    float gain(uint32_t row) const { return dsv4::f16_to_f32(gains[row]); }
    uint64_t payload_bytes() const { return uint64_t(rows) * 2 + uint64_t(rows) * mrow + (has_plane ? uint64_t(rows) * prow : 0); }
};

// Expand the shared E4M3 codebook once per layer. E4M3 has 3 mantissa bits, so every value is exactly
// representable in bf16 and the f32 bits are just the bf16 bits shifted up.
void expand_codebook(const uint8_t* e4m3, uint32_t nc, std::vector<uint16_t>& out) {
    out.resize(size_t(nc) * 8);
    for (size_t i = 0; i < out.size(); i++) {
        const float f = dsv4::e4m3fn_to_f32(e4m3[i]);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        out[i] = static_cast<uint16_t>(u >> 16);
    }
}

bool cpu_view(const VqMatrix& m, const uint16_t* cb, CpuVq* out) {
    out->stream = m.stream;
    out->plane = m.plane;
    out->gains = reinterpret_cast<const uint16_t*>(m.gains);
    out->cb = cb;
    out->rows = m.rows;
    out->cols = m.cols;
    out->nidx = m.nidx_row();
    out->mrow = m.v3_mrow();
    out->prow = m.v3_prow();
    out->has_plane = m.plane != nullptr;
    return true;
}

// ---- the kernel ----------------------------------------------------------------------------------------

// One row of one matrix over one column block, accumulated into out[t*os + row]. The block is small enough
// that its activations for every token in the tile stay in L1 while the block sweeps all rows; without that,
// each row re-reads the whole activation tile from L2/L3 and the kernel runs at a quarter of its port limit.
// Each codeword is fetched and decoded once; its codebook word then feeds N fused multiply-adds, one per
// token. The stream packs two 12-bit indices in every 3 bytes, so they are decoded in pairs; the 13th bit,
// when present, is one bit each from the row's plane byte. The per-row gain is applied by the caller.
constexpr uint32_t kColBlock = 64;  // codewords, i.e. 512 columns: 16 KB of activation per 8-token tile
constexpr uint32_t kTileCols = kColBlock * 8;

// Copy the tile's activations for one column block into a compact [token][kTileCols] buffer. The token
// stride is then a compile-time constant, so the eight dot-product loads are constant displacements off
// one base register instead of eight runtime addresses (which the compiler spills every iteration).
template <int N>
inline void gather_tile(const float* __restrict__ xb, size_t xs, const uint32_t* __restrict__ toks,
                        uint32_t i0, uint32_t i1, float* __restrict__ xblk) {
    const uint32_t n = (i1 - i0) * 8;
    for (int t = 0; t < N; t++)
        std::memcpy(xblk + size_t(t) * kTileCols, xb + xs * toks[t] + size_t(i0) * 8, size_t(n) * sizeof(float));
}

template <int N, bool Plane>
inline void rowdot_block(const CpuVq& m, uint32_t row, uint32_t i0, uint32_t i1,
                         const float* __restrict__ xblk, float* __restrict__ out, size_t os) {
    __m256 acc[N];
#pragma GCC unroll 8
    for (int t = 0; t < N; t++) acc[t] = _mm256_setzero_ps();

    const uint8_t* __restrict__ rs = m.stream + size_t(row) * m.mrow;
    const uint8_t* __restrict__ rp = Plane ? m.plane + size_t(row) * m.prow : nullptr;
    (void)rp;

    for (uint32_t ip = i0; ip < i1; ip += 2) {
        uint32_t w;
        std::memcpy(&w, rs + (ip >> 1) * 3, 4);  // 24 bits wanted; the fourth is masked by the shifts
        uint32_t ia = w & 0xFFFu;
        uint32_t ib = (w >> 12) & 0xFFFu;
        if (Plane) {
            const uint32_t bits = (rp[ip >> 3] >> (ip & 7u)) & 3u;  // ip is even, so both bits share a byte
            ia |= (bits & 1u) << 12;
            ib |= (bits >> 1) << 12;
        }

        const float* __restrict__ xg = xblk + size_t(ip - i0) * 8;
        const __m256 wa = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(m.cb + size_t(ia) * 8))), 16));
#pragma GCC unroll 8
        for (int t = 0; t < N; t++)
            acc[t] = _mm256_fmadd_ps(wa, _mm256_loadu_ps(xg + size_t(t) * kTileCols), acc[t]);

        const __m256 wb = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(m.cb + size_t(ib) * 8))), 16));
        xg += 8;
#pragma GCC unroll 8
        for (int t = 0; t < N; t++)
            acc[t] = _mm256_fmadd_ps(wb, _mm256_loadu_ps(xg + size_t(t) * kTileCols), acc[t]);
    }

#pragma GCC unroll 8
    for (int t = 0; t < N; t++) out[size_t(t) * os] += hsum256(acc[t]);
}

// The full expert for N <= 8 tokens: gate and up sweeps, SwiGLU (bf16 mid), down sweep. Layouts are
// [token][channel] so a token's activation group is 8 contiguous floats.
// The stream word for a row is only 1.5 lines, and the next row is a fixed stride away, so a row starts
// with a cold miss that the out-of-order window cannot cover (a row is 32 pairs of work, the window holds
// about four).  Touch the row kPf ahead so the miss overlaps the current row's arithmetic.
constexpr uint32_t kPrefetchRows = 6;

inline void prefetch_row(const CpuVq& m, uint32_t row, uint32_t i0) {
    const char* p = reinterpret_cast<const char*>(m.stream + size_t(row) * m.mrow + (i0 >> 1) * 3);
    _mm_prefetch(p, _MM_HINT_T1);
    _mm_prefetch(p + 64, _MM_HINT_T1);
    if (m.plane) _mm_prefetch(reinterpret_cast<const char*>(m.plane + size_t(row) * m.prow + (i0 >> 3)), _MM_HINT_T1);
}

template <int N, bool Plane>
void expert_tile(const CpuVq& cg, const CpuVq& cu, const CpuVq& cd, const float* __restrict__ xb, size_t xs,
                 const uint32_t* __restrict__ toks, float* __restrict__ gb, float* __restrict__ ub,
                 float* __restrict__ hb, float* __restrict__ ob, float* __restrict__ csum) {
    const uint32_t mid = cg.rows;
    alignas(32) float xblk[8 * kTileCols];
    std::fill(gb, gb + size_t(N) * mid, 0.0f);
    std::fill(ub, ub + size_t(N) * cu.rows, 0.0f);
    for (uint32_t i0 = 0; i0 < cg.nidx; i0 += kColBlock) {
        const uint32_t i1 = std::min(i0 + kColBlock, cg.nidx);
        gather_tile<N>(xb, xs, toks, i0, i1, xblk);  // one copy, shared by the gate and up sweeps
        for (uint32_t r = 0; r < mid; r++) {
            if (r + kPrefetchRows < mid) {
                prefetch_row(cg, r + kPrefetchRows, i0);
                prefetch_row(cu, r + kPrefetchRows, i0);
            }
            rowdot_block<N, Plane>(cg, r, i0, i1, xblk, gb + r, mid);
            rowdot_block<N, Plane>(cu, r, i0, i1, xblk, ub + r, cu.rows);
        }
    }
    for (int t = 0; t < N; t++) {
        for (uint32_t r = 0; r < mid; r++) {
            float g = gb[size_t(t) * mid + r] * cg.gain(r);
            float uu = ub[size_t(t) * cu.rows + r] * cu.gain(r);
            g = std::fmin(g, kClamp);
            uu = std::fmin(std::fmax(uu, -kClamp), kClamp);
            const float v = (g / (1.0f + std::exp(-g))) * uu;
            hb[size_t(t) * cd.cols + r] = bf16_round(v);
        }
    }
    std::fill(ob, ob + size_t(N) * cd.rows, 0.0f);
    // hb is per-tile: the down sweep indexes it by the token's position in the tile, not its global id
    uint32_t local[8];
    for (int t = 0; t < N; t++) local[t] = static_cast<uint32_t>(t);
    for (uint32_t i0 = 0; i0 < cd.nidx; i0 += kColBlock) {
        const uint32_t i1 = std::min(i0 + kColBlock, cd.nidx);
        gather_tile<N>(hb, cd.cols, local, i0, i1, xblk);
        for (uint32_t o = 0; o < cd.rows; o++) {
            if (o + kPrefetchRows < cd.rows) prefetch_row(cd, o + kPrefetchRows, i0);
            rowdot_block<N, Plane>(cd, o, i0, i1, xblk, ob + o, cd.rows);
        }
    }

    // sparse: this only has to stop the compiler from eliding the down sweep, not be a real reduction
    float s = 0.0f;
    for (int t = 0; t < N; t++)
        for (uint32_t o = 0; o < cd.rows; o += 64) s += ob[size_t(t) * cd.rows + o] * cd.gain(o);
    *csum += s;
}

template <int N>
void expert_tile_plane(const CpuVq& cg, const CpuVq& cu, const CpuVq& cd, const float* xb, size_t xs,
                       const uint32_t* toks, bool plane, float* gb, float* ub, float* hb, float* ob,
                       float* csum) {
    if (plane)
        expert_tile<N, true>(cg, cu, cd, xb, xs, toks, gb, ub, hb, ob, csum);
    else
        expert_tile<N, false>(cg, cu, cd, xb, xs, toks, gb, ub, hb, ob, csum);
}

void expert_any(const CpuVq& cg, const CpuVq& cu, const CpuVq& cd, const float* xb, size_t xs, uint32_t* toks,
                uint32_t n, float* gb, float* ub, float* hb, float* ob, float* csum, bool plane) {
    switch (n) {
        case 1: expert_tile_plane<1>(cg, cu, cd, xb, xs, toks, plane, gb, ub, hb, ob, csum); break;
        case 2: expert_tile_plane<2>(cg, cu, cd, xb, xs, toks, plane, gb, ub, hb, ob, csum); break;
        case 3: expert_tile_plane<3>(cg, cu, cd, xb, xs, toks, plane, gb, ub, hb, ob, csum); break;
        case 4: expert_tile_plane<4>(cg, cu, cd, xb, xs, toks, plane, gb, ub, hb, ob, csum); break;
        case 5: expert_tile_plane<5>(cg, cu, cd, xb, xs, toks, plane, gb, ub, hb, ob, csum); break;
        case 6: expert_tile_plane<6>(cg, cu, cd, xb, xs, toks, plane, gb, ub, hb, ob, csum); break;
        case 7: expert_tile_plane<7>(cg, cu, cd, xb, xs, toks, plane, gb, ub, hb, ob, csum); break;
        case 8: expert_tile_plane<8>(cg, cu, cd, xb, xs, toks, plane, gb, ub, hb, ob, csum); break;
        default: std::abort();
    }
}

// ---- a synthetic layer in the real container layout -----------------------------------------------------

struct Layer {
    std::vector<uint8_t> blob;
    uint32_t nc = 8192;
    uint32_t nexp = kExperts;
    std::vector<uint16_t> cb16;
    std::vector<CpuVq> gate, up, down;
    uint64_t bytes_per_expert = 0;
};

uint32_t three_rows(uint32_t which) { return which == 2 ? kOutDim : kMidDim; }
uint32_t three_cols(uint32_t which) { return which == 2 ? kMidDim : kInDim; }

bool build_layer(uint32_t nc, uint64_t seed, Layer* out, const char** err) {
    const bool plane = nc > 4096;
    const uint32_t nidx_g = kInDim / 8;
    const uint32_t nidx_d = kMidDim / 8;
    const uint32_t mrow_g = (nidx_g * 12 + 7) / 8;
    const uint32_t mrow_d = (nidx_d * 12 + 7) / 8;
    const uint32_t prow_g = (nidx_g + 7) / 8;
    const uint32_t prow_d = (nidx_d + 7) / 8;

    struct Slot {
        size_t off;
        uint64_t size;
    };
    std::vector<Slot> slots(size_t(out->nexp) * 3);
    size_t off = 16 + size_t(out->nexp) * 3 * 8;
    uint64_t bytes_per_expert = 0;
    for (uint32_t e = 0; e < out->nexp; e++) {
        for (uint32_t w = 0; w < 3; w++) {
            const uint32_t rows = three_rows(w);
            const uint32_t mrow = w == 2 ? mrow_d : mrow_g;
            const uint32_t prow = w == 2 ? prow_d : prow_g;
            const uint64_t size = 32 + uint64_t(rows) * 2 + uint64_t(rows) * mrow + (plane ? uint64_t(rows) * prow : 0);
            slots[size_t(e) * 3 + w] = {off, size};
            off += size;
            if (w == 0) bytes_per_expert = 0;
            bytes_per_expert += uint64_t(rows) * 2 + uint64_t(rows) * mrow + (plane ? uint64_t(rows) * prow : 0);
        }
    }
    const size_t cb_off = off;
    const size_t total = cb_off + size_t(nc) * 8 + 64;
    out->blob.assign(total, 0);
    out->nc = nc;
    out->bytes_per_expert = bytes_per_expert;

    auto put32 = [&](size_t o, uint32_t v) { std::memcpy(out->blob.data() + o, &v, 4); };
    auto put64 = [&](size_t o, uint64_t v) { std::memcpy(out->blob.data() + o, &v, 8); };
    auto put16 = [&](size_t o, uint16_t v) { std::memcpy(out->blob.data() + o, &v, 2); };

    put32(0, dsv4::kBlobMagic);
    put32(4, 3);
    put32(8, 0);
    put32(12, out->nexp);
    for (uint32_t e = 0; e < out->nexp; e++)
        for (uint32_t w = 0; w < 3; w++) put64(16 + (size_t(e) * 3 + w) * 8, slots[size_t(e) * 3 + w].off);

    uint64_t rng = seed;
    for (uint32_t e = 0; e < out->nexp; e++) {
        for (uint32_t w = 0; w < 3; w++) {
            const Slot& s = slots[size_t(e) * 3 + w];
            const uint32_t rows = three_rows(w);
            const uint32_t cols = three_cols(w);
            const uint32_t mrow = w == 2 ? mrow_d : mrow_g;
            const uint32_t prow = w == 2 ? prow_d : prow_g;
            put32(s.off, dsv4::kMat3Magic);
            put16(s.off + 4, 8);
            put16(s.off + 6, static_cast<uint16_t>(nc));
            put32(s.off + 8, rows);
            put32(s.off + 12, cols);
            put32(s.off + 16, dsv4::kV3FlagE4M3 | (plane ? dsv4::kV3FlagPlane : 0));
            put32(s.off + 20, 12);
            put64(s.off + 24, cb_off);

            size_t o = s.off + 32;
            for (uint32_t r = 0; r < rows; r++, o += 2) {
                // gains around 2^-7 .. 2^-6, the scale the artifact uses
                put16(o, static_cast<uint16_t>(0x2000u | (next_u32(rng) & 0x3FFu)));
            }
            for (uint32_t r = 0; r < rows; r++, o += mrow) {
                for (uint32_t i = 0; i < mrow; i += 8) {
                    uint64_t v = xs64(rng);
                    const size_t n = std::min<size_t>(8, mrow - i);
                    std::memcpy(out->blob.data() + o + i, &v, n);
                }
            }
            if (plane) {
                for (uint32_t r = 0; r < rows; r++, o += prow) {
                    for (uint32_t i = 0; i < prow; i += 8) {
                        uint64_t v = xs64(rng);
                        const size_t n = std::min<size_t>(8, prow - i);
                        std::memcpy(out->blob.data() + o + i, &v, n);
                    }
                }
            }
        }
    }
    for (uint32_t i = 0; i < nc * 8; i++) {
        uint32_t v = next_u32(rng) & 0xFF;
        if ((v & 0x7Fu) == 0x7Fu) v &= 0xFEu;  // no NaN/Inf in the codebook
        out->blob[cb_off + i] = static_cast<uint8_t>(v);
    }

    // Open every matrix through the production host decoder: the layout is valid, not merely intended.
    const uint8_t* cb = out->blob.data() + cb_off;
    expand_codebook(cb, nc, out->cb16);
    out->gate.resize(out->nexp);
    out->up.resize(out->nexp);
    out->down.resize(out->nexp);
    for (uint32_t e = 0; e < out->nexp; e++) {
        VqMatrix m;
        if (!VqMatrix::open(out->blob.data(), out->blob.size(), e, 0, kMidDim, kInDim, &m, err)) return false;
        cpu_view(m, out->cb16.data(), &out->gate[e]);
        if (!VqMatrix::open(out->blob.data(), out->blob.size(), e, 1, kMidDim, kInDim, &m, err)) return false;
        cpu_view(m, out->cb16.data(), &out->up[e]);
        if (!VqMatrix::open(out->blob.data(), out->blob.size(), e, 2, kOutDim, kMidDim, &m, err)) return false;
        cpu_view(m, out->cb16.data(), &out->down[e]);
    }
    return true;
}

// ---- correctness: the AVX2 row dot against the host decoder ---------------------------------------------

bool check_rowdot(const Layer& layer) {
    const CpuVq& m = layer.gate[0];
    const VqMatrix vm = [&] {
        VqMatrix t;
        const char* err = "";
        const bool ok = VqMatrix::open(layer.blob.data(), layer.blob.size(), 0, 0, kMidDim, kInDim, &t, &err);
        if (!ok) {
            std::fprintf(stderr, "check: open failed: %s\n", err);
            std::abort();
        }
        return t;
    }();

    std::vector<float> x(static_cast<size_t>(kInDim));
    uint64_t rng = 0x12345678u;
    for (uint32_t i = 0; i < kInDim; i++) {
        union {
            uint32_t u;
            float f;
        } v;
        v.u = 0x3F000000u | (next_u32(rng) & 0x007FFFFFu);  // [0.5, 1)
        x[i] = bf16_round(v.f) * ((next_u32(rng) & 1u) ? 1.0f : -1.0f);
    }

    // The whole row, decoded by the reference host decoder, in order.
    const float gain = vm.gain(0);
    float ref = 0.0f;
    for (uint32_t i = 0; i < vm.nidx_row(); i++) {
        float cb[8];
        vm.codebook_word(vm.index(0, i), cb);
        for (uint32_t d = 0; d < 8; d++) ref += cb[d] * x[size_t(i) * 8 + d];
    }
    ref *= gain;

    const uint32_t toks[1] = {0};
    alignas(32) float xblk[kTileCols];
    float got = 0.0f;
    for (uint32_t i0 = 0; i0 < vm.nidx_row(); i0 += kColBlock) {
        const uint32_t i1 = std::min(i0 + kColBlock, vm.nidx_row());
        gather_tile<1>(x.data(), kInDim, toks, i0, i1, xblk);
        if (layer.nc > 4096)
            rowdot_block<1, true>(m, 0, i0, i1, xblk, &got, 1);
        else
            rowdot_block<1, false>(m, 0, i0, i1, xblk, &got, 1);
    }
    got *= gain;

    const float denom = std::max(1e-6f, std::fabs(ref));
    const float rel = std::fabs(got - ref) / denom;
    const bool ok = std::isfinite(got) && rel < 2e-3f;
    std::printf("check  row0 dot: decoder %.6f  avx2 %.6f  rel %.2e  %s\n", ref, got, rel, ok ? "ok" : "FAILED");
    return ok;
}

// ---- the timed layer -----------------------------------------------------------------------------------

struct Timing {
    double sec = 0.0;       // one layer
    double checksum = 0.0;
};

template <typename Fn>
Timing run_threads(uint32_t threads, Fn&& fn) {
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> go{false};
    std::vector<double> sums(threads, 0.0);
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (uint32_t t = 0; t < threads; t++) {
        pool.emplace_back([&, t] {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) _mm_pause();
            sums[t] = fn(t);
        });
    }
    while (ready.load(std::memory_order_acquire) != threads) _mm_pause();
    const double t0 = now_s();
    go.store(true, std::memory_order_release);
    for (auto& th : pool) th.join();
    const double t1 = now_s();
    Timing out;
    out.sec = t1 - t0;
    for (double s : sums) out.checksum += s;
    return out;
}

Timing run_layer(const Layer& layer, uint32_t threads, uint32_t t_per_expert, const std::vector<float>& x) {
    const uint32_t mid = kMidDim, out_dim = kOutDim;
    return run_threads(threads, [&](uint32_t tid) {
        std::vector<float> gb(size_t(8) * mid), ub(size_t(8) * mid), hb(size_t(8) * mid), ob(size_t(8) * out_dim);
        std::vector<uint32_t> toks(t_per_expert);
        std::iota(toks.begin(), toks.end(), 0u);
        float sum = 0.0f;
        for (uint32_t e = tid; e < layer.nexp; e += threads) {
            for (uint32_t t0 = 0; t0 < t_per_expert; t0 += 8) {
                const uint32_t n = std::min(8u, t_per_expert - t0);
                expert_any(layer.gate[e], layer.up[e], layer.down[e], x.data(), kInDim, toks.data() + t0, n,
                           gb.data(), ub.data(), hb.data(), ob.data(), &sum, layer.nc > 4096);
            }
        }
        if (std::getenv("DSV4_TRACE") != nullptr) std::printf("thr %u t %u sum %.6g\n", tid, t_per_expert, sum);
        return static_cast<double>(sum);
    });
}
void report(const Layer& layer, uint32_t threads, uint32_t t_per_expert, const Timing& tm) {
    const uint32_t n_tok = layer.nexp * t_per_expert / kTopk;
    const double model_s = tm.sec * kLayers;
    const double tok_s = n_tok / model_s;
    const double bytes = double(layer.bytes_per_expert) * layer.nexp;
    const double gbs = bytes / tm.sec / 1e9;
    const double cw = double(layer.nexp) *
                      (2.0 * kMidDim * (kInDim / 8.0) + kOutDim * (kMidDim / 8.0));  // codewords per layer
    std::printf("T=%2u (%4u-token block)  %2u thr  layer %7.3f s  %6.1f GB read  %5.1f GB/s  "
                "%.1f Gcodeword/s  model %6.2f s  %6.1f tok/s  sum %.4g\n",
                t_per_expert, n_tok, threads, tm.sec, bytes / 1e9, gbs, cw / tm.sec / 1e9, model_s, tok_s,
                tm.checksum);
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t threads = 28;
    uint32_t nc = 8192;
    std::vector<uint32_t> configs;
    if (argc > 1) threads = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 0));
    if (argc > 2) nc = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 0));
    for (int i = 3; i < argc; i++) configs.push_back(static_cast<uint32_t>(std::strtoul(argv[i], nullptr, 0)));
    if (configs.empty()) configs = {8, 16, 32, 64};
    if (nc != 4096 && nc != 8192) {
        std::fprintf(stderr, "nc must be 4096 (12-bit) or 8192 (13-bit)\n");
        return 2;
    }
    if (threads == 0 || threads > 256) {
        std::fprintf(stderr, "threads out of range\n");
        return 2;
    }

    Layer layer;
    const char* err = "";
    const double tb0 = now_s();
    const uint64_t seed = 0x9E3779B97F4A7C15ull ^ nc;
    if (!build_layer(nc, seed, &layer, &err)) {
        std::fprintf(stderr, "build_layer failed: %s\n", err);
        return 1;
    }
    std::printf("layer   %u experts, in %u, mid %u, out %u, %s codebook %u, %.2f MB/expert, blob %.2f GB (built %.1f s)\n",
                layer.nexp, kInDim, kMidDim, kOutDim, nc > 4096 ? "13-bit" : "12-bit", nc,
                layer.bytes_per_expert / 1e6, layer.blob.size() / 1e9, now_s() - tb0);

    if (!check_rowdot(layer)) return 1;

    // activations for the largest tile, but never fewer than the 16 the scaling run uses; bf16-rounded like
    // the packed x the GPU path consumes
    const uint32_t max_t = std::max<uint32_t>(16u, *std::max_element(configs.begin(), configs.end()));
    std::vector<float> x(size_t(max_t) * kInDim);
    uint64_t rng = 0xABCDEF01u;
    for (size_t i = 0; i < x.size(); i++) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        union {
            uint32_t u;
            float f;
        } v;
        v.u = 0x3F800000u | (static_cast<uint32_t>(rng) & 0x007FFFFFu);
        v.u &= 0xFFFF0000u;  // bf16
        x[i] = (v.f - 1.5f);
    }

    // optional: find the first expert whose tile goes non-finite
    if (std::getenv("DSV4_SCAN") != nullptr) {
        std::vector<float> gb(size_t(8) * kMidDim), ub(size_t(8) * kMidDim), hb(size_t(8) * kMidDim),
            ob(size_t(8) * kOutDim);
        std::vector<uint32_t> toks(16);
        std::iota(toks.begin(), toks.end(), 0u);
        for (uint32_t e = 0; e < layer.nexp; e++) {
            for (uint32_t t0 = 0; t0 < 16; t0 += 8) {
            float sum = 0.0f;
            expert_any(layer.gate[e], layer.up[e], layer.down[e], x.data(), kInDim, toks.data() + t0, 8, gb.data(),
                       ub.data(), hb.data(), ob.data(), &sum, layer.nc > 4096);
            if (!std::isfinite(sum)) {
                std::printf("scan  expert %u tile %u sum %f (first non-finite)\n", e, t0, sum);
                for (int t = 0; t < 8; t++) {
                    for (uint32_t r = 0; r < kMidDim; r++) {
                        const float g = gb[size_t(t) * kMidDim + r];
                        const float uu = ub[size_t(t) * kMidDim + r];
                        const float h = hb[size_t(t) * kMidDim + r];
                        if (!std::isfinite(g) || !std::isfinite(uu) || !std::isfinite(h)) {
                            std::printf("  t %d r %u  g %g u %g h %g\n", t, r, g, uu, h);
                            t = 99;
                            break;
                        }
                    }
                }
                return 1;
            }
            }
        }
        std::printf("scan  all %u experts finite\n", layer.nexp);
        // exactly the single-threaded timed path: the checksum thread carries one float across all experts
        {
            float acc = 0.0f;
            for (uint32_t e = 0; e < layer.nexp; e++) {
                for (uint32_t t0 = 0; t0 < 16; t0 += 8) {
                    float s = 0.0f;
                    expert_any(layer.gate[e], layer.up[e], layer.down[e], x.data(), kInDim, toks.data() + t0, 8,
                               gb.data(), ub.data(), hb.data(), ob.data(), &s, layer.nc > 4096);
                    acc += s;
                }
                if (!std::isfinite(acc)) {
                    std::printf("accum non-finite after expert %u (acc %g)\n", e, acc);
                    return 1;
                }
            }
            std::printf("accum total %.6g\n", acc);
        }
    }

    // warm the codebook and the first experts
    (void)run_layer(layer, std::min(threads, 7u), 8, x);

    std::printf("\n-- one layer at %u threads, best of 2 --\n", threads);
    for (uint32_t t : configs) {
        Timing best;
        best.sec = 1e30;
        for (int rep = 0; rep < 2; rep++) {
            const Timing tm = run_layer(layer, threads, t, x);
            if (tm.sec < best.sec) best = tm;
        }
        report(layer, threads, t, best);
    }

    std::printf("\n-- thread scaling, T=16 (1024-token block), one run --\n");
    for (uint32_t th : {1u, 7u, 14u, 28u, 56u}) {
        if (th > threads * 2) continue;
        const Timing tm = run_layer(layer, th, 16, x);
        report(layer, th, 16, tm);
    }
    return 0;
}
