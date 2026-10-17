// include/strata/kernels/deepseek4_vq.hpp - DeepSeek-V4.1-Flash routed-expert VQ matrices on the GPU.
//
// The format is `tools/dsv4/vq_decode.hpp`'s (the CPU decoder P0 checked against the engine's own fixtures):
// one GGUF tensor per layer is a `DQVL` container holding every routed expert's gate/up/down matrix as a
// `DQVQ` (v2) or `DQV3` payload.  A codeword is 8 values from a shared codebook, times a per-row gain:
//
//     value[row][col] = codebook[index(row, col/8)][col % 8] * gain[row]
//
// v2 keeps the codebook per payload as f16 and packs the index stream at `index_bits(nc)` per group; v3
// keeps ONE codebook per layer as E4M3 (behind the blob header), writes a fixed 12-bit main stream per row,
// and puts the 13th bit of a 13-bit index in a separate 1-bit plane.  The two hard parts a port gets wrong
// are the index window (three bytes may cross the row start, and the last window needs three readable bytes
// past its row's end - the format reserves them) and the plane's row stride, (nidx_row+7)/8 bytes.
//
// WHY THE OPEN IS `__host__ __device__`.  The CPU decoder validates a payload against the host blob; the
// engine's expert cache holds the blob in device memory and the MoE kernel must open the payload from there
// (the reference's `v41_vq_open` is device-only for that reason).  One parser, two address spaces: it reads
// bytes one at a time and performs no allocation, so the same code is valid on both sides.  `deepseek4_vq_open`
// is the host wrapper that turns a failed parse into a named error for callers that hold host memory.
//
// THE GEMV READS THE PACKED STREAM DIRECTLY - it never materialises the matrix.  One warp per row, lane j
// takes index groups j, j+32, j+64, ... (the reference's single-token mapping), and the final gain multiply
// and the 5-step xor reduction are its order.  `x` is packed bf16, two values per uint32 (low half first),
// because the engine's activation path rounds to bf16 before the VQ multiply; the reference measured that
// change as throughput, not accuracy, since the values already sit on the bf16 grid.
//
// The codebook conversions here are exact in both directions and written as integer/float arithmetic rather
// than hardware instructions: E4M3 is a subset of f16, and the scalar table is numerically identical to
// `cvt.rn.f16x2.e4m3x2` (the engine measured the instruction's cost, not its semantics).  Keeping it portable
// is what lets one source file serve CUDA (sm_86 included, which has no FP8 hardware conversion) and HIP.
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_DSV4_VQ_HD __host__ __device__
#else
#define STRATA_DSV4_VQ_HD
#endif

namespace strata::kernels {

// Why a payload was refused.  Kept on the matrix so the parser can stay one function for host and device:
// the host wrapper turns the code into the CPU decoder's wording, and a device caller can just test ok.
enum Deepseek4VqError : uint32_t {
    kDsv4VqOk = 0,
    kDsv4VqBadContainer,
    kDsv4VqExpertOutOfRange,
    kDsv4VqSlotAbsent,
    kDsv4VqPayloadOutside,
    kDsv4VqPayloadMagic,
    kDsv4VqShape,
    kDsv4VqDim,
    kDsv4VqV3Flags,
    kDsv4VqV3PlaneFlag,
    kDsv4VqV3Codebook,
    kDsv4VqV3Gains,
    kDsv4VqV3Stream,
    kDsv4VqV3PlaneTrunc,
    kDsv4VqV2CodebookGains,
    kDsv4VqV2Stream,
};

// One expert matrix, opened from its payload.  All pointers stay inside the blob; the caller keeps the blob
// alive for as long as the matrix is used.  Pointers are whatever address space the blob lives in (host for
// `deepseek4_vq_open`, device for the `__device__` parse the engine calls).
struct Deepseek4VqMatrix {
    const uint8_t* codebook = nullptr;       // nc*8 f16 (v2) or nc*8 E4M3 (v3)
    const uint8_t* gains = nullptr;          // rows f16, payload order
    const uint8_t* stream = nullptr;         // main index stream
    const uint8_t* plane = nullptr;          // v3 13th-bit plane, null when the width is 12
    const float* gain_override = nullptr;    // rows f32, replaces the payload gain when set (the engine's sidecar)
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t nidx_row = 0;                   // cols / 8
    uint32_t nbit = 0;                       // index width from nc
    uint32_t nc = 0;
    uint32_t v3 = 0;
    int ok = 0;
    uint32_t err = kDsv4VqOk;
};

// ---- the shared parser --------------------------------------------------------------------------------
//
// Same checks and the same order as dsv4::VqMatrix::open (tools/dsv4/vq_decode.hpp), so a payload the CPU
// decoder accepts is accepted here with identical pointers, and one it refuses is refused.  `size` is the
// blob's length in bytes.  Returns ok=0 on any mismatch; never reads outside [blob, blob+size).

STRATA_DSV4_VQ_HD inline uint16_t deepseek4_vq_rd_u16(const uint8_t* p) {
    return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8));
}

STRATA_DSV4_VQ_HD inline uint32_t deepseek4_vq_rd_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

STRATA_DSV4_VQ_HD inline uint64_t deepseek4_vq_rd_u64(const uint8_t* p) {
    return static_cast<uint64_t>(deepseek4_vq_rd_u32(p)) |
           (static_cast<uint64_t>(deepseek4_vq_rd_u32(p + 4)) << 32);
}

STRATA_DSV4_VQ_HD inline uint32_t deepseek4_vq_index_bits(uint32_t nc) {
    uint32_t nbit = 0;
    while ((1u << nbit) < nc) nbit++;
    return nbit < 1u ? 1u : nbit;
}

STRATA_DSV4_VQ_HD inline Deepseek4VqMatrix deepseek4_vq_parse(const uint8_t* blob, uint64_t size,
                                                             uint32_t e, uint32_t which, uint32_t rows,
                                                             uint32_t cols) {
    constexpr uint32_t kBlobMagic = 0x4C565144u;  // 'DQVL'
    constexpr uint32_t kMatMagic = 0x51565144u;   // 'DQVQ'
    constexpr uint32_t kMat3Magic = 0x33565144u;  // 'DQV3'
    constexpr uint64_t kV3Header = 32;
    constexpr uint32_t kV3MainBits = 12;
    constexpr uint32_t kVqDim = 8;

    Deepseek4VqMatrix m;
    if (blob == nullptr || size < 16) { m.err = kDsv4VqBadContainer; return m; }
    if (deepseek4_vq_rd_u32(blob) != kBlobMagic) { m.err = kDsv4VqBadContainer; return m; }
    const uint32_t ver = deepseek4_vq_rd_u32(blob + 4);
    if (ver < 2 || ver > 3) { m.err = kDsv4VqBadContainer; return m; }
    const uint32_t nexp = deepseek4_vq_rd_u32(blob + 12);
    if (nexp < 1 || nexp > 4096 || size < 16 + static_cast<uint64_t>(nexp) * 3 * 8) {
        m.err = kDsv4VqBadContainer;
        return m;
    }
    if (e >= nexp) { m.err = kDsv4VqExpertOutOfRange; return m; }

    const uint64_t off = deepseek4_vq_rd_u64(blob + 16 + (static_cast<uint64_t>(e) * 3 + which) * 8);
    if (off == 0) { m.err = kDsv4VqSlotAbsent; return m; }
    if (off > size || size - off < kV3Header) { m.err = kDsv4VqPayloadOutside; return m; }
    const uint8_t* pay = blob + off;
    const uint32_t magic = deepseek4_vq_rd_u32(pay);
    const uint32_t v3 = magic == kMat3Magic ? 1u : 0u;
    if (magic != kMatMagic && magic != kMat3Magic) { m.err = kDsv4VqPayloadMagic; return m; }
    if (deepseek4_vq_rd_u16(pay + 4) != kVqDim) { m.err = kDsv4VqDim; return m; }
    const uint32_t nc = deepseek4_vq_rd_u16(pay + 6);
    if (deepseek4_vq_rd_u32(pay + 8) != rows || deepseek4_vq_rd_u32(pay + 12) != cols) {
        m.err = kDsv4VqShape;
        return m;
    }

    m.rows = rows;
    m.cols = cols;
    m.nidx_row = cols / kVqDim;
    m.nbit = deepseek4_vq_index_bits(nc);
    m.nc = nc;
    m.v3 = v3;
    if (m.nidx_row == 0) { m.err = kDsv4VqShape; return m; }

    if (v3) {
        const uint32_t flags = deepseek4_vq_rd_u32(pay + 16);
        const uint32_t main_bits = deepseek4_vq_rd_u32(pay + 20);
        const uint64_t cb_off = deepseek4_vq_rd_u64(pay + 24);
        if (main_bits != kV3MainBits || (flags & 1u) == 0) { m.err = kDsv4VqV3Flags; return m; }
        const bool plane_present = (flags & 2u) != 0;
        if (plane_present != (m.nbit > kV3MainBits)) { m.err = kDsv4VqV3PlaneFlag; return m; }
        const uint64_t cb_len = static_cast<uint64_t>(nc) * kVqDim;
        if (cb_off > size || cb_len > size - cb_off) { m.err = kDsv4VqV3Codebook; return m; }
        m.codebook = blob + cb_off;
        const uint64_t gains_off = off + kV3Header;
        const uint64_t gains_len = static_cast<uint64_t>(rows) * 2;
        if (gains_len > size - gains_off) { m.err = kDsv4VqV3Gains; return m; }
        m.gains = blob + gains_off;
        const uint8_t* stream = blob + gains_off + gains_len;
        const uint64_t stream_len = size - (gains_off + gains_len);
        const uint64_t mrow = (static_cast<uint64_t>(m.nidx_row) * kV3MainBits + 7) / 8;
        const uint64_t main_len = static_cast<uint64_t>(rows) * mrow;
        const uint64_t last_window =
            static_cast<uint64_t>(rows - 1) * mrow + ((static_cast<uint64_t>(m.nidx_row) - 1) * kV3MainBits >> 3) + 3;
        const uint64_t need = main_len > last_window ? main_len : last_window;
        if (stream_len < need) { m.err = kDsv4VqV3Stream; return m; }
        m.stream = stream;
        if (plane_present) {
            const uint64_t prow = (static_cast<uint64_t>(m.nidx_row) + 7) / 8;
            const uint64_t plane_len = static_cast<uint64_t>(rows) * prow;
            if (stream_len < main_len + plane_len) { m.err = kDsv4VqV3PlaneTrunc; return m; }
            m.plane = stream + main_len;
        }
    } else {
        const uint64_t cb_len = static_cast<uint64_t>(nc) * kVqDim * 2;
        const uint64_t gains_off = off + 16;
        const uint64_t gains_len = static_cast<uint64_t>(rows) * 2;
        if (cb_len > size - gains_off || gains_len > size - gains_off - cb_len) {
            m.err = kDsv4VqV2CodebookGains;
            return m;
        }
        m.codebook = blob + gains_off;
        m.gains = blob + gains_off + cb_len;
        m.stream = blob + gains_off + cb_len + gains_len;
        const uint64_t total = static_cast<uint64_t>(rows) * m.nidx_row;
        const uint64_t need = m.nbit == 8 ? total : (((total - 1) * m.nbit) >> 3) + 3;
        if (size - (gains_off + cb_len + gains_len) < need) { m.err = kDsv4VqV2Stream; return m; }
    }
    m.ok = 1;
    m.err = kDsv4VqOk;
    return m;
}

// ---- the kernels --------------------------------------------------------------------------------------

// Host-side opener with a named reason.  `blob` is host memory; the returned matrix points into it.
// `which` is 0 w1 (gate), 1 w3 (up), 2 w2 (down).
bool deepseek4_vq_open(const uint8_t* blob, uint64_t size, uint32_t expert, uint32_t which, uint32_t rows,
                       uint32_t cols, Deepseek4VqMatrix* out, const char** err);

// Decode the whole matrix row-major: out[row*cols + i*8 + d] = codebook[index(row,i)][d] * gain(row), with
// `gain_override` replacing the payload gain when set.  Bit-identical to dsv4::VqMatrix::dequant_f32 - the
// P2 gate - because every step (index extraction, f16/E4M3 conversion, the final multiply) is the CPU
// decoder's, not a hardware shortcut.  A null stream launches on the default stream and synchronizes.
void deepseek4_vq_dequant(const Deepseek4VqMatrix& m, float* out, void* stream);

// y[row] = gain(row) * sum_col W[row][col] * x[col], reading the packed index stream directly (no decode).
// `x_bf16_pairs` holds cols values as bf16, two per uint32, low half = the even column.  One warp per row.
// Returns false, launching nothing, when the geometry has no compiled specialization (v2 index widths 8-12,
// v3 widths 12 and 13 - the widths the artifact and the engine's fixtures use).
bool deepseek4_vq_gemv_bf16(const Deepseek4VqMatrix& m, const uint32_t* x_bf16_pairs, float* y, void* stream);

}  // namespace strata::kernels
