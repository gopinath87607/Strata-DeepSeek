// tools/dsv4/vq_decode.hpp - DeepSeek V4.1 routed-expert VQ blob, decoded.
//
// The blob is one GGUF tensor per layer (`blk.L.ffn_exps_vq.blob`, GGML type
// 42); it holds every routed expert's gate/up/down matrix in a vector-
// quantized form. The layout is defined by the C engine at /data/YoungAi
// (vq_fmt.h for the container and the v2 payload, src/cuda/cuda_vq_row.inc.cu
// for the v3 payload) and ported here from Baekpica/ds4-dfm-rs
// crates/ds4-core/src/vq.rs (MIT), which cites those files line by line.
//
// Container: [u32 'DQVL'][u32 ver][u32 layer][u32 nexp]
//            [nexp*3 u64 payload offsets, 0 = absent]
//            ... payloads ...
// Slot (e, which): which 0 = w1 (gate), 1 = w3 (up), 2 = w2 (down).
// Value rule: value[row][col] = codebook[idx][col % 8] * gain[row].
//
// v2 payload 'DQVQ': [u32 magic][u16 dim][u16 nc][u32 rows][u32 cols]
//                    [nc*dim f16 codebook][rows f16 gains][index stream]
// v3 payload 'DQV3': [u32 magic][u16 dim][u16 nc][u32 rows][u32 cols]
//                    [u32 flags][u32 main_nbits][u64 codebook offset]
//                    [rows f16 gains][12-bit main stream][1-bit plane]
// flags bit 0: the codebook is E4M3 (one per layer, shared by every matrix);
// flags bit 1: a 13th-bit plane follows the main stream. The main stream is
// always 12 bits, packed per row.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace dsv4 {

constexpr uint32_t kBlobMagic = 0x4C565144u;  // 'DQVL'
constexpr uint32_t kMatMagic = 0x51565144u;   // 'DQVQ', v2 payload
constexpr uint32_t kMat3Magic = 0x33565144u;  // 'DQV3', v3 payload
constexpr uint32_t kVerMin = 2;
constexpr uint32_t kVerMax = 3;
constexpr uint16_t kVqDim = 8;
constexpr uint32_t kV3MainBits = 12;
constexpr uint32_t kV3FlagE4M3 = 1u;
constexpr uint32_t kV3FlagPlane = 2u;
constexpr size_t kV2Header = 16;
constexpr size_t kV3Header = 32;

// f16 -> f32 with subnormals kept exact (C ds4vq_f16).
inline float f16_to_f32(uint16_t h) {
    uint32_t s = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {
            e = 112;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e--;
            }
            m &= 0x3FFu;
            f = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

// E4M3FN -> f32 (C ds4_e4m3fn_to_f32): 0x7f/0xff are NaN; exp 0 is the
// subnormal mant * 2^-9.
inline float e4m3fn_to_f32(uint8_t x) {
    const uint32_t abs = x & 0x7Fu;
    const bool sign = (x & 0x80u) != 0;
    if (abs == 0) return sign ? -0.0f : 0.0f;
    if (abs == 0x7Fu) return NAN;
    const int exp = static_cast<int>((x >> 3) & 0x0Fu);
    const int man = static_cast<int>(x & 0x07u);
    float value;
    if (exp == 0) {
        value = static_cast<float>(man) * 0.001953125f;  // 2^-9
    } else {
        value = (1.0f + static_cast<float>(man) / 8.0f) *
                static_cast<float>(1 << (exp - 1)) * (1.0f / 64.0f);
    }
    return sign ? -value : value;
}

inline uint16_t rd_u16(const uint8_t* b, size_t off) {
    uint16_t v;
    std::memcpy(&v, b + off, 2);
    return v;
}

inline uint32_t rd_u32(const uint8_t* b, size_t off) {
    uint32_t v;
    std::memcpy(&v, b + off, 4);
    return v;
}

inline uint64_t rd_u64(const uint8_t* b, size_t off) {
    uint64_t v;
    std::memcpy(&v, b + off, 8);
    return v;
}

inline uint32_t blob_version(const uint8_t* blob, size_t size) {
    return size >= 8 ? rd_u32(blob, 4) : 0;
}

inline uint32_t blob_nexp(const uint8_t* blob, size_t size) {
    return size >= 16 ? rd_u32(blob, 12) : 0;
}

// Magic, accepted version and a slot table that fits, as the engine requires.
// An unknown version is refused rather than decoded as this one: the fields
// would line up and produce plausible wrong weights.
inline bool blob_ok(const uint8_t* blob, size_t size) {
    if (blob == nullptr || size < 16) return false;
    if (rd_u32(blob, 0) != kBlobMagic) return false;
    const uint32_t ver = rd_u32(blob, 4);
    if (ver < kVerMin || ver > kVerMax) return false;
    const uint32_t nexp = rd_u32(blob, 12);
    return nexp >= 1 && nexp <= 4096 && size >= 16 + static_cast<size_t>(nexp) * 3 * 8;
}

inline uint64_t blob_slot(const uint8_t* blob, uint32_t e, uint32_t which) {
    return rd_u64(blob, 16 + (static_cast<size_t>(e) * 3 + which) * 8);
}

inline uint32_t index_bits(uint32_t nc) {
    uint32_t nbit = 0;
    while ((1u << nbit) < nc) nbit++;
    return nbit < 1 ? 1 : nbit;
}

// One expert matrix, opened from its payload. Pointers stay inside `blob`;
// the caller keeps the blob alive for as long as the matrix is used.
struct VqMatrix {
    const uint8_t* codebook = nullptr;  // nc*dim f16 (v2) or nc*dim E4M3 (v3)
    const uint8_t* gains = nullptr;     // rows f16
    const uint8_t* stream = nullptr;    // main index stream
    const uint8_t* plane = nullptr;     // 13th-bit plane, v3 with flags bit 1
    bool v3 = false;
    uint32_t nc = 0;
    uint32_t nbit = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;

    uint32_t nidx_row() const { return cols / kVqDim; }
    uint32_t v3_mrow() const { return (nidx_row() * kV3MainBits + 7) / 8; }
    uint32_t v3_prow() const { return (nidx_row() + 7) / 8; }

    float gain(uint32_t row) const { return f16_to_f32(rd_u16(gains, static_cast<size_t>(row) * 2)); }

    // Index of the i-th group of 8 columns in `row`.
    uint32_t index(uint32_t row, uint32_t i) const {
        if (!v3) {
            const uint32_t gi = row * nidx_row() + i;
            if (nbit == 8) return stream[gi];
            const size_t bit = static_cast<size_t>(gi) * nbit;
            const uint32_t w = static_cast<uint32_t>(stream[bit >> 3]) |
                               (static_cast<uint32_t>(stream[(bit >> 3) + 1]) << 8) |
                               (static_cast<uint32_t>(stream[(bit >> 3) + 2]) << 16);
            const uint32_t imsk = (1u << nbit) - 1u;
            return (w >> (bit & 7)) & imsk;
        }
        const uint32_t mrow = v3_mrow();
        const uint8_t* row_stream = stream + static_cast<size_t>(row) * mrow;
        const size_t bit = static_cast<size_t>(i) * kV3MainBits;
        const uint32_t w = static_cast<uint32_t>(row_stream[bit >> 3]) |
                           (static_cast<uint32_t>(row_stream[(bit >> 3) + 1]) << 8) |
                           (static_cast<uint32_t>(row_stream[(bit >> 3) + 2]) << 16);
        uint32_t v = (w >> (bit & 7)) & ((1u << kV3MainBits) - 1u);
        if (plane != nullptr) {
            const uint8_t byte = plane[static_cast<size_t>(row) * v3_prow() + (i >> 3)];
            v |= static_cast<uint32_t>((byte >> (i & 7)) & 1u) << kV3MainBits;
        }
        return v;
    }

    void codebook_word(uint32_t idx, float out[kVqDim]) const {
        if (v3) {
            for (uint32_t d = 0; d < kVqDim; d++) out[d] = e4m3fn_to_f32(codebook[static_cast<size_t>(idx) * kVqDim + d]);
        } else {
            for (uint32_t d = 0; d < kVqDim; d++) out[d] = f16_to_f32(rd_u16(codebook, static_cast<size_t>(idx) * kVqDim * 2 + d * 2));
        }
    }

    // Open expert `e`'s matrix `which` (0 w1, 1 w3, 2 w2), asserting the
    // payload's own shape against the caller's expectation. Returns false and
    // sets `err` when anything does not add up.
    static bool open(const uint8_t* blob, size_t size, uint32_t e, uint32_t which,
                     uint32_t rows, uint32_t cols, VqMatrix* out, const char** err) {
        *out = VqMatrix{};
        if (!blob_ok(blob, size)) {
            *err = "not a DQVL v2/v3 blob";
            return false;
        }
        const uint64_t off64 = blob_slot(blob, e, which);
        if (off64 == 0) {
            *err = "expert matrix absent";
            return false;
        }
        if (off64 > size || size - off64 < kV3Header) {
            *err = "payload offset outside the blob";
            return false;
        }
        const size_t off = static_cast<size_t>(off64);
        const uint8_t* pay = blob + off;
        const uint32_t magic = rd_u32(pay, 0);
        bool v3;
        if (magic == kMatMagic) {
            v3 = false;
        } else if (magic == kMat3Magic) {
            v3 = true;
        } else {
            *err = "payload magic is neither DQVQ nor DQV3";
            return false;
        }
        const uint16_t dim = rd_u16(pay, 4);
        const uint32_t nc = rd_u16(pay, 6);
        const uint32_t p_rows = rd_u32(pay, 8);
        const uint32_t p_cols = rd_u32(pay, 12);
        if (p_rows != rows || p_cols != cols) {
            *err = "payload shape does not match the caller";
            return false;
        }
        if (dim != kVqDim) {
            *err = "payload dim is not 8";
            return false;
        }
        out->v3 = v3;
        out->nc = nc;
        out->nbit = index_bits(nc);
        out->rows = rows;
        out->cols = cols;
        if (v3) {
            const uint32_t flags = rd_u32(pay, 16);
            const uint32_t mnb = rd_u32(pay, 20);
            const uint64_t cb_off = rd_u64(pay, 24);
            if (mnb != kV3MainBits || (flags & kV3FlagE4M3) == 0) {
                *err = "v3 flags/main width not understood";
                return false;
            }
            const bool plane_present = (flags & kV3FlagPlane) != 0;
            if (plane_present != (out->nbit > kV3MainBits)) {
                *err = "v3 plane flag disagrees with the index width";
                return false;
            }
            const size_t cb_len = static_cast<size_t>(nc) * kVqDim;
            if (cb_off > size || cb_len > size - cb_off) {
                *err = "v3 codebook outside the blob";
                return false;
            }
            out->codebook = blob + cb_off;
            const size_t gains_off = off + kV3Header;
            const size_t gains_len = static_cast<size_t>(rows) * 2;
            if (gains_len > size - gains_off) {
                *err = "v3 gains outside the blob";
                return false;
            }
            out->gains = blob + gains_off;
            const uint8_t* stream = blob + gains_off + gains_len;
            const size_t stream_len = size - (gains_off + gains_len);
            const size_t mrow = (static_cast<size_t>(out->nidx_row()) * kV3MainBits + 7) / 8;
            const size_t main_len = static_cast<size_t>(rows) * mrow;
            const size_t last_window = static_cast<size_t>(rows - 1) * mrow +
                                       ((static_cast<size_t>(out->nidx_row()) - 1) * kV3MainBits >> 3) + 3;
            if (stream_len < (main_len > last_window ? main_len : last_window)) {
                *err = "v3 index stream truncated";
                return false;
            }
            out->stream = stream;
            if (plane_present) {
                const size_t prow = (static_cast<size_t>(out->nidx_row()) + 7) / 8;
                const size_t plane_len = static_cast<size_t>(rows) * prow;
                if (stream_len < main_len + plane_len) {
                    *err = "v3 plane truncated";
                    return false;
                }
                out->plane = stream + main_len;
            }
        } else {
            const size_t cb_len = static_cast<size_t>(nc) * kVqDim * 2;
            const size_t gains_off = off + kV2Header;
            const size_t gains_len = static_cast<size_t>(rows) * 2;
            if (cb_len > size - gains_off || gains_len > size - gains_off - cb_len) {
                *err = "v2 codebook/gains outside the blob";
                return false;
            }
            out->codebook = blob + gains_off;
            out->gains = blob + gains_off + cb_len;
            out->stream = blob + gains_off + cb_len + gains_len;
            const size_t total = static_cast<size_t>(rows) * out->nidx_row();
            const size_t need = out->nbit == 8 ? total : (((total - 1) * out->nbit) >> 3) + 3;
            if (size - (gains_off + cb_len + gains_len) < need) {
                *err = "v2 index stream truncated";
                return false;
            }
        }
        return true;
    }

    // Dequantise the whole matrix row-major. `gain_override` replaces the
    // payload's per-row gain when non-null (the engine's gr_Lnn.bin path).
    void dequant_f32(float* out, const float* gain_override = nullptr) const {
        const uint32_t nidx = nidx_row();
        for (uint32_t r = 0; r < rows; r++) {
            const float g = gain_override != nullptr ? gain_override[r] : gain(r);
            float* orow = out + static_cast<size_t>(r) * cols;
            for (uint32_t i = 0; i < nidx; i++) {
                float cb[kVqDim];
                codebook_word(index(r, i), cb);
                float* o = orow + static_cast<size_t>(i) * kVqDim;
                for (uint32_t d = 0; d < kVqDim; d++) o[d] = cb[d] * g;
            }
        }
    }
};

}  // namespace dsv4
