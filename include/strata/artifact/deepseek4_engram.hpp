// include/strata/artifact/deepseek4_engram.hpp - DeepSeek-V4.1-Flash's engram sidecar: the n-gram hash, the
// O_DIRECT row reader, the fp8 row dequant and the gate's host emulation (docs/DEEPSEEK4.md §4, P4).
//
// The engram is not part of the trunk.  Layers 1 and 14 each own a table of 384M fp8 rows on the side, addressed
// by a hash of the trailing 4 tokens; per token each layer reads 24 random rows (8 heads x 3 n-gram lengths),
// dequantizes them to bf16, multiplies by the layer's fp8 `wkv` matrix and folds the result into the
// hyper-connection state through a gated update.  The four hash tables ARE in the GGUF; the row tables and the
// q/k/wkv planes are in two safetensors shards, which the metadata names by path, offset and row count.
//
// The source of truth is the C engine (core_v41_engram.c), ported line by line by the reference runtime
// (`Baekpica/ds4-dfm-rs`, MIT: crates/ds4-core/src/engram.rs and cuda/ds41_engram.cuh).  Three things are easy
// to get wrong and are the reason this file exists in this shape:
//
//   * the multipliers and primes are flat, LAYER-MAJOR (`mult[ei*G+k]`, `prim[(ei*(G-1)+i-1)*H+h]`); their GGUF
//     dimensions suggest a row-major order that silently produces different rows, not an error;
//   * the multiply wraps unsigned (`(uint64_t)cid * (uint64_t)mult`) before the modulo, and the modulo is the C
//     remainder with a negative correction;
//   * a 264-byte row is not O_DIRECT-aligned, so a read is an aligned 4 KiB superset copied out of a bounce
//     buffer, and a filesystem that refuses O_DIRECT falls back to a plain read plus FADV_RANDOM rather than
//     erroring (the reference measured why: 264-byte reads through the page cache evict the model mapping and
//     make temperature-0 runs non-reproducible).
//
// The arithmetic below is `__host__ __device__` so the GPU kernel and this host oracle run the same code: P4's
// gate requires the row dequant to be bit-exact, and a second copy is how that stops being true.
#pragma once

#include "strata/artifact/deepseek4.hpp"
#include "strata/artifact/deepseek4_engram_bits.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strata::deepseek4 {

// ---- The hash ---------------------------------------------------------------------------------------------
//
// All four tables are required when the wire carries engram layers (the engine's `weights_bind_v41` does the
// same).  Sizes come from the geometry: token_map `vocab_size`, multipliers `max_ngram * layers`, primes and
// offsets `(max_ngram - 1) * heads * layers`.

struct EngramHash {
    std::vector<int32_t> token_map;    ///< model token -> compressed id
    std::vector<int64_t> multipliers;  ///< layers * max_ngram, layer-major flat
    std::vector<int64_t> primes;       ///< layers * (max_ngram-1) * heads, layer-major flat
    std::vector<int64_t> offsets;      ///< the same shape as primes
    uint32_t max_ngram = 0;            ///< G: the n-gram length (4)
    uint32_t heads = 0;                ///< H (8)
    int32_t pad = 0;                   ///< the compressed pad id (2)
    uint32_t n_vocab = 0;              ///< the model vocabulary the token map covers

    /// Rows per position: (max_ngram - 1) heads.
    uint32_t cols() const { return max_ngram > 0 ? (max_ngram - 1) * heads : 0; }

    /// Table sizes must be exactly the geometry's; anything else is refused before a row is computed.
    bool valid(uint32_t layer_count, std::string& err) const {
        if (max_ngram < 2 || max_ngram > 8) {
            err = "engram max_ngram is not 2..8";
            return false;
        }
        if (heads == 0 || heads > 64) {
            err = "engram head count is not 1..64";
            return false;
        }
        if (token_map.size() != n_vocab) {
            err = "engram token_map holds " + std::to_string(token_map.size()) + " entries, vocabulary is " +
                  std::to_string(n_vocab);
            return false;
        }
        const uint64_t mult = static_cast<uint64_t>(layer_count) * max_ngram;
        const uint64_t prim = static_cast<uint64_t>(layer_count) * (max_ngram - 1) * heads;
        if (multipliers.size() != mult || primes.size() != prim || offsets.size() != prim) {
            err = "engram hash tables do not match " + std::to_string(layer_count) + " layers x " +
                  std::to_string(max_ngram) + "-gram x " + std::to_string(heads) + " heads";
            return false;
        }
        return true;
    }

    /// The hash of position `p` for engram index `ei` (not the layer number): `cols()` row ids into the layer's
    /// table.  `hist` is the token history by absolute position; a position past `hist_len` reads as the pad id,
    /// like the engine's zero-filled history.  `image_mask`, when non-null, marks image positions: an image
    /// position and every older n-gram position it reaches are the pad id, so an image's own tokens never hash
    /// across its edge.  A null mask, or one shorter than the position, reads as text.
    void rows_masked(const int32_t* hist, size_t hist_len, const uint8_t* image_mask, size_t image_mask_len,
                     int64_t p, uint32_t ei, int64_t* out) const {
        const uint32_t g = max_ngram;
        const uint32_t h = heads;
        int64_t prod[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // valid() bounds max_ngram to 8
        bool blocked = false;
        for (uint32_t k = 0; k < g; ++k) {
            const int64_t back = p - static_cast<int64_t>(k);
            int64_t cid;
            if (back < 0 || blocked) {
                cid = pad;
            } else if (image_mask != nullptr && static_cast<size_t>(back) < image_mask_len && image_mask[back] != 0) {
                blocked = true;
                cid = pad;
            } else {
                const int32_t tok = static_cast<size_t>(back) < hist_len ? hist[back] : -1;
                cid = (tok >= 0 && static_cast<uint32_t>(tok) < n_vocab) ? token_map[static_cast<size_t>(tok)]
                                                                        : static_cast<int64_t>(pad);
            }
            // C `(int64_t)((uint64_t)cid * (uint64_t)mult)`: wrap-around unsigned multiply, reinterpreted signed.
            prod[k] = static_cast<int64_t>(static_cast<uint64_t>(cid) * static_cast<uint64_t>(multipliers[static_cast<size_t>(ei) * g + k]));
        }
        int64_t rolling = prod[0];
        size_t o = 0;
        for (uint32_t i = 1; i < g; ++i) {
            rolling ^= prod[i];
            for (uint32_t head = 0; head < h; ++head) {
                const int64_t pr = primes[(static_cast<size_t>(ei) * (g - 1) + (i - 1)) * h + head];
                int64_t r = rolling % pr;
                if (r < 0) r += pr;
                out[o++] = r + offsets[static_cast<size_t>(ei) * (g - 1) * h + (i - 1) * h + head];
            }
        }
    }

    void rows(const int32_t* hist, size_t hist_len, int64_t p, uint32_t ei, int64_t* out) const {
        rows_masked(hist, hist_len, nullptr, 0, p, ei, out);
    }
};

/// Reads the four hash tensors (types 26 and 27) out of the GGUF image.  Returns "" on success, or an error
/// naming the tensor and what is wrong with it.  Nothing is read when the geometry carries no engram layer.
inline std::string load_engram_hash(const GgufFile& g, const Geometry& geo, EngramHash& out) {
    out = EngramHash{};
    const size_t layers = geo.engram_layer_ids.size();
    if (layers == 0) return {};

    const uint32_t g_ngram = static_cast<uint32_t>(geo.engram_max_ngram_size);
    const uint32_t heads = static_cast<uint32_t>(geo.engram_head_count);
    out.max_ngram = g_ngram;
    out.heads = heads;
    out.pad = static_cast<int32_t>(geo.engram_pad_id_compressed);
    out.n_vocab = static_cast<uint32_t>(geo.vocab_size);
    out.token_map.resize(static_cast<size_t>(geo.vocab_size));
    out.multipliers.resize(layers * g_ngram);
    out.primes.resize(layers * (g_ngram - 1) * heads);
    out.offsets.resize(layers * (g_ngram - 1) * heads);

    const uint64_t payload = g.file_size() - g.data_start();
    auto load = [&](const char* name, uint32_t type, uint64_t elems, uint32_t elem_bytes, void* dst,
                    std::string& err) {
        const TensorInfo* t = g.find(name);
        if (t == nullptr) {
            err = "missing tensor '" + std::string(name) + "'";
            return false;
        }
        if (t->type != type) {
            err = "tensor '" + std::string(name) + "' is type " + std::to_string(t->type) + ", expected " +
                  std::to_string(type);
            return false;
        }
        const uint64_t bytes = elems * elem_bytes;
        if (payload_bytes(*t) != bytes) {
            err = "tensor '" + std::string(name) + "' is " + std::to_string(payload_bytes(*t)) + " bytes, expected " +
                  std::to_string(bytes);
            return false;
        }
        if (t->offset > payload || bytes > payload - t->offset) {
            err = "tensor '" + std::string(name) + "' runs past the end of the file";
            return false;
        }
        std::memcpy(dst, g.tensor_data(*t), bytes);
        return true;
    };

    std::string err;
    if (!load("engram.token_map", 26, geo.vocab_size, 4, out.token_map.data(), err) ||
        !load("engram.multipliers", 27, static_cast<uint64_t>(layers) * g_ngram, 8, out.multipliers.data(), err) ||
        !load("engram.primes", 27, static_cast<uint64_t>(layers) * (g_ngram - 1) * heads, 8, out.primes.data(),
              err) ||
        !load("engram.offsets", 27, static_cast<uint64_t>(layers) * (g_ngram - 1) * heads, 8, out.offsets.data(),
              err))
        return "deepseek4: " + err;
    std::string bad;
    if (!out.valid(static_cast<uint32_t>(layers), bad)) return "deepseek4: " + bad;
    return {};
}

// ---- The sidecar shard reader -----------------------------------------------------------------------------
//
// Row `r` of a layer is `head_dim` E4M3 weight bytes at `weight_off + r*head_dim` and `head_dim/32` UE8M0 scale
// bytes at `scale_off + r*(head_dim/32)`.  The planes are separate and far apart (the scale plane starts 98 GB
// into a 101.5 GB shard), so one row costs two reads and random ids do not coalesce.

class EngramShard {
public:
    static constexpr uint32_t kAlign = 4096;   ///< V41_EDIO_ALIGN (core_v41.h:18)
    static uint32_t row_stride(uint32_t head_dim) { return head_dim + head_dim / 32; }

    EngramShard() = default;
    ~EngramShard() { close(); }
    EngramShard(const EngramShard&) = delete;
    EngramShard& operator=(const EngramShard&) = delete;
    EngramShard(EngramShard&& o) noexcept { move_from(o); }
    EngramShard& operator=(EngramShard&& o) noexcept {
        if (this != &o) {
            close();
            move_from(o);
        }
        return *this;
    }

    /// How the shard reads.  `kAuto` is the runtime's path: O_DIRECT first, then a plain read plus
    /// FADV_RANDOM (the fallback is a reproducibility risk the reference names explicitly, not a performance
    /// note).  The rest are the test seams that keep both read paths exercised on any filesystem: `kDirect`
    /// refuses unless the filesystem takes O_DIRECT, `kBuffered` reads the exact span, and `kAligned` uses the
    /// aligned-bounce read (what O_DIRECT needs) over a plain descriptor.
    enum class Io { kAuto, kDirect, kBuffered, kAligned };

    bool open(const std::string& path, std::string& err, Io io = Io::kAuto) {
        close();
#if defined(_WIN32)
        if (io == Io::kDirect) {
            err = "engram shard: O_DIRECT is not available on this platform until P8";
            return false;
        }
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            err = "engram shard: cannot open " + path;
            return false;
        }
        if (_fseeki64(f, 0, SEEK_END) != 0) {
            std::fclose(f);
            err = "engram shard: cannot size " + path;
            return false;
        }
        const long long size = _ftelli64(f);
        if (size < 0 || _fseeki64(f, 0, SEEK_SET) != 0) {
            std::fclose(f);
            err = "engram shard: cannot size " + path;
            return false;
        }
        file_ = f;
        direct_ = false;   // the Windows no-buffering path is P8's; file_ is the page-cache fallback here
        aligned_ = io == Io::kAligned;
        size_ = static_cast<uint64_t>(size);
        path_ = path;
        return true;
#else
        int fd = -1;
        bool direct = false, aligned = io == Io::kAligned;
#if defined(__linux__)
        if (io == Io::kAuto || io == Io::kDirect) {
            fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
            if (fd >= 0) {
                direct = true;
                aligned = true;
            }
        }
        if (fd < 0 && io == Io::kDirect) {
            err = "engram shard: O_DIRECT refused for " + path;
            return false;
        }
#endif
        if (fd < 0) {
            if (io == Io::kDirect) {
                err = "engram shard: O_DIRECT is not available on this platform";
                return false;
            }
            fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        }
        if (fd < 0) {
            err = "engram shard: cannot open " + path + " (errno " + std::to_string(errno) + ")";
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 0) {
            ::close(fd);
            err = "engram shard: cannot size " + path;
            return false;
        }
        if (!direct) (void)posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
        fd_ = fd;
        direct_ = direct;
        aligned_ = aligned;
        size_ = static_cast<uint64_t>(st.st_size);
        path_ = path;
        return true;
#endif
    }

    void close() {
#if defined(_WIN32)
        if (file_ != nullptr) std::fclose(file_);
        file_ = nullptr;
#else
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
        direct_ = false;
        aligned_ = false;
        size_ = 0;
        path_.clear();
        std::vector<uint8_t>().swap(scales_);
        scale_base_ = 0;
        scale_span_ = 0;
    }

    bool is_open() const {
#if defined(_WIN32)
        return file_ != nullptr;
#else
        return fd_ >= 0;
#endif
    }
    bool is_direct() const { return direct_; }      ///< the descriptor really is O_DIRECT
    bool is_aligned() const { return aligned_; }    ///< reads take the aligned-bounce path
    uint64_t size() const { return size_; }
    const std::string& path() const { return path_; }

    /// The engine's span check (`v41_ejob_create`): both planes' rows must end inside the file.  A shard shorter
    /// than the metadata says is refused here, before any read.
    bool check_span(uint64_t weight_off, uint64_t scale_off, uint64_t rows, uint64_t head_dim,
                    std::string& err) const {
        if (!is_open()) {
            err = "engram shard: not open";
            return false;
        }
        if (head_dim == 0 || head_dim % 32 != 0) {
            err = "engram shard: head_dim is not a multiple of 32";
            return false;
        }
        const uint64_t nsc = head_dim / 32;
        if (rows != 0 && (rows * head_dim / head_dim != rows || rows * nsc / nsc != rows)) {
            err = "engram shard: row count overflows";
            return false;
        }
        if (weight_off > size_ || rows * head_dim > size_ - weight_off) {
            err = "engram table runs past the shard (weight plane)";
            return false;
        }
        if (scale_off > size_ || rows * nsc > size_ - scale_off) {
            err = "engram table runs past the shard (scale plane)";
            return false;
        }
        return true;
    }

    /// Opt-in (`--engram-scale-cache`): hold the table's scale plane in host RAM.  A row costs two random 4 KiB
    /// preads -- 256 weight bytes and 8 scale bytes -- and the scale plane is only rows*(head_dim/32) bytes
    /// (3.1 GB a table here), so reading it once in order removes one of the two reads for every row after.
    bool cache_scales(uint64_t scale_off, uint64_t rows, uint32_t head_dim, std::string& err) {
        if (!is_open()) {
            err = "engram shard: not open";
            return false;
        }
        if (head_dim == 0 || head_dim % 32 != 0) {
            err = "engram shard: head_dim is not a multiple of 32";
            return false;
        }
        const uint64_t nsc = head_dim / 32;
        if (rows != 0 && rows * nsc / nsc != rows) {
            err = "engram shard: row count overflows";
            return false;
        }
        const uint64_t plane = rows * nsc;
        if (scale_off > size_ || plane > size_ - scale_off) {
            err = "engram shard: scale plane runs past the shard";
            return false;
        }
        std::vector<uint8_t> buf;
        try {
            buf.resize(static_cast<size_t>(plane));
        } catch (const std::bad_alloc&) {
            err = "engram shard: not enough RAM for the scale cache";
            return false;
        }
#if defined(_WIN32)
        if (!read_exact(buf.data(), static_cast<size_t>(plane), scale_off, err)) return false;
#else
        // A separate buffered descriptor for the one sequential pass: the read has no O_DIRECT alignment
        // rules, and readahead is what this access pattern wants.  The shard's own descriptor stays O_DIRECT
        // for the random 256-byte rows.
        const int bfd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (bfd < 0) {
            err = "engram shard: cannot open the shard for the scale cache";
            return false;
        }
        size_t got = 0;
        while (got < plane) {
            const size_t chunk = std::min<size_t>(static_cast<size_t>(plane - got), size_t{8} << 20);
            const ssize_t n = ::pread(bfd, buf.data() + got, chunk, static_cast<off_t>(scale_off + got));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                ::close(bfd);
                err = "engram shard: scale plane read failed";
                return false;
            }
            got += static_cast<size_t>(n);
        }
        (void)posix_fadvise(bfd, static_cast<off_t>(scale_off), static_cast<off_t>(plane), POSIX_FADV_DONTNEED);
        ::close(bfd);
#endif
        scales_ = std::move(buf);
        scale_base_ = scale_off;
        scale_span_ = plane;
        return true;
    }

    bool scales_cached() const { return !scales_.empty(); }

    /// One row's raw bytes: `head_dim` weights and `head_dim/32` scales.
    bool row(uint64_t weight_off, uint64_t scale_off, uint64_t r, uint32_t head_dim, uint8_t* weights,
             uint8_t* scale, std::string& err) const {
        if (head_dim == 0 || head_dim % 32 != 0) {
            err = "engram shard: head_dim is not a multiple of 32";
            return false;
        }
        const uint64_t hd = head_dim;
        const uint64_t nsc = hd / 32;
        if (r > (UINT64_MAX - weight_off) / hd || r > (UINT64_MAX - scale_off) / nsc) {
            err = "engram shard: row offset overflows";
            return false;
        }
        if (!read_exact(weights, hd, weight_off + r * hd, err)) return false;
        if (!scales_.empty()) {
            // The cache holds the plane from scale_base_: any scale_off inside it maps to an offset.
            if (scale_off < scale_base_ || scale_off - scale_base_ > scale_span_) {
                err = "engram shard: the scale cache does not cover that row";
                return false;
            }
            const uint64_t in = scale_off - scale_base_;
            const uint64_t avail = scale_span_ - in;
            if (avail < nsc || r > (avail - nsc) / nsc) {
                err = "engram shard: the scale cache does not cover that row";
                return false;
            }
            std::memcpy(scale, scales_.data() + in + r * nsc, static_cast<size_t>(nsc));
            return true;
        }
        return read_exact(scale, nsc, scale_off + r * nsc, err);
    }

    /// `n_ids` rows as the forward consumes them: row i lands at out[i*row_stride()] as weights then scales.
    ///
    /// A prompt block's IDs are hashed, so consecutive rows land anywhere in a multi-hundred-gigabyte shard and
    /// one pread at a time leaves the drive's queue empty: measured 9 s per table per 1024 tokens on the
    /// reference box, the single largest prefill cost.  The descriptor is only pread from, so a long call reads
    /// contiguous ID slices on several threads.  Decode-sized calls keep the serial loop; the bytes and the
    /// errors are the same either way.
    bool rows(uint64_t weight_off, uint64_t scale_off, const int64_t* ids, size_t n_ids, uint32_t head_dim,
              uint8_t* out, std::string& err) const {
        const uint32_t stride = row_stride(head_dim);
        auto one = [&](size_t i, std::string& e) -> bool {
            const int64_t r = ids[i];
            if (r < 0) {
                e = "engram shard: negative row id";
                return false;
            }
            return row(weight_off, scale_off, static_cast<uint64_t>(r), head_dim, out + i * stride,
                       out + i * stride + head_dim, e);
        };
        uint32_t nthreads = 1;
#if !defined(_WIN32)
        // The read path is pread on a shared descriptor there; Windows seeks a shared FILE* and stays serial.
        if (n_ids >= 4096) {
            // 8 matched the reference box's QLC drive; STRATA_V4_ENGRAM_ROWS_THREADS overrides for a sweep.
            static const uint32_t rows_threads_env = [] {
                const char* v = std::getenv("STRATA_V4_ENGRAM_ROWS_THREADS");
                if (v != nullptr) {
                    const long t = std::strtol(v, nullptr, 10);
                    if (t >= 1 && t <= 64) return static_cast<uint32_t>(t);
                }
                return 8u;
            }();
            nthreads = rows_threads_env;
            if (nthreads > n_ids / 1024) nthreads = static_cast<uint32_t>(n_ids / 1024);
            if (nthreads == 0) nthreads = 1;
        }
#endif
        if (nthreads <= 1) {
            for (size_t i = 0; i < n_ids; ++i)
                if (!one(i, err)) return false;
            return true;
        }
        const size_t per = (n_ids + nthreads - 1) / nthreads;
        std::atomic<bool> ok{true};
        std::mutex mu;
        std::string first_err;
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (uint32_t t = 0; t < nthreads; ++t) {
            const size_t i0 = std::min(n_ids, static_cast<size_t>(t) * per);
            const size_t i1 = std::min(n_ids, i0 + per);
            if (i0 >= i1) break;
            pool.emplace_back([&, i0, i1]() {
                std::string e;
                for (size_t i = i0; i < i1; ++i) {
                    if (!ok.load(std::memory_order_relaxed)) return;
                    if (!one(i, e)) {
                        std::lock_guard<std::mutex> g(mu);
                        if (ok.exchange(false)) first_err = e;
                        return;
                    }
                }
            });
        }
        for (auto& th : pool) th.join();
        if (!ok.load()) {
            err = first_err;
            return false;
        }
        return true;
    }

private:
    void move_from(EngramShard& o) {
#if defined(_WIN32)
        file_ = o.file_;
        o.file_ = nullptr;
#else
        fd_ = o.fd_;
        o.fd_ = -1;
#endif
        direct_ = o.direct_;
        aligned_ = o.aligned_;
        size_ = o.size_;
        path_ = o.path_;
        scales_ = std::move(o.scales_);
        o.scales_ = {};
        scale_base_ = o.scale_base_;
        scale_span_ = o.scale_span_;
        o.scale_base_ = 0;
        o.scale_span_ = 0;
        o.direct_ = false;
        o.aligned_ = false;
        o.size_ = 0;
    }

    /// C `v41_edio_pread`: an O_DIRECT read starts and ends on a block boundary, so read the aligned superset
    /// into a bounce buffer and copy out the requested span.  A short aligned read is an error, not a partial
    /// result - every engram plane is followed by more of the shard, and the engine refuses the same way.
    bool read_exact(uint8_t* dst, size_t len, uint64_t off, std::string& err) const {
        if (len == 0) return true;
        if (!is_open()) {
            err = "engram shard: not open";
            return false;
        }
        if (off > size_ || len > size_ - off) {
            err = "engram shard: read past the end of the file";
            return false;
        }
#if defined(_WIN32)
        if (_fseeki64(file_, static_cast<long long>(off), SEEK_SET) != 0 ||
            std::fread(dst, 1, len, file_) != len) {
            err = "engram shard: read failed";
            return false;
        }
        return true;
#else
        if (!aligned_) {
            size_t got = 0;
            while (got < len) {
                const ssize_t n = ::pread(fd_, dst + got, len - got, static_cast<off_t>(off + got));
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) {
                    err = "engram shard: read failed";
                    return false;
                }
                got += static_cast<size_t>(n);
            }
            return true;
        }
        const uint64_t a = off & ~(static_cast<uint64_t>(kAlign) - 1);
        const uint64_t span = off - a + len;
        const uint64_t nb = (span + kAlign - 1) / kAlign * kAlign;
        // Per-thread now that rows() reads on several threads: one reused aligned scratch per thread instead of
        // a fresh allocation per pread.
        thread_local std::vector<uint8_t> bounce;
        bounce.resize(static_cast<size_t>(nb) + kAlign);
        uint8_t* buf = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(bounce.data()) + kAlign - 1) &
                                                  ~(static_cast<uintptr_t>(kAlign) - 1));
        size_t got = 0;
        while (got < nb) {
            const ssize_t n = ::pread(fd_, buf + got, static_cast<size_t>(nb - got), static_cast<off_t>(a + got));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                err = "engram shard: O_DIRECT read returned a short aligned span";
                return false;
            }
            got += static_cast<size_t>(n);
        }
        std::memcpy(dst, buf + (off - a), len);
        return true;
#endif
    }

#if defined(_WIN32)
    std::FILE* file_ = nullptr;
#else
    int fd_ = -1;
#endif
    bool direct_ = false;
    bool aligned_ = false;
    uint64_t size_ = 0;
    std::string path_;
    // --engram-scale-cache: the scale plane, read once; empty when the tier is off.
    std::vector<uint8_t> scales_;
    uint64_t scale_base_ = 0;   // the file offset scales_[0] came from
    uint64_t scale_span_ = 0;   // bytes held, from scale_base_
};

/// Opens engram index `ei`'s shard from `dir`: the metadata records the writer's path, so the runtime's
/// `--engram-dir` supplies the directory and the recorded file name is kept.  The span is checked with the
/// metadata's offsets and row count, so a mismatched or truncated shard is refused here.
inline bool open_engram_table(const Geometry& geo, size_t ei, const std::string& dir, EngramShard& shard,
                              std::string& err, bool cache_scales = false) {
    if (ei >= geo.engram_tables.size()) {
        err = "engram index " + std::to_string(ei) + " is not in the geometry";
        return false;
    }
    const EngramTable& t = geo.engram_tables[ei];
    const std::string path = (std::filesystem::path(dir) / std::filesystem::path(t.path).filename()).string();
    std::string e;
    if (!shard.open(path, e)) {
        err = "engram table " + std::to_string(ei) + ": " + e;
        return false;
    }
    if (!shard.check_span(t.weight_offset, t.scale_offset, t.rows, geo.engram_head_dim, e)) {
        err = "engram table " + std::to_string(ei) + ": " + e;
        return false;
    }
    if (cache_scales && !shard.cache_scales(t.scale_offset, t.rows, geo.engram_head_dim, e)) {
        err = "engram table " + std::to_string(ei) + ": " + e;
        return false;
    }
    return true;
}

// ---- The row dequant and the gate -------------------------------------------------------------------------
//
// The on-disk row is E4M3 with one UE8M0 scale per 32 values (ds4_fp8.h semantics), then the engine's RNE bf16
// round.  Every product is exact and both sides run the same conversion (deepseek4_engram_bits.hpp, which the
// device kernel in strata/kernels/deepseek4_engram.cu includes too), so the dequant is bit-exact - and P4's
// gate requires that bit equality, not a tolerance.

/// Decodes `n_rows` raw rows (stride `head_dim + head_dim/32`) to the bf16 grid, `n_rows * head_dim` floats.
inline bool engram_rows_f32(const uint8_t* raw, uint32_t n_rows, uint32_t head_dim, float* out) {
    if (raw == nullptr || out == nullptr || head_dim == 0 || head_dim % 32 != 0) return false;
    const uint32_t stride = EngramShard::row_stride(head_dim);
    for (uint32_t r = 0; r < n_rows; ++r) {
        const uint8_t* row = raw + static_cast<size_t>(r) * stride;
        for (uint32_t d = 0; d < head_dim; ++d) out[static_cast<size_t>(r) * head_dim + d] = engram_row_value(row, head_dim, d);
    }
    return true;
}

/// The gate of the official Engram.forward, in f64: for every (token, hyper-connection row) it computes
/// sh = sum h^2, sk = sum key^2, sd = sum h*(qw*kw)*key, then
///   rstd = rsqrt(sh/E + eps) * rsqrt(sk/E + eps), dot = sd * rstd * rsqrt(E),
///   mag = sqrt(max(|dot|, 1e-6)), gate = sigmoid(copysign(mag, dot)), out[d] = bf16r(h[d] + gate*val[d]).
/// This is the reference emulation (`ds41_engram_ref.rs`) and the oracle the CUDA kernel is judged against; the
/// device's f32 lane/tree sums and its expf differ, which is why the compare there is at bf16-ulp distance.
/// `out` may alias `hc` (the sums are finished before any element is written).
inline void engram_gate_emulate(const float* hc, const float* kv, const float* qw, const float* kw, uint32_t e,
                                uint32_t n_hc, uint32_t n_tok, float eps, float* out) {
    const double de = static_cast<double>(e);
    for (uint32_t t = 0; t < n_tok; ++t) {
        for (uint32_t c = 0; c < n_hc; ++c) {
            const float* hrow = hc + (static_cast<size_t>(t) * n_hc + c) * e;
            const float* krow = kv + (static_cast<size_t>(t) * (n_hc + 1) + c) * e;
            const float* vrow = kv + (static_cast<size_t>(t) * (n_hc + 1) + n_hc) * e;
            const float* qc = qw + static_cast<size_t>(c) * e;
            const float* kc = kw + static_cast<size_t>(c) * e;
            double sh = 0.0, sk = 0.0, sd = 0.0;
            for (uint32_t d = 0; d < e; ++d) {
                const double w = static_cast<double>(qc[d]) * static_cast<double>(kc[d]);
                const double hv = static_cast<double>(hrow[d]);
                const double kvv = static_cast<double>(krow[d]);
                sh += hv * hv;
                sk += kvv * kvv;
                sd += hv * w * kvv;
            }
            const double rstd = (1.0 / std::sqrt(sh / de + static_cast<double>(eps))) *
                                (1.0 / std::sqrt(sk / de + static_cast<double>(eps)));
            const double dot = sd * rstd / std::sqrt(de);
            const double mag = std::sqrt(std::fmax(std::fabs(dot), 1e-6));
            const double z = std::copysign(mag, dot);
            const double gate = 1.0 / (1.0 + std::exp(-z));
            float* orow = out + (static_cast<size_t>(t) * n_hc + c) * e;
            for (uint32_t d = 0; d < e; ++d) {
                const float v = static_cast<float>(static_cast<double>(hrow[d]) + gate * static_cast<double>(vrow[d]));
                orow[d] = engram_bf16r(v);
            }
        }
    }
}

}  // namespace strata::deepseek4
