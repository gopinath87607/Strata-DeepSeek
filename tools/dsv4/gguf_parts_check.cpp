// tools/dsv4/gguf_parts_check.cpp - check a byte-split DeepSeek-V4.1-Flash download without assembling it.
//
// The 113.6 GB GGUF is published as `name.partNN-of-MM` files that are byte chunks of one file (not GGUF's own
// per-shard naming), so no single file can be opened by the loader until it is assembled - which needs another
// 113.6 GB.  This tool reads the header from part 1 and checks every tensor's range against the length the
// parts add up to, then reads each expert blob's ~7 MB container across the part boundaries and runs the
// loader's own container gate (check_blob) on it.  Payload bytes of the non-blob tensors are not read; the
// inventory checks their ranges and the blobs' containers carry the structural checks.
//
// Build:
//   g++ -O2 -std=c++17 -Iinclude tools/dsv4/gguf_parts_check.cpp -o /tmp/dsv4_gguf_parts
// Run:
//   /tmp/dsv4_gguf_parts /mnt/nvme/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf.part01-of-40
//
// The part names are derived from the first one; every part must exist and be the same size except the last.

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "strata/artifact/deepseek4.hpp"

namespace ds = strata::deepseek4;

namespace {

struct Part {
    std::string path;
    int fd = -1;
    uint64_t size = 0;
    uint64_t begin = 0;   // logical offset of this part in the assembled file
};

std::string base_name(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// "stem.partNN-of-MM" -> stem, NN, MM and the digit width NN was written with.
bool split_part_name(const std::string& path, std::string* stem, unsigned* first, unsigned* total, int* width) {
    const size_t dot = path.rfind(".part");
    if (dot == std::string::npos) return false;
    const char* p = path.c_str() + dot;
    int n = 0;
    unsigned a = 0, b = 0;
    if (std::sscanf(p, ".part%u-of-%u%n", &a, &b, &n) != 2 || p[n] != '\0') return false;
    if (a == 0 || b == 0 || a > b) return false;
    const char* digits = p + 5;   // after ".part"
    int w = 0;
    while (digits[w] >= '0' && digits[w] <= '9') w++;
    if (w < 1 || w > 6) return false;   // a part count is not a six-digit number
    *stem = path.substr(0, dot);
    *first = a;
    *total = b;
    *width = w;
    return true;
}

std::string part_path(const std::string& stem, unsigned i, unsigned total, int width) {
    char buf[64];
    std::snprintf(buf, sizeof buf, ".part%0*u-of-%0*u", width, i, width, total);
    return stem + buf;
}

// Copies `len` bytes at assembled offset `off` into `dst`, walking across the parts.
bool read_logical(const std::vector<Part>& parts, uint64_t off, uint64_t len, uint8_t* dst, std::string& err) {
    while (len != 0) {
        // The part holding `off`: the last one whose begin is <= off.
        const auto it = std::upper_bound(parts.begin(), parts.end(), off,
                                         [](uint64_t o, const Part& p) { return o < p.begin; });
        if (it == parts.begin()) {
            err = "offset " + std::to_string(off) + " is before the first part";
            return false;
        }
        const Part& p = *(it - 1);
        const uint64_t within = off - p.begin;
        if (within >= p.size) {
            err = "offset " + std::to_string(off) + " is in the gap after " + p.path;
            return false;
        }
        const uint64_t take = std::min<uint64_t>(len, p.size - within);
        size_t done = 0;
        while (done < take) {
            const ssize_t got = ::pread(p.fd, dst + done, take - done, static_cast<off_t>(within + done));
            if (got < 0) {
                err = "pread " + p.path + ": " + std::strerror(errno);
                return false;
            }
            if (got == 0) {
                err = "short read at " + std::to_string(within + done) + " in " + p.path;
                return false;
            }
            done += static_cast<size_t>(got);
        }
        dst += take;
        off += take;
        len -= take;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <first part, name.part01-of-MM>\n", argv[0]);
        return 2;
    }
    std::string stem;
    unsigned first = 0, total = 0;
    int width = 0;
    if (!split_part_name(argv[1], &stem, &first, &total, &width)) {
        std::fprintf(stderr, "%s: not a .partNN-of-MM name\n", argv[1]);
        return 2;
    }
    if (first != 1) {
        std::fprintf(stderr, "%s: this is part %u; give part 1\n", argv[1], first);
        return 2;
    }

    std::vector<Part> parts;
    parts.reserve(total);
    int failures = 0;
    uint64_t chunk = 0;
    for (unsigned i = 1; i <= total; i++) {
        Part p;
        p.path = part_path(stem, i, total, width);
        struct stat st {};
        if (::stat(p.path.c_str(), &st) != 0) {
            std::fprintf(stderr, "MISSING %s\n", p.path.c_str());
            failures++;
            continue;
        }
        p.size = static_cast<uint64_t>(st.st_size);
        if (i == 1) chunk = p.size;
        parts.push_back(std::move(p));
    }
    if (failures != 0) {
        std::printf("%d part errors\n", failures);
        return 1;
    }
    // The uploader's split is even with the remainder given to the leading parts: floor(total/n) bytes each,
    // and the first total%n parts one byte longer.  Any other size pattern means a missing or short part.
    uint64_t assembled = 0;
    for (const Part& p : parts) assembled += p.size;
    const uint64_t even = assembled / total, extra = assembled % total;
    for (unsigned i = 0; i < total; i++) {
        const uint64_t want = even + (i < extra ? 1u : 0u);
        if (parts[i].size != want) {
            std::fprintf(stderr,
                         "ERROR %s is %llu bytes; an even split of %llu over %u parts wants %llu\n",
                         base_name(parts[i].path).c_str(), (unsigned long long) parts[i].size,
                         (unsigned long long) assembled, total, (unsigned long long) want);
            failures++;
        }
    }
    if (failures != 0) {
        std::printf("%d part errors\n", failures);
        return 1;
    }
    chunk = even;
    assembled = 0;
    for (Part& p : parts) {
        p.begin = assembled;
        assembled += p.size;
        p.fd = ::open(p.path.c_str(), O_RDONLY);
        if (p.fd < 0) {
            std::fprintf(stderr, "cannot open %s: %s\n", p.path.c_str(), std::strerror(errno));
            return 1;
        }
    }
    std::printf("parts: %u, chunk %llu B (%llu B in the first %llu), assembled %llu B\n", total,
                (unsigned long long) chunk, (unsigned long long) (chunk + (extra ? 1 : 0)),
                (unsigned long long) extra, (unsigned long long) assembled);

    try {
        // The header is parsed from part 1 alone; every read below is bounded by the header's own sizes.
        strata::GgufFile g(parts[0].path);
        if (parts[0].size < g.data_start()) {
            std::fprintf(stderr, "part 1 ends before the GGUF data section starts\n");
            return 1;
        }
        if (strata::model_family(g) != strata::ModelFamily::Deepseek4) {
            std::fprintf(stderr, "not a deepseek4 file\n");
            return 1;
        }
        ds::Geometry geo;
        const std::string geom_err = ds::read_geometry(g, geo);
        if (!geom_err.empty()) {
            std::fprintf(stderr, "%s\n", geom_err.c_str());
            return 1;
        }

        const ds::Inventory inv = ds::check_inventory(g, geo, /*allow_partial=*/false, assembled);
        std::printf("inventory: %llu tensors against the assembled length, %llu beyond its end\n",
                    (unsigned long long) inv.tensors, (unsigned long long) inv.beyond_eof);
        std::printf("  types:");
        for (const auto& kv : inv.type_counts) std::printf(" %s x%llu", ds::type_name(kv.first),
                                                           (unsigned long long) kv.second);
        std::printf("\n");
        for (const std::string& e : inv.errors) std::printf("  ERROR %s\n", e.c_str());
        failures += static_cast<int>(inv.errors.size());

        // The data section's own end: the last tensor's payload end, and what the split leaves after it.
        uint64_t max_end = g.data_start();
        for (const strata::TensorInfo& t : g.tensors()) max_end = std::max(max_end, g.data_start() + t.offset +
                                                                                        ds::payload_bytes(t));
        std::printf("data section ends at %llu, parts leave %llu B after it\n", (unsigned long long) max_end,
                    (unsigned long long) (assembled >= max_end ? assembled - max_end : 0));
        if (assembled < max_end) {
            std::printf("  ERROR the parts are shorter than the data section\n");
            failures++;
        }

        std::printf("expert blobs\n");
        uint64_t checked = 0;
        for (const strata::TensorInfo& t : g.tensors()) {
            if (t.type != ds::kTypeVqBlob) continue;
            uint32_t layer = 0;
            bool main = false;
            if (std::sscanf(t.name.c_str(), "blk.%u.ffn_exps_vq.blob", &layer) == 1)
                main = true;
            else if (std::sscanf(t.name.c_str(), "mtp.%u.ffn_exps_vq.blob", &layer) != 1) {
                std::printf("  ERROR %s: type-42 tensor whose name the loader cannot place\n", t.name.c_str());
                failures++;
                continue;
            }
            const uint64_t bytes = ds::payload_bytes(t);
            std::vector<uint8_t> buf(bytes);
            std::string err;
            if (!read_logical(parts, g.data_start() + t.offset, bytes, buf.data(), err)) {
                std::printf("  ERROR %s: %s\n", t.name.c_str(), err.c_str());
                failures++;
                continue;
            }
            ds::BlobInfo info;
            if (!ds::check_blob(buf.data(), bytes, layer, main, geo, &info, err)) {
                std::printf("  ERROR %s: %s\n", t.name.c_str(), err.c_str());
                failures++;
                continue;
            }
            checked++;
            std::printf("  %s: v%u layer %u, %u experts, nc %u (%u-bit), %llu payloads checked across parts\n",
                        t.name.c_str(), info.version, info.layer, info.experts, info.nc, info.nbit,
                        (unsigned long long) info.payloads_checked);
        }
        std::printf("  %llu blobs checked\n", (unsigned long long) checked);
        if (checked != 43) {
            std::printf("  ERROR expected 43 expert blobs, found %llu\n", (unsigned long long) checked);
            failures++;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        for (Part& p : parts)
            if (p.fd >= 0) ::close(p.fd);
        return 1;
    }

    for (Part& p : parts)
        if (p.fd >= 0) ::close(p.fd);
    if (failures == 0) {
        std::printf("OK\n");
        return 0;
    }
    std::printf("%d errors\n", failures);
    return 1;
}
