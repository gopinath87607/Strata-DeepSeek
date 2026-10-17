#!/usr/bin/env python3
"""Write a metadata-only GGUF (0 tensors) from the header of a DeepSeek-V4.1-Flash GGUF.

The tokenizer, the geometry and the inventory all live in the metadata, which ends a few megabytes into
the 113.6 GB file.  This copies that block byte for byte into a small GGUF that the reference tokenizer
oracle and `tools/strata_tokenizer.py` can open, so the tokenizer half of P1 is testable without the
weights.  It reads only as much of the source as the metadata needs, so a header prefix (the first 64 MiB)
is enough.

    python3 tools/dsv4/make_tokenizer_gguf.py <model.gguf-or-prefix> <out.gguf>
    python3 tools/strata_tokenizer.py --gguf <out.gguf> --out /tmp/ds4-pack --check

The output is not a model: it has no tensors and cannot be loaded by an engine.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

# GGUF metadata value types -> struct format.  7 (bool), 8 (string) and 9 (array) are handled separately.
FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}


class NeedMore(Exception):
    """The buffer ends before the metadata does: read more of the source and try again."""


class Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise NeedMore()
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def num(self, t: int):
        fmt = FIXED[t]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = struct.unpack("<Q", self.take(8))[0]
        return self.take(n).decode("utf-8")

    def value(self, t: int):
        if t in FIXED:
            return self.num(t)
        if t == 7:
            return self.take(1) != b"\0"
        if t == 8:
            return self.string()
        if t == 9:
            et, n = struct.unpack("<I", self.take(4))[0], struct.unpack("<Q", self.take(8))[0]
            for _ in range(n):
                self.value(et)
            return None
        raise ValueError("metadata value type %d is not a GGUF type" % t)


def metadata_block(data: bytes) -> tuple[bytes, int, int, int]:
    """Return (raw KV block, version, n_tensors, n_keys) or raise NeedMore."""
    r = Reader(data)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF (no magic)")
    version = struct.unpack("<I", r.take(4))[0]
    n_tensors = struct.unpack("<Q", r.take(8))[0]
    n_kv = struct.unpack("<Q", r.take(8))[0]
    start = r.pos
    for _ in range(n_kv):
        r.string()
        r.value(struct.unpack("<I", r.take(4))[0])
    return data[start:r.pos], version, n_tensors, n_kv


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.exit(__doc__)
    src, out = Path(argv[1]), Path(argv[2])
    with src.open("rb") as f:
        chunk, data = 8 << 20, b""
        while True:
            more = f.read(chunk)
            data += more
            try:
                kv, version, n_tensors, n_kv = metadata_block(data)
                break
            except NeedMore:
                if not more:
                    sys.exit("%s ends before the metadata does (%d bytes read)" % (src, len(data)))
                chunk = min(chunk * 2, 256 << 20)
        if n_tensors:
            print("note: the source declares %d tensors; only their names and shapes are read, and the "
                  "metadata-only output drops them" % n_tensors)
    body = b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", 0) + struct.pack("<Q", n_kv) + kv
    out.write_bytes(body)
    print("wrote %s (%d bytes): GGUF v%d, %d metadata keys, 0 tensors" % (out, len(body), version, n_kv))
    print("metadata ends at %d in the source" % (len(kv) + 4 + 4 + 8 + 8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
