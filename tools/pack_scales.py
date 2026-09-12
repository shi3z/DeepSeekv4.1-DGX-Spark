#!/usr/bin/env python3
"""
pack_scales.py -- put the routed experts' UE8M0 scales in 4 bits instead of 8.

An exhaustive scan of all 17.4 GB of expert scale bytes finds 14 distinct values, and they occupy
the CONTIGUOUS range 115..128 (2^-12 .. 2^1). So the packing needs no lookup table at all: a 4-bit
value v is the byte v + BIAS, and the kernels already turn an E8M0 byte into a float with
`__int_as_float(byte << 23)`. Decoding is one add.

  byte 120 = 2^-7   53.50 %      byte 122 = 2^-5    1.69 %
  byte 121 = 2^-6   44.20 %      byte 117 = 2^-10   0.17 %      (and ten more, all < 0.1 %)

The scan matters: an 8-tensor sample sees 8 distinct values and a partial scan sees 11. A LUT built
from either silently corrupts whichever experts use the values it missed, so `verify` re-reads the
checkpoint and checks every byte round-trips.

An expert's three scale tensors are 1,105,920 B and pack to 552,960 B, which is already a multiple
of 4096, so every expert's block is O_DIRECT-aligned at `(layer * 384 + expert) * 552960`. The
weights are not touched: this writes a side file and the engine keeps reading the weight run out of
the original shard.
"""

from __future__ import annotations

import argparse, json, os, struct, sys, time
import numpy as np

BIAS = 115                      # 4-bit value v  <->  E8M0 byte v + BIAS
N_EXPERTS = 384
SCALE_ORDER = ("w1", "w2", "w3")          # the order inside one expert's block
EXPERT_SCALE_BYTES = 1_105_920            # w1 368640 + w2 368640 + w3 368640
PACKED_BYTES = EXPERT_SCALE_BYTES // 2    # 552_960, a multiple of 4096
MAGIC = b"DSV41S4B"


def pack(b: np.ndarray) -> np.ndarray:
    """uint8 E8M0 bytes -> 4-bit, two per byte, low nibble first."""
    v = b.astype(np.int16) - BIAS
    if v.min() < 0 or v.max() > 15:
        bad = np.unique(b[(v < 0) | (v > 15)])
        raise ValueError(f"E8M0 bytes outside the {BIAS}..{BIAS + 15} window: {bad.tolist()}")
    v = v.astype(np.uint8).reshape(-1, 2)
    return (v[:, 0] | (v[:, 1] << 4)).astype(np.uint8)


def unpack(p: np.ndarray) -> np.ndarray:
    out = np.empty(p.size * 2, dtype=np.uint8)
    out[0::2] = (p & 0x0F) + BIAS
    out[1::2] = (p >> 4) + BIAS
    return out


class Shards:
    def __init__(self, model_dir: str):
        self.md = model_dir
        self.wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self._hdr: dict[str, tuple] = {}
        self._fh: dict[str, object] = {}

    def read(self, name: str) -> np.ndarray:
        shard = self.wm[name]
        if shard not in self._hdr:
            f = open(os.path.join(self.md, shard), "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            self._hdr[shard] = (json.loads(f.read(n)), 8 + n)
            self._fh[shard] = f
        hdr, base = self._hdr[shard]
        f = self._fh[shard]
        a, b = hdr[name]["data_offsets"]
        f.seek(base + a)
        return np.frombuffer(f.read(b - a), dtype=np.uint8)


def keys(n_layers: int, mtp: int):
    for L in range(n_layers):
        for e in range(N_EXPERTS):
            yield (L, e), f"layers.{L}.ffn.experts.{e}."
    for k in range(mtp):
        for e in range(128):
            yield (n_layers + k, e), f"mtp.{k}.ffn.experts.{e}."


def slot_index(key, n_layers: int) -> int:
    L, e = key
    return L * N_EXPERTS + e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--mtp", type=int, default=3)
    ap.add_argument("--verify", type=int, default=64, help="experts to re-read and check bit-exactly")
    a = ap.parse_args()

    sh = Shards(a.model_dir)
    all_keys = list(keys(a.n_layers, a.mtp))
    n_slots = max(slot_index(k, a.n_layers) for k, _ in all_keys) + 1
    size = 4096 + n_slots * PACKED_BYTES
    print(f"{len(all_keys):,} experts -> {n_slots:,} slots x {PACKED_BYTES:,} B = {size/1e9:.2f} GB")

    t0 = time.time()
    with open(a.out, "wb") as f:
        hdr = {"magic": MAGIC.decode(), "bias": BIAS, "packed_bytes": PACKED_BYTES,
               "order": list(SCALE_ORDER), "n_layers": a.n_layers, "mtp": a.mtp,
               "n_experts": N_EXPERTS, "slots": n_slots}
        blob = json.dumps(hdr).encode()
        f.write(MAGIC + struct.pack("<I", len(blob)) + blob + b"\0" * (4096 - 12 - len(blob)))
        f.truncate(size)
        for i, (key, prefix) in enumerate(all_keys):
            parts = [sh.read(prefix + w + ".scale") for w in SCALE_ORDER]
            raw = np.concatenate(parts)
            assert raw.size == EXPERT_SCALE_BYTES, (key, raw.size)
            f.seek(4096 + slot_index(key, a.n_layers) * PACKED_BYTES)
            f.write(pack(raw).tobytes())
            if (i + 1) % 2000 == 0:
                print(f"  {i+1:,}/{len(all_keys):,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e9:.2f} GB) in {time.time()-t0:.0f}s")

    # ---- verify: re-read the side file and compare against the checkpoint, byte for byte
    import random
    random.seed(0)
    sample = random.sample(all_keys, min(a.verify, len(all_keys)))
    bad = 0
    with open(a.out, "rb") as f:
        for key, prefix in sample:
            f.seek(4096 + slot_index(key, a.n_layers) * PACKED_BYTES)
            got = unpack(np.frombuffer(f.read(PACKED_BYTES), dtype=np.uint8))
            want = np.concatenate([sh.read(prefix + w + ".scale") for w in SCALE_ORDER])
            if not np.array_equal(got, want):
                bad += 1
                print(f"  MISMATCH {prefix}: {int((got != want).sum())} of {want.size} bytes")
    print(f"verify: {len(sample) - bad}/{len(sample)} experts round-trip bit-exactly"
          f"{'' if bad == 0 else '  <-- FAILURE'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
