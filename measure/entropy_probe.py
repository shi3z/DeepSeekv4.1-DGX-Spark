"""What can be squeezed out of the experts losslessly?

Before writing a codec, measure the ceiling. Shannon entropy of the FP4 nibble stream sets the
limit for any order-0 entropy coder (Huffman/rANS/FSE); the zstd/lz4 numbers say what an
off-the-shelf dictionary coder finds on top of that. Anything above ~0.9x is not worth decoding.
"""
import json, os, sys, struct, collections, time
import numpy as np

MD = "models/DeepSeek-V4.1-Flash"
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]

_hdr_cache = {}
def read_tensor(name):
    shard = idx[name]
    path = os.path.join(MD, shard)
    if shard not in _hdr_cache:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            _hdr_cache[shard] = (json.loads(f.read(n)), 8 + n)
    hdr, base = _hdr_cache[shard]
    t = hdr[name]
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        return np.frombuffer(f.read(b - a), dtype=np.uint8), t["shape"], t["dtype"]

def H(sym, k):
    c = np.bincount(sym, minlength=k).astype(np.float64)
    p = c / c.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def nibbles(b):
    return np.concatenate([b & 0xF, b >> 4])

import zstandard, lz4.frame
def ratios(buf):
    out = {}
    for lvl in (1, 3, 9):
        out[f"zstd{lvl}"] = len(zstandard.ZstdCompressor(level=lvl).compress(buf)) / len(buf)
    out["lz4"] = len(lz4.frame.compress(buf)) / len(buf)
    return out

LAYERS = [0, 13, 27, 39]
EXPERTS = [0, 1, 2, 7]
print(f"{'tensor':28s} {'MB':>6s} {'nib H':>6s} {'byte H':>7s} {'zstd1':>6s} {'zstd3':>6s} {'zstd9':>6s} {'lz4':>6s}")
tot = collections.defaultdict(float); totb = 0.0
for L in LAYERS:
    for e in EXPERTS[:2]:
        for w in ("w1", "w2", "w3"):
            nm = f"layers.{L}.ffn.experts.{e}.{w}.weight"
            if nm not in idx: continue
            b, shape, dt = read_tensor(nm)
            nh = H(nibbles(b), 16); bh = H(b, 256)
            r = ratios(b.tobytes())
            print(f"L{L:02d}.e{e}.{w}{'':12s} {len(b)/1e6:6.2f} {nh:6.3f} {bh:7.3f} "
                  f"{r['zstd1']:6.3f} {r['zstd3']:6.3f} {r['zstd9']:6.3f} {r['lz4']:6.3f}")
            for k, v in r.items(): tot[k] += v * len(b)
            tot["nibH"] += nh / 4 * len(b); totb += len(b)
print(f"\nweight payload, weighted over {totb/1e6:.0f} MB:")
for k in ("nibH", "zstd1", "zstd3", "zstd9", "lz4"):
    print(f"  {k:6s} {tot[k]/totb:.4f}x" + ("   <- order-0 entropy limit" if k == "nibH" else ""))

print("\n--- UE8M0 scales ---")
st = collections.defaultdict(float); stb = 0.0
for L in LAYERS:
    for e in EXPERTS[:2]:
        for w in ("w1", "w2", "w3"):
            nm = f"layers.{L}.ffn.experts.{e}.{w}.scale"
            if nm not in idx: continue
            b, shape, dt = read_tensor(nm)
            a = b.astype(np.int16)
            d = np.diff(a.reshape(shape[0], -1), axis=1).ravel() + 255
            h_abs, h_del = H(b, 256), H(d.astype(np.int64), 512)
            r = ratios(b.tobytes())
            rd = ratios((np.diff(a.reshape(shape[0], -1), axis=1).astype(np.int8)).tobytes())
            print(f"L{L:02d}.e{e}.{w}.scale{'':6s} {len(b)/1e6:6.3f} absH {h_abs:5.2f} deltaH {h_del:5.2f} "
                  f"zstd3 {r['zstd3']:.3f} | delta+zstd3 {rd['zstd3']:.3f}")
            st["abs"] += r["zstd3"] * len(b); st["delta"] += rd["zstd3"] * len(b)
            st["absH"] += h_abs / 8 * len(b); st["delH"] += h_del / 8 * len(b); stb += len(b)
print(f"\nscales, weighted over {stb/1e6:.2f} MB:  absH {st['absH']/stb:.3f}x  deltaH {st['delH']/stb:.3f}x  "
      f"zstd3 {st['abs']/stb:.3f}x  delta+zstd3 {st['delta']/stb:.3f}x")

print("\n--- adjacent-expert XOR (same layer, same matrix) ---")
for L in (0, 27):
    for w in ("w1", "w2"):
        a, _, _ = read_tensor(f"layers.{L}.ffn.experts.0.{w}.weight")
        b, _, _ = read_tensor(f"layers.{L}.ffn.experts.1.{w}.weight")
        x = (a ^ b)
        zn = float((nibbles(x) == 0).mean())
        print(f"L{L:02d}.{w}: XOR nibble-zero {zn:.4f} (uncorrelated = 0.0625), "
              f"nib H {H(nibbles(x),16):.3f}, zstd3 {ratios(x.tobytes())['zstd3']:.3f}")
