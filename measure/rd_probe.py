"""Rate-distortion of the expert weights at ~2 bits, without any calibration.

AQLM's advantage over a scalar codebook is vector quantisation: coding groups of weights jointly.
That part is measurable on the weights alone, in minutes, with no calibration data. If joint coding
is much better than scalar here, AQLM has room to work and the calibration time is worth spending;
if it is not, these already-FP4 weights have no 2-bit structure to find and data-aware calibration
would have to supply all of it.

Distortion is relative to the dequantised FP4 the checkpoint stores, which is what any 2-bit scheme
has to reproduce. At 2 bits per weight a group of `dim` weights needs 2**(2*dim) centroids, so the
centroid update has to be a scatter-add rather than a loop over k: 4,096 at dim=6.
"""
import json, os, struct, sys, time
import numpy as np
import torch

MD = "models/DeepSeek-V4.1-Flash"
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
_h = {}


def rd(name):
    sh = idx[name]
    p = os.path.join(MD, sh)
    if sh not in _h:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            _h[sh] = (json.loads(f.read(n)), 8 + n)
    hdr, base = _h[sh]
    a, b = hdr[name]["data_offsets"]
    with open(p, "rb") as f:
        f.seek(base + a)
        return np.frombuffer(f.read(b - a), dtype=np.uint8)


FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])


def dequant(w, s, K):
    codes = np.stack([w & 0xF, w >> 4], -1).reshape(w.shape[0], -1)
    v = FP4[torch.from_numpy(codes.astype(np.int64))]
    sc = torch.exp2(torch.from_numpy(s.astype(np.int16)).float() - 127.0)
    return (v.view(w.shape[0], -1, 32) * sc[:, :, None]).view(w.shape[0], K)


def nrmse(a, b):
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a))


def kmeans1d(x, k, iters=20):
    """per-row scalar k-means; x [N, K] -> the quantised x"""
    q = torch.quantile(x, torch.linspace(0, 1, k + 2, device=x.device)[1:-1], dim=1).T.contiguous()
    for _ in range(iters):
        a = (x[:, :, None] - q[:, None, :]).abs().argmin(-1)
        oh = torch.nn.functional.one_hot(a, k).to(x.dtype)
        cnt = oh.sum(1)
        s = torch.einsum("nk,nkj->nj", x, oh)
        q = torch.where(cnt > 0, s / cnt.clamp_min(1), q)
    a = (x[:, :, None] - q[:, None, :]).abs().argmin(-1)
    return torch.gather(q, 1, a)


def vq(x, dim, bits, iters=12, sample=100_000, chunk=4096):
    """vector-quantise groups of `dim` weights with 2**bits centroids, one codebook per matrix."""
    g = x.reshape(-1, dim)
    k = 1 << bits
    perm = torch.randperm(g.shape[0], device=g.device)
    sel = g[perm[:sample]]
    c = sel[torch.randperm(sel.shape[0], device=g.device)[:k]].clone()

    def assign(t):
        out = torch.empty(t.shape[0], dtype=torch.long, device=t.device)
        for i in range(0, t.shape[0], chunk):
            out[i:i + chunk] = torch.cdist(t[i:i + chunk], c).argmin(1)
        return out

    for _ in range(iters):
        a = assign(sel)
        num = torch.zeros_like(c)
        cnt = torch.zeros(k, device=g.device, dtype=c.dtype)
        num.index_add_(0, a, sel)
        cnt.index_add_(0, a, torch.ones(a.shape[0], device=g.device, dtype=c.dtype))
        keep = cnt > 0
        c[keep] = num[keep] / cnt[keep, None]
    out = torch.empty_like(g)
    for i in range(0, g.shape[0], chunk):
        out[i:i + chunk] = c[torch.cdist(g[i:i + chunk], c).argmin(1)]
    return out.reshape(x.shape)


dev = "cuda"
hdr = f"{'tensor':16s} {'CB2-style':>10s} {'free 1D':>9s} {'VQ pair':>9s} {'VQ quad':>9s} {'VQ 6-tup':>9s}"
print(hdr)
print(f"{'':16s} {'2.25 bpw':>10s} {'2.25 bpw':>9s} {'2.0 bpw':>9s} {'2.0 bpw':>9s} {'2.0 bpw':>9s}")
rows = []
for (L, e, w, N, K) in ((0, 0, "w1", 2304, 5120), (20, 5, "w1", 2304, 5120), (39, 2, "w2", 5120, 2304)):
    t0 = time.time()
    W = rd(f"layers.{L}.ffn.experts.{e}.{w}.weight")
    S = rd(f"layers.{L}.ffn.experts.{e}.{w}.scale")
    x = dequant(W.reshape(N, -1), S.reshape(N, -1), K).to(dev)
    sc = torch.exp2(torch.from_numpy(S.reshape(N, -1).astype(np.int16)).float() - 127.0).to(dev)
    xs = (x.view(N, -1, 32) / sc[:, :, None]).view(N, K)     # grid units
    grid = FP4.to(dev)
    q = kmeans1d(xs, 4)
    q = grid[(q[:, :, None] - grid[None, None, :]).abs().argmin(-1)]   # snap to the FP4 grid
    cb2 = (q.view(N, -1, 32) * sc[:, :, None]).view(N, K)
    r = [nrmse(x, cb2), nrmse(x, kmeans1d(x, 4)),
         nrmse(x, vq(x, 2, 4)), nrmse(x, vq(x, 4, 8)), nrmse(x, vq(x, 6, 12))]
    rows.append(r)
    print(f"L{L:02d}.e{e}.{w:3s}{'':6s} {r[0]:10.4f} {r[1]:9.4f} {r[2]:9.4f} {r[3]:9.4f} {r[4]:9.4f}"
          f"   ({time.time()-t0:.0f}s)", flush=True)
m = np.array(rows).mean(0)
print(f"\n{'mean':16s} {m[0]:10.4f} {m[1]:9.4f} {m[2]:9.4f} {m[3]:9.4f} {m[4]:9.4f}")
print(f"\n  free scalar vs the FP4-grid codebook : {m[0]/m[1]:.2f}x lower error")
print(f"  6-tuple VQ vs the FP4-grid codebook  : {m[0]/m[4]:.2f}x lower error")
print(f"  6-tuple VQ vs free scalar            : {m[1]/m[4]:.2f}x lower error   <- this is the part")
print(f"       AQLM's vector quantisation buys before any calibration or residual codebook.")
