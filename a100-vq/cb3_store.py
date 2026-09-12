"""Read pre-packed CB3 expert slots straight off NVMe, so a miss costs neither the FP4->CB3 pack
nor the wider FP4 read.

Measured on gx10-b872 (unpruned stream, ARENA_GB=94, DSV41_BLOCK=1):

    FP4 arena            108.7 ms/tok   NVMe 0.233 GB/tok   hit 0.9585    9.16 tok/s
    CB3 arena, pack on   148.5 ms/tok   NVMe 0.069 GB/tok   hit 0.9872    6.71 tok/s
      every miss                        (the CB3 arena holds 31 % more experts, so the traffic
                                         collapses -- and the pack, 20.8 ms/expert, more than
                                         eats it: ~3.7 misses/token is ~76 ms of fill)

This module removes the fill: the record written by a100-vq/pack_store.py is exactly the 12 slot
tensors of cb3_moe.CB3ArenaV2 concatenated, so a miss is one O_DIRECT pread of 14,454,784 B (a
multiple of 4096, so no alignment slack) into a pinned buffer plus 12 H2D slice copies.

Opt-in: the engine only attaches a store when DSV41_CB3_STORE names one, so nothing changes for a
run that does not ask for it.
"""

from __future__ import annotations

import json
import os
import threading

import torch

ALIGN = 4096


class CB3Store:
    def __init__(self, path: str, device, punched_path: str = ""):
        meta = json.load(open(path + ".json"))
        assert meta["format"] == "cb3_v2", meta["format"]
        self.stride = int(meta["stride"])
        assert self.stride % ALIGN == 0, f"record stride {self.stride} is not {ALIGN}-aligned"
        self.offsets = {k: (int(o), int(n), tuple(s)) for k, (o, n, s) in meta["offsets"].items()}
        self.records = {tuple(int(x) for x in k.split(",")): int(v) for k, v in meta["records"].items()}
        self.fd = os.open(path + ".bin", os.O_RDONLY | os.O_DIRECT)
        self.device = device
        self._tls = threading.local()
        self.n_hits = 0
        # experts whose FP4 bytes were freed by a100-vq/punch_fp4.py: reading them would return
        # zeros, so a miss on one that is NOT in the store has to fail loudly instead
        self.punched = set()
        if punched_path and os.path.exists(punched_path):
            self.punched = {tuple(x) for x in json.load(open(punched_path))["punched"]}

    def guard(self, key) -> None:
        k = (int(key[0]), int(key[1]))
        if k in self.punched:
            raise RuntimeError(
                f"expert {k} is not in the CB3 store but its FP4 bytes were punched out of the "
                f"checkpoint; the store and models/fp4_punched.json disagree. Re-pack it with "
                f"a100-vq/pack_store.py or restore the shard.")

    def has(self, key) -> bool:
        return (int(key[0]), int(key[1])) in self.records

    def _buf(self):
        b = getattr(self._tls, "buf", None)
        if b is None:
            # pinned and 4096-aligned: O_DIRECT refuses anything else
            raw = torch.empty(self.stride + ALIGN, dtype=torch.uint8, pin_memory=True)
            off = (-raw.data_ptr()) % ALIGN
            b = self._tls.buf = raw[off:off + self.stride]
            self._tls.raw = raw
            self._tls.mv = b.numpy().data
        return b, self._tls.mv

    def load(self, arena, slot: int, key, stream) -> int:
        buf, mv = self._buf()
        rec = self.records[(int(key[0]), int(key[1]))]
        got = os.preadv(self.fd, [mv], rec * self.stride)
        assert got == self.stride, (got, self.stride)
        compute = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            stream.wait_stream(compute)
            for name, (o, n, shape) in self.offsets.items():
                getattr(arena, name)[slot].view(-1).copy_(buf[o:o + n], non_blocking=True)
        stream.synchronize()
        self.n_hits += 1
        return slot


def maybe_open(device):
    """CB3Store (or MultiStore) named by DSV41_CB3_STORE (colon separated), or None."""
    p = os.environ.get("DSV41_CB3_STORE", "")
    if not p:
        return None
    p = os.path.expanduser(p)
    stores = [x for x in p.split(":") if os.path.exists(x + ".bin") and os.path.exists(x + ".json")]
    if not stores:
        return None
    punched = os.environ.get("DSV41_FP4_PUNCHED", os.path.join(os.path.dirname(stores[0]),
                                                               "fp4_punched.json"))
    if len(stores) == 1:
        return CB3Store(stores[0], device, punched)
    return MultiStore([CB3Store(s, device) for s in stores], punched)


class MultiStore:
    """Several record files behind one interface, so a store can be extended without rewriting it."""

    def __init__(self, stores, punched_path: str = ""):
        self.stores = stores
        self.records = {}
        for st in stores:
            for k in st.records:
                self.records[k] = st
        self.punched = set()
        if punched_path and os.path.exists(punched_path):
            self.punched = {tuple(x) for x in json.load(open(punched_path))["punched"]}

    def has(self, key) -> bool:
        return (int(key[0]), int(key[1])) in self.records

    def guard(self, key) -> None:
        CB3Store.guard(self, key)

    def load(self, arena, slot: int, key, stream) -> int:
        return self.records[(int(key[0]), int(key[1]))].load(arena, slot, key, stream)
