"""Is splitting an expert read into chunks helping or hurting?

engine/experts.py cuts each expert's 17.7 MB weight run into DSV41_READ_CHUNK_MB pieces and issues
them across a thread pool, on the reasoning that a decode layer misses only a few experts so the
parallelism has to come from inside one expert. The device says otherwise: a single large O_DIRECT
read is faster than the same bytes as several smaller ones.
"""
import os, time, mmap, threading, statistics
from concurrent.futures import ThreadPoolExecutor
ALIGN = 4096
F = "models/DeepSeek-V4.1-Flash/model-00003-of-00048.safetensors"
SZ = os.path.getsize(F)
EXPERT = 17_694_720          # the weight run of one expert, rounded to alignment below
EXPERT = EXPERT // ALIGN * ALIGN
fd = os.open(F, os.O_RDONLY | os.O_DIRECT)

def buf(n):
    return mmap.mmap(-1, n)   # page-aligned

def read_chunks(off, total, chunk, pool):
    b = buf(total); v = memoryview(b)
    jobs = []
    o = 0
    while o < total:
        n = min(chunk, total - o)
        jobs.append((v[o:o + n], off + o, n)); o += n
    def rd(j):
        view, o2, n = j
        got = 0
        while got < n:
            r = os.preadv(fd, [view[got:]], o2 + got)
            if r <= 0: break
            got += r
    if len(jobs) == 1: rd(jobs[0])
    else: list(pool.map(rd, jobs))
    del jobs, v
    b.close()

import random
def bench(chunk_mb, n_expert, threads, iters=6):
    chunk = int(chunk_mb * 1024 * 1024) // ALIGN * ALIGN if chunk_mb else EXPERT
    pool = ThreadPoolExecutor(threads)
    ts = []
    for _ in range(iters):
        offs = [random.randrange(0, (SZ - EXPERT) // ALIGN) * ALIGN for _ in range(n_expert)]
        t0 = time.perf_counter()
        if n_expert == 1:
            read_chunks(offs[0], EXPERT, chunk, pool)
        else:
            outer = ThreadPoolExecutor(n_expert)
            list(outer.map(lambda o: read_chunks(o, EXPERT, chunk, pool), offs))
            outer.shutdown()
        ts.append(time.perf_counter() - t0)
    pool.shutdown()
    t = statistics.median(ts)
    return EXPERT * n_expert / 1e9 / t, t * 1000

print(f"{'experts in flight':>18s} {'chunk':>8s} {'threads':>8s} {'GB/s':>7s} {'ms':>8s}")
for n_expert in (1, 2, 4):
    for chunk_mb, threads in ((None, 1), (8, 8), (4, 24), (2, 24), (1, 24)):
        gbs, ms = bench(chunk_mb, n_expert, threads)
        label = "whole (17.7 MB)" if chunk_mb is None else f"{chunk_mb} MB"
        star = "  <- engine default" if (chunk_mb == 4 and threads == 24) else ""
        print(f"{n_expert:18d} {label:>8s} {threads:8d} {gbs:7.2f} {ms:8.2f}{star}")
os.close(fd)
