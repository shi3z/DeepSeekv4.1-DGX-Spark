"""Does a big arena make the expert kernel slower? Their in-engine CB3 is 99 us/expert (146 GB/s);
the same kernel on a 40-slot arena measures 85 us (171 GB/s). If the gap is the arena, it is TLB
reach over a 90 GB random-access working set, not the kernel."""
import sys, time, torch
sys.path.insert(0,'work/tools'); sys.path.insert(0,'work')
import fp4_moe as F4, cb3_moe as C3
torch.manual_seed(0); dev='cuda'
T,K=6,6
x=torch.randn(T,5120,dtype=torch.bfloat16,device=dev); w=torch.rand(T,K,device=dev)
def bench(fn,it=25):
    for _ in range(6): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
print(f"{'slots':>7s} {'arenaGB':>8s} {'ms':>8s} {'us/expert':>10s} {'GB/s':>8s}")
for NS in (40, 200, 800, 1600):
    try:
        ar=F4.ExpertArena(NS,dev)
        # touch the memory so it is really backed
        ar.w1.random_(0,256); ar.w2.random_(0,256); ar.w3.random_(0,256)
        for s in (ar.s1,ar.s2,ar.s3): s.random_(120,131)
        bps=ar.bytes_per_slot
        # experts spread over the WHOLE arena, as routing does
        pool=torch.randperm(NS)[:16]
        loc=pool[torch.stack([torch.randperm(16)[:K] for _ in range(T)])].to(dev).int()
        nu=int(torch.unique(loc).numel())
        ms=bench(lambda: F4.moe_forward(x,loc,w,ar))
        print(f"{NS:7d} {NS*bps/1e9:8.1f} {ms:8.3f} {ms/nu*1e3:10.1f} {nu*bps/1e9/(ms/1e3):8.1f}")
        del ar; torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"{NS:7d}  OOM/{str(e)[:60]}"); break
