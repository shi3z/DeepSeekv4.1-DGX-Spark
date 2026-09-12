import os, sys, time, torch
sys.path.insert(0,'/home/shi3z/dsv41-spark/work/tools')
sys.path.insert(0,'/home/shi3z/dsv41-spark/work')
import fp4_moe as F4, cb3_moe as C3
torch.manual_seed(0)
dev='cuda'
DIM,INTER=F4.DIM,F4.INTER

def fill_rand(t, lo=0, hi=256):
    t.random_(lo,hi)

def make_slots(T,K,nuniq,nslots):
    pool=torch.randperm(nslots)[:nuniq]
    idx=torch.randint(0,nuniq,(T,K))
    # ensure each token's K experts distinct
    for t in range(T):
        p=torch.randperm(nuniq)[:K]; idx[t]=p
    return pool[idx].to(dev).int()

def bench(fn, it=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it

T,K = 6,6
NSLOT=40
results={}
for name, mk, fwd in [
    ("FP4 ", lambda: F4.ExpertArena(NSLOT, dev), F4.moe_forward),
    ("CB3v3", lambda: C3.CB3ArenaV2(NSLOT, dev), C3.moe_forward_v3),
    ("CB2 ", lambda: C3.CB2ArenaV2(NSLOT, dev), C3.moe_forward_cb2),
]:
    ar=mk()
    for attr in dir(ar):
        t=getattr(ar,attr,None)
        if isinstance(t,torch.Tensor) and t.dtype==torch.uint8:
            if attr.endswith('_cb'): t.random_(0,16)
            else: t.random_(0,256)
    bps=ar.bytes_per_slot
    x=torch.randn(T,DIM,dtype=torch.bfloat16,device=dev)
    w=torch.rand(T,K,device=dev)
    for nuniq in (6, 16, 36):
        nu=min(nuniq,NSLOT)
        slots=make_slots(T,K,nu,NSLOT)
        real_uniq=int(torch.unique(slots).numel())
        try:
            ms=bench(lambda: fwd(x,slots,w,ar))*1e3
            gbs=real_uniq*bps/1e9/(ms/1e3)
            print(f"{name} uniq={real_uniq:2d} bytes/slot={bps/1e6:6.2f}MB  {ms:7.3f} ms  {gbs:7.1f} GB/s  per-expert {ms/real_uniq*1e3:6.1f} us")
            results[(name,real_uniq)]=(ms,bps)
        except Exception as e:
            print(f"{name} uniq={nu}: ERROR {type(e).__name__} {str(e)[:150]}")
    del ar; torch.cuda.empty_cache()
print()
print("--- wall-time per expert at uniq=16 (what a layer costs) ---")
for n in ("FP4 ","CB3v3","CB2 "):
    if (n,16) in results:
        ms,bps=results[(n,16)]
        print(f"{n}: {ms:6.3f} ms for 16 experts = {ms/16*1000:6.1f} us/expert, {bps/1e6:.2f} MB -> relative {ms:.3f}")
