import os, sys, time, torch
sys.path.insert(0,'/home/shi3z/dsv41-spark/work/tools'); sys.path.insert(0,'/home/shi3z/dsv41-spark/work')
import fp4_moe as F4, cb3_moe as C3
torch.manual_seed(0); dev='cuda'; DIM,INTER=F4.DIM,F4.INTER
NSLOT=40; T,K=6,6
def fill(ar):
    for a in dir(ar):
        t=getattr(ar,a,None)
        if isinstance(t,torch.Tensor) and t.dtype==torch.uint8:
            t.random_(0,16) if a.endswith('_cb') else t.random_(0,256)
pool=torch.randperm(NSLOT)[:16]
idx=torch.stack([torch.randperm(16)[:K] for _ in range(T)])
slots=pool[idx].to(dev).int(); nu=int(torch.unique(slots).numel())
x=torch.randn(T,DIM,dtype=torch.bfloat16,device=dev); w=torch.rand(T,K,device=dev)
def bench(fn,it=20):
    for _ in range(4): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it
CAND=[(bn,nw,ns) for bn in (32,64,128,256) for nw in (2,4,8) for ns in (1,2,3,4)]
for label, cls, fwd, defu, defd in (
        ("CB2", C3.CB2ArenaV2, C3.moe_forward_cb2, C3.CB2_UP_CFG[16], C3.CB2_DOWN_CFG[16]),
        ("CB3", C3.CB3ArenaV2, C3.moe_forward_v3,  C3.CB3_UP_CFG[16], C3.CB3_DOWN_CFG[16]),
        ("FP4", F4.ExpertArena, F4.moe_forward,    F4._UP_CFG[16],    F4._DOWN_CFG[16])):
    ar=cls(NSLOT,dev); fill(ar); bps=ar.bytes_per_slot
    base=bench(lambda: fwd(x,slots,w,ar))*1e3
    ups=[]
    for c in CAND:
        try: ups.append((bench(lambda: fwd(x,slots,w,ar,cfg_up=c,cfg_down=defd),it=10)*1e3, c))
        except Exception: pass
    ups.sort(); bestu=ups[0][1]
    downs=[]
    for c in CAND:
        try: downs.append((bench(lambda: fwd(x,slots,w,ar,cfg_up=bestu,cfg_down=c),it=10)*1e3, c))
        except Exception: pass
    downs.sort(); bestd=downs[0][1]
    final=bench(lambda: fwd(x,slots,w,ar,cfg_up=bestu,cfg_down=bestd))*1e3
    print("%s: default %s/%s = %6.3f ms (%6.1f GB/s)" % (label,defu,defd,base,nu*bps/1e9/(base/1e3)))
    print("      best    %s/%s = %6.3f ms (%6.1f GB/s)  speedup %.3fx" % (bestu,bestd,final,nu*bps/1e9/(final/1e3),base/final), flush=True)
    del ar; torch.cuda.empty_cache()
