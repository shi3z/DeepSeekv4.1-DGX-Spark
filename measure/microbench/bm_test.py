import sys, time, torch
sys.path.insert(0,'/home/shi3z/dsv41-spark/work/tools'); sys.path.insert(0,'/home/shi3z/dsv41-spark/work')
import fp4_moe as F4, cb3_moe as C3
torch.manual_seed(0); dev='cuda'
def fill(ar,seed=1):
    g=torch.Generator().manual_seed(seed)
    for a in sorted(dir(ar)):
        t=getattr(ar,a,None)
        if isinstance(t,torch.Tensor) and t.dtype==torch.uint8:
            if a.endswith('_cb'): t.copy_(torch.randint(0,16,t.shape,generator=g,dtype=torch.uint8))
            elif a.startswith('s') and len(a)==2: t.copy_(torch.randint(120,131,t.shape,generator=g,dtype=torch.uint8))
            else: t.copy_(torch.randint(0,256,t.shape,generator=g,dtype=torch.uint8))
def bench(fn,it=40):
    for _ in range(8): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
N=40
for label, cls, fwd, cfgU, cfgD, mb in (("FP4", F4.ExpertArena, F4.moe_forward, F4._UP_CFG, F4._DOWN_CFG, 18.80),
                                        ("CB2", C3.CB2ArenaV2, C3.moe_forward_cb2, C3.CB2_UP_CFG, C3.CB2_DOWN_CFG, 9.99)):
    ar=cls(N,dev); fill(ar)
    for T in (6, 4):
        K=6
        x=torch.randn(T,5120,dtype=torch.bfloat16,device=dev); w=torch.rand(T,K,device=dev)
        loc=torch.stack([torch.randperm(16)[:K] for _ in range(T)]).to(dev).int()
        nu=int(torch.unique(loc).numel())
        print(f"--- {label} T={T} K={K} uniq={nu} ---")
        for BM in (8,16,32):
            if BM not in cfgU: 
                cu,cd = cfgU[16], cfgD[16]
            else:
                cu,cd = cfgU[BM], cfgD[BM]
            try:
                ms=bench(lambda: fwd(x,loc,w,ar,block_m=BM,cfg_up=cu,cfg_down=cd))
                print(f"   BM={BM:2d} cfg{cu}/{cd}: {ms:7.3f} ms  {nu*mb/1e3/(ms/1e3):7.1f} GB/s  {nu and ms/nu*1e3:6.1f} us/expert")
            except Exception as e:
                print(f"   BM={BM}: ERR {type(e).__name__} {str(e)[:90]}")
    del ar; torch.cuda.empty_cache()
