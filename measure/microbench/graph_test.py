import sys, torch, time
sys.path.insert(0,'work/tools'); sys.path.insert(0,'work')
import tiered_moe as TM
from engine.codebook_sim import CodebookSim
torch.manual_seed(0); dev='cuda'; sim=CodebookSim(2,dev)
a=TM.TieredArena([('cb2h',60),('cb2',30),('fp4',10)],dev,sims={'cb2':sim},inter_h=768)
def fill(ar,seed):
    g=torch.Generator().manual_seed(seed)
    for n in sorted(dir(ar)):
        t=getattr(ar,n,None)
        if isinstance(t,torch.Tensor) and t.dtype==torch.uint8:
            if n.endswith('_cb'): t.copy_(torch.randint(0,16,t.shape,generator=g,dtype=torch.uint8))
            elif n.startswith('s') and len(n)==2: t.copy_(torch.randint(120,131,t.shape,generator=g,dtype=torch.uint8))
            else: t.copy_(torch.randint(0,256,t.shape,generator=g,dtype=torch.uint8))
for i,t in enumerate(a.tiers): fill(t.arena,i+1)
T,K=6,6
x=torch.randn(T,5120,dtype=torch.bfloat16,device=dev)
w=torch.rand(T,K,device=dev)
slots=torch.randint(0,a.slots,(T,K),device=dev,dtype=torch.int32)
# warm up on a side stream (required before capture)
s=torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): y=TM.moe_forward_tiered(x,slots,w,a)
torch.cuda.current_stream().wait_stream(s)
eager=TM.moe_forward_tiered(x,slots,w,a).clone()
pool=torch.cuda.graph_pool_handle()
g=torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g, pool=pool):
        out=TM.moe_forward_tiered(x,slots,w,a)
    print("CAPTURE OK")
    g.replay(); torch.cuda.synchronize()
    print("replay max|diff| vs eager:", float((out.float()-eager.float()).abs().max()))
    # change the routing and replay: the graph must follow the new slots
    slots.copy_(torch.randint(0,a.slots,(T,K),device=dev,dtype=torch.int32))
    g.replay(); torch.cuda.synchronize()
    ref=TM.moe_forward_tiered(x,slots,w,a)
    print("after new routing, replay vs eager:", float((out.float()-ref.float()).abs().max()), "(0 = the graph re-routes correctly)")
    def bench(fn,it=50):
        for _ in range(5): fn()
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(it): fn()
        torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
    print(f"eager {bench(lambda: TM.moe_forward_tiered(x,slots,w,a)):.3f} ms   graph {bench(lambda: g.replay()):.3f} ms")
except Exception as e:
    print("CAPTURE FAILED:", type(e).__name__, str(e)[:300])
