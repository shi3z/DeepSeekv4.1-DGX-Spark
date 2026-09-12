import sys, torch
sys.path.insert(0,'/home/shi3z/dsv41-spark/work/tools'); sys.path.insert(0,'/home/shi3z/dsv41-spark/work')
import fp4_moe as F4, cb3_moe as C3, tiered_moe as TM
torch.manual_seed(0); dev='cuda'; DIM,INTER=F4.DIM,F4.INTER
T,K=6,6
def fill(ar, seed):
    g=torch.Generator(device='cpu').manual_seed(seed)
    for a in sorted(dir(ar)):
        t=getattr(ar,a,None)
        if isinstance(t,torch.Tensor) and t.dtype==torch.uint8:
            if a.endswith('_cb'):
                t.copy_(torch.randint(0,16,t.shape,generator=g,dtype=torch.uint8))
            elif a.startswith('s') and len(a)==2:      # UE8M0 scales: keep near 2^0
                t.copy_(torch.randint(120,131,t.shape,generator=g,dtype=torch.uint8))
            else:
                t.copy_(torch.randint(0,256,t.shape,generator=g,dtype=torch.uint8))
N0,N1=20,20
tiered=TM.TieredArena([("fp4",N0),("cb2",N1)],dev)
fp4_ref=F4.ExpertArena(N0,dev); cb2_ref=C3.CB2ArenaV2(N1,dev)
fill(tiered.tiers[0].arena, 1); fill(fp4_ref, 1)
fill(tiered.tiers[1].arena, 2); fill(cb2_ref, 2)
x=torch.randn(T,DIM,dtype=torch.bfloat16,device=dev); w=torch.rand(T,K,device=dev)

# A: every pair in tier0 (fp4)
loc=torch.stack([torch.randperm(N0)[:K] for _ in range(T)]).to(dev).int()
a=TM.moe_forward_tiered(x, loc.clone(), w, tiered)
b=F4.moe_forward(x, loc.clone(), w, fp4_ref)
print("A fp4-only  max|diff| =", (a.float()-b.float()).abs().max().item(), " ref absmax", b.float().abs().max().item())

# B: every pair in tier1 (cb2)
gl=(loc+N0)
a=TM.moe_forward_tiered(x, gl.clone(), w, tiered)
b=C3.moe_forward_cb2(x, loc.clone(), w, cb2_ref)
print("B cb2-only  max|diff| =", (a.float()-b.float()).abs().max().item(), " ref absmax", b.float().abs().max().item())

# C: mixed -- half the pairs fp4, half cb2; compare to the sum of two separate masked calls
mix=loc.clone(); mask=torch.rand(T,K,device=dev)<0.5
mixg=torch.where(mask, loc, loc+N0)
a=TM.moe_forward_tiered(x, mixg.clone(), w, tiered)
# reference: run each tier alone with the other tier's pairs zero-weighted
w0=torch.where(mask, w, torch.zeros_like(w)); w1=torch.where(mask, torch.zeros_like(w), w)
r0=F4.moe_forward(x, loc.clone(), w0, fp4_ref)
r1=C3.moe_forward_cb2(x, loc.clone(), w1, cb2_ref)
b=(r0.float()+r1.float())
print("C mixed     max|diff| =", (a.float()-b).abs().max().item(), " ref absmax", b.abs().max().item())
print("tier bytes:", tiered.bytes_per_slot, "total MB", tiered.total_bytes()/1e6)
