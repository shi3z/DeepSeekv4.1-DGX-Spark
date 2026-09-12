"""Indexing check for CB2HalfArena: with the dropped channels forced to zero, the half-width
expert must reproduce the full-width one. A wrong w2 byte-column stride shows up as ~100 % error."""
import sys, torch
sys.path.insert(0,'work/tools'); sys.path.insert(0,'work')
import fp4_moe as F4, cb3_moe as C3, tiered_moe as TM, cb2half as CBH
from engine.codebook_sim import CodebookSim
torch.manual_seed(0); dev='cuda'; DIM,INTER=F4.DIM,F4.INTER
IH=1280; NG=IH//32
sim=CodebookSim(2,dev)
g=torch.Generator().manual_seed(7)
def u8(*sh, lo=0, hi=256): return torch.randint(lo,hi,sh,generator=g,dtype=torch.uint8).to(dev)
w1=u8(INTER,DIM//2); w3=u8(INTER,DIM//2); w2=u8(DIM,INTER//2)
s1=u8(INTER,DIM//32,lo=124,hi=130); s3=u8(INTER,DIM//32,lo=124,hi=130); s2=u8(DIM,INTER//32,lo=124,hi=130)
# zero every channel outside the first IH so both widths must agree
w1[IH:]=0; w3[IH:]=0; w2[:, IH//2:]=0
full=C3.CB2ArenaV2(1,dev); full.sim=sim
full.load_slot(0,w1,s1,w2,s2,w3,s3)
half=CBH.CB2HalfArena(1,dev,inter_h=IH); half.sim=sim
half.select_groups=lambda a,b: torch.arange(NG,device=dev)     # force the first NG groups
half.load_slot(0,w1,s1,w2,s2,w3,s3)
T,K=6,1
x=torch.randn(T,DIM,dtype=torch.bfloat16,device=dev)*0.02
w=torch.ones(T,K,device=dev)
slots=torch.zeros(T,K,device=dev,dtype=torch.int32)
af=TM.TieredArena([('cb2',1)],dev,sims={'cb2':sim}); af.tiers[0].arena=full
ah=TM.TieredArena([('cb2h',1)],dev,sims={'cb2':sim},inter_h=IH); ah.tiers[0].arena=half
yf=TM.moe_forward_tiered(x,slots,w,af).float()
yh=TM.moe_forward_tiered(x,slots,w,ah).float()
rel=(yf-yh).abs().max()/yf.abs().max().clamp_min(1e-9)
print(f"full-width  absmax {yf.abs().max():.4f}")
print(f"half-width  absmax {yh.abs().max():.4f}")
print(f"max relative difference: {rel:.5f}   -> {'INDEXING OK' if rel < 0.05 else 'INDEXING BROKEN'}")
# and the selection itself: groups with tiny scales must be the ones dropped
s1b=u8(INTER,DIM//32,lo=124,hi=130); s3b=u8(INTER,DIM//32,lo=124,hi=130)
s1b[:32*10]=100; s3b[:32*10]=100      # first 10 groups made tiny
sel=CBH.CB2HalfArena(1,dev,inter_h=IH).select_groups(s1b,s3b)
print("select_groups dropped the 10 deliberately-tiny groups:", not bool((sel<10).any()))
