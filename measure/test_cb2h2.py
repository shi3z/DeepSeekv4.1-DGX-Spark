import sys, torch
sys.path.insert(0,'work/tools'); sys.path.insert(0,'work')
import fp4_moe as F4, cb3_moe as C3, tiered_moe as TM, cb2half as CBH
from cb3 import dequant_cb2
from engine.codebook_sim import CodebookSim
torch.manual_seed(0); dev='cuda'; DIM,INTER=F4.DIM,F4.INTER
IH=1280; NG=IH//32
g=torch.Generator().manual_seed(7)
# every nibble in 0..3 -> the per-row CB2 codebook is exactly {0,1,2,3}: packing is LOSSLESS and
# identical for any subset of the row, so a difference can only be an indexing error.
def lowcodes(*sh):
    a=torch.randint(0,4,sh,generator=g,dtype=torch.uint8); b=torch.randint(0,4,sh,generator=g,dtype=torch.uint8)
    return (a|(b<<4)).to(dev)
def sc(*sh): return torch.full(sh,127,dtype=torch.uint8,device=dev)
w1=lowcodes(INTER,DIM//2); w3=lowcodes(INTER,DIM//2); w2=lowcodes(DIM,INTER//2)
s1=sc(INTER,DIM//32); s3=sc(INTER,DIM//32); s2=sc(DIM,INTER//32)
sim=CodebookSim(2,dev)
half=CBH.CB2HalfArena(1,dev,inter_h=IH); half.sim=sim
sel=torch.arange(NG,device=dev)*1   # force groups 0..NG-1 but NOT contiguous: use a strided set
sel=torch.arange(0,72,72//NG,device=dev)[:NG]
half.select_groups=lambda a,b: sel
half.load_slot(0,w1,s1,w2,s2,w3,s3)
# expected: dequantised half weights == the selected channels of the full dequantised weights
full=C3.CB2ArenaV2(1,dev); full.sim=sim; full.load_slot(0,w1,s1,w2,s2,w3,s3)
fw1,fw2,fw3 = full.dequant_slot(0)
hw1,hw2,hw3 = half.dequant_slot(0)
ar32=torch.arange(32,device=dev)
rows=(sel[:,None]*32+ar32).reshape(-1)
ok1=torch.equal(hw1.float(), fw1[rows].float())
ok3=torch.equal(hw3.float(), fw3[rows].float())
ok2=torch.equal(hw2.float(), fw2[:,rows].float())
print("w1 rows  gathered correctly:", ok1)
print("w3 rows  gathered correctly:", ok3)
print("w2 cols  gathered correctly:", ok2)
if not ok2:
    d=(hw2.float()-fw2[:,rows].float()).abs()
    bad=(d>0).float().mean().item()
    print(f"   w2 mismatch fraction {bad:.3f}; first bad col {int((d>0).any(0).nonzero()[0]) if (d>0).any() else -1}")
