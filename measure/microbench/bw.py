import torch, time
d='cuda'
print('dev', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('total mem GB', torch.cuda.get_device_properties(0).total_memory/1e9)
p=torch.cuda.get_device_properties(0)
print('SMs', p.multi_processor_count)

def bench(fn, warm=3, it=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it

# 1. pure read bandwidth: sum over a big tensor
for GB in (2, 8, 16):
    n = int(GB*1e9//2)
    x = torch.empty(n, dtype=torch.bfloat16, device=d).normal_()
    t = bench(lambda: x.sum())
    print(f'read-only sum  {GB:2d}GB : {t*1e3:8.2f} ms -> {GB/t:7.1f} GB/s')
    del x; torch.cuda.empty_cache()

# 2. copy (read+write)
n=int(4e9//2)
a=torch.empty(n,dtype=torch.bfloat16,device=d).normal_(); b=torch.empty_like(a)
t=bench(lambda: b.copy_(a))
print(f'copy 4GB (r+w 8GB): {t*1e3:8.2f} ms -> {8/t:7.1f} GB/s')
del a,b; torch.cuda.empty_cache()

# 3. bf16 GEMV (weight-stationary read) - emulates dense decode
for (M,K,N) in [(1,5120,5120),(1,5120,16384),(4,5120,16384)]:
    w=torch.randn(K,N,dtype=torch.bfloat16,device=d); x=torch.randn(M,K,dtype=torch.bfloat16,device=d)
    t=bench(lambda: x@w)
    gb=K*N*2/1e9
    print(f'bf16 GEMV M={M} {K}x{N} ({gb:.2f}GB): {t*1e6:8.1f} us -> {gb/t:7.1f} GB/s')
    del w,x; torch.cuda.empty_cache()

# 4. kernel launch overhead
x=torch.randn(1024,device=d)
t=bench(lambda: x.add_(1.0), warm=50, it=2000)
print(f'tiny kernel launch: {t*1e6:.2f} us')
