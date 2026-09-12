#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>
__global__ void __launch_bounds__(256) read_u4(const uint4* __restrict__ p, size_t n, float* out){
  uint32_t acc=0; size_t stride=(size_t)gridDim.x*blockDim.x;
  for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=stride){
    uint4 v=__ldg(p+i); acc ^= v.x^v.y^v.z^v.w;
  }
  if(acc==0xFFFFFFFFu) out[0]=1.0f;
}
static void bench(const char*name, uint4* d, size_t bytes){
  size_t n=bytes/16; float* o; cudaMalloc(&o,4);
  int grid=48*8;
  read_u4<<<grid,256>>>(d,n,o); cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
  cudaEventRecord(a); for(int r=0;r<5;r++) read_u4<<<grid,256>>>(d,n,o);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b); ms/=5;
  cudaError_t e=cudaGetLastError();
  printf("%-22s %8.3f ms -> %7.1f GB/s   %s\n",name,ms,bytes/1e9/(ms/1e3), e?cudaGetErrorString(e):"");
  cudaFree(o);
}
int main(){
  size_t bytes=4ull<<30;
  uint4 *dev=nullptr,*host=nullptr,*man=nullptr;
  cudaMalloc(&dev,bytes); cudaMemset(dev,1,bytes); bench("cudaMalloc(device)",dev,bytes); cudaFree(dev);
  cudaError_t e1=cudaHostAlloc((void**)&host,bytes,cudaHostAllocMapped);
  if(e1==cudaSuccess){ cudaMemset(host,1,bytes); cudaDeviceSynchronize();
     printf("  host ptr %p page-aligned=%d\n",(void*)host,((uintptr_t)host%4096)==0);
     bench("cudaHostAlloc(pinned)",host,bytes); cudaFreeHost(host);} else printf("cudaHostAlloc failed: %s\n",cudaGetErrorString(e1));
  cudaError_t e2=cudaMallocManaged((void**)&man,bytes);
  if(e2==cudaSuccess){ cudaMemset(man,1,bytes); cudaDeviceSynchronize();
     printf("  managed ptr %p page-aligned=%d\n",(void*)man,((uintptr_t)man%4096)==0);
     bench("cudaMallocManaged",man,bytes); cudaFree(man);} else printf("cudaMallocManaged failed: %s\n",cudaGetErrorString(e2));
  return 0;
}
