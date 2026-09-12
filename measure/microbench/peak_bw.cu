#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>
// pure streaming read: each thread reads uint4 (16B), strided grid loop, reduces to avoid DCE
__global__ void __launch_bounds__(256) read_u4(const uint4* __restrict__ p, size_t n, float* out){
  uint32_t acc=0; size_t stride=(size_t)gridDim.x*blockDim.x;
  for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=stride){
    uint4 v=__ldg(p+i); acc ^= v.x^v.y^v.z^v.w;
  }
  if(acc==0xFFFFFFFFu) out[0]=1.0f;
}
int main(int argc,char**argv){
  size_t GB = argc>1?atol(argv[1]):8;
  size_t bytes = GB<<30; size_t n = bytes/16;
  uint4* d; cudaMalloc(&d,bytes); cudaMemset(d,1,bytes);
  float* o; cudaMalloc(&o,4);
  int dev; cudaGetDevice(&dev); cudaDeviceProp pr; cudaGetDeviceProperties(&pr,dev);
  printf("SMs=%d memBusWidth=%d L2=%d KB\n",pr.multiProcessorCount,pr.memoryBusWidth,pr.l2CacheSize/1024);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  for(int blocks_per_sm : {1,2,4,8,16,32}){
    int grid = pr.multiProcessorCount*blocks_per_sm;
    read_u4<<<grid,256>>>(d,n,o); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for(int r=0;r<5;r++) read_u4<<<grid,256>>>(d,n,o);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b); ms/=5;
    printf("grid=%5d (%2d/SM)  %8.3f ms  -> %7.1f GB/s\n",grid,blocks_per_sm,ms, bytes/1e9/(ms/1e3));
  }
  return 0;
}
