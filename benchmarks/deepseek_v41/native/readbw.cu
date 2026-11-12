// Floor of a streaming read on this V100: sum uint4 words of a buffer, back
// to back in one CUDA graph, for the V4.1 dense weight sizes.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* out) {
  unsigned acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = __ldcs(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678) out[0] = acc;
}
int main() {
  size_t maxb = 64ull << 20;
  uint4* p;
  unsigned* o;
  cudaMalloc(&p, maxb);
  cudaMalloc(&o, 4);
  cudaMemset(p, 1, maxb);
  cudaStream_t s;
  cudaStreamCreate(&s);
  double mbs[] = {1, 2.6, 2.9, 4.2, 5.9, 9.2, 10.5, 20, 40};
  for (double mb : mbs) {
    size_t n = (size_t)(mb * 1e6) / 16;
    for (int grid : {160, 320, 640}) {
      cudaGraph_t g;
      cudaGraphExec_t ge;
      cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
      for (int i = 0; i < 20; ++i) rd<<<grid, 256, 0, s>>>(p, n, o);
      cudaStreamEndCapture(s, &g);
      cudaGraphInstantiate(&ge, g, 0);
      cudaGraphLaunch(ge, s);
      cudaStreamSynchronize(s);
      cudaEvent_t a, b;
      cudaEventCreate(&a);
      cudaEventCreate(&b);
      cudaEventRecord(a, s);
      for (int r = 0; r < 20; ++r) cudaGraphLaunch(ge, s);
      cudaEventRecord(b, s);
      cudaEventSynchronize(b);
      float ms;
      cudaEventElapsedTime(&ms, a, b);
      double us = ms * 1000 / 400;
      printf("%5.1f MB grid %4d: %6.2f us  %5.0f GB/s\n", mb, grid, us,
             mb * 1e6 / us / 1e3);
    }
  }
}
