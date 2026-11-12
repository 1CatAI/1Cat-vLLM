// Access-pattern floor of the skinny MoE decode kernel: grid (tiles, experts),
// SPLIT warps per CTA, each warp streams G/SPLIT groups of 32 lanes x W bytes.
#include <cstdio>
#include <cuda_runtime.h>
template <int SPLIT, typename T>
__global__ void pat(const T* __restrict__ w, int G, int tiles,
                    const int* experts, unsigned* out) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int e = experts[blockIdx.y], tile = blockIdx.x;
  const int Gq = G / SPLIT, g0 = warp * Gq;
  const T* cb = w + ((size_t)e * tiles + tile) * G * 32 + lane;
  unsigned acc = 0;
#pragma unroll 4
  for (int g = g0; g < g0 + Gq; ++g) {
    T v = __ldcs(cb + (size_t)g * 32);
    acc ^= reinterpret_cast<unsigned*>(&v)[0];
  }
  if (acc == 0x1234567) out[0] = acc;
}
int main() {
  const int E = 384, H = 5120,
            I2 = 1152;  // w13 per rank: N = 2*576 = 1152, K = 5120
  const int tiles = I2 / 32, K = H;
  size_t bytes = (size_t)E * I2 * K / 2;  // FP4
  char* w;
  cudaMalloc(&w, bytes);
  cudaMemset(w, 1, bytes);
  int hexp[19];
  for (int i = 0; i < 19; ++i) hexp[i] = (i * 37) % E;
  int* dexp;
  cudaMalloc(&dexp, sizeof(hexp));
  cudaMemcpy(dexp, hexp, sizeof(hexp), cudaMemcpyHostToDevice);
  unsigned* o;
  cudaMalloc(&o, 4);
  cudaStream_t s;
  cudaStreamCreate(&s);
  double mb = 19.0 * I2 * K / 2 / 1e6;
  auto run = [&](const char* name, auto launch) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 10; ++i) launch();
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
    double us = ms * 1000 / 200;
    printf("%-28s %7.1f us %6.0f GB/s\n", name, us, mb * 1e6 / us / 1e3);
  };
  dim3 grid(tiles, 19);
  // uint2 per lane, G = K/16 groups of 16 FP4 values (8 B per lane)
  run("uint2 split16 (kernel now)", [&] {
    pat<16, uint2>
        <<<grid, 512, 0, s>>>((const uint2*)w, K / 16, tiles, dexp, o);
  });
  run("uint2 split8", [&] {
    pat<8, uint2><<<grid, 256, 0, s>>>((const uint2*)w, K / 16, tiles, dexp, o);
  });
  run("uint2 split32", [&] {
    pat<32, uint2>
        <<<grid, 1024, 0, s>>>((const uint2*)w, K / 16, tiles, dexp, o);
  });
  // uint4 per lane: 2 groups per load
  run("uint4 split16", [&] {
    pat<16, uint4>
        <<<grid, 512, 0, s>>>((const uint4*)w, K / 32, tiles, dexp, o);
  });
  run("uint4 split8", [&] {
    pat<8, uint4><<<grid, 256, 0, s>>>((const uint4*)w, K / 32, tiles, dexp, o);
  });
  run("uint4 split20", [&] {
    pat<20, uint4>
        <<<grid, 640, 0, s>>>((const uint4*)w, K / 32, tiles, dexp, o);
  });
}
