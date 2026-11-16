// Host shim: runs CUDA kernel source as one std::thread per CUDA thread.
// CTAs run one at a time, so `__shared__` = function-local static.
#pragma once
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __restrict__
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))

struct dim3 { unsigned x = 0, y = 0, z = 0; };
inline thread_local dim3 threadIdx, blockIdx;
inline dim3 blockDim, gridDim;

struct alignas(16) int4 { int x, y, z, w; };
struct alignas(8) uint2 { unsigned x, y; };

struct half { uint16_t x; };
using __half = half;
inline half __float2half_rn(float f) { _Float16 h = (_Float16)f; half r; std::memcpy(&r.x, &h, 2); return r; }
inline float __half2float(half h) { _Float16 v; std::memcpy(&v, &h.x, 2); return (float)v; }
inline half __ushort_as_half(unsigned short u) { half r; r.x = u; return r; }

template <class T> inline T __ldg(const T* p) { return *p; }
inline uint32_t __funnelshift_r(uint32_t lo, uint32_t hi, uint32_t sh) {
  return uint32_t(((uint64_t(hi) << 32) | lo) >> (sh & 31));
}
inline int __popc(unsigned x) { return __builtin_popcount(x); }
inline int __dp4a(int a, int b, int c) {
  for (int i = 0; i < 4; ++i) c += int(int8_t(uint32_t(a) >> (8 * i))) * int(int8_t(uint32_t(b) >> (8 * i)));
  return c;
}
inline unsigned __vadd4(unsigned a, unsigned b) {
  unsigned r = 0;
  for (int i = 0; i < 4; ++i) r |= (((a >> (8 * i)) + (b >> (8 * i))) & 255u) << (8 * i);
  return r;
}
inline unsigned __vsub4(unsigned a, unsigned b) {
  unsigned r = 0;
  for (int i = 0; i < 4; ++i) r |= (((a >> (8 * i)) - (b >> (8 * i))) & 255u) << (8 * i);
  return r;
}
inline int atomicAdd(int* p, int v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }

// Sense-reversing spin barrier with yield (std::barrier's futex waits make
// the 32-lane shuffle barriers very slow).
struct SpinBarrier {
  explicit SpinBarrier(int n) : n_(n) {}
  void arrive_and_wait() {
    const unsigned gen = gen_.load(std::memory_order_acquire);
    if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
      count_.store(0, std::memory_order_relaxed);
      gen_.fetch_add(1, std::memory_order_acq_rel);
    } else {
      while (gen_.load(std::memory_order_acquire) == gen) std::this_thread::yield();
    }
  }
  int n_;
  std::atomic<int> count_{0};
  std::atomic<unsigned> gen_{0};
};
struct Cta {
  std::unique_ptr<SpinBarrier> bar;
  std::vector<std::unique_ptr<SpinBarrier>> wbar;
  uint32_t slot[32][32];
};
inline Cta* g_cta = nullptr;
inline void __syncthreads() { g_cta->bar->arrive_and_wait(); }
template <class T> inline T shfl_from(T v, int src_lane) {
  static_assert(sizeof(T) == 4);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  std::memcpy(&g_cta->slot[warp][lane], &v, 4);
  g_cta->wbar[warp]->arrive_and_wait();
  T r;
  std::memcpy(&r, &g_cta->slot[warp][src_lane & 31], 4);
  g_cta->wbar[warp]->arrive_and_wait();
  return r;
}
template <class T> inline T __shfl_xor_sync(unsigned, T v, int o) { return shfl_from(v, (threadIdx.x & 31) ^ o); }
template <class T> inline T __shfl_sync(unsigned, T v, int src) { return shfl_from(v, src); }

// Runs every CTA (in the given order) with one team of `block` threads.
template <class F>
void emu_launch(int grid, int block, F f, const std::vector<int>& order) {
  blockDim.x = block;
  gridDim.x = grid;
  Cta cta;
  cta.bar = std::make_unique<SpinBarrier>(block);
  for (int w = 0; w < block / 32; ++w) cta.wbar.push_back(std::make_unique<SpinBarrier>(32));
  SpinBarrier edge(block);
  g_cta = &cta;
  std::vector<std::thread> th;
  th.reserve(block);
  for (int t = 0; t < block; ++t)
    th.emplace_back([&, t] {
      threadIdx.x = t;
      for (int b : order) {
        blockIdx.x = b;
        f();
        edge.arrive_and_wait();  // CTA boundary: shared statics are reused
      }
    });
  for (auto& x : th) x.join();
  g_cta = nullptr;
}

// 2-D grid (gx x gy), CTAs in the given order of linear ids x + gx * y.
template <class F>
void emu_launch2(int gx, int gy, int block, F f, const std::vector<int>& order) {
  gridDim.y = gy;
  std::vector<int> xs(order.size());
  for (size_t i = 0; i < order.size(); ++i) xs[i] = order[i];
  emu_launch(gx * gy, block, [&] {
    const int id = int(blockIdx.x);
    blockIdx.x = id % gx;
    blockIdx.y = id / gx;
    f();
    blockIdx.x = id;
  }, xs);
  gridDim.x = gx;
}
inline float __fdiv_rn(float a, float b) { return a / b; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fmul_rn(float a, float b) { return a * b; }

// Old sign_mask (LatticeDot<22>::sign_mask): prmt(nibble * 0x10204080, 0, 0xba98).
inline uint32_t sign_mask(int nibble) {
  const uint32_t bits = uint32_t(nibble) * 0x10204080U;
  uint32_t r = 0;
  for (int n = 0; n < 4; ++n) { uint32_t v = (bits >> (8 * n)) & 255; r |= ((v & 128) ? 255u : 0u) << (8 * n); }
  return r;
}
