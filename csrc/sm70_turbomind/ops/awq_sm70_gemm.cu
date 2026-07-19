/*
 * SM70 AWQ GEMM integration using TurboMind s884h kernels.
 * Adapted from LMDeploy TurboMind (Apache-2.0).
 */

#include <torch/all.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <atomic>
#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "custom_all_reduce.cuh"

namespace turbomind {
void unpack_awq_gemm(uint4_t* dst, const uint4_t* src, int rows, int cols,
                     cudaStream_t st);
}  // namespace turbomind

namespace {

__global__ void sm70_silu_and_mul_fp16_kernel(__half* out, const __half* input,
                                              int rows, int d, int input_stride,
                                              int out_stride) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const __half* row_in = input + row * input_stride;
  const __half* gate = row_in;
  const __half* up = row_in + d;
  __half* row_out = out + row * out_stride;
  for (int col = threadIdx.x; col < d; col += blockDim.x) {
    const float gate_f = __half2float(gate[col]);
    const __half silu = __float2half(gate_f / (1.0f + expf(-gate_f)));
    row_out[col] = __hmul(silu, up[col]);
  }
}

void sm70_silu_and_mul_fp16_out(torch::Tensor out, torch::Tensor input) {
  const int rows = static_cast<int>(out.size(0));
  const int d = static_cast<int>(out.size(1));
  const int input_stride = static_cast<int>(input.stride(0));
  const int out_stride = static_cast<int>(out.stride(0));
  constexpr int kThreads = 256;
  sm70_silu_and_mul_fp16_kernel<<<rows, kThreads, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()), rows, d,
      input_stride, out_stride);
}

__global__ void sm70_silu_and_mul_interleaved_fp16_kernel(__half* out,
                                                          const __half* input,
                                                          int rows, int d,
                                                          int input_stride,
                                                          int out_stride) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const __half* row_in = input + row * input_stride;
  __half* row_out = out + row * out_stride;
  for (int col = threadIdx.x; col < d; col += blockDim.x) {
    const int gate_col = col * 2;
    const float gate_f = __half2float(row_in[gate_col]);
    const __half silu = __float2half(gate_f / (1.0f + expf(-gate_f)));
    row_out[col] = __hmul(silu, row_in[gate_col + 1]);
  }
}

void sm70_silu_and_mul_interleaved_fp16_out(torch::Tensor out,
                                            torch::Tensor input) {
  const int rows = static_cast<int>(out.size(0));
  const int d = static_cast<int>(out.size(1));
  const int input_stride = static_cast<int>(input.stride(0));
  const int out_stride = static_cast<int>(out.stride(0));
  constexpr int kThreads = 256;
  sm70_silu_and_mul_interleaved_fp16_kernel<<<
      rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()), rows, d,
      input_stride, out_stride);
}

bool sm70_profile_trace_enabled() {
  const char* value = std::getenv("VLLM_SM70_PROFILE_TRACE");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

const char* sm70_scalar_type_name(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Float:
      return "float32";
    case at::ScalarType::Half:
      return "float16";
    case at::ScalarType::BFloat16:
      return "bfloat16";
    default:
      return "other";
  }
}

const char* sm70_capture_status_name(cudaStreamCaptureStatus status) {
  switch (status) {
    case cudaStreamCaptureStatusNone:
      return "none";
    case cudaStreamCaptureStatusActive:
      return "active";
    case cudaStreamCaptureStatusInvalidated:
      return "invalidated";
    default:
      return "unknown";
  }
}

unsigned sm70_capture_status_bit(cudaStreamCaptureStatus status) {
  switch (status) {
    case cudaStreamCaptureStatusNone:
      return 1u;
    case cudaStreamCaptureStatusActive:
      return 2u;
    case cudaStreamCaptureStatusInvalidated:
      return 4u;
    default:
      return 8u;
  }
}

void maybe_log_sm70_moe_route_once(std::atomic<unsigned>& logged_mask,
                                   const char* route,
                                   const torch::Tensor& input, int64_t tokens,
                                   int64_t experts_or_top_k) {
  if (!sm70_profile_trace_enabled()) {
    return;
  }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
  const unsigned bit = sm70_capture_status_bit(capture_status);
  const unsigned previous =
      logged_mask.fetch_or(bit, std::memory_order_relaxed);
  if ((previous & bit) != 0u) {
    return;
  }
  std::cerr << route << " tokens=" << tokens
            << " experts_or_top_k=" << experts_or_top_k
            << " dtype=" << sm70_scalar_type_name(input.scalar_type())
            << " capture=" << sm70_capture_status_name(capture_status)
            << std::endl;
}

}  // namespace

void silu_and_mul(torch::Tensor& out, torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(),
              "silu_and_mul SM70 compatibility op expects CUDA tensors.");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half &&
                  out.scalar_type() == at::ScalarType::Half,
              "silu_and_mul SM70 compatibility op only supports float16.");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) % 2 == 0,
              "silu_and_mul expects the last input dimension to be even.");
  TORCH_CHECK(out.numel() * 2 == input.numel(),
              "silu_and_mul output shape must be half of input last dim.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  sm70_silu_and_mul_fp16_out(out, input);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void silu_and_mul_interleaved(torch::Tensor& out, torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(),
              "silu_and_mul_interleaved SM70 op expects CUDA tensors.");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half &&
                  out.scalar_type() == at::ScalarType::Half,
              "silu_and_mul_interleaved only supports float16.");
  TORCH_CHECK(input.dim() >= 1,
              "silu_and_mul_interleaved expects input rank >= 1.");
  TORCH_CHECK(input.size(-1) == out.size(-1) * 2,
              "silu_and_mul_interleaved input last dim must be twice output.");
  TORCH_CHECK(input.numel() == out.numel() * 2,
              "silu_and_mul_interleaved batch shape mismatch.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  sm70_silu_and_mul_interleaved_fp16_out(out, input);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace turbomind::gemm {

template <class Base>
struct Top1Epilogue {
  using Tc = typename Base::Tc;
  using Dtype = typename Base::Dtype;

  static constexpr auto kOrder = Base::kOrder;
  static constexpr auto kMode = Base::kMode;
  static constexpr bool SplitK = Base::SplitK;
  static constexpr int TM = Base::TM;
  static constexpr int TN = Base::TN;
  static constexpr int S = Base::S;
  static constexpr int C = Base::C;
  static constexpr int kAccess = Base::kAccess;

  using SharedStorage = typename Base::SharedStorage;
  using Map = typename Base::Map;

  template <class T>
  using OutputC = typename Base::template OutputC<T>;

  template <class FragC>
  __device__ void operator()(FragC& frag_C, const int4& tile_offset,
                             const int2& extents, int, int, bool,
                             const EpilogueParam& param,
                             SharedStorage& storage) {
    static_assert(std::is_same_v<Tc, half_t>);
    // This epilogue reduces each row in the current M/N tile to one
    // partial top1 pair. The wrapper stores partials as [M, num_n_tiles].
    Base base{};
    OutputC<Dtype> tmp_C[S][C];
    base.Rearrange(frag_C, storage, tmp_C);

    const int2 cta_cs = mk2cs<kOrder>(tile_offset.x * TM, tile_offset.y * TN);
    const int2 end_cs = mk2cs<kOrder>(extents);
    const int2 thr_cs =
        Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);

    float best_values[TM];
    int64_t best_indices[TM];
    PRAGMA_UNROLL
    for (int row = 0; row < TM; ++row) {
      best_values[row] = -FLT_MAX;
      best_indices[row] = INT64_MAX;
    }

    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
      PRAGMA_UNROLL
      for (int c = 0; c < C; ++c) {
        const int ss = thr_cs.y + s * Map::kDeltaS;
        const int cc = thr_cs.x + c * Map::kDeltaC;
        if (ss >= end_cs.y || cc >= end_cs.x) {
          continue;
        }
        const auto vals_half = cast<Tc>(tmp_C[s][c]);
        const auto vals = cast<float>(vals_half);
        PRAGMA_UNROLL
        for (int i = 0; i < kAccess; ++i) {
          const int2 mn = cs2mk<kOrder>(cta_cs.x + cc + i, cta_cs.y + ss);
          const int row = mn.x;
          const int col = mn.y;
          const int local_row = row - tile_offset.x * TM;
          if (local_row < 0 || local_row >= extents.x) {
            continue;
          }
          const float val = vals[i];
          const int64_t idx = static_cast<int64_t>(param.c.stride) + col;
          if (val > best_values[local_row] || (val == best_values[local_row] &&
                                               idx < best_indices[local_row])) {
            best_values[local_row] = val;
            best_indices[local_row] = idx;
          }
        }
      }
    }

    __shared__ float thread_values[256];
    __shared__ int64_t thread_indices[256];
    const int num_tiles_n = param.partials.stride;
    PRAGMA_UNROLL
    for (int local_row = 0; local_row < TM; ++local_row) {
      if (local_row >= extents.x) {
        break;
      }
      thread_values[threadIdx.x] = best_values[local_row];
      thread_indices[threadIdx.x] = best_indices[local_row];
      __syncthreads();

      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          const float other_val = thread_values[threadIdx.x + stride];
          const int64_t other_idx = thread_indices[threadIdx.x + stride];
          if (other_val > thread_values[threadIdx.x] ||
              (other_val == thread_values[threadIdx.x] &&
               other_idx < thread_indices[threadIdx.x])) {
            thread_values[threadIdx.x] = other_val;
            thread_indices[threadIdx.x] = other_idx;
          }
        }
        __syncthreads();
      }

      if (threadIdx.x == 0) {
        const int global_row = tile_offset.x * TM + local_row;
        const int partial_offset = global_row * num_tiles_n + tile_offset.y;
        reinterpret_cast<float*>(param.c.ptr)[partial_offset] =
            thread_values[0];
        reinterpret_cast<int64_t*>(param.partials.ptr)[partial_offset] =
            thread_indices[0];
      }
      __syncthreads();
    }
  }
};

template <class Base>
struct Top20Epilogue {
  using Tc = typename Base::Tc;
  using Dtype = typename Base::Dtype;

  static constexpr auto kOrder = Base::kOrder;
  static constexpr auto kMode = Base::kMode;
  static constexpr bool SplitK = Base::SplitK;
  static constexpr int TM = Base::TM;
  static constexpr int TN = Base::TN;
  static constexpr int S = Base::S;
  static constexpr int C = Base::C;
  static constexpr int kAccess = Base::kAccess;
  static constexpr int kTopK = 20;

  using SharedStorage = typename Base::SharedStorage;
  using Map = typename Base::Map;

  template <class T>
  using OutputC = typename Base::template OutputC<T>;

  __device__ static bool Better(float val, int64_t idx, float best_val,
                                int64_t best_idx) {
    return val > best_val || (val == best_val && idx < best_idx);
  }

  template <class FragC>
  __device__ void operator()(FragC& frag_C, const int4& tile_offset,
                             const int2& extents, int, int, bool,
                             const EpilogueParam& param,
                             SharedStorage& storage) {
    static_assert(std::is_same_v<Tc, half_t>);
    static_assert(TN == 256);
    Base base{};
    OutputC<Dtype> tmp_C[S][C];
    base.Rearrange(frag_C, storage, tmp_C);

    __shared__ float tile_values[TM * TN];
    for (int idx = threadIdx.x; idx < TM * TN; idx += blockDim.x) {
      tile_values[idx] = -FLT_MAX;
    }
    __syncthreads();

    const int2 cta_cs = mk2cs<kOrder>(tile_offset.x * TM, tile_offset.y * TN);
    const int2 end_cs = mk2cs<kOrder>(extents);
    const int2 thr_cs =
        Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);

    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
      PRAGMA_UNROLL
      for (int c = 0; c < C; ++c) {
        const int ss = thr_cs.y + s * Map::kDeltaS;
        const int cc = thr_cs.x + c * Map::kDeltaC;
        if (ss >= end_cs.y || cc >= end_cs.x) {
          continue;
        }
        const auto vals_half = cast<Tc>(tmp_C[s][c]);
        const auto vals = cast<float>(vals_half);
        PRAGMA_UNROLL
        for (int i = 0; i < kAccess; ++i) {
          const int2 mn = cs2mk<kOrder>(cta_cs.x + cc + i, cta_cs.y + ss);
          const int local_row = mn.x - tile_offset.x * TM;
          const int local_col = mn.y - tile_offset.y * TN;
          if (local_row >= 0 && local_row < extents.x && local_col >= 0 &&
              local_col < TN) {
            tile_values[local_row * TN + local_col] = vals[i];
          }
        }
      }
    }
    __syncthreads();

    const int local_row = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (local_row >= extents.x) {
      return;
    }

    const int num_tiles_n = param.partials.stride;
    const int global_row = tile_offset.x * TM + local_row;
    const int partial_base = (global_row * num_tiles_n + tile_offset.y) * kTopK;
    const int tile_col_start = tile_offset.y * TN;
    auto* partial_values = reinterpret_cast<float*>(param.c.ptr);
    auto* partial_indices = reinterpret_cast<int64_t*>(param.partials.ptr);

    PRAGMA_UNROLL
    for (int rank = 0; rank < kTopK; ++rank) {
      float best_val = -FLT_MAX;
      int64_t best_idx = INT64_MAX;
      int best_col = -1;
      for (int col = lane; col < TN; col += WARP_SIZE) {
        const float val = tile_values[local_row * TN + col];
        const int64_t token =
            static_cast<int64_t>(param.c.stride) + tile_col_start + col;
        if (Better(val, token, best_val, best_idx)) {
          best_val = val;
          best_idx = token;
          best_col = col;
        }
      }
      for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        const float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        const int64_t other_idx =
            __shfl_down_sync(0xffffffff, best_idx, offset);
        const int other_col = __shfl_down_sync(0xffffffff, best_col, offset);
        if (Better(other_val, other_idx, best_val, best_idx)) {
          best_val = other_val;
          best_idx = other_idx;
          best_col = other_col;
        }
      }
      if (lane == 0) {
        partial_values[partial_base + rank] = best_val;
        partial_indices[partial_base + rank] = best_idx;
        if (best_col >= 0) {
          tile_values[local_row * TN + best_col] = -FLT_MAX;
        }
      }
      __syncwarp();
    }
  }
};

namespace sm70_s884 {

template <class A, class TransformA, class U, class B, class TransformB,
          class V, Order order_C, class Tc, Order raster_order, int group_axis>
struct Sm70_s884_top1 {
  static_assert(A::SmemCopyAtom::K == B::SmemCopyAtom::K);

  static constexpr int SMEM_M = A::SmemCopyAtom::M / A::SmemCopyAtom::kFragNum;
  static constexpr int SMEM_N = B::SmemCopyAtom::M / B::SmemCopyAtom::kFragNum;
  static constexpr int SMEM_K = A::SmemCopyAtom::K;

  static constexpr auto MODE_ =
      group_axis >= 0 ? Striding::kBlocked : Striding::kFlat;
  static constexpr auto MODE_A = group_axis == 0 ? Striding::kIndexed : MODE_;
  static constexpr auto MODE_B = group_axis == 1 ? Striding::kIndexed : MODE_;
  static constexpr auto MODE_C = MODE_;

  template <int CTA_M, int CTA_N, int CTA_K, int TG_M, int TG_N, int TG_K,
            class PolicyA, class PolicyB, int Stages, bool SplitK,
            int GroupSizeU = 1, int GroupSizeV = 1, int TILE_C_M_ = -1,
            int TILE_C_N_ = -1>
  struct Type {
    using MMA_Atom = SM70_MMA_884;
    using Partition = Blocked<TG_M, TG_N, kColMajor>;
    using MMA_Map =
        MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, SMEM_N, SMEM_K, Partition, TG_K>;
    using MMA = Tiled_MMA_v2<MMA_Atom, MMA_Map>;

    using Mainloop =
        MainloopSm70<MMA, A, IteratorSm70<MODE_A, PolicyA>, TransformA, U,
                     GroupSizeU, B, IteratorSm70<MODE_B, PolicyB>, TransformB,
                     V, GroupSizeV, Stages, true>;

    static constexpr int CHUNK_K =
        std::lcm(std::lcm(GroupSizeU, GroupSizeV), CTA_K);
    using Scheduler = SchedulerSm70<raster_order, CTA_M, CTA_N, CTA_K, CHUNK_K,
                                    SplitK, group_axis>;

    static constexpr int TILE_C_M = TILE_C_M_ == -1 ? CTA_M : TILE_C_M_;
    static constexpr int TILE_C_N = TILE_C_N_ == -1 ? CTA_N : TILE_C_N_;
    using BaseEpilogue =
        gemm::Epilogue_<Tc, CTA_M, CTA_N, TILE_C_M, TILE_C_N, MMA::kThreadCount,
                        Rearrange<MMA>, Operand_C<float, order_C>, MODE_C,
                        SplitK>;
    using Epilogue = gemm::Top1Epilogue<BaseEpilogue>;
    using Kernel = GemmUniversal<Sm70, Mainloop, Epilogue, Scheduler>;
  };
};

template <Order raster_order, int group_axis = -1>
using Config_F16_Top1 =
    Sm70_s884_top1<Operand_A<half>, Transform_Default, VoidOperand,
                   Operand_B_Pack<half>, Transform_Default, VoidOperand,
                   kRowMajor, half, raster_order, group_axis>;

template <class A, class TransformA, class U, class B, class TransformB,
          class V, Order order_C, class Tc, Order raster_order, int group_axis>
struct Sm70_s884_top20 {
  static_assert(A::SmemCopyAtom::K == B::SmemCopyAtom::K);

  static constexpr int SMEM_M = A::SmemCopyAtom::M / A::SmemCopyAtom::kFragNum;
  static constexpr int SMEM_N = B::SmemCopyAtom::M / B::SmemCopyAtom::kFragNum;
  static constexpr int SMEM_K = A::SmemCopyAtom::K;

  static constexpr auto MODE_ =
      group_axis >= 0 ? Striding::kBlocked : Striding::kFlat;
  static constexpr auto MODE_A = group_axis == 0 ? Striding::kIndexed : MODE_;
  static constexpr auto MODE_B = group_axis == 1 ? Striding::kIndexed : MODE_;
  static constexpr auto MODE_C = MODE_;

  template <int CTA_M, int CTA_N, int CTA_K, int TG_M, int TG_N, int TG_K,
            class PolicyA, class PolicyB, int Stages, bool SplitK,
            int GroupSizeU = 1, int GroupSizeV = 1, int TILE_C_M_ = -1,
            int TILE_C_N_ = -1>
  struct Type {
    using MMA_Atom = SM70_MMA_884;
    using Partition = Blocked<TG_M, TG_N, kColMajor>;
    using MMA_Map =
        MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, SMEM_N, SMEM_K, Partition, TG_K>;
    using MMA = Tiled_MMA_v2<MMA_Atom, MMA_Map>;

    using Mainloop =
        MainloopSm70<MMA, A, IteratorSm70<MODE_A, PolicyA>, TransformA, U,
                     GroupSizeU, B, IteratorSm70<MODE_B, PolicyB>, TransformB,
                     V, GroupSizeV, Stages, true>;

    static constexpr int CHUNK_K =
        std::lcm(std::lcm(GroupSizeU, GroupSizeV), CTA_K);
    using Scheduler = SchedulerSm70<raster_order, CTA_M, CTA_N, CTA_K, CHUNK_K,
                                    SplitK, group_axis>;

    static constexpr int TILE_C_M = TILE_C_M_ == -1 ? CTA_M : TILE_C_M_;
    static constexpr int TILE_C_N = TILE_C_N_ == -1 ? CTA_N : TILE_C_N_;
    using BaseEpilogue =
        gemm::Epilogue_<Tc, CTA_M, CTA_N, TILE_C_M, TILE_C_N, MMA::kThreadCount,
                        Rearrange<MMA>, Operand_C<float, order_C>, MODE_C,
                        SplitK>;
    using Epilogue = gemm::Top20Epilogue<BaseEpilogue>;
    using Kernel = GemmUniversal<Sm70, Mainloop, Epilogue, Scheduler>;
  };
};

template <Order raster_order, int group_axis = -1>
using Config_F16_Top20 =
    Sm70_s884_top20<Operand_A<half>, Transform_Default, VoidOperand,
                    Operand_B_Pack<half>, Transform_Default, VoidOperand,
                    kRowMajor, half, raster_order, group_axis>;

}  // namespace sm70_s884
}  // namespace turbomind::gemm

namespace vllm {
namespace awq_sm70 {

__device__ __constant__ float kNvfp4E2m1Values[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __constant__ unsigned short kNvfp4E2m1HalfBits[16] = {
    0x0000, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400, 0x4600,
    0x8000, 0xb800, 0xbc00, 0xbe00, 0xc000, 0xc200, 0xc400, 0xc600,
};

__device__ __forceinline__ __half nvfp4_e2m1_to_half(uint32_t value) {
  return __ushort_as_half(kNvfp4E2m1HalfBits[value & 0xf]);
}

__device__ __forceinline__ __half2 nvfp4_e2m1_pair_to_half2(uint32_t packed,
                                                            int low_shift) {
  return __halves2half2(nvfp4_e2m1_to_half(packed >> low_shift),
                        nvfp4_e2m1_to_half(packed >> (low_shift + 4)));
}

template <int Threads>
__global__ void nvfp4_raw_gemv_partial_kernel(
    __half* __restrict__ out, float* __restrict__ partials,
    const __half* __restrict__ input,
    const uint32_t* __restrict__ qweight_packed,
    const __half* __restrict__ scales, int k, int n, int num_groups,
    int split_k) {
  const int qwords = n / 8;
  const int qword = static_cast<int>(blockIdx.x) * Threads + threadIdx.x;
  if (qword >= qwords) {
    return;
  }

  const int split = static_cast<int>(blockIdx.y);
  const int groups_per_split = (num_groups + split_k - 1) / split_k;
  const int group_begin = split * groups_per_split;
  const int group_end = min(num_groups, group_begin + groups_per_split);
  const int col = qword * 8;

  float acc0 = 0.f;
  float acc1 = 0.f;
  float acc2 = 0.f;
  float acc3 = 0.f;
  float acc4 = 0.f;
  float acc5 = 0.f;
  float acc6 = 0.f;
  float acc7 = 0.f;

  for (int group = group_begin; group < group_end; ++group) {
    const __half* scale_ptr = scales + group * n + col;
    const float scale0 = __half2float(scale_ptr[0]);
    const float scale1 = __half2float(scale_ptr[1]);
    const float scale2 = __half2float(scale_ptr[2]);
    const float scale3 = __half2float(scale_ptr[3]);
    const float scale4 = __half2float(scale_ptr[4]);
    const float scale5 = __half2float(scale_ptr[5]);
    const float scale6 = __half2float(scale_ptr[6]);
    const float scale7 = __half2float(scale_ptr[7]);

    const int k_base = group * 16;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const int kk = k_base + r;
      if (kk >= k) {
        break;
      }
      const float x = __half2float(input[kk]);
      const uint32_t packed = qweight_packed[kk * qwords + qword];
      acc0 += x * scale0 * kNvfp4E2m1Values[(packed >> 0) & 0xf];
      acc1 += x * scale1 * kNvfp4E2m1Values[(packed >> 4) & 0xf];
      acc2 += x * scale2 * kNvfp4E2m1Values[(packed >> 8) & 0xf];
      acc3 += x * scale3 * kNvfp4E2m1Values[(packed >> 12) & 0xf];
      acc4 += x * scale4 * kNvfp4E2m1Values[(packed >> 16) & 0xf];
      acc5 += x * scale5 * kNvfp4E2m1Values[(packed >> 20) & 0xf];
      acc6 += x * scale6 * kNvfp4E2m1Values[(packed >> 24) & 0xf];
      acc7 += x * scale7 * kNvfp4E2m1Values[(packed >> 28) & 0xf];
    }
  }

  if (split_k == 1) {
    __half* out_ptr = out + col;
    out_ptr[0] = __float2half(acc0);
    out_ptr[1] = __float2half(acc1);
    out_ptr[2] = __float2half(acc2);
    out_ptr[3] = __float2half(acc3);
    out_ptr[4] = __float2half(acc4);
    out_ptr[5] = __float2half(acc5);
    out_ptr[6] = __float2half(acc6);
    out_ptr[7] = __float2half(acc7);
  } else {
    float* partial_ptr = partials + split * n + col;
    partial_ptr[0] = acc0;
    partial_ptr[1] = acc1;
    partial_ptr[2] = acc2;
    partial_ptr[3] = acc3;
    partial_ptr[4] = acc4;
    partial_ptr[5] = acc5;
    partial_ptr[6] = acc6;
    partial_ptr[7] = acc7;
  }
}

template <int Threads>
__global__ void nvfp4_raw_gemv_reduce_kernel(__half* __restrict__ out,
                                             const float* __restrict__ partials,
                                             int n, int split_k) {
  const int qwords = n / 8;
  const int qword = static_cast<int>(blockIdx.x) * Threads + threadIdx.x;
  if (qword >= qwords) {
    return;
  }
  const int col = qword * 8;
  float acc0 = 0.f;
  float acc1 = 0.f;
  float acc2 = 0.f;
  float acc3 = 0.f;
  float acc4 = 0.f;
  float acc5 = 0.f;
  float acc6 = 0.f;
  float acc7 = 0.f;
  for (int split = 0; split < split_k; ++split) {
    const float* partial_ptr = partials + split * n + col;
    acc0 += partial_ptr[0];
    acc1 += partial_ptr[1];
    acc2 += partial_ptr[2];
    acc3 += partial_ptr[3];
    acc4 += partial_ptr[4];
    acc5 += partial_ptr[5];
    acc6 += partial_ptr[6];
    acc7 += partial_ptr[7];
  }
  __half* out_ptr = out + col;
  out_ptr[0] = __float2half(acc0);
  out_ptr[1] = __float2half(acc1);
  out_ptr[2] = __float2half(acc2);
  out_ptr[3] = __float2half(acc3);
  out_ptr[4] = __float2half(acc4);
  out_ptr[5] = __float2half(acc5);
  out_ptr[6] = __float2half(acc6);
  out_ptr[7] = __float2half(acc7);
}

template <int Threads>
__global__ void nvfp4_raw_gemv_h2_partial_kernel(
    __half* __restrict__ out, __half* __restrict__ partials,
    const __half* __restrict__ input,
    const uint32_t* __restrict__ qweight_packed,
    const __half* __restrict__ scales, int k, int n, int num_groups,
    int split_k) {
  const int qwords = n / 8;
  const int qword = static_cast<int>(blockIdx.x) * Threads + threadIdx.x;
  if (qword >= qwords) {
    return;
  }

  const int split = static_cast<int>(blockIdx.y);
  const int groups_per_split = (num_groups + split_k - 1) / split_k;
  const int group_begin = split * groups_per_split;
  const int group_end = min(num_groups, group_begin + groups_per_split);
  const int col = qword * 8;

  __half2 acc01 = __float2half2_rn(0.f);
  __half2 acc23 = __float2half2_rn(0.f);
  __half2 acc45 = __float2half2_rn(0.f);
  __half2 acc67 = __float2half2_rn(0.f);

  for (int group = group_begin; group < group_end; ++group) {
    const __half* scale_ptr = scales + group * n + col;
    const __half2 scale01 = *reinterpret_cast<const __half2*>(scale_ptr + 0);
    const __half2 scale23 = *reinterpret_cast<const __half2*>(scale_ptr + 2);
    const __half2 scale45 = *reinterpret_cast<const __half2*>(scale_ptr + 4);
    const __half2 scale67 = *reinterpret_cast<const __half2*>(scale_ptr + 6);

    const int k_base = group * 16;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const int kk = k_base + r;
      if (kk >= k) {
        break;
      }
      const __half2 x = __halves2half2(input[kk], input[kk]);
      const uint32_t packed = qweight_packed[kk * qwords + qword];
      acc01 = __hfma2(x, __hmul2(scale01, nvfp4_e2m1_pair_to_half2(packed, 0)),
                      acc01);
      acc23 = __hfma2(x, __hmul2(scale23, nvfp4_e2m1_pair_to_half2(packed, 8)),
                      acc23);
      acc45 = __hfma2(x, __hmul2(scale45, nvfp4_e2m1_pair_to_half2(packed, 16)),
                      acc45);
      acc67 = __hfma2(x, __hmul2(scale67, nvfp4_e2m1_pair_to_half2(packed, 24)),
                      acc67);
    }
  }

  __half* out_ptr = split_k == 1 ? out + col : partials + split * n + col;
  *reinterpret_cast<__half2*>(out_ptr + 0) = acc01;
  *reinterpret_cast<__half2*>(out_ptr + 2) = acc23;
  *reinterpret_cast<__half2*>(out_ptr + 4) = acc45;
  *reinterpret_cast<__half2*>(out_ptr + 6) = acc67;
}

template <int Threads>
__global__ void nvfp4_raw_gemv_h2_reduce_kernel(
    __half* __restrict__ out, const __half* __restrict__ partials, int n,
    int split_k) {
  const int qwords = n / 8;
  const int qword = static_cast<int>(blockIdx.x) * Threads + threadIdx.x;
  if (qword >= qwords) {
    return;
  }
  const int col = qword * 8;
  __half2 acc01 = __float2half2_rn(0.f);
  __half2 acc23 = __float2half2_rn(0.f);
  __half2 acc45 = __float2half2_rn(0.f);
  __half2 acc67 = __float2half2_rn(0.f);
  for (int split = 0; split < split_k; ++split) {
    const __half* partial_ptr = partials + split * n + col;
    acc01 = __hadd2(acc01, *reinterpret_cast<const __half2*>(partial_ptr + 0));
    acc23 = __hadd2(acc23, *reinterpret_cast<const __half2*>(partial_ptr + 2));
    acc45 = __hadd2(acc45, *reinterpret_cast<const __half2*>(partial_ptr + 4));
    acc67 = __hadd2(acc67, *reinterpret_cast<const __half2*>(partial_ptr + 6));
  }
  __half* out_ptr = out + col;
  *reinterpret_cast<__half2*>(out_ptr + 0) = acc01;
  *reinterpret_cast<__half2*>(out_ptr + 2) = acc23;
  *reinterpret_cast<__half2*>(out_ptr + 4) = acc45;
  *reinterpret_cast<__half2*>(out_ptr + 6) = acc67;
}

template <int Threads>
__global__ void nvfp4_raw_gemv_warp_kernel(
    __half* __restrict__ out, const __half* __restrict__ input,
    const uint32_t* __restrict__ qweight_packed,
    const __half* __restrict__ scales, int n, int num_groups) {
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = Threads / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int qword = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
  const int qwords = n / 8;
  if (qword >= qwords) {
    return;
  }

  const int half = lane >> 4;
  const int lane_in_half = lane & 15;
  const int col = qword * 8;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;

  float acc0 = 0.f;
  float acc1 = 0.f;
  float acc2 = 0.f;
  float acc3 = 0.f;
  float acc4 = 0.f;
  float acc5 = 0.f;
  float acc6 = 0.f;
  float acc7 = 0.f;

  for (int group_base = 0; group_base < num_groups; group_base += 2) {
    const int group = group_base + half;
    if (group >= num_groups) {
      continue;
    }

    const int source_lane_base = half * 16;
    const __half* scale_ptr = scales + group * n + col;
    const float loaded_scale =
        lane_in_half < 8 ? __half2float(scale_ptr[lane_in_half]) : 0.f;
    const float scale0 =
        __shfl_sync(half_mask, loaded_scale, source_lane_base + 0);
    const float scale1 =
        __shfl_sync(half_mask, loaded_scale, source_lane_base + 1);
    const float scale2 =
        __shfl_sync(half_mask, loaded_scale, source_lane_base + 2);
    const float scale3 =
        __shfl_sync(half_mask, loaded_scale, source_lane_base + 3);
    const float scale4 =
        __shfl_sync(half_mask, loaded_scale, source_lane_base + 4);
    const float scale5 =
        __shfl_sync(half_mask, loaded_scale, source_lane_base + 5);
    const float scale6 =
        __shfl_sync(half_mask, loaded_scale, source_lane_base + 6);
    const float scale7 =
        __shfl_sync(half_mask, loaded_scale, source_lane_base + 7);

    const int kk = group * 16 + lane_in_half;
    const float x = __half2float(input[kk]);
    const uint32_t packed = qweight_packed[kk * qwords + qword];
    acc0 += x * scale0 * kNvfp4E2m1Values[(packed >> 0) & 0xf];
    acc1 += x * scale1 * kNvfp4E2m1Values[(packed >> 4) & 0xf];
    acc2 += x * scale2 * kNvfp4E2m1Values[(packed >> 8) & 0xf];
    acc3 += x * scale3 * kNvfp4E2m1Values[(packed >> 12) & 0xf];
    acc4 += x * scale4 * kNvfp4E2m1Values[(packed >> 16) & 0xf];
    acc5 += x * scale5 * kNvfp4E2m1Values[(packed >> 20) & 0xf];
    acc6 += x * scale6 * kNvfp4E2m1Values[(packed >> 24) & 0xf];
    acc7 += x * scale7 * kNvfp4E2m1Values[(packed >> 28) & 0xf];
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
    acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
    acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    acc4 += __shfl_down_sync(0xffffffffu, acc4, offset);
    acc5 += __shfl_down_sync(0xffffffffu, acc5, offset);
    acc6 += __shfl_down_sync(0xffffffffu, acc6, offset);
    acc7 += __shfl_down_sync(0xffffffffu, acc7, offset);
  }

  if (lane == 0) {
    __half* out_ptr = out + col;
    out_ptr[0] = __float2half(acc0);
    out_ptr[1] = __float2half(acc1);
    out_ptr[2] = __float2half(acc2);
    out_ptr[3] = __float2half(acc3);
    out_ptr[4] = __float2half(acc4);
    out_ptr[5] = __float2half(acc5);
    out_ptr[6] = __float2half(acc6);
    out_ptr[7] = __float2half(acc7);
  }
}

namespace {

struct WorkspaceHolder {
  torch::Tensor barriers;
  torch::Tensor partials;
  torch::Tensor tensormaps;
  torch::Tensor flags;
  turbomind::gemm::Workspace workspace{};
};

struct GemmHolder {
  std::unique_ptr<turbomind::gemm::Gemm> gemm;
};

enum class TuneKeyKind : int {
  kGenericDense = 0,
  kAwqDense = 1,
  kFp8Dense = 2,
  kGenericMoe = 3,
  kAwqMoe = 4,
  kFp8Moe = 5,
  kMxfp4Dense = 6,
  kNvfp4Dense = 7,
};

struct DenseTuneKey {
  TuneKeyKind kind;
  int device;
  int m;
  int n;
  int k;
  int group_size;

  bool operator==(const DenseTuneKey& other) const {
    return kind == other.kind && device == other.device && m == other.m &&
           n == other.n && k == other.k && group_size == other.group_size;
  }
};

struct DenseTuneKeyHash {
  std::size_t operator()(const DenseTuneKey& key) const {
    std::size_t h = std::hash<int>()(static_cast<int>(key.kind));
    h ^= std::hash<int>()(key.device) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.m) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.group_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct MoeTuneKey {
  TuneKeyKind kind;
  int device;
  int total_tokens;
  int n;
  int k;
  int num_experts;
  int group_size;

  bool operator==(const MoeTuneKey& other) const {
    return kind == other.kind && device == other.device &&
           total_tokens == other.total_tokens && n == other.n && k == other.k &&
           num_experts == other.num_experts && group_size == other.group_size;
  }
};

struct MoeTuneKeyHash {
  std::size_t operator()(const MoeTuneKey& key) const {
    std::size_t h = std::hash<int>()(static_cast<int>(key.kind));
    h ^= std::hash<int>()(key.device) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.total_tokens) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.num_experts) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.group_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct Sm70F16WeightCacheKey {
  int device;
  const void* tensor_impl;
  int64_t rows;
  int64_t cols;

  bool operator==(const Sm70F16WeightCacheKey& other) const {
    return device == other.device && tensor_impl == other.tensor_impl &&
           rows == other.rows && cols == other.cols;
  }
};

struct Sm70F16WeightCacheKeyHash {
  std::size_t operator()(const Sm70F16WeightCacheKey& key) const {
    std::size_t h = std::hash<int>()(key.device);
    h ^= std::hash<const void*>()(key.tensor_impl) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<int64_t>()(key.rows) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>()(key.cols) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct Sm70F16WeightCacheEntry {
  torch::Tensor tm_weight;
  int64_t k_ld;
};

// Per-stream workspace management to eliminate mutex contention
struct StreamWorkspaceKey {
  int device;
  cudaStream_t stream;

  bool operator==(const StreamWorkspaceKey& other) const {
    return device == other.device && stream == other.stream;
  }
};

struct StreamWorkspaceKeyHash {
  std::size_t operator()(const StreamWorkspaceKey& k) const {
    return std::hash<int>()(k.device) ^
           (std::hash<cudaStream_t>()(k.stream) << 1);
  }
};

std::mutex workspace_mutex;
std::mutex gemm_mutex;
std::mutex tune_mutex;
std::mutex sm70_f16_weight_cache_mutex;
std::unordered_map<StreamWorkspaceKey, WorkspaceHolder, StreamWorkspaceKeyHash>
    workspace_cache;
std::unordered_map<int, GemmHolder> gemm_cache;
std::unordered_set<DenseTuneKey, DenseTuneKeyHash> dense_tuned_shapes;
std::unordered_set<MoeTuneKey, MoeTuneKeyHash> moe_tuned_shapes;
std::unordered_set<int> imported_cache_devices;
std::unordered_map<Sm70F16WeightCacheKey, Sm70F16WeightCacheEntry,
                   Sm70F16WeightCacheKeyHash>
    sm70_f16_weight_cache;

turbomind::gemm::DispatchPolicy select_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream);
turbomind::gemm::DispatchPolicy select_awq_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream);
turbomind::gemm::DispatchPolicy select_fp8_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream);
turbomind::gemm::DispatchPolicy select_mxfp4_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream);
turbomind::gemm::DispatchPolicy select_nvfp4_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream);
turbomind::gemm::DispatchPolicy select_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream);
turbomind::gemm::DispatchPolicy select_fp8_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream);

bool tune_small_shapes_enabled() {
  const char* raw = std::getenv("VLLM_SM70_AWQ_TUNE_SMALL_SHAPES");
  return raw == nullptr || std::atoi(raw) != 0;
}

bool awq_tune_small_shapes_enabled() {
  const char* raw = std::getenv("VLLM_SM70_AWQ_TUNE_SMALL_SHAPES");
  return raw != nullptr && std::atoi(raw) != 0;
}

bool fp8_tune_small_shapes_enabled() {
  const char* raw = std::getenv("VLLM_SM70_FP8_TUNE_SMALL_SHAPES");
  return raw == nullptr || std::atoi(raw) != 0;
}

bool mxfp4_tune_small_shapes_enabled() {
  const char* raw = std::getenv("VLLM_SM70_MXFP4_TUNE_SMALL_SHAPES");
  return raw == nullptr || std::atoi(raw) != 0;
}

bool nvfp4_tune_small_shapes_enabled() {
  const char* raw = std::getenv("VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES");
  return raw == nullptr || std::atoi(raw) != 0;
}

bool fp8_moe_single_token_per_expert_dispatch_enabled() {
  const char* raw =
      std::getenv("VLLM_SM70_FP8_MOE_SINGLE_TOKEN_PER_EXPERT_DISPATCH");
  return raw != nullptr && std::atoi(raw) != 0;
}

bool fp8_0dot3_dense_selector_enabled() {
  const char* raw = std::getenv("VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR");
  return raw != nullptr && std::atoi(raw) != 0;
}

bool fp8_safe_fast_selector_enabled() {
  const char* raw = std::getenv("VLLM_SM70_FP8_SAFE_FAST_SELECTOR");
  return raw != nullptr && std::atoi(raw) != 0;
}

bool awq_reuse_imported_cache_enabled() {
  const char* raw = std::getenv("VLLM_SM70_AWQ_REUSE_IMPORTED_CACHE");
  return raw != nullptr && std::atoi(raw) != 0;
}

bool awq_preserve_default_splits_enabled() {
  const char* raw = std::getenv("VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS");
  return raw == nullptr || std::atoi(raw) != 0;
}

bool awq_preserve_default_splits_only_enabled() {
  const char* raw = std::getenv("VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS_ONLY");
  return raw != nullptr && std::atoi(raw) != 0;
}

bool fp8_preserve_default_splits_enabled() {
  const char* raw = std::getenv("VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS");
  return raw == nullptr || std::atoi(raw) != 0;
}

bool fp8_preserve_default_splits_only_enabled() {
  const char* raw = std::getenv("VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY");
  return raw != nullptr && std::atoi(raw) != 0;
}

inline turbomind::gemm::DispatchPolicy maybe_preserve_default_splits(
    turbomind::gemm::DispatchPolicy policy) {
  if (policy == turbomind::gemm::DispatchPolicy::kMeasure ||
      policy == turbomind::gemm::DispatchPolicy::kReuse) {
    if (awq_preserve_default_splits_only_enabled()) {
      return policy |
             turbomind::gemm::DispatchPolicy::kPreserveDefaultSplitCount;
    }
    if (!awq_preserve_default_splits_enabled()) {
      return policy;
    }
    return policy | turbomind::gemm::DispatchPolicy::kPreserveDefaultSplits;
  }
  return policy;
}

inline turbomind::gemm::DispatchPolicy maybe_preserve_fp8_default_splits(
    turbomind::gemm::DispatchPolicy policy) {
  if (policy == turbomind::gemm::DispatchPolicy::kMeasure ||
      policy == turbomind::gemm::DispatchPolicy::kReuse) {
    if (fp8_preserve_default_splits_only_enabled()) {
      return policy |
             turbomind::gemm::DispatchPolicy::kPreserveDefaultSplitCount;
    }
    if (!fp8_preserve_default_splits_enabled()) {
      return policy;
    }
    return policy | turbomind::gemm::DispatchPolicy::kPreserveDefaultSplits;
  }
  return policy;
}

std::optional<turbomind::gemm::DispatchPolicy>
dispatch_policy_override_from_env(const char* env_name) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || std::strcmp(raw, "") == 0) {
    return std::nullopt;
  }
  if (std::strcmp(raw, "default") == 0) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  if (std::strcmp(raw, "reuse") == 0) {
    return turbomind::gemm::DispatchPolicy::kReuse;
  }
  if (std::strcmp(raw, "measure") == 0) {
    return turbomind::gemm::DispatchPolicy::kMeasure;
  }
  TORCH_CHECK(false, env_name, " must be one of: default, reuse, measure.");
  return std::nullopt;
}

std::optional<turbomind::gemm::DispatchPolicy>
awq_moe_dispatch_policy_override() {
  return dispatch_policy_override_from_env("VLLM_SM70_AWQ_MOE_DISPATCH_POLICY");
}

int awq_dense_tune_max_m() {
  const char* raw = std::getenv("VLLM_SM70_AWQ_DENSE_TUNE_MAX_M");
  return raw ? std::max(std::atoi(raw), 0) : 16;
}

int generic_dense_tune_max_m() {
  const char* raw = std::getenv("VLLM_SM70_F16_DENSE_TUNE_MAX_M");
  return raw ? std::max(std::atoi(raw), 0) : 16;
}

int fp8_dense_tune_max_m() {
  const char* raw = std::getenv("VLLM_SM70_FP8_DENSE_TUNE_MAX_M");
  return raw ? std::max(std::atoi(raw), 0) : 16;
}

int mxfp4_dense_tune_max_m() {
  const char* raw = std::getenv("VLLM_SM70_MXFP4_DENSE_TUNE_MAX_M");
  return raw ? std::max(std::atoi(raw), 0) : 16;
}

int nvfp4_dense_tune_max_m() {
  const char* raw = std::getenv("VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M");
  return raw ? std::max(std::atoi(raw), 0) : 16;
}

int moe_tune_max_tokens() {
  const char* raw = std::getenv("VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS");
  return raw ? std::max(std::atoi(raw), 0) : 128;
}

int sm70_f16_dense_max_m() {
  const char* raw = std::getenv("VLLM_SM70_F16_DENSE_MAX_M");
  return raw ? std::max(std::atoi(raw), 0) : 64;
}

bool is_stream_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  const auto ec = cudaStreamIsCapturing(stream, &status);
  if (ec != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return status != cudaStreamCaptureStatusNone;
}

bool has_imported_cache(int device) {
  std::lock_guard<std::mutex> lock(tune_mutex);
  return imported_cache_devices.find(device) != imported_cache_devices.end();
}

turbomind::gemm::DispatchPolicy select_dense_dispatch_policy_impl(
    int device, int m, int n, int k, int group_size, cudaStream_t stream,
    TuneKeyKind kind, bool tune_enabled, bool reuse_imported_cache, int max_m) {
  if (m > max_m) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  if (reuse_imported_cache && has_imported_cache(device)) {
    return turbomind::gemm::DispatchPolicy::kReuse;
  }
  if (!tune_enabled) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }

  DenseTuneKey key{kind, device, m, n, k, group_size};
  std::lock_guard<std::mutex> lock(tune_mutex);
  if (dense_tuned_shapes.find(key) != dense_tuned_shapes.end()) {
    return turbomind::gemm::DispatchPolicy::kReuse;
  }
  if (is_stream_capturing(stream)) {
    if (has_imported_cache(device)) {
      return turbomind::gemm::DispatchPolicy::kReuse;
    }
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  dense_tuned_shapes.insert(key);
  return turbomind::gemm::DispatchPolicy::kMeasure;
}

turbomind::gemm::DispatchPolicy select_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  return select_dense_dispatch_policy_impl(
      device, m, n, k, group_size, stream, TuneKeyKind::kGenericDense,
      tune_small_shapes_enabled(), false, generic_dense_tune_max_m());
}

turbomind::gemm::DispatchPolicy select_awq_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  return maybe_preserve_default_splits(select_dense_dispatch_policy_impl(
      device, m, n, k, group_size, stream, TuneKeyKind::kAwqDense,
      awq_tune_small_shapes_enabled(), awq_reuse_imported_cache_enabled(),
      awq_dense_tune_max_m()));
}

turbomind::gemm::DispatchPolicy select_fp8_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  if (fp8_0dot3_dense_selector_enabled()) {
    return select_dense_dispatch_policy_impl(
        device, m, n, k, group_size, stream, TuneKeyKind::kGenericDense,
        tune_small_shapes_enabled(), false, generic_dense_tune_max_m());
  }
  auto policy = select_dense_dispatch_policy_impl(
      device, m, n, k, group_size, stream, TuneKeyKind::kFp8Dense,
      fp8_tune_small_shapes_enabled(), false, fp8_dense_tune_max_m());
  if (!fp8_safe_fast_selector_enabled()) {
    return policy;
  }
  return maybe_preserve_fp8_default_splits(policy);
}

turbomind::gemm::DispatchPolicy select_mxfp4_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  return select_dense_dispatch_policy_impl(
      device, m, n, k, group_size, stream, TuneKeyKind::kMxfp4Dense,
      mxfp4_tune_small_shapes_enabled(), false, mxfp4_dense_tune_max_m());
}

turbomind::gemm::DispatchPolicy select_nvfp4_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  return select_dense_dispatch_policy_impl(
      device, m, n, k, group_size, stream, TuneKeyKind::kNvfp4Dense,
      nvfp4_tune_small_shapes_enabled(), false, nvfp4_dense_tune_max_m());
}

turbomind::gemm::DispatchPolicy select_moe_dispatch_policy_impl(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream, TuneKeyKind kind, bool tune_enabled) {
  if (!tune_enabled || total_tokens > moe_tune_max_tokens()) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }

  MoeTuneKey key{kind, device, total_tokens, n, k, num_experts, group_size};
  std::lock_guard<std::mutex> lock(tune_mutex);
  if (moe_tuned_shapes.find(key) != moe_tuned_shapes.end()) {
    return turbomind::gemm::DispatchPolicy::kReuse;
  }
  if (is_stream_capturing(stream)) {
    if (has_imported_cache(device)) {
      return turbomind::gemm::DispatchPolicy::kReuse;
    }
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  moe_tuned_shapes.insert(key);
  return turbomind::gemm::DispatchPolicy::kMeasure;
}

turbomind::gemm::DispatchPolicy select_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream) {
  return select_moe_dispatch_policy_impl(
      device, total_tokens, n, k, num_experts, group_size, stream,
      TuneKeyKind::kGenericMoe, tune_small_shapes_enabled());
}

turbomind::gemm::DispatchPolicy select_fp8_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream) {
  return select_moe_dispatch_policy_impl(
      device, total_tokens, n, k, num_experts, group_size, stream,
      TuneKeyKind::kFp8Moe, fp8_tune_small_shapes_enabled());
}

WorkspaceHolder& get_workspace(int device, cudaStream_t stream) {
  thread_local int cached_device = -1;
  thread_local cudaStream_t cached_stream = nullptr;
  thread_local WorkspaceHolder* cached_holder = nullptr;
  if (cached_holder != nullptr && cached_device == device &&
      cached_stream == stream) {
    return *cached_holder;
  }

  StreamWorkspaceKey key{device, stream};

  // Fast path: check if workspace exists without lock
  {
    std::lock_guard<std::mutex> lock(workspace_mutex);
    auto it = workspace_cache.find(key);
    if (it != workspace_cache.end()) {
      cached_device = device;
      cached_stream = stream;
      cached_holder = &it->second;
      return it->second;
    }
  }

  // Slow path: create new workspace
  WorkspaceHolder holder;
  auto byte_opts = torch::TensorOptions()
                       .device(torch::Device(torch::kCUDA, device))
                       .dtype(torch::kUInt8);
  auto int_opts = torch::TensorOptions()
                      .device(torch::Device(torch::kCUDA, device))
                      .dtype(torch::kInt32);

  holder.barriers = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kBarriersSize}, byte_opts);
  holder.partials = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kPartialsSize}, byte_opts);
  // Keep same tensormap size as TurboMind LlamaLinear.
  holder.tensormaps = torch::empty({(long long)(8192 * 128)}, byte_opts);
  holder.flags = torch::zeros({1}, int_opts);

  holder.workspace.barriers = holder.barriers.data_ptr();
  holder.workspace.barriers_size = holder.barriers.numel();
  holder.workspace.partials = holder.partials.data_ptr();
  holder.workspace.partials_size = holder.partials.numel();
  holder.workspace.tensormaps = holder.tensormaps.data_ptr();
  holder.workspace.tensormaps_size = holder.tensormaps.numel();
  holder.workspace.flags = holder.flags.data_ptr<int>();

  std::lock_guard<std::mutex> lock(workspace_mutex);
  auto [insert_it, _] = workspace_cache.emplace(key, std::move(holder));
  cached_device = device;
  cached_stream = stream;
  cached_holder = &insert_it->second;
  return insert_it->second;
}

turbomind::gemm::Gemm& get_gemm(int device) {
  thread_local int cached_device = -1;
  thread_local turbomind::gemm::Gemm* cached_gemm = nullptr;
  if (cached_gemm != nullptr && cached_device == device) {
    return *cached_gemm;
  }

  std::lock_guard<std::mutex> lock(gemm_mutex);
  auto it = gemm_cache.find(device);
  if (it != gemm_cache.end()) {
    cached_device = device;
    cached_gemm = it->second.gemm.get();
    return *it->second.gemm;
  }
  GemmHolder holder;
  holder.gemm = std::make_unique<turbomind::gemm::Gemm>();
  auto [insert_it, _] = gemm_cache.emplace(device, std::move(holder));
  cached_device = device;
  cached_gemm = insert_it->second.gemm.get();
  return *insert_it->second.gemm;
}

void validate_awq_inputs(const torch::Tensor& qweight,
                         const torch::Tensor& scales,
                         const torch::Tensor& qzeros) {
  TORCH_CHECK(qweight.is_cuda(), "awq_sm70_prepare: qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), "awq_sm70_prepare: scales must be CUDA.");
  TORCH_CHECK(qzeros.is_cuda(), "awq_sm70_prepare: qzeros must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32,
              "awq_sm70_prepare: qweight must be int32.");
  TORCH_CHECK(qzeros.scalar_type() == torch::kInt32,
              "awq_sm70_prepare: qzeros must be int32.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16,
              "awq_sm70_prepare: scales must be float16.");
}

void validate_uint4_inputs(const torch::Tensor& qweight,
                           const torch::Tensor& scales,
                           const torch::Tensor& qzeros, const char* op_name) {
  TORCH_CHECK(qweight.is_cuda(), op_name, ": qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), op_name, ": scales must be CUDA.");
  TORCH_CHECK(qzeros.is_cuda(), op_name, ": qzeros must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8, op_name,
              ": qweight must be unpacked uint8.");
  TORCH_CHECK(qzeros.scalar_type() == torch::kFloat16, op_name,
              ": qzeros must be float16.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16, op_name,
              ": scales must be float16.");
  TORCH_CHECK(qweight.dim() == 2, op_name, ": qweight must be 2D.");
  TORCH_CHECK(scales.dim() == 2, op_name, ": scales must be 2D.");
  TORCH_CHECK(qzeros.dim() == 2, op_name, ": qzeros must be 2D.");
}

void validate_mxfp4_inputs(const torch::Tensor& qweight,
                           const torch::Tensor& scales, int64_t group_size,
                           const char* op_name) {
  TORCH_CHECK(qweight.is_cuda(), op_name, ": qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), op_name, ": scales must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8, op_name,
              ": qweight must be unpacked uint8.");
  TORCH_CHECK(scales.scalar_type() == torch::kUInt8, op_name,
              ": scales must be uint8 E8M0.");
  TORCH_CHECK(qweight.dim() == 2, op_name, ": qweight must be 2D.");
  TORCH_CHECK(scales.dim() == 2, op_name, ": scales must be 2D.");
  TORCH_CHECK(group_size == 32, op_name, ": only group_size=32 is supported.");
}

void validate_nvfp4_inputs(const torch::Tensor& qweight,
                           const torch::Tensor& scales, int64_t group_size,
                           const char* op_name) {
  TORCH_CHECK(qweight.is_cuda(), op_name, ": qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), op_name, ": scales must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8, op_name,
              ": qweight must be unpacked uint8.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16, op_name,
              ": scales must be float16.");
  TORCH_CHECK(qweight.dim() == 2, op_name, ": qweight must be 2D.");
  TORCH_CHECK(scales.dim() == 2, op_name, ": scales must be 2D.");
  TORCH_CHECK(group_size == 16, op_name, ": only group_size=16 is supported.");
}

void validate_fp8_inputs(const torch::Tensor& qweight,
                         const torch::Tensor& scales, int64_t group_size) {
  TORCH_CHECK(qweight.is_cuda(), "fp8_sm70_prepare: qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), "fp8_sm70_prepare: scales must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_sm70_prepare: qweight must be float8_e4m3fn.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "fp8_sm70_prepare: scales must be float32.");
  TORCH_CHECK(qweight.dim() == 2, "fp8_sm70_prepare: qweight must be 2D.");
  TORCH_CHECK(scales.dim() == 2, "fp8_sm70_prepare: scales must be 2D.");
  TORCH_CHECK(group_size == 128,
              "fp8_sm70_prepare: only group_size=128 is supported.");
}

void validate_f16_weight(const torch::Tensor& weight, const char* op_name) {
  TORCH_CHECK(weight.is_cuda(), op_name, ": weight must be CUDA.");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16, op_name,
              ": weight must be float16.");
  TORCH_CHECK(weight.dim() == 2, op_name, ": weight must be 2D.");
}

void validate_f16_input(const torch::Tensor& in_feats,
                        const torch::Tensor& tm_weight,
                        const torch::Tensor& out, bool gated_silu,
                        const char* op_name) {
  TORCH_CHECK(in_feats.is_cuda(), op_name, ": input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), op_name, ": weight must be CUDA.");
  TORCH_CHECK(out.is_cuda(), op_name, ": output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16, op_name,
              ": input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kFloat16, op_name,
              ": weight must be float16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16, op_name,
              ": output must be float16.");
  TORCH_CHECK(in_feats.dim() == 2, op_name, ": input must be 2D.");
  TORCH_CHECK(tm_weight.dim() == 2, op_name, ": weight must be 2D.");
  TORCH_CHECK(out.dim() == 2, op_name, ": output must be 2D.");
}

void validate_f16_gate_mul_input(const torch::Tensor& out,
                                 const torch::Tensor& in_feats,
                                 const torch::Tensor& gate_weight) {
  TORCH_CHECK(out.is_cuda(), "sm70_f16_gate_mul_out: out must be CUDA.");
  TORCH_CHECK(in_feats.is_cuda(), "sm70_f16_gate_mul_out: input must be CUDA.");
  TORCH_CHECK(gate_weight.is_cuda(),
              "sm70_f16_gate_mul_out: gate weight must be CUDA.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "sm70_f16_gate_mul_out: out must be float16.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "sm70_f16_gate_mul_out: input must be float16.");
  TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat16,
              "sm70_f16_gate_mul_out: gate weight must be float16.");
  TORCH_CHECK(out.dim() == 2, "sm70_f16_gate_mul_out: out must be 2D.");
  TORCH_CHECK(in_feats.dim() == 2, "sm70_f16_gate_mul_out: input must be 2D.");
  TORCH_CHECK(gate_weight.dim() == 2,
              "sm70_f16_gate_mul_out: gate weight must be 2D.");
  TORCH_CHECK(gate_weight.size(0) == 1,
              "sm70_f16_gate_mul_out: gate weight must have one output row.");
  TORCH_CHECK(out.size(0) == in_feats.size(0),
              "sm70_f16_gate_mul_out: out/input batch mismatch.");
  TORCH_CHECK(out.stride(1) == 1,
              "sm70_f16_gate_mul_out: out must be row-major contiguous.");
  TORCH_CHECK(in_feats.stride(1) == 1,
              "sm70_f16_gate_mul_out: input must be row-major contiguous.");
  TORCH_CHECK(gate_weight.stride(1) == 1,
              "sm70_f16_gate_mul_out: gate weight must be contiguous.");
  TORCH_CHECK(in_feats.size(1) == gate_weight.size(1),
              "sm70_f16_gate_mul_out: input/gate K mismatch.");
}

void validate_f16_lm_head_top1_input(const torch::Tensor& values_out,
                                     const torch::Tensor& indices_out,
                                     const torch::Tensor& in_feats,
                                     const torch::Tensor& weight, int64_t k_ld,
                                     int64_t num_vocab_padding, int64_t max_m) {
  TORCH_CHECK(values_out.is_cuda(),
              "sm70_f16_lm_head_top1_out: values_out must be CUDA.");
  TORCH_CHECK(indices_out.is_cuda(),
              "sm70_f16_lm_head_top1_out: indices_out must be CUDA.");
  TORCH_CHECK(in_feats.is_cuda(),
              "sm70_f16_lm_head_top1_out: input must be CUDA.");
  TORCH_CHECK(weight.is_cuda(),
              "sm70_f16_lm_head_top1_out: weight must be CUDA.");
  TORCH_CHECK(values_out.scalar_type() == torch::kFloat32,
              "sm70_f16_lm_head_top1_out: values_out must be float32.");
  TORCH_CHECK(indices_out.scalar_type() == torch::kInt64,
              "sm70_f16_lm_head_top1_out: indices_out must be int64.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "sm70_f16_lm_head_top1_out: input must be float16.");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16,
              "sm70_f16_lm_head_top1_out: weight must be float16.");
  TORCH_CHECK(values_out.dim() == 1,
              "sm70_f16_lm_head_top1_out: values_out must be 1D.");
  TORCH_CHECK(indices_out.dim() == 1,
              "sm70_f16_lm_head_top1_out: indices_out must be 1D.");
  TORCH_CHECK(in_feats.dim() == 2,
              "sm70_f16_lm_head_top1_out: input must be 2D.");
  TORCH_CHECK(weight.dim() == 2,
              "sm70_f16_lm_head_top1_out: weight must be 2D.");
  TORCH_CHECK(values_out.size(0) == in_feats.size(0),
              "sm70_f16_lm_head_top1_out: values/input batch mismatch.");
  TORCH_CHECK(indices_out.size(0) == in_feats.size(0),
              "sm70_f16_lm_head_top1_out: indices/input batch mismatch.");
  TORCH_CHECK(in_feats.size(1) == weight.size(1),
              "sm70_f16_lm_head_top1_out: input/weight K mismatch.");
  TORCH_CHECK(in_feats.size(0) >= 1 && in_feats.size(0) <= max_m,
              "sm70_f16_lm_head_top1_out: unsupported decode M.");
  TORCH_CHECK(values_out.is_contiguous(),
              "sm70_f16_lm_head_top1_out: values_out must be contiguous.");
  TORCH_CHECK(indices_out.is_contiguous(),
              "sm70_f16_lm_head_top1_out: indices_out must be contiguous.");
  TORCH_CHECK(in_feats.stride(1) == 1,
              "sm70_f16_lm_head_top1_out: input must be row-major.");
  TORCH_CHECK(weight.stride(1) == 1,
              "sm70_f16_lm_head_top1_out: weight must be row-major.");
  TORCH_CHECK(k_ld == 0 || k_ld >= weight.size(1),
              "sm70_f16_lm_head_top1_out: invalid weight leading dim.");
  TORCH_CHECK(num_vocab_padding >= 0 && num_vocab_padding < weight.size(0),
              "sm70_f16_lm_head_top1_out: invalid vocab padding.");
}

void validate_f16_lm_head_top20_input(const torch::Tensor& values_out,
                                      const torch::Tensor& indices_out,
                                      const torch::Tensor& in_feats,
                                      const torch::Tensor& weight, int64_t k_ld,
                                      int64_t num_vocab_padding,
                                      int64_t max_m) {
  TORCH_CHECK(values_out.is_cuda() && indices_out.is_cuda() &&
                  in_feats.is_cuda() && weight.is_cuda(),
              "sm70_f16_lm_head_top20_tc_out: tensors must be CUDA.");
  TORCH_CHECK(values_out.scalar_type() == torch::kFloat32,
              "sm70_f16_lm_head_top20_tc_out: values must be float32.");
  TORCH_CHECK(indices_out.scalar_type() == torch::kInt64,
              "sm70_f16_lm_head_top20_tc_out: indices must be int64.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16 &&
                  weight.scalar_type() == torch::kFloat16,
              "sm70_f16_lm_head_top20_tc_out: input/weight must be float16.");
  TORCH_CHECK(values_out.dim() == 2 && values_out.size(1) == 20,
              "sm70_f16_lm_head_top20_tc_out: values must be [M, 20].");
  TORCH_CHECK(indices_out.sizes() == values_out.sizes(),
              "sm70_f16_lm_head_top20_tc_out: output shapes must match.");
  TORCH_CHECK(in_feats.dim() == 2 && weight.dim() == 2,
              "sm70_f16_lm_head_top20_tc_out: input/weight must be 2D.");
  TORCH_CHECK(values_out.size(0) == in_feats.size(0),
              "sm70_f16_lm_head_top20_tc_out: batch mismatch.");
  TORCH_CHECK(in_feats.size(1) == weight.size(1),
              "sm70_f16_lm_head_top20_tc_out: K mismatch.");
  TORCH_CHECK(in_feats.size(0) >= 1 && in_feats.size(0) <= max_m,
              "sm70_f16_lm_head_top20_tc_out: unsupported decode M.");
  TORCH_CHECK(values_out.is_contiguous() && indices_out.is_contiguous() &&
                  in_feats.stride(1) == 1 && weight.stride(1) == 1,
              "sm70_f16_lm_head_top20_tc_out: tensors must be contiguous.");
  TORCH_CHECK(k_ld >= weight.size(1),
              "sm70_f16_lm_head_top20_tc_out: invalid weight leading dim.");
  TORCH_CHECK(num_vocab_padding >= 0 && num_vocab_padding < weight.size(0),
              "sm70_f16_lm_head_top20_tc_out: invalid vocab padding.");
}

Sm70F16WeightCacheKey make_sm70_f16_weight_cache_key(
    const torch::Tensor& weight) {
  return Sm70F16WeightCacheKey{
      weight.get_device(),
      static_cast<const void*>(weight.unsafeGetTensorImpl()),
      weight.size(0),
      weight.size(1),
  };
}

Sm70F16WeightCacheEntry prepare_sm70_f16_weight(torch::Tensor weight,
                                                cudaStream_t stream) {
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1);

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kHalf, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  TORCH_CHECK(conv_w, "sm70_f16_prepare: no compatible TurboMind converter.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::kHalf;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty_like(weight);
  TORCH_CHECK(conv_w->Convert(weight.data_ptr(), w_desc, tm_weight.data_ptr(),
                              k_desc, stream) == 0,
              "sm70_f16_prepare: weight conversion failed.");

  return {std::move(tm_weight), static_cast<int64_t>(k_desc.ld)};
}

Sm70F16WeightCacheEntry get_sm70_f16_cached_weight(torch::Tensor weight,
                                                   cudaStream_t stream) {
  weight = weight.contiguous();
  const auto key = make_sm70_f16_weight_cache_key(weight);

  {
    std::lock_guard<std::mutex> lock(sm70_f16_weight_cache_mutex);
    auto it = sm70_f16_weight_cache.find(key);
    if (it != sm70_f16_weight_cache.end()) {
      return it->second;
    }
  }

  TORCH_CHECK(!is_stream_capturing(stream),
              "sm70_f16_prepare: cache miss during CUDA graph capture.");

  auto entry = prepare_sm70_f16_weight(weight, stream);

  std::lock_guard<std::mutex> lock(sm70_f16_weight_cache_mutex);
  auto [it, _] = sm70_f16_weight_cache.emplace(key, entry);
  return it->second;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <int THREADS>
__global__ void sm70_f16_gate_mul_kernel(half* out, const half* in_feats,
                                         const half* gate_weight, int out_rows,
                                         int out_cols, int k,
                                         int64_t out_row_stride,
                                         int64_t in_row_stride) {
  const int row = blockIdx.x;
  if (row >= out_rows) {
    return;
  }

  const int tid = threadIdx.x;
  const half* x_row = in_feats + row * in_row_stride;

  float dot = 0.f;
  const int k2 = k >> 1;
  const half2* x2 = reinterpret_cast<const half2*>(x_row);
  const half2* w2 = reinterpret_cast<const half2*>(gate_weight);

  for (int idx = tid; idx < k2; idx += THREADS) {
    const half2 a = x2[idx];
    const half2 b = w2[idx];
    const half2 prod = __hmul2(a, b);
    dot += __half2float(__low2half(prod));
    dot += __half2float(__high2half(prod));
  }
  if ((k & 1) && tid == 0) {
    dot += __half2float(x_row[k - 1]) * __half2float(gate_weight[k - 1]);
  }

  dot = warp_reduce_sum(dot);

  __shared__ float block_sum[THREADS / 32];
  if ((tid & 31) == 0) {
    block_sum[tid >> 5] = dot;
  }
  __syncthreads();

  if (tid < 32) {
    float sum = tid < (THREADS / 32) ? block_sum[tid] : 0.f;
    sum = warp_reduce_sum(sum);
    if (tid == 0) {
      block_sum[0] = 1.f / (1.f + __expf(-sum));
    }
  }
  __syncthreads();

  const float gate = block_sum[0];
  const half2 gate2 = __float2half2_rn(gate);
  half* out_row = out + row * out_row_stride;
  half2* out_row2 = reinterpret_cast<half2*>(out_row);
  const int out_cols2 = out_cols >> 1;
  for (int idx = tid; idx < out_cols2; idx += THREADS) {
    out_row2[idx] = __hmul2(out_row2[idx], gate2);
  }
  if ((out_cols & 1) && tid == 0) {
    out_row[out_cols - 1] =
        __float2half(__half2float(out_row[out_cols - 1]) * gate);
  }
}

__device__ __forceinline__ bool top1_better(float val, int64_t idx,
                                            float best_val, int64_t best_idx) {
  return val > best_val || (val == best_val && idx < best_idx);
}

template <int ROWS_PER_BLOCK>
__global__ void sm70_f16_lm_head_top1_stage1_kernel(
    float* partial_values, int64_t* partial_indices, const half* in_feats,
    const half* weight, int m, int n, int k, int valid_n, int64_t in_row_stride,
    int64_t weight_row_stride, int64_t vocab_start_index, int num_blocks_n) {
  const int row = blockIdx.y;
  const int block_n = blockIdx.x;
  if (row >= m) {
    return;
  }

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int local_col = block_n * ROWS_PER_BLOCK + warp;
  const bool valid_col = warp < ROWS_PER_BLOCK && local_col < valid_n;

  float dot = 0.f;
  if (valid_col) {
    const half* x_row = in_feats + row * in_row_stride;
    const half* w_row = weight + local_col * weight_row_stride;
    const int k2 = k >> 1;
    const half2* x2 = reinterpret_cast<const half2*>(x_row);
    const half2* w2 = reinterpret_cast<const half2*>(w_row);
    for (int idx = lane; idx < k2; idx += 32) {
      const half2 prod = __hmul2(x2[idx], w2[idx]);
      dot += __half2float(__low2half(prod));
      dot += __half2float(__high2half(prod));
    }
    if ((k & 1) && lane == 0) {
      dot += __half2float(x_row[k - 1]) * __half2float(w_row[k - 1]);
    }
  }

  dot = warp_reduce_sum(dot);

  __shared__ float warp_values[ROWS_PER_BLOCK];
  __shared__ int64_t warp_indices[ROWS_PER_BLOCK];
  if (lane == 0 && warp < ROWS_PER_BLOCK) {
    if (valid_col) {
      warp_values[warp] = __half2float(__float2half_rn(dot));
      warp_indices[warp] = vocab_start_index + local_col;
    } else {
      warp_values[warp] = -3.402823466e38F;
      warp_indices[warp] = INT64_MAX;
    }
  }
  __syncthreads();

  if (warp == 0) {
    float best_val =
        lane < ROWS_PER_BLOCK ? warp_values[lane] : -3.402823466e38F;
    int64_t best_idx = lane < ROWS_PER_BLOCK ? warp_indices[lane] : INT64_MAX;
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
      const int64_t other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
      if (top1_better(other_val, other_idx, best_val, best_idx)) {
        best_val = other_val;
        best_idx = other_idx;
      }
    }
    if (lane == 0) {
      const int out_offset = row * num_blocks_n + block_n;
      partial_values[out_offset] = best_val;
      partial_indices[out_offset] = best_idx;
    }
  }
}

template <int THREADS>
__global__ void sm70_f16_lm_head_top1_stage2_kernel(
    float* values_out, int64_t* indices_out, const float* partial_values,
    const int64_t* partial_indices, int m, int num_blocks_n) {
  const int row = blockIdx.x;
  if (row >= m) {
    return;
  }

  const int tid = threadIdx.x;
  float best_val = -3.402823466e38F;
  int64_t best_idx = INT64_MAX;
  const int base = row * num_blocks_n;
  for (int idx = tid; idx < num_blocks_n; idx += THREADS) {
    const float val = partial_values[base + idx];
    const int64_t token = partial_indices[base + idx];
    if (top1_better(val, token, best_val, best_idx)) {
      best_val = val;
      best_idx = token;
    }
  }

  __shared__ float thread_values[THREADS];
  __shared__ int64_t thread_indices[THREADS];
  thread_values[tid] = best_val;
  thread_indices[tid] = best_idx;
  __syncthreads();

  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_val = thread_values[tid + stride];
      const int64_t other_idx = thread_indices[tid + stride];
      if (top1_better(other_val, other_idx, thread_values[tid],
                      thread_indices[tid])) {
        thread_values[tid] = other_val;
        thread_indices[tid] = other_idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    values_out[row] = thread_values[0];
    indices_out[row] = thread_indices[0];
  }
}

__global__ void sm70_f16_lm_head_top20_stage2_kernel(
    float* values_out, int64_t* indices_out, float* partial_values,
    const int64_t* partial_indices, int m, int num_candidates,
    int64_t vocab_start_index, int valid_n) {
  constexpr int kTopK = 20;
  const int row = blockIdx.x;
  if (row >= m) {
    return;
  }

  const int lane = threadIdx.x;
  const int base = row * num_candidates;
  for (int rank = 0; rank < kTopK; ++rank) {
    float best_val = -FLT_MAX;
    int64_t best_idx = INT64_MAX;
    int best_pos = -1;
    for (int pos = lane; pos < num_candidates; pos += WARP_SIZE) {
      const float raw_val = partial_values[base + pos];
      const int64_t token = partial_indices[base + pos];
      if (token < vocab_start_index || token >= vocab_start_index + valid_n) {
        continue;
      }
      const float val = isfinite(raw_val) ? raw_val : -FLT_MAX;
      if (top1_better(val, token, best_val, best_idx)) {
        best_val = val;
        best_idx = token;
        best_pos = pos;
      }
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      const float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
      const int64_t other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
      const int other_pos = __shfl_down_sync(0xffffffff, best_pos, offset);
      if (top1_better(other_val, other_idx, best_val, best_idx)) {
        best_val = other_val;
        best_idx = other_idx;
        best_pos = other_pos;
      }
    }
    if (lane == 0) {
      if (best_pos < 0) {
        best_val = -FLT_MAX;
        best_idx = vocab_start_index + rank % valid_n;
      }
      values_out[row * kTopK + rank] = best_val;
      indices_out[row * kTopK + rank] = best_idx;
      if (best_pos >= 0) {
        partial_values[base + best_pos] = -INFINITY;
      }
    }
    __syncwarp();
  }
}

__global__ void sm70_merge_tail_top20_pack_kernel(
    float* pairs_out, const float* base_values, const int64_t* base_indices,
    const int64_t* base_token_id_map, const half* tail_logits,
    const int64_t* tail_token_ids, int base_token_id_count, int tail_size,
    int tail_row_start) {
  constexpr int kTopK = 20;
  const int lane = threadIdx.x;
  const int candidate_count = kTopK + tail_size;
  extern __shared__ float candidate_values[];
  for (int pos = lane; pos < candidate_count; pos += WARP_SIZE) {
    candidate_values[pos] =
        pos < kTopK ? base_values[pos] : __half2float(tail_logits[pos - kTopK]);
  }
  __syncwarp();

  for (int rank = 0; rank < kTopK; ++rank) {
    float best_val = -FLT_MAX;
    int64_t best_idx = INT64_MAX;
    int best_pos = -1;
    for (int pos = lane; pos < candidate_count; pos += WARP_SIZE) {
      const bool from_base = pos < kTopK;
      const int tail_pos = pos - kTopK;
      const int64_t base_index = from_base ? base_indices[pos] : -1;
      const bool valid_base_index =
          from_base && base_index >= 0 && base_index < base_token_id_count;
      const int64_t token = valid_base_index
                                ? base_token_id_map[base_index]
                                : (from_base ? -1 : tail_token_ids[tail_pos]);
      const float raw_val = candidate_values[pos];
      const float val = token < 0 || !isfinite(raw_val) ? -FLT_MAX : raw_val;
      if (token < 0) {
        continue;
      }
      if (top1_better(val, token, best_val, best_idx)) {
        best_val = val;
        best_idx = token;
        best_pos = pos;
      }
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      const float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
      const int64_t other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
      const int other_pos = __shfl_down_sync(0xffffffff, best_pos, offset);
      if (top1_better(other_val, other_idx, best_val, best_idx)) {
        best_val = other_val;
        best_idx = other_idx;
        best_pos = other_pos;
      }
    }
    if (lane == 0) {
      const bool use_fallback = best_pos < 0;
      if (use_fallback) {
        best_pos = rank;
        best_val = -FLT_MAX;
        best_idx = rank;
      }
      const bool from_base = best_pos < kTopK;
      const int64_t reduced_row =
          use_fallback ? rank
                       : (from_base ? base_indices[best_pos]
                                    : tail_row_start + best_pos - kTopK);
      pairs_out[rank * 3] = best_val;
      pairs_out[rank * 3 + 1] = static_cast<float>(best_idx);
      pairs_out[rank * 3 + 2] = static_cast<float>(reduced_row);
      if (best_pos < candidate_count) {
        candidate_values[best_pos] = -INFINITY;
      }
    }
    __syncwarp();
  }
}

__global__ void sm70_sample_packed_top20_kernel(
    int64_t* sampled_token_out, int64_t* sparse_ids_out,
    float* sparse_probs_out, const float* gathered_pairs, int candidate_count,
    const float* exponential, int exponential_size, float top_p) {
  constexpr int kTopK = 20;
  const int lane = threadIdx.x;
  __shared__ int selected_positions[kTopK];
  __shared__ float selected_values[kTopK];
  __shared__ int64_t selected_ids[kTopK];
  __shared__ int selected_rows[kTopK];

  for (int rank = 0; rank < kTopK; ++rank) {
    float best_val = -FLT_MAX;
    int64_t best_idx = INT64_MAX;
    int best_pos = -1;
    for (int pos = lane; pos < candidate_count; pos += WARP_SIZE) {
      bool selected = false;
      for (int previous = 0; previous < rank; ++previous) {
        selected |= selected_positions[previous] == pos;
      }
      if (selected) {
        continue;
      }
      const float raw_val = gathered_pairs[pos * 3];
      const int64_t token = static_cast<int64_t>(gathered_pairs[pos * 3 + 1]);
      if (token < 0) {
        continue;
      }
      const float val = isfinite(raw_val) ? raw_val : -FLT_MAX;
      if (top1_better(val, token, best_val, best_idx)) {
        best_val = val;
        best_idx = token;
        best_pos = pos;
      }
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      const float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
      const int64_t other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
      const int other_pos = __shfl_down_sync(0xffffffff, best_pos, offset);
      if (top1_better(other_val, other_idx, best_val, best_idx)) {
        best_val = other_val;
        best_idx = other_idx;
        best_pos = other_pos;
      }
    }
    if (lane == 0) {
      if (best_pos < 0) {
        best_pos = -rank - 1;
        best_val = -FLT_MAX;
        best_idx = rank;
      }
      selected_positions[rank] = best_pos;
      selected_values[rank] = best_val;
      selected_ids[rank] = best_idx;
      selected_rows[rank] =
          best_pos >= 0 ? static_cast<int>(gathered_pairs[best_pos * 3 + 2])
                        : rank % exponential_size;
    }
    __syncwarp();
  }

  if (lane == 0) {
    const float max_value = selected_values[0];
    float probability_sum = 0.f;
    for (int index = 0; index < kTopK; ++index) {
      const float probability = __expf(selected_values[index] - max_value);
      sparse_probs_out[index] = probability;
      probability_sum += probability;
      sparse_ids_out[index] = selected_ids[index];
    }

    float cumulative_probability = 0.f;
    float filtered_probability_sum = 0.f;
    for (int index = 0; index < kTopK; ++index) {
      const float probability = sparse_probs_out[index] / probability_sum;
      const bool keep = index == 0 || cumulative_probability < top_p;
      cumulative_probability += probability;
      sparse_probs_out[index] = keep ? probability : 0.f;
      filtered_probability_sum += sparse_probs_out[index];
    }

    float best_score = -FLT_MAX;
    int64_t sampled_token = selected_ids[0];
    for (int index = 0; index < kTopK; ++index) {
      const float probability =
          sparse_probs_out[index] / filtered_probability_sum;
      sparse_probs_out[index] = probability;
      const int row = selected_rows[index];
      const float random_value =
          row >= 0 && row < exponential_size ? exponential[row] : INFINITY;
      const float score = probability / random_value;
      if (score > best_score) {
        best_score = score;
        sampled_token = selected_ids[index];
      }
    }
    sampled_token_out[0] = sampled_token;
  }
}

constexpr int kDynamicDraftVocabMaxTailCapacity = 512;

__global__ void sm70_dynamic_draft_vocab_update_tail_kernel(
    int64_t* lru_token_ids, int64_t* local_tail_token_ids,
    int64_t* source_row_indices, const int32_t* observed_output_ids,
    int64_t observed_count, const int64_t* target_candidate_ids,
    int64_t target_count, const bool* base_token_mask, int full_vocab_size,
    int local_shard_start, int local_shard_end, int tail_capacity) {
  __shared__ int64_t active_lru[kDynamicDraftVocabMaxTailCapacity];
  __shared__ int64_t shifted_lru[kDynamicDraftVocabMaxTailCapacity];
  __shared__ int64_t token_id;
  __shared__ int active_count;
  __shared__ int existing_index;
  __shared__ int token_is_valid;
  const int thread_id = threadIdx.x;

  for (int slot = thread_id; slot < tail_capacity; slot += blockDim.x) {
    active_lru[slot] = lru_token_ids[slot];
  }
  __syncthreads();

  if (thread_id == 0) {
    active_count = 0;
    while (active_count < tail_capacity && active_lru[active_count] >= 0) {
      ++active_count;
    }
  }
  __syncthreads();
  for (int slot = active_count + thread_id; slot < tail_capacity;
       slot += blockDim.x) {
    active_lru[slot] = -1;
  }
  __syncthreads();

  const int64_t total_count = observed_count + target_count;
  for (int64_t input_index = 0; input_index < total_count; ++input_index) {
    if (thread_id == 0) {
      token_id = input_index < observed_count
                     ? static_cast<int64_t>(observed_output_ids[input_index])
                     : target_candidate_ids[input_index - observed_count];
      token_is_valid = token_id >= 0 && token_id < full_vocab_size &&
                       token_id >= local_shard_start &&
                       token_id < local_shard_end && !base_token_mask[token_id];
      existing_index = tail_capacity;
    }
    __syncthreads();

    if (token_is_valid) {
      for (int slot = thread_id; slot < active_count; slot += blockDim.x) {
        if (active_lru[slot] == token_id) {
          atomicMin(&existing_index, slot);
        }
      }
      __syncthreads();

      if (existing_index < active_count) {
        for (int slot = thread_id; slot < tail_capacity; slot += blockDim.x) {
          if (slot < existing_index) {
            shifted_lru[slot] = active_lru[slot];
          } else if (slot + 1 < active_count) {
            shifted_lru[slot] = active_lru[slot + 1];
          } else if (slot == active_count - 1) {
            shifted_lru[slot] = token_id;
          } else {
            shifted_lru[slot] = active_lru[slot];
          }
        }
        __syncthreads();
        for (int slot = thread_id; slot < tail_capacity; slot += blockDim.x) {
          active_lru[slot] = shifted_lru[slot];
        }
      } else if (active_count < tail_capacity) {
        if (thread_id == 0) {
          active_lru[active_count++] = token_id;
        }
      } else {
        for (int slot = thread_id; slot < tail_capacity; slot += blockDim.x) {
          shifted_lru[slot] =
              slot + 1 < tail_capacity ? active_lru[slot + 1] : token_id;
        }
        __syncthreads();
        for (int slot = thread_id; slot < tail_capacity; slot += blockDim.x) {
          active_lru[slot] = shifted_lru[slot];
        }
      }
    }
    __syncthreads();
  }

  for (int slot = thread_id; slot < tail_capacity; slot += blockDim.x) {
    lru_token_ids[slot] = active_lru[slot];
  }
  __syncthreads();

  if (thread_id == 0) {
    for (int slot = 0; slot < tail_capacity; ++slot) {
      local_tail_token_ids[slot] = -1;
      source_row_indices[slot] = -1;
    }
    int local_count = 0;
    for (int slot = 0; slot < active_count; ++slot) {
      const int64_t local_token_id = active_lru[slot];
      if (local_shard_start <= local_token_id &&
          local_token_id < local_shard_end) {
        local_tail_token_ids[local_count] = local_token_id;
        source_row_indices[local_count] = local_token_id - local_shard_start;
        ++local_count;
      }
    }
  }
}

__global__ void sm70_dynamic_draft_vocab_refresh_tail_weight_kernel(
    half* local_tail_weight, const half* source_weight,
    const int64_t* source_row_indices, int source_row_count, int hidden_size) {
  const int slot = static_cast<int>(blockIdx.y);
  const int hidden_index =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (hidden_index >= hidden_size) {
    return;
  }

  const int64_t source_row = source_row_indices[slot];
  local_tail_weight[slot * hidden_size + hidden_index] =
      source_row >= 0 && source_row < source_row_count
          ? source_weight[source_row * hidden_size + hidden_index]
          : __float2half(0.f);
}

}  // namespace

torch::Tensor interleave_gated_silu_cols(torch::Tensor tensor) {
  const int64_t n = tensor.size(-1);
  TORCH_CHECK((n % 2) == 0,
              "awq_sm70_prepare: gated_silu interleave requires even columns.");
  const int64_t half = n / 2;
  auto first = tensor.slice(-1, 0, half);
  auto second = tensor.slice(-1, half, n);
  return torch::stack({first, second}, -1).reshape(tensor.sizes());
}

std::vector<torch::Tensor> awq_sm70_prepare(torch::Tensor qweight,
                                            torch::Tensor scales,
                                            torch::Tensor qzeros,
                                            int64_t group_size,
                                            bool interleave_gated_silu) {
  validate_awq_inputs(qweight, scales, qzeros);

  qweight = qweight.contiguous();
  scales = scales.contiguous();
  qzeros = qzeros.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1) * 8;
  const int64_t num_groups = scales.size(0);

  TORCH_CHECK(scales.size(1) == n, "awq_sm70_prepare: scales shape mismatch.");
  TORCH_CHECK(qzeros.size(0) == num_groups,
              "awq_sm70_prepare: qzeros group mismatch.");
  TORCH_CHECK(qzeros.size(1) * 8 == n,
              "awq_sm70_prepare: qzeros shape mismatch.");
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0,
              "awq_sm70_prepare: K and N must be multiples of 8.");
  TORCH_CHECK(k % num_groups == 0,
              "awq_sm70_prepare: input dim must be divisible by groups.");

  if (group_size <= 0) {
    group_size = k / num_groups;
  }
  TORCH_CHECK(k / num_groups == group_size,
              "awq_sm70_prepare: group_size mismatch with scales.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_sm70_prepare: SM70 AWQ supports group_size=32/64/128, got ",
              group_size, ".");

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_sm70_prepare: no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  auto packed_weight = torch::empty_like(qweight);
  turbomind::unpack_awq_gemm(
      reinterpret_cast<turbomind::uint4_t*>(packed_weight.data_ptr<int>()),
      reinterpret_cast<const turbomind::uint4_t*>(qweight.data_ptr<int>()),
      static_cast<int>(k), static_cast<int>(n), stream);

  auto u16_opts =
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt16);
  auto tmp_u16 = torch::empty({k, n}, u16_opts);
  turbomind::extend_to_u16(
      reinterpret_cast<uint16_t*>(tmp_u16.data_ptr<int16_t>()),
      reinterpret_cast<const turbomind::uint4_t*>(
          packed_weight.data_ptr<int>()),
      tmp_u16.numel(), stream);
  if (interleave_gated_silu) {
    tmp_u16 = interleave_gated_silu_cols(tmp_u16);
  }

  torch::Tensor tmp_u16_conv = tmp_u16;
  if (order_w == turbomind::gemm::kRowMajor) {
    tmp_u16_conv = tmp_u16.transpose(0, 1).contiguous();
  }

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };

  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::data_type_v<turbomind::uint4_t>;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty_like(qweight);
  TORCH_CHECK(conv_w->Convert(tmp_u16_conv.data_ptr(), w_desc,
                              tm_weight.data_ptr(), k_desc, stream) == 0,
              "awq_sm70_prepare: weight conversion failed.");

  // Unpack AWQ zeros using PyTorch tensor ops (matches lmdeploy's Python
  // approach).  The C++ unpack_awq_gemm() requires rows%8==0 which fails
  // when num_groups < 8 (e.g. Qwen3-30B-A3B w2: K=768, num_groups=6).
  const int awq_order[] = {0, 4, 1, 5, 2, 6, 3, 7};
  std::vector<torch::Tensor> zslices;
  auto zz = qzeros;
  for (int i = 0; i < 8; ++i) {
    zslices.push_back((zz & 0xF).to(torch::kUInt8));
    zz = zz.__rshift__(4);
  }
  std::vector<torch::Tensor> zordered;
  for (int i = 0; i < 8; ++i) {
    zordered.push_back(zslices[awq_order[i]]);
  }
  auto zeros_half =
      torch::stack(zordered, -1).reshape({num_groups, n}).to(torch::kFloat16);
  if (interleave_gated_silu) {
    scales = interleave_gated_silu_cols(scales);
    zeros_half = interleave_gated_silu_cols(zeros_half);
  }

  auto fused = torch::empty(
      {num_groups, n * 2},
      torch::TensorOptions().device(scales.device()).dtype(torch::kFloat16));
  turbomind::fuse_scales_and_zeros(
      reinterpret_cast<half*>(fused.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(zeros_half.data_ptr<at::Half>()), scales.numel(),
      stream);

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout q_desc = s_desc;
  q_desc.pack = conv_s->pack;
  if (is_A_s) {
    q_desc = turbomind::gemm::transpose(q_desc);
  }

  auto tm_scales = torch::empty(
      {num_groups, n},
      torch::TensorOptions().device(scales.device()).dtype(torch::kInt32));
  TORCH_CHECK(conv_s->Convert(fused.data_ptr(), s_desc, tm_scales.data_ptr(),
                              q_desc, stream) == 0,
              "awq_sm70_prepare: scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

std::vector<torch::Tensor> uint4_sm70_prepare(torch::Tensor qweight,
                                              torch::Tensor scales,
                                              torch::Tensor zeros,
                                              int64_t group_size,
                                              bool interleave_gated_silu) {
  constexpr const char* op_name = "uint4_sm70_prepare";
  validate_uint4_inputs(qweight, scales, zeros, op_name);

  qweight = qweight.contiguous();
  scales = scales.contiguous();
  zeros = zeros.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1);
  const int64_t num_groups = scales.size(0);

  TORCH_CHECK(scales.size(1) == n, op_name, ": scales shape mismatch.");
  TORCH_CHECK(zeros.size(0) == num_groups, op_name, ": zeros group mismatch.");
  TORCH_CHECK(zeros.size(1) == n, op_name, ": zeros shape mismatch.");
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0, op_name,
              ": K and N must be multiples of 8.");
  TORCH_CHECK(num_groups > 0 && k % num_groups == 0, op_name,
              ": input dim must be divisible by groups.");

  if (group_size <= 0) {
    group_size = k / num_groups;
  }
  TORCH_CHECK(k / num_groups == group_size, op_name,
              ": group_size mismatch with scales.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              op_name, ": SM70 uint4 supports group_size=32/64/128, got ",
              group_size, ".");

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s, op_name,
              ": no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  auto u16_opts =
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt16);
  auto tmp_u16 = torch::empty({k, n}, u16_opts);
  turbomind::extend_to_u16(
      reinterpret_cast<uint16_t*>(tmp_u16.data_ptr<int16_t>()),
      reinterpret_cast<const uint8_t*>(qweight.data_ptr()), tmp_u16.numel(),
      stream);
  if (interleave_gated_silu) {
    tmp_u16 = interleave_gated_silu_cols(tmp_u16);
  }

  torch::Tensor tmp_u16_conv = tmp_u16;
  if (order_w == turbomind::gemm::kRowMajor) {
    tmp_u16_conv = tmp_u16.transpose(0, 1).contiguous();
  }

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::data_type_v<turbomind::uint4_t>;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty(
      {k, n / 8},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt32));
  TORCH_CHECK(conv_w->Convert(tmp_u16_conv.data_ptr(), w_desc,
                              tm_weight.data_ptr(), k_desc, stream) == 0,
              op_name, ": weight conversion failed.");

  if (interleave_gated_silu) {
    scales = interleave_gated_silu_cols(scales);
    zeros = interleave_gated_silu_cols(zeros);
  }

  auto fused = torch::empty(
      {num_groups, n * 2},
      torch::TensorOptions().device(scales.device()).dtype(torch::kFloat16));
  turbomind::fuse_scales_and_zeros(
      reinterpret_cast<half*>(fused.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(zeros.data_ptr<at::Half>()), scales.numel(),
      stream);

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout q_desc = s_desc;
  q_desc.pack = conv_s->pack;
  if (is_A_s) {
    q_desc = turbomind::gemm::transpose(q_desc);
  }

  auto tm_scales = torch::empty(
      {num_groups, n},
      torch::TensorOptions().device(scales.device()).dtype(torch::kInt32));
  TORCH_CHECK(conv_s->Convert(fused.data_ptr(), s_desc, tm_scales.data_ptr(),
                              q_desc, stream) == 0,
              op_name, ": scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

std::vector<torch::Tensor> fp8_sm70_prepare(torch::Tensor qweight,
                                            torch::Tensor scales,
                                            int64_t group_size,
                                            bool interleave_gated_silu) {
  validate_fp8_inputs(qweight, scales, group_size);

  qweight = qweight.contiguous();
  scales = scales.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t n = qweight.size(0);
  const int64_t k = qweight.size(1);
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0,
              "fp8_sm70_prepare: K and N must be multiples of 8.");
  TORCH_CHECK(scales.size(0) == (n + group_size - 1) / group_size,
              "fp8_sm70_prepare: output scale block mismatch.");
  TORCH_CHECK(scales.size(1) == (k + group_size - 1) / group_size,
              "fp8_sm70_prepare: input scale block mismatch.");

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "fp8_sm70_prepare: no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  auto qweight_kn = qweight.transpose(0, 1).contiguous();
  auto i16_opts =
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt16);
  auto tmp_u16 = torch::empty({k, n}, i16_opts);
  turbomind::extend_to_u16(
      reinterpret_cast<uint16_t*>(tmp_u16.data_ptr<int16_t>()),
      reinterpret_cast<const uint8_t*>(qweight_kn.data_ptr()), tmp_u16.numel(),
      stream);
  if (interleave_gated_silu) {
    tmp_u16 = interleave_gated_silu_cols(tmp_u16);
  }

  torch::Tensor tmp_u16_conv = tmp_u16;
  if (order_w == turbomind::gemm::kRowMajor) {
    tmp_u16_conv = tmp_u16.transpose(0, 1).contiguous();
  }

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::data_type_v<uint8_t>;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty(
      {k, n},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kUInt8));
  TORCH_CHECK(conv_w->Convert(tmp_u16_conv.data_ptr(), w_desc,
                              tm_weight.data_ptr(), k_desc, stream) == 0,
              "fp8_sm70_prepare: weight conversion failed.");

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  const int64_t num_groups = (k + group_size - 1) / group_size;

  auto group_scales = scales.transpose(0, 1)
                          .contiguous()
                          .to(torch::kFloat16)
                          .repeat_interleave(group_size, 1)
                          .slice(1, 0, n)
                          .contiguous();
  if (interleave_gated_silu) {
    group_scales = interleave_gated_silu_cols(group_scales);
  }

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout q_desc = s_desc;
  q_desc.pack = conv_s->pack;
  if (is_A_s) {
    q_desc = turbomind::gemm::transpose(q_desc);
  }

  auto tm_scales = torch::empty_like(group_scales);
  TORCH_CHECK(conv_s->Convert(group_scales.data_ptr(), s_desc,
                              tm_scales.data_ptr(), q_desc, stream) == 0,
              "fp8_sm70_prepare: scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

std::vector<torch::Tensor> mxfp4_sm70_prepare(torch::Tensor qweight,
                                              torch::Tensor scales,
                                              int64_t group_size,
                                              bool interleave_gated_silu) {
  constexpr const char* op_name = "mxfp4_sm70_prepare";
  validate_mxfp4_inputs(qweight, scales, group_size, op_name);

  qweight = qweight.contiguous();
  scales = scales.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1);
  const int64_t num_groups = scales.size(0);
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0, op_name,
              ": K and N must be multiples of 8.");
  TORCH_CHECK(k % group_size == 0, op_name,
              ": K must be divisible by group size.");
  TORCH_CHECK(k / group_size == num_groups, op_name,
              ": scales group mismatch.");
  TORCH_CHECK(scales.size(1) == n, op_name, ": scales shape mismatch.");

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat4_e2m1, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s, op_name,
              ": no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  auto u16_opts =
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt16);
  auto tmp_u16 = torch::empty({k, n}, u16_opts);
  turbomind::extend_to_u16(
      reinterpret_cast<uint16_t*>(tmp_u16.data_ptr<int16_t>()),
      reinterpret_cast<const uint8_t*>(qweight.data_ptr()), tmp_u16.numel(),
      stream);
  if (interleave_gated_silu) {
    tmp_u16 = interleave_gated_silu_cols(tmp_u16);
  }

  torch::Tensor tmp_u16_conv = tmp_u16;
  if (order_w == turbomind::gemm::kRowMajor) {
    tmp_u16_conv = tmp_u16.transpose(0, 1).contiguous();
  }

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::kFloat4_e2m1;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty(
      {k, n / 8},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt32));
  TORCH_CHECK(conv_w->Convert(tmp_u16_conv.data_ptr(), w_desc,
                              tm_weight.data_ptr(), k_desc, stream) == 0,
              op_name, ": weight conversion failed.");

  if (interleave_gated_silu) {
    scales = interleave_gated_silu_cols(scales);
  }
  auto adjusted_scales = scales.clone();
  turbomind::AdjustUe8m0ScaleForHalf(
      reinterpret_cast<uint8_t*>(adjusted_scales.data_ptr()),
      adjusted_scales.numel(), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint8,   order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout q_desc = s_desc;
  q_desc.pack = conv_s->pack;
  if (is_A_s) {
    q_desc = turbomind::gemm::transpose(q_desc);
  }

  auto tm_scales = torch::empty_like(adjusted_scales);
  TORCH_CHECK(conv_s->Convert(adjusted_scales.data_ptr(), s_desc,
                              tm_scales.data_ptr(), q_desc, stream) == 0,
              op_name, ": scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

std::vector<torch::Tensor> nvfp4_sm70_prepare(torch::Tensor qweight,
                                              torch::Tensor scales,
                                              int64_t group_size,
                                              bool interleave_gated_silu) {
  constexpr const char* op_name = "nvfp4_sm70_prepare";
  validate_nvfp4_inputs(qweight, scales, group_size, op_name);

  qweight = qweight.contiguous();
  scales = scales.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1);
  const int64_t num_groups = scales.size(0);
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0, op_name,
              ": K and N must be multiples of 8.");
  TORCH_CHECK(k % group_size == 0, op_name,
              ": K must be divisible by group size.");
  TORCH_CHECK(k / group_size == num_groups, op_name,
              ": scales group mismatch.");
  TORCH_CHECK(scales.size(1) == n, op_name, ": scales shape mismatch.");

  const auto fp4_converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat4_e2m1, turbomind::kHalf, true, 70);
  const auto fp8_converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = fp4_converters[0];
  const auto* conv_s = fp8_converters[1];
  TORCH_CHECK(conv_w && conv_s, op_name,
              ": no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  auto u16_opts =
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt16);
  auto tmp_u16 = torch::empty({k, n}, u16_opts);
  turbomind::extend_to_u16(
      reinterpret_cast<uint16_t*>(tmp_u16.data_ptr<int16_t>()),
      reinterpret_cast<const uint8_t*>(qweight.data_ptr()), tmp_u16.numel(),
      stream);
  if (interleave_gated_silu) {
    tmp_u16 = interleave_gated_silu_cols(tmp_u16);
  }

  torch::Tensor tmp_u16_conv = tmp_u16;
  if (order_w == turbomind::gemm::kRowMajor) {
    tmp_u16_conv = tmp_u16.transpose(0, 1).contiguous();
  }

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::kFloat4_e2m1;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty(
      {k, n / 8},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt32));
  TORCH_CHECK(conv_w->Convert(tmp_u16_conv.data_ptr(), w_desc,
                              tm_weight.data_ptr(), k_desc, stream) == 0,
              op_name, ": weight conversion failed.");

  if (interleave_gated_silu) {
    scales = interleave_gated_silu_cols(scales);
  }

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout q_desc = s_desc;
  q_desc.pack = conv_s->pack;
  if (is_A_s) {
    q_desc = turbomind::gemm::transpose(q_desc);
  }

  auto tm_scales = torch::empty_like(scales);
  TORCH_CHECK(conv_s->Convert(scales.data_ptr(), s_desc, tm_scales.data_ptr(),
                              q_desc, stream) == 0,
              op_name, ": scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

std::vector<torch::Tensor> sm70_f16_prepare(torch::Tensor weight) {
  validate_f16_weight(weight, "sm70_f16_prepare");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto entry = get_sm70_f16_cached_weight(weight, stream);

  auto meta = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, entry.k_ld);
  return {entry.tm_weight, meta};
}

void awq_gemm_sm70_out(torch::Tensor out, torch::Tensor in_feats,
                       torch::Tensor tm_weight, torch::Tensor tm_scales,
                       int64_t group_size, int64_t k_ld, int64_t q_ld,
                       bool gated_silu,
                       const turbomind::gemm::TileAllReduceParam* tile_reduce) {
  TORCH_CHECK(in_feats.is_cuda(), "awq_gemm_sm70: input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), "awq_gemm_sm70: weight must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "awq_gemm_sm70: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "awq_gemm_sm70: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "awq_gemm_sm70: input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kInt32,
              "awq_gemm_sm70: weight must be int32.");
  TORCH_CHECK(tm_scales.scalar_type() == torch::kInt32,
              "awq_gemm_sm70: scales must be int32.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "awq_gemm_sm70: output must be float16.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1) * 8;
  TORCH_CHECK(tm_weight.size(0) == k, "awq_gemm_sm70: weight shape mismatch.");
  TORCH_CHECK(k % group_size == 0,
              "awq_gemm_sm70: input dim must be divisible by group size.");
  TORCH_CHECK(tm_scales.size(0) == k / group_size,
              "awq_gemm_sm70: scale groups mismatch.");
  TORCH_CHECK(tm_scales.size(1) == n, "awq_gemm_sm70: scale shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "awq_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "awq_gemm_sm70: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "awq_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "awq_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n, "awq_gemm_sm70: output cols must match n.");
  }

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_gemm_sm70: no compatible TurboMind converters.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,    turbomind::gemm::kRowMajor, static_cast<int>(m),
      static_cast<int>(k), static_cast<int>(k),
  };
  turbomind::gemm::MatrixLayout desc_U{};

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::data_type_v<turbomind::uint4_t>;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  const int64_t num_groups_raw = k / group_size;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups_raw),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = static_cast<int>(q_ld);

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,    turbomind::gemm::kRowMajor,      static_cast<int>(m),
      static_cast<int>(n), static_cast<int>(out.stride(0)),
  };

  turbomind::gemm::Operation op{};
  op.dispatch = select_awq_dense_dispatch_policy(
      device, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  if (tile_reduce != nullptr) {
    TORCH_CHECK(!gated_silu,
                "awq_gemm_sm70: tile all-reduce cannot combine gated_silu.");
    op.epilogue = static_cast<turbomind::gemm::Epilogue>(
        static_cast<int>(op.epilogue) |
        static_cast<int>(turbomind::gemm::Epilogue::kTileAllReduce));
    op.tile_allreduce =
        const_cast<turbomind::gemm::TileAllReduceParam*>(tile_reduce);
  }
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = get_workspace(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op, 1.f, in_feats.data_ptr(), desc_A, nullptr, desc_U,
                          tm_weight.data_ptr(), desc_B, tm_scales.data_ptr(),
                          desc_V, 0.f, out.data_ptr(), desc_D, out.data_ptr(),
                          desc_D, workspace_holder.workspace, stream);
  TORCH_CHECK(ec == 0, "awq_gemm_sm70: TurboMind GEMM failed.");
}

void awq_gemm_sm70_out(torch::Tensor out, torch::Tensor in_feats,
                       torch::Tensor tm_weight, torch::Tensor tm_scales,
                       int64_t group_size, int64_t k_ld, int64_t q_ld,
                       bool gated_silu) {
  awq_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, group_size, k_ld, q_ld,
                    gated_silu, nullptr);
}

void awq_gemm_sm70_out_tile_reduce(
    torch::Tensor out, torch::Tensor staging, torch::Tensor in_feats,
    torch::Tensor tm_weight, torch::Tensor tm_scales, int64_t group_size,
    int64_t k_ld, int64_t q_ld, int64_t fa_ptr, int64_t tile_numel,
    int64_t reducer_blocks, int64_t kernel_reducer_blocks, bool overlap) {
  TORCH_CHECK(out.is_cuda() && staging.is_cuda() && in_feats.is_cuda(),
              "awq_gemm_sm70_tile_reduce: tensors must be CUDA.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  staging.scalar_type() == torch::kFloat16 &&
                  in_feats.scalar_type() == torch::kFloat16,
              "awq_gemm_sm70_tile_reduce: out/staging/input must be fp16.");
  TORCH_CHECK(out.sizes() == staging.sizes(),
              "awq_gemm_sm70_tile_reduce: out and staging shape mismatch.");
  TORCH_CHECK(out.dim() == 2 && staging.dim() == 2 && out.size(0) == 1 &&
                  staging.size(0) == 1,
              "awq_gemm_sm70_tile_reduce: only M=1 is supported.");
  TORCH_CHECK(out.stride(1) == 1 && staging.stride(1) == 1,
              "awq_gemm_sm70_tile_reduce: outputs must be row-major.");
  TORCH_CHECK(fa_ptr != 0,
              "awq_gemm_sm70_tile_reduce: custom all-reduce handle is null.");
  TORCH_CHECK(
      tile_numel > 0 && reducer_blocks >= 0 && kernel_reducer_blocks >= 0,
      "awq_gemm_sm70_tile_reduce: invalid tile runtime config.");
  if (kernel_reducer_blocks > 0) {
    TORCH_CHECK(overlap,
                "awq_gemm_sm70_tile_reduce: kernel reducer requires overlap.");
    TORCH_CHECK((tile_numel & 1) == 0 && (staging.numel() & 1) == 0,
                "awq_gemm_sm70_tile_reduce: kernel reducer requires half2 "
                "aligned tile/output sizes.");
  }

  auto* fa = reinterpret_cast<::vllm::CustomAllreduce*>(fa_ptr);
  TORCH_CHECK(fa->world_size_ == 2,
              "awq_gemm_sm70_tile_reduce: only TP2 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  auto current_stream = at::cuda::getCurrentCUDAStream(device);
  cudaStreamCaptureStatus capture_status;
  AT_CUDA_CHECK(
      cudaStreamIsCapturing(current_stream.stream(), &capture_status));
  TORCH_CHECK(capture_status == cudaStreamCaptureStatusActive,
              "awq_gemm_sm70_tile_reduce requires CUDA graph capture.");

  turbomind::gemm::TileAllReduceParam tile_reduce{};
  for (int i = 0; i < fa->world_size_; ++i) {
    tile_reduce.signals[i] = fa->sg_.signals[i];
  }
  tile_reduce.self_signal = fa->self_sg_;
  tile_reduce.rank_data = nullptr;
  tile_reduce.output = nullptr;
  tile_reduce.rank = fa->rank_;
  tile_reduce.world_size = fa->world_size_;
  tile_reduce.tile_numel = static_cast<int>(tile_numel);
  tile_reduce.output_numel = static_cast<int>(staging.numel());
  tile_reduce.reducer_blocks = static_cast<int>(reducer_blocks);
  tile_reduce.kernel_reducer_blocks =
      overlap ? static_cast<int>(kernel_reducer_blocks) : 0;

  if (overlap) {
    tile_reduce.rank_data =
        fa->rank_data_for_buffer(current_stream.stream(), staging.data_ptr(),
                                 "awq_gemm_sm70_tile_reduce");
    tile_reduce.output = out.data_ptr();
    awq_gemm_sm70_out(staging, in_feats, tm_weight, tm_scales, group_size, k_ld,
                      q_ld, false, &tile_reduce);
    return;
  }

  if (!overlap) {
    awq_gemm_sm70_out(staging, in_feats, tm_weight, tm_scales, group_size, k_ld,
                      q_ld, false, &tile_reduce);
    fa->tile_runtime_wait_reduce<half>(
        current_stream.stream(), reinterpret_cast<half*>(staging.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()), out.numel(), tile_numel,
        reducer_blocks);
    return;
  }
}

void fp8_gemm_sm70_out(torch::Tensor out, torch::Tensor in_feats,
                       torch::Tensor tm_weight, torch::Tensor tm_scales,
                       int64_t group_size, int64_t k_ld, int64_t q_ld,
                       bool gated_silu) {
  TORCH_CHECK(in_feats.is_cuda(), "fp8_gemm_sm70: input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), "fp8_gemm_sm70: weight must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "fp8_gemm_sm70: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "fp8_gemm_sm70: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "fp8_gemm_sm70: input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kUInt8,
              "fp8_gemm_sm70: weight must be uint8.");
  TORCH_CHECK(tm_scales.scalar_type() == torch::kFloat16,
              "fp8_gemm_sm70: scales must be float16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "fp8_gemm_sm70: output must be float16.");
  TORCH_CHECK(group_size == 128,
              "fp8_gemm_sm70: only group_size=128 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1);
  TORCH_CHECK(tm_weight.size(0) == k, "fp8_gemm_sm70: weight shape mismatch.");
  TORCH_CHECK(tm_scales.size(0) == (k + group_size - 1) / group_size,
              "fp8_gemm_sm70: scale groups mismatch.");
  TORCH_CHECK(tm_scales.size(1) == n, "fp8_gemm_sm70: scale shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "fp8_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "fp8_gemm_sm70: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "fp8_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "fp8_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "fp8_gemm_sm70: output cols must match weight output dim.");
  }

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "fp8_gemm_sm70: no compatible TurboMind converters.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(in_feats.stride(0)),
  };
  turbomind::gemm::MatrixLayout desc_U{};

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;
  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::kFloat8_e4m3;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  const int64_t num_groups = (k + group_size - 1) / group_size;
  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = static_cast<int>(q_ld);

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,    turbomind::gemm::kRowMajor,      static_cast<int>(m),
      static_cast<int>(n), static_cast<int>(out.stride(0)),
  };

  turbomind::gemm::Operation op{};
  op.dispatch = select_fp8_dense_dispatch_policy(
      device, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = get_workspace(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op, 1.f, in_feats.data_ptr(), desc_A, nullptr, desc_U,
                          tm_weight.data_ptr(), desc_B, tm_scales.data_ptr(),
                          desc_V, 0.f, out.data_ptr(), desc_D, out.data_ptr(),
                          desc_D, workspace_holder.workspace, stream);
  TORCH_CHECK(ec == 0, "fp8_gemm_sm70: TurboMind GEMM failed.");
}

void mxfp4_gemm_sm70_out(torch::Tensor out, torch::Tensor in_feats,
                         torch::Tensor tm_weight, torch::Tensor tm_scales,
                         int64_t group_size, int64_t k_ld, int64_t q_ld,
                         bool gated_silu) {
  TORCH_CHECK(in_feats.is_cuda(), "mxfp4_gemm_sm70: input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), "mxfp4_gemm_sm70: weight must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "mxfp4_gemm_sm70: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "mxfp4_gemm_sm70: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "mxfp4_gemm_sm70: input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kInt32,
              "mxfp4_gemm_sm70: weight must be int32.");
  TORCH_CHECK(tm_scales.scalar_type() == torch::kUInt8,
              "mxfp4_gemm_sm70: scales must be uint8.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "mxfp4_gemm_sm70: output must be float16.");
  TORCH_CHECK(group_size == 32,
              "mxfp4_gemm_sm70: only group_size=32 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1) * 8;
  TORCH_CHECK(tm_weight.size(0) == k,
              "mxfp4_gemm_sm70: weight shape mismatch.");
  TORCH_CHECK(tm_scales.size(0) == k / group_size,
              "mxfp4_gemm_sm70: scale groups mismatch.");
  TORCH_CHECK(tm_scales.size(1) == n, "mxfp4_gemm_sm70: scale shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "mxfp4_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "mxfp4_gemm_sm70: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "mxfp4_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "mxfp4_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "mxfp4_gemm_sm70: output cols must match weight output dim.");
  }

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat4_e2m1, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "mxfp4_gemm_sm70: no compatible TurboMind converters.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(in_feats.stride(0)),
  };
  turbomind::gemm::MatrixLayout desc_U{};

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;
  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::kFloat4_e2m1;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  const int64_t num_groups = k / group_size;
  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint8,   order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = static_cast<int>(q_ld);

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,    turbomind::gemm::kRowMajor,      static_cast<int>(m),
      static_cast<int>(n), static_cast<int>(out.stride(0)),
  };

  turbomind::gemm::Operation op{};
  op.dispatch = select_mxfp4_dense_dispatch_policy(
      device, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = get_workspace(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op, 1.f, in_feats.data_ptr(), desc_A, nullptr, desc_U,
                          tm_weight.data_ptr(), desc_B, tm_scales.data_ptr(),
                          desc_V, 0.f, out.data_ptr(), desc_D, out.data_ptr(),
                          desc_D, workspace_holder.workspace, stream);
  TORCH_CHECK(ec == 0, "mxfp4_gemm_sm70: TurboMind GEMM failed.");
}

void nvfp4_gemm_sm70_out(torch::Tensor out, torch::Tensor in_feats,
                         torch::Tensor tm_weight, torch::Tensor tm_scales,
                         int64_t group_size, int64_t k_ld, int64_t q_ld,
                         bool gated_silu) {
  TORCH_CHECK(in_feats.is_cuda(), "nvfp4_gemm_sm70: input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), "nvfp4_gemm_sm70: weight must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "nvfp4_gemm_sm70: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "nvfp4_gemm_sm70: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "nvfp4_gemm_sm70: input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kInt32,
              "nvfp4_gemm_sm70: weight must be int32.");
  TORCH_CHECK(tm_scales.scalar_type() == torch::kFloat16,
              "nvfp4_gemm_sm70: scales must be float16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "nvfp4_gemm_sm70: output must be float16.");
  TORCH_CHECK(group_size == 16,
              "nvfp4_gemm_sm70: only group_size=16 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1) * 8;
  TORCH_CHECK(tm_weight.size(0) == k,
              "nvfp4_gemm_sm70: weight shape mismatch.");
  TORCH_CHECK(k % group_size == 0,
              "nvfp4_gemm_sm70: input dim must be divisible by group size.");
  TORCH_CHECK(tm_scales.size(0) == k / group_size,
              "nvfp4_gemm_sm70: scale groups mismatch.");
  TORCH_CHECK(tm_scales.size(1) == n, "nvfp4_gemm_sm70: scale shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "nvfp4_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "nvfp4_gemm_sm70: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "nvfp4_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "nvfp4_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "nvfp4_gemm_sm70: output cols must match weight output dim.");
  }

  const auto fp4_converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat4_e2m1, turbomind::kHalf, true, 70);
  const auto fp8_converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = fp4_converters[0];
  const auto* conv_s = fp8_converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "nvfp4_gemm_sm70: no compatible TurboMind converters.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(in_feats.stride(0)),
  };
  turbomind::gemm::MatrixLayout desc_U{};

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;
  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::kFloat4_e2m1;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  const int64_t num_groups = k / group_size;
  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = static_cast<int>(q_ld);

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,    turbomind::gemm::kRowMajor,      static_cast<int>(m),
      static_cast<int>(n), static_cast<int>(out.stride(0)),
  };

  turbomind::gemm::Operation op{};
  op.dispatch = select_nvfp4_dense_dispatch_policy(
      device, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = get_workspace(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op, 1.f, in_feats.data_ptr(), desc_A, nullptr, desc_U,
                          tm_weight.data_ptr(), desc_B, tm_scales.data_ptr(),
                          desc_V, 0.f, out.data_ptr(), desc_D, out.data_ptr(),
                          desc_D, workspace_holder.workspace, stream);
  TORCH_CHECK(ec == 0, "nvfp4_gemm_sm70: TurboMind GEMM failed.");
}

void nvfp4_gemv_sm70_raw_out(torch::Tensor out, torch::Tensor in_feats,
                             torch::Tensor qweight_packed, torch::Tensor scales,
                             torch::Tensor partials, int64_t group_size,
                             int64_t split_k) {
  TORCH_CHECK(in_feats.is_cuda(), "nvfp4_gemv_sm70_raw: input must be CUDA.");
  TORCH_CHECK(qweight_packed.is_cuda(),
              "nvfp4_gemv_sm70_raw: weight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), "nvfp4_gemv_sm70_raw: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "nvfp4_gemv_sm70_raw: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "nvfp4_gemv_sm70_raw: input must be float16.");
  TORCH_CHECK(qweight_packed.scalar_type() == torch::kInt32,
              "nvfp4_gemv_sm70_raw: packed weight must be int32.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16,
              "nvfp4_gemv_sm70_raw: scales must be float16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "nvfp4_gemv_sm70_raw: output must be float16.");
  TORCH_CHECK(group_size == 16,
              "nvfp4_gemv_sm70_raw: only group_size=16 is supported.");
  TORCH_CHECK(split_k >= 1 && split_k <= 256,
              "nvfp4_gemv_sm70_raw: split_k must be in [1, 256].");
  TORCH_CHECK(in_feats.dim() == 2 && in_feats.size(0) == 1,
              "nvfp4_gemv_sm70_raw: only M=1 input is supported.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == 1,
              "nvfp4_gemv_sm70_raw: output must be [1, N].");
  TORCH_CHECK(out.stride(1) == 1,
              "nvfp4_gemv_sm70_raw: output must be row-major contiguous.");
  TORCH_CHECK(qweight_packed.dim() == 2,
              "nvfp4_gemv_sm70_raw: packed weight must be 2D.");
  TORCH_CHECK(scales.dim() == 2, "nvfp4_gemv_sm70_raw: scales must be 2D.");

  const int64_t k = in_feats.size(1);
  const int64_t n = out.size(1);
  TORCH_CHECK(k % group_size == 0,
              "nvfp4_gemv_sm70_raw: K must be divisible by group_size.");
  TORCH_CHECK(n % 8 == 0, "nvfp4_gemv_sm70_raw: N must be divisible by 8.");
  TORCH_CHECK(qweight_packed.size(0) == k && qweight_packed.size(1) == n / 8,
              "nvfp4_gemv_sm70_raw: packed weight shape mismatch.");
  TORCH_CHECK(scales.size(0) == k / group_size && scales.size(1) == n,
              "nvfp4_gemv_sm70_raw: scales shape mismatch.");
  if (split_k > 1) {
    TORCH_CHECK(partials.is_cuda(),
                "nvfp4_gemv_sm70_raw: partials must be CUDA.");
    TORCH_CHECK(partials.scalar_type() == torch::kFloat32,
                "nvfp4_gemv_sm70_raw: partials must be float32.");
    TORCH_CHECK(partials.dim() == 2 && partials.size(0) == split_k &&
                    partials.size(1) == n,
                "nvfp4_gemv_sm70_raw: partials shape mismatch.");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 128;
  const int qwords = static_cast<int>(n / 8);
  const int num_groups = static_cast<int>(k / group_size);
  const dim3 partial_grid((qwords + kThreads - 1) / kThreads,
                          static_cast<unsigned int>(split_k));
  float* partial_ptr = split_k > 1 ? partials.data_ptr<float>() : nullptr;
  nvfp4_raw_gemv_partial_kernel<kThreads>
      <<<partial_grid, kThreads, 0, stream>>>(
          reinterpret_cast<__half*>(out.data_ptr<at::Half>()), partial_ptr,
          reinterpret_cast<const __half*>(in_feats.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(qweight_packed.data_ptr<int32_t>()),
          reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
          static_cast<int>(k), static_cast<int>(n), num_groups,
          static_cast<int>(split_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (split_k > 1) {
    const dim3 reduce_grid((qwords + kThreads - 1) / kThreads);
    nvfp4_raw_gemv_reduce_kernel<kThreads>
        <<<reduce_grid, kThreads, 0, stream>>>(
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), partial_ptr,
            static_cast<int>(n), static_cast<int>(split_k));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void nvfp4_gemv_sm70_warp_out(torch::Tensor out, torch::Tensor in_feats,
                              torch::Tensor qweight_packed,
                              torch::Tensor scales, int64_t group_size) {
  TORCH_CHECK(in_feats.is_cuda(), "nvfp4_gemv_sm70_warp: input must be CUDA.");
  TORCH_CHECK(qweight_packed.is_cuda(),
              "nvfp4_gemv_sm70_warp: weight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), "nvfp4_gemv_sm70_warp: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "nvfp4_gemv_sm70_warp: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "nvfp4_gemv_sm70_warp: input must be float16.");
  TORCH_CHECK(qweight_packed.scalar_type() == torch::kInt32,
              "nvfp4_gemv_sm70_warp: packed weight must be int32.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16,
              "nvfp4_gemv_sm70_warp: scales must be float16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "nvfp4_gemv_sm70_warp: output must be float16.");
  TORCH_CHECK(group_size == 16,
              "nvfp4_gemv_sm70_warp: only group_size=16 is supported.");
  TORCH_CHECK(in_feats.dim() == 2 && in_feats.size(0) == 1,
              "nvfp4_gemv_sm70_warp: only M=1 input is supported.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == 1,
              "nvfp4_gemv_sm70_warp: output must be [1, N].");
  TORCH_CHECK(out.stride(1) == 1,
              "nvfp4_gemv_sm70_warp: output must be row-major contiguous.");
  TORCH_CHECK(qweight_packed.dim() == 2,
              "nvfp4_gemv_sm70_warp: packed weight must be 2D.");
  TORCH_CHECK(scales.dim() == 2, "nvfp4_gemv_sm70_warp: scales must be 2D.");

  const int64_t k = in_feats.size(1);
  const int64_t n = out.size(1);
  TORCH_CHECK(k % group_size == 0,
              "nvfp4_gemv_sm70_warp: K must be divisible by group_size.");
  TORCH_CHECK(n % 8 == 0, "nvfp4_gemv_sm70_warp: N must be divisible by 8.");
  TORCH_CHECK(qweight_packed.size(0) == k && qweight_packed.size(1) == n / 8,
              "nvfp4_gemv_sm70_warp: packed weight shape mismatch.");
  TORCH_CHECK(scales.size(0) == k / group_size && scales.size(1) == n,
              "nvfp4_gemv_sm70_warp: scales shape mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 128;
  constexpr int kWarpsPerBlock = kThreads / 32;
  const int qwords = static_cast<int>(n / 8);
  const dim3 grid((qwords + kWarpsPerBlock - 1) / kWarpsPerBlock);
  nvfp4_raw_gemv_warp_kernel<kThreads><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(in_feats.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(qweight_packed.data_ptr<int32_t>()),
      reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
      static_cast<int>(n), static_cast<int>(k / group_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_gemv_sm70_h2_out(torch::Tensor out, torch::Tensor in_feats,
                            torch::Tensor qweight_packed, torch::Tensor scales,
                            torch::Tensor partials, int64_t group_size,
                            int64_t split_k) {
  TORCH_CHECK(in_feats.is_cuda(), "nvfp4_gemv_sm70_h2: input must be CUDA.");
  TORCH_CHECK(qweight_packed.is_cuda(),
              "nvfp4_gemv_sm70_h2: weight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), "nvfp4_gemv_sm70_h2: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "nvfp4_gemv_sm70_h2: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "nvfp4_gemv_sm70_h2: input must be float16.");
  TORCH_CHECK(qweight_packed.scalar_type() == torch::kInt32,
              "nvfp4_gemv_sm70_h2: packed weight must be int32.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16,
              "nvfp4_gemv_sm70_h2: scales must be float16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "nvfp4_gemv_sm70_h2: output must be float16.");
  TORCH_CHECK(group_size == 16,
              "nvfp4_gemv_sm70_h2: only group_size=16 is supported.");
  TORCH_CHECK(split_k >= 1 && split_k <= 256,
              "nvfp4_gemv_sm70_h2: split_k must be in [1, 256].");
  TORCH_CHECK(in_feats.dim() == 2 && in_feats.size(0) == 1,
              "nvfp4_gemv_sm70_h2: only M=1 input is supported.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == 1,
              "nvfp4_gemv_sm70_h2: output must be [1, N].");
  TORCH_CHECK(out.stride(1) == 1,
              "nvfp4_gemv_sm70_h2: output must be row-major contiguous.");
  TORCH_CHECK(qweight_packed.dim() == 2,
              "nvfp4_gemv_sm70_h2: packed weight must be 2D.");
  TORCH_CHECK(scales.dim() == 2, "nvfp4_gemv_sm70_h2: scales must be 2D.");

  const int64_t k = in_feats.size(1);
  const int64_t n = out.size(1);
  TORCH_CHECK(k % group_size == 0,
              "nvfp4_gemv_sm70_h2: K must be divisible by group_size.");
  TORCH_CHECK(n % 8 == 0, "nvfp4_gemv_sm70_h2: N must be divisible by 8.");
  TORCH_CHECK(qweight_packed.size(0) == k && qweight_packed.size(1) == n / 8,
              "nvfp4_gemv_sm70_h2: packed weight shape mismatch.");
  TORCH_CHECK(scales.size(0) == k / group_size && scales.size(1) == n,
              "nvfp4_gemv_sm70_h2: scales shape mismatch.");
  if (split_k > 1) {
    TORCH_CHECK(partials.is_cuda(),
                "nvfp4_gemv_sm70_h2: partials must be CUDA.");
    TORCH_CHECK(partials.scalar_type() == torch::kFloat16,
                "nvfp4_gemv_sm70_h2: partials must be float16.");
    TORCH_CHECK(partials.dim() == 2 && partials.size(0) == split_k &&
                    partials.size(1) == n,
                "nvfp4_gemv_sm70_h2: partials shape mismatch.");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 128;
  const int qwords = static_cast<int>(n / 8);
  const int num_groups = static_cast<int>(k / group_size);
  const dim3 partial_grid((qwords + kThreads - 1) / kThreads,
                          static_cast<unsigned int>(split_k));
  __half* partial_ptr =
      split_k > 1 ? reinterpret_cast<__half*>(partials.data_ptr<at::Half>())
                  : nullptr;
  nvfp4_raw_gemv_h2_partial_kernel<kThreads>
      <<<partial_grid, kThreads, 0, stream>>>(
          reinterpret_cast<__half*>(out.data_ptr<at::Half>()), partial_ptr,
          reinterpret_cast<const __half*>(in_feats.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(qweight_packed.data_ptr<int32_t>()),
          reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
          static_cast<int>(k), static_cast<int>(n), num_groups,
          static_cast<int>(split_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (split_k > 1) {
    const dim3 reduce_grid((qwords + kThreads - 1) / kThreads);
    nvfp4_raw_gemv_h2_reduce_kernel<kThreads>
        <<<reduce_grid, kThreads, 0, stream>>>(
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), partial_ptr,
            static_cast<int>(n), static_cast<int>(split_k));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void fp8_gemm_sm70_out_auto(torch::Tensor out, torch::Tensor in_feats,
                            torch::Tensor tm_weight, torch::Tensor tm_scales) {
  constexpr int64_t group_size = 128;
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1);

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "fp8_gemm_sm70: no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;
  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::kFloat8_e4m3;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  const int64_t num_groups = (k + group_size - 1) / group_size;
  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }

  fp8_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, group_size, desc_B.ld,
                    desc_V.ld, false);
}

void fp8_gemm_sm70_out_meta(torch::Tensor out, torch::Tensor in_feats,
                            torch::Tensor tm_weight, torch::Tensor tm_scales,
                            torch::Tensor meta, bool gated_silu) {
  TORCH_CHECK(meta.scalar_type() == torch::kInt64,
              "fp8_gemm_sm70: meta must be int64.");
  TORCH_CHECK(meta.numel() >= 2, "fp8_gemm_sm70: meta must have two values.");
  auto meta_cpu = meta.device().is_cpu() ? meta.contiguous()
                                         : meta.to(torch::kCPU).contiguous();
  const int64_t* meta_ptr = meta_cpu.data_ptr<int64_t>();
  fp8_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, 128, meta_ptr[0],
                    meta_ptr[1], gated_silu);
}

void sm70_f16_gemm_out(torch::Tensor out, torch::Tensor in_feats,
                       torch::Tensor tm_weight, int64_t k_ld, bool gated_silu) {
  validate_f16_input(in_feats, tm_weight, out, gated_silu, "sm70_f16_gemm");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(0);

  TORCH_CHECK(tm_weight.size(1) == k, "sm70_f16_gemm: weight shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "sm70_f16_gemm: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "sm70_f16_gemm: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "sm70_f16_gemm: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "sm70_f16_gemm: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n, "sm70_f16_gemm: output cols must match n.");
  }

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(in_feats.stride(0)),
  };
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kHalf, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  TORCH_CHECK(conv_w, "sm70_f16_gemm: no compatible TurboMind converter.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);
  turbomind::gemm::MatrixLayout desc_U{};
  turbomind::gemm::MatrixLayout desc_V{};
  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,    turbomind::gemm::kRowMajor,      static_cast<int>(m),
      static_cast<int>(n), static_cast<int>(out.stride(0)),
  };

  turbomind::gemm::Operation op{};
  op.dispatch = select_dense_dispatch_policy(device, static_cast<int>(m),
                                             static_cast<int>(n),
                                             static_cast<int>(k), 0, stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kNone, 0};
  op.batch_dim = 0;

  auto& workspace_holder = get_workspace(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op, 1.f, in_feats.data_ptr(), desc_A, nullptr, desc_U,
                          tm_weight.data_ptr(), desc_B, nullptr, desc_V, 0.f,
                          out.data_ptr(), desc_D, out.data_ptr(), desc_D,
                          workspace_holder.workspace, stream);
  TORCH_CHECK(ec == 0, "sm70_f16_gemm: TurboMind GEMM failed.");
}

void sm70_f16_lm_head_top1_out(torch::Tensor values_out,
                               torch::Tensor indices_out,
                               torch::Tensor in_feats, torch::Tensor weight,
                               int64_t k_ld, int64_t vocab_start_index,
                               int64_t num_vocab_padding) {
  constexpr int64_t max_m = 1;
  validate_f16_lm_head_top1_input(values_out, indices_out, in_feats, weight,
                                  k_ld, num_vocab_padding, max_m);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t n = weight.size(0);
  const int64_t k = in_feats.size(1);
  if (m == 0 || n == 0) {
    return;
  }

  constexpr int rows_per_block = 8;
  constexpr int stage1_threads = rows_per_block * 32;
  constexpr int stage2_threads = 256;
  const int64_t valid_n = n - num_vocab_padding;
  const int64_t num_blocks_n = (valid_n + rows_per_block - 1) / rows_per_block;
  auto partial_values = torch::empty(
      {m, num_blocks_n},
      torch::TensorOptions().dtype(torch::kFloat32).device(in_feats.device()));
  auto partial_indices = torch::empty(
      {m, num_blocks_n},
      torch::TensorOptions().dtype(torch::kInt64).device(in_feats.device()));

  const int64_t weight_row_stride =
      k_ld > 0 ? k_ld : static_cast<int64_t>(weight.stride(0));

  dim3 grid1(static_cast<unsigned int>(num_blocks_n),
             static_cast<unsigned int>(m));
  sm70_f16_lm_head_top1_stage1_kernel<rows_per_block>
      <<<grid1, stage1_threads, 0, stream>>>(
          partial_values.data_ptr<float>(), partial_indices.data_ptr<int64_t>(),
          reinterpret_cast<const half*>(in_feats.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
          static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
          static_cast<int>(valid_n), in_feats.stride(0), weight_row_stride,
          vocab_start_index, static_cast<int>(num_blocks_n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  sm70_f16_lm_head_top1_stage2_kernel<stage2_threads>
      <<<static_cast<unsigned int>(m), stage2_threads, 0, stream>>>(
          values_out.data_ptr<float>(), indices_out.data_ptr<int64_t>(),
          partial_values.data_ptr<float>(), partial_indices.data_ptr<int64_t>(),
          static_cast<int>(m), static_cast<int>(num_blocks_n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_f16_lm_head_top1_tc_out(torch::Tensor values_out,
                                  torch::Tensor indices_out,
                                  torch::Tensor in_feats,
                                  torch::Tensor tm_weight, int64_t k_ld,
                                  int64_t vocab_start_index,
                                  int64_t num_vocab_padding) {
  constexpr int64_t max_m = 17;
  validate_f16_lm_head_top1_input(values_out, indices_out, in_feats, tm_weight,
                                  k_ld, num_vocab_padding, max_m);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t n = tm_weight.size(0);
  const int64_t k = in_feats.size(1);
  if (m == 0 || n == 0) {
    return;
  }

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(in_feats.stride(0)),
  };
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kHalf, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  TORCH_CHECK(conv_w, "sm70_f16_lm_head_top1_tc_out: no compatible converter.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);
  turbomind::gemm::MatrixLayout desc_U{};
  turbomind::gemm::MatrixLayout desc_V{};

  constexpr int cta_n = 256;
  constexpr int stage2_threads = 256;
  const int64_t valid_n = n - num_vocab_padding;
  const int64_t num_tiles_n = (valid_n + cta_n - 1) / cta_n;
  auto partial_values = torch::empty(
      {m, num_tiles_n},
      torch::TensorOptions().dtype(torch::kFloat32).device(in_feats.device()));
  auto partial_indices = torch::empty(
      {m, num_tiles_n},
      torch::TensorOptions().dtype(torch::kInt64).device(in_feats.device()));

  using namespace turbomind::gemm;
  using C = sm70_s884::Config_F16_Top1<kColMajor, 0>;
  using D = turbomind::cache_policy::Default;
  using S = turbomind::cache_policy::Stream;
  using Gemm = typename C::template Type<8, cta_n, 64, 1, 4, 1, D, S, 2, true,
                                         1, 1>::Kernel;
  using Sched = typename Gemm::Scheduler;

  GemmParam param{
      to_param(in_feats.data_ptr(), desc_A),
      to_param(tm_weight.data_ptr(), transpose(desc_B)),
      to_param(nullptr, desc_U),
      to_param(nullptr, desc_V),
  };
  TORCH_CHECK(vocab_start_index <= std::numeric_limits<int>::max(),
              "sm70_f16_lm_head_top1_tc_out: vocab start too large.");
  TORCH_CHECK(valid_n <= std::numeric_limits<int>::max(),
              "sm70_f16_lm_head_top1_tc_out: vocab shard too large.");
  EpilogueParam epi_param{};
  epi_param.c = {partial_values.data_ptr<float>(),
                 static_cast<int>(vocab_start_index), nullptr, nullptr};
  epi_param.partials = {partial_indices.data_ptr<int64_t>(),
                        static_cast<int>(num_tiles_n), nullptr, nullptr};

  constexpr int log_tile = 4;
  Sched sched{
      {static_cast<int>(m), static_cast<int>(valid_n), static_cast<int>(k), 1},
      log_tile,
      1};
  sched.offsets_ = nullptr;

  const auto grid = sched.get_grid_shape();
  const auto block = Gemm::Impl::WARPS * WARP_SIZE;
  constexpr int dynamic_smem_size =
      static_cast<int>(sizeof(typename Gemm::SharedStorage));
  auto kernel = gemm_kernel<Gemm, GemmParam, EpilogueParam, Sched>;
  if constexpr (dynamic_smem_size > (48 << 10)) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         dynamic_smem_size);
  }
  kernel<<<grid, block, dynamic_smem_size, stream>>>(param, epi_param, sched);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  sm70_f16_lm_head_top1_stage2_kernel<stage2_threads>
      <<<static_cast<unsigned int>(m), stage2_threads, 0, stream>>>(
          values_out.data_ptr<float>(), indices_out.data_ptr<int64_t>(),
          partial_values.data_ptr<float>(), partial_indices.data_ptr<int64_t>(),
          static_cast<int>(m), static_cast<int>(num_tiles_n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_f16_lm_head_top20_tc_out(torch::Tensor values_out,
                                   torch::Tensor indices_out,
                                   torch::Tensor in_feats,
                                   torch::Tensor tm_weight, int64_t k_ld,
                                   int64_t vocab_start_index,
                                   int64_t num_vocab_padding) {
  constexpr int64_t max_m = 8;
  constexpr int top_k = 20;
  validate_f16_lm_head_top20_input(values_out, indices_out, in_feats, tm_weight,
                                   k_ld, num_vocab_padding, max_m);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t n = tm_weight.size(0);
  const int64_t k = in_feats.size(1);
  if (m == 0 || n == 0) {
    return;
  }

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(in_feats.stride(0)),
  };
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kHalf, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  TORCH_CHECK(conv_w,
              "sm70_f16_lm_head_top20_tc_out: no compatible converter.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);
  turbomind::gemm::MatrixLayout desc_U{};
  turbomind::gemm::MatrixLayout desc_V{};

  constexpr int cta_n = 256;
  const int64_t valid_n = n - num_vocab_padding;
  const int64_t num_tiles_n = (valid_n + cta_n - 1) / cta_n;
  auto partial_values = torch::empty(
      {m, num_tiles_n, top_k},
      torch::TensorOptions().dtype(torch::kFloat32).device(in_feats.device()));
  auto partial_indices = torch::empty(
      {m, num_tiles_n, top_k},
      torch::TensorOptions().dtype(torch::kInt64).device(in_feats.device()));

  using namespace turbomind::gemm;
  using C = sm70_s884::Config_F16_Top20<kColMajor, 0>;
  using D = turbomind::cache_policy::Default;
  using S = turbomind::cache_policy::Stream;
  using Gemm = typename C::template Type<8, cta_n, 64, 1, 4, 1, D, S, 2, true,
                                         1, 1>::Kernel;
  using Sched = typename Gemm::Scheduler;

  GemmParam param{
      to_param(in_feats.data_ptr(), desc_A),
      to_param(tm_weight.data_ptr(), transpose(desc_B)),
      to_param(nullptr, desc_U),
      to_param(nullptr, desc_V),
  };
  TORCH_CHECK(vocab_start_index <= std::numeric_limits<int>::max(),
              "sm70_f16_lm_head_top20_tc_out: vocab start too large.");
  TORCH_CHECK(valid_n <= std::numeric_limits<int>::max(),
              "sm70_f16_lm_head_top20_tc_out: vocab shard too large.");
  EpilogueParam epi_param{};
  epi_param.c = {partial_values.data_ptr<float>(),
                 static_cast<int>(vocab_start_index), nullptr, nullptr};
  epi_param.partials = {partial_indices.data_ptr<int64_t>(),
                        static_cast<int>(num_tiles_n), nullptr, nullptr};

  constexpr int log_tile = 4;
  Sched sched{
      {static_cast<int>(m), static_cast<int>(valid_n), static_cast<int>(k), 1},
      log_tile,
      1};
  sched.offsets_ = nullptr;

  const auto grid = sched.get_grid_shape();
  const auto block = Gemm::Impl::WARPS * WARP_SIZE;
  constexpr int dynamic_smem_size =
      static_cast<int>(sizeof(typename Gemm::SharedStorage));
  auto kernel = gemm_kernel<Gemm, GemmParam, EpilogueParam, Sched>;
  if constexpr (dynamic_smem_size > (48 << 10)) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         dynamic_smem_size);
  }
  kernel<<<grid, block, dynamic_smem_size, stream>>>(param, epi_param, sched);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t num_candidates = num_tiles_n * top_k;
  sm70_f16_lm_head_top20_stage2_kernel<<<static_cast<unsigned int>(m),
                                         WARP_SIZE, 0, stream>>>(
      values_out.data_ptr<float>(), indices_out.data_ptr<int64_t>(),
      partial_values.data_ptr<float>(), partial_indices.data_ptr<int64_t>(),
      static_cast<int>(m), static_cast<int>(num_candidates), vocab_start_index,
      static_cast<int>(valid_n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_merge_tail_top20_pack_out(torch::Tensor pairs_out,
                                    torch::Tensor base_values,
                                    torch::Tensor base_indices,
                                    torch::Tensor base_token_id_map,
                                    torch::Tensor tail_logits,
                                    torch::Tensor tail_token_ids,
                                    int64_t tail_row_start) {
  constexpr int top_k = 20;
  TORCH_CHECK(pairs_out.is_cuda() && base_values.is_cuda() &&
                  base_indices.is_cuda() && base_token_id_map.is_cuda() &&
                  tail_logits.is_cuda() && tail_token_ids.is_cuda(),
              "sm70_merge_tail_top20_pack_out: tensors must be CUDA.");
  TORCH_CHECK(pairs_out.scalar_type() == torch::kFloat32 &&
                  base_values.scalar_type() == torch::kFloat32 &&
                  base_indices.scalar_type() == torch::kInt64 &&
                  base_token_id_map.scalar_type() == torch::kInt64 &&
                  tail_logits.scalar_type() == torch::kFloat16 &&
                  tail_token_ids.scalar_type() == torch::kInt64,
              "sm70_merge_tail_top20_pack_out: invalid tensor dtype.");
  TORCH_CHECK(pairs_out.numel() == top_k * 3 && base_values.numel() == top_k &&
                  base_indices.numel() == top_k &&
                  base_token_id_map.numel() >= top_k &&
                  tail_logits.numel() == tail_token_ids.numel(),
              "sm70_merge_tail_top20_pack_out: invalid tensor shape.");
  TORCH_CHECK(tail_row_start >= 0,
              "sm70_merge_tail_top20_pack_out: invalid tail row start.");
  TORCH_CHECK(pairs_out.is_contiguous() && base_values.is_contiguous() &&
                  base_indices.is_contiguous() &&
                  base_token_id_map.is_contiguous() &&
                  tail_logits.is_contiguous() && tail_token_ids.is_contiguous(),
              "sm70_merge_tail_top20_pack_out: tensors must be contiguous.");
  TORCH_CHECK(tail_logits.numel() >= top_k,
              "sm70_merge_tail_top20_pack_out: tail must have 20 rows.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(base_values));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const size_t shared_bytes =
      static_cast<size_t>(top_k + tail_logits.numel()) * sizeof(float);
  sm70_merge_tail_top20_pack_kernel<<<1, WARP_SIZE, shared_bytes, stream>>>(
      pairs_out.data_ptr<float>(), base_values.data_ptr<float>(),
      base_indices.data_ptr<int64_t>(), base_token_id_map.data_ptr<int64_t>(),
      reinterpret_cast<const half*>(tail_logits.data_ptr<at::Half>()),
      tail_token_ids.data_ptr<int64_t>(),
      static_cast<int>(base_token_id_map.numel()),
      static_cast<int>(tail_logits.numel()), static_cast<int>(tail_row_start));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_sample_packed_top20_out(torch::Tensor sampled_token_out,
                                  torch::Tensor sparse_ids_out,
                                  torch::Tensor sparse_probs_out,
                                  torch::Tensor gathered_pairs,
                                  torch::Tensor exponential, double top_p) {
  constexpr int top_k = 20;
  TORCH_CHECK(sampled_token_out.is_cuda() && sparse_ids_out.is_cuda() &&
                  sparse_probs_out.is_cuda() && gathered_pairs.is_cuda() &&
                  exponential.is_cuda(),
              "sm70_sample_packed_top20_out: tensors must be CUDA.");
  TORCH_CHECK(sampled_token_out.scalar_type() == torch::kInt64 &&
                  sparse_ids_out.scalar_type() == torch::kInt64 &&
                  sparse_probs_out.scalar_type() == torch::kFloat32 &&
                  gathered_pairs.scalar_type() == torch::kFloat32 &&
                  exponential.scalar_type() == torch::kFloat32,
              "sm70_sample_packed_top20_out: invalid tensor dtype.");
  TORCH_CHECK(
      sampled_token_out.numel() == 1 && sparse_ids_out.numel() == top_k &&
          sparse_probs_out.numel() == top_k &&
          gathered_pairs.numel() >= top_k * 3 &&
          gathered_pairs.numel() % 3 == 0 && exponential.numel() >= top_k,
      "sm70_sample_packed_top20_out: invalid tensor shape.");
  TORCH_CHECK(sampled_token_out.is_contiguous() &&
                  sparse_ids_out.is_contiguous() &&
                  sparse_probs_out.is_contiguous() &&
                  gathered_pairs.is_contiguous() && exponential.is_contiguous(),
              "sm70_sample_packed_top20_out: tensors must be contiguous.");
  TORCH_CHECK(top_p > 0.0 && top_p <= 1.0,
              "sm70_sample_packed_top20_out: top_p must be in (0, 1].");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(gathered_pairs));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  sm70_sample_packed_top20_kernel<<<1, WARP_SIZE, 0, stream>>>(
      sampled_token_out.data_ptr<int64_t>(), sparse_ids_out.data_ptr<int64_t>(),
      sparse_probs_out.data_ptr<float>(), gathered_pairs.data_ptr<float>(),
      static_cast<int>(gathered_pairs.numel() / 3),
      exponential.data_ptr<float>(), static_cast<int>(exponential.numel()),
      static_cast<float>(top_p));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_dynamic_draft_vocab_update_tail_out(
    torch::Tensor lru_token_ids, torch::Tensor local_tail_token_ids,
    torch::Tensor source_row_indices, torch::Tensor observed_output_ids,
    torch::Tensor target_candidate_ids, torch::Tensor base_token_mask,
    int64_t full_vocab_size, int64_t local_shard_start,
    int64_t local_shard_end) {
  TORCH_CHECK(
      lru_token_ids.is_cuda() && local_tail_token_ids.is_cuda() &&
          source_row_indices.is_cuda() && observed_output_ids.is_cuda() &&
          target_candidate_ids.is_cuda() && base_token_mask.is_cuda(),
      "sm70_dynamic_draft_vocab_update_tail_out: tensors must be CUDA.");
  TORCH_CHECK(
      lru_token_ids.scalar_type() == torch::kInt64 &&
          local_tail_token_ids.scalar_type() == torch::kInt64 &&
          source_row_indices.scalar_type() == torch::kInt64 &&
          observed_output_ids.scalar_type() == torch::kInt32 &&
          target_candidate_ids.scalar_type() == torch::kInt64 &&
          base_token_mask.scalar_type() == torch::kBool,
      "sm70_dynamic_draft_vocab_update_tail_out: invalid tensor dtype.");
  TORCH_CHECK(
      lru_token_ids.dim() == 1 && local_tail_token_ids.dim() == 1 &&
          source_row_indices.dim() == 1 && observed_output_ids.dim() == 1 &&
          target_candidate_ids.dim() == 1 && base_token_mask.dim() == 1,
      "sm70_dynamic_draft_vocab_update_tail_out: invalid tensor shape.");
  TORCH_CHECK(
      lru_token_ids.is_contiguous() && local_tail_token_ids.is_contiguous() &&
          source_row_indices.is_contiguous() &&
          observed_output_ids.is_contiguous() &&
          target_candidate_ids.is_contiguous() &&
          base_token_mask.is_contiguous(),
      "sm70_dynamic_draft_vocab_update_tail_out: tensors must be contiguous.");
  TORCH_CHECK(
      lru_token_ids.numel() > 0 &&
          local_tail_token_ids.numel() == lru_token_ids.numel() &&
          source_row_indices.numel() == lru_token_ids.numel(),
      "sm70_dynamic_draft_vocab_update_tail_out: invalid tail buffers.");
  TORCH_CHECK(
      lru_token_ids.numel() <= kDynamicDraftVocabMaxTailCapacity,
      "sm70_dynamic_draft_vocab_update_tail_out: tail capacity exceeds ",
      kDynamicDraftVocabMaxTailCapacity, ".");
  TORCH_CHECK(
      full_vocab_size > 0 && base_token_mask.numel() == full_vocab_size &&
          0 <= local_shard_start && local_shard_start < local_shard_end &&
          local_shard_end <= full_vocab_size,
      "sm70_dynamic_draft_vocab_update_tail_out: invalid vocabulary range.");
  const int device = lru_token_ids.get_device();
  TORCH_CHECK(
      local_tail_token_ids.get_device() == device &&
          source_row_indices.get_device() == device &&
          observed_output_ids.get_device() == device &&
          target_candidate_ids.get_device() == device &&
          base_token_mask.get_device() == device,
      "sm70_dynamic_draft_vocab_update_tail_out: tensors must share a device.");
  TORCH_CHECK(lru_token_ids.numel() <= std::numeric_limits<int>::max() &&
                  full_vocab_size <= std::numeric_limits<int>::max(),
              "sm70_dynamic_draft_vocab_update_tail_out: tensor is too large.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(lru_token_ids));
  const auto stream = at::cuda::getCurrentCUDAStream();
  sm70_dynamic_draft_vocab_update_tail_kernel<<<1, 256, 0, stream>>>(
      lru_token_ids.data_ptr<int64_t>(),
      local_tail_token_ids.data_ptr<int64_t>(),
      source_row_indices.data_ptr<int64_t>(),
      observed_output_ids.data_ptr<int32_t>(), observed_output_ids.numel(),
      target_candidate_ids.data_ptr<int64_t>(), target_candidate_ids.numel(),
      base_token_mask.data_ptr<bool>(), static_cast<int>(full_vocab_size),
      static_cast<int>(local_shard_start), static_cast<int>(local_shard_end),
      static_cast<int>(lru_token_ids.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_dynamic_draft_vocab_refresh_tail_weight_out(
    torch::Tensor local_tail_weight, torch::Tensor source_weight,
    torch::Tensor source_row_indices) {
  TORCH_CHECK(local_tail_weight.is_cuda() && source_weight.is_cuda() &&
                  source_row_indices.is_cuda(),
              "sm70_dynamic_draft_vocab_refresh_tail_weight_out: tensors must "
              "be CUDA.");
  TORCH_CHECK(local_tail_weight.scalar_type() == torch::kFloat16 &&
                  source_weight.scalar_type() == torch::kFloat16 &&
                  source_row_indices.scalar_type() == torch::kInt64,
              "sm70_dynamic_draft_vocab_refresh_tail_weight_out: invalid "
              "tensor dtype.");
  TORCH_CHECK(local_tail_weight.dim() == 2 && source_weight.dim() == 2 &&
                  source_row_indices.dim() == 1 &&
                  local_tail_weight.size(0) == source_row_indices.numel() &&
                  local_tail_weight.size(1) == source_weight.size(1) &&
                  source_weight.size(0) > 0,
              "sm70_dynamic_draft_vocab_refresh_tail_weight_out: invalid "
              "tensor shape.");
  TORCH_CHECK(local_tail_weight.is_contiguous() &&
                  source_weight.is_contiguous() &&
                  source_row_indices.is_contiguous(),
              "sm70_dynamic_draft_vocab_refresh_tail_weight_out: tensors must "
              "be contiguous.");
  const int device = local_tail_weight.get_device();
  TORCH_CHECK(source_weight.get_device() == device &&
                  source_row_indices.get_device() == device,
              "sm70_dynamic_draft_vocab_refresh_tail_weight_out: tensors must "
              "share a device.");
  TORCH_CHECK(
      local_tail_weight.size(0) <= std::numeric_limits<int>::max() &&
          source_weight.size(0) <= std::numeric_limits<int>::max() &&
          source_weight.size(1) <= std::numeric_limits<int>::max(),
      "sm70_dynamic_draft_vocab_refresh_tail_weight_out: tensor is too large.");

  constexpr int kThreads = 256;
  const dim3 grid(static_cast<unsigned int>(
                      (source_weight.size(1) + kThreads - 1) / kThreads),
                  static_cast<unsigned int>(local_tail_weight.size(0)));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(local_tail_weight));
  const auto stream = at::cuda::getCurrentCUDAStream();
  sm70_dynamic_draft_vocab_refresh_tail_weight_kernel<<<grid, kThreads, 0,
                                                        stream>>>(
      reinterpret_cast<half*>(local_tail_weight.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(source_weight.data_ptr<at::Half>()),
      source_row_indices.data_ptr<int64_t>(),
      static_cast<int>(source_weight.size(0)),
      static_cast<int>(source_weight.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_f16_gate_mul_out(torch::Tensor out, torch::Tensor in_feats,
                           torch::Tensor gate_weight) {
  validate_f16_gate_mul_input(out, in_feats, gate_weight);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = out.size(1);
  if (m == 0 || n == 0) {
    return;
  }

  constexpr int kThreads = 256;
  sm70_f16_gate_mul_kernel<kThreads>
      <<<static_cast<unsigned int>(m), kThreads, 0, stream>>>(
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(in_feats.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(gate_weight.data_ptr<at::Half>()),
          static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
          out.stride(0), in_feats.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor awq_gemm_sm70(torch::Tensor in_feats, torch::Tensor tm_weight,
                            torch::Tensor tm_scales, int64_t group_size,
                            int64_t k_ld, int64_t q_ld) {
  const int64_t n = tm_weight.size(1) * 8;
  auto out = torch::empty(
      {in_feats.size(0), n},
      torch::TensorOptions().dtype(in_feats.dtype()).device(in_feats.device()));
  awq_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, group_size, k_ld, q_ld,
                    false);
  return out;
}

torch::Tensor sm70_f16_gemm(torch::Tensor in_feats, torch::Tensor weight) {
  validate_f16_weight(weight, "sm70_f16_gemm");
  TORCH_CHECK(in_feats.is_cuda(), "sm70_f16_gemm: input must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "sm70_f16_gemm: input must be float16.");
  TORCH_CHECK(in_feats.dim() == 2, "sm70_f16_gemm: input must be 2D.");
  TORCH_CHECK(in_feats.size(1) == weight.size(1),
              "sm70_f16_gemm: input/weight K mismatch.");
  if (in_feats.size(0) > sm70_f16_dense_max_m()) {
    return torch::mm(in_feats, weight.transpose(0, 1));
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto entry = get_sm70_f16_cached_weight(weight, stream);
  const int64_t n = weight.size(0);
  auto out = torch::empty(
      {in_feats.size(0), n},
      torch::TensorOptions().dtype(in_feats.dtype()).device(in_feats.device()));
  sm70_f16_gemm_out(out, in_feats, entry.tm_weight, entry.k_ld, false);
  return out;
}

turbomind::gemm::DispatchPolicy awq_select_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream) {
  return select_moe_dispatch_policy(device, total_tokens, n, k, num_experts,
                                    group_size, stream);
}

turbomind::gemm::DispatchPolicy select_awq_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream) {
  const auto override_policy = awq_moe_dispatch_policy_override();
  if (override_policy.has_value()) {
    return override_policy.value();
  }
  return select_moe_dispatch_policy_impl(
      device, total_tokens, n, k, num_experts, group_size, stream,
      TuneKeyKind::kAwqMoe, awq_tune_small_shapes_enabled());
}

}  // namespace awq_sm70
}  // namespace vllm

std::vector<torch::Tensor> awq_sm70_prepare(torch::Tensor _kernel,
                                            torch::Tensor _scaling_factors,
                                            torch::Tensor _zeros,
                                            int64_t group_size,
                                            bool interleave_gated_silu) {
  return vllm::awq_sm70::awq_sm70_prepare(_kernel, _scaling_factors, _zeros,
                                          group_size, interleave_gated_silu);
}

std::vector<torch::Tensor> uint4_sm70_prepare(torch::Tensor _kernel,
                                              torch::Tensor _scaling_factors,
                                              torch::Tensor _zeros,
                                              int64_t group_size,
                                              bool interleave_gated_silu) {
  return vllm::awq_sm70::uint4_sm70_prepare(_kernel, _scaling_factors, _zeros,
                                            group_size, interleave_gated_silu);
}

std::vector<torch::Tensor> fp8_sm70_prepare(torch::Tensor _kernel,
                                            torch::Tensor _scaling_factors,
                                            int64_t group_size,
                                            bool interleave_gated_silu) {
  return vllm::awq_sm70::fp8_sm70_prepare(_kernel, _scaling_factors, group_size,
                                          interleave_gated_silu);
}

std::vector<torch::Tensor> mxfp4_sm70_prepare(torch::Tensor _kernel,
                                              torch::Tensor _scaling_factors,
                                              int64_t group_size,
                                              bool interleave_gated_silu) {
  return vllm::awq_sm70::mxfp4_sm70_prepare(_kernel, _scaling_factors,
                                            group_size, interleave_gated_silu);
}

std::vector<torch::Tensor> nvfp4_sm70_prepare(torch::Tensor _kernel,
                                              torch::Tensor _scaling_factors,
                                              int64_t group_size,
                                              bool interleave_gated_silu) {
  return vllm::awq_sm70::nvfp4_sm70_prepare(_kernel, _scaling_factors,
                                            group_size, interleave_gated_silu);
}

std::vector<torch::Tensor> sm70_f16_prepare(torch::Tensor _kernel) {
  return vllm::awq_sm70::sm70_f16_prepare(_kernel);
}

torch::Tensor awq_gemm_sm70(torch::Tensor _in_feats, torch::Tensor _kernel,
                            torch::Tensor _scaling_factors, int64_t group_size,
                            int64_t k_ld, int64_t q_ld) {
  return vllm::awq_sm70::awq_gemm_sm70(_in_feats, _kernel, _scaling_factors,
                                       group_size, k_ld, q_ld);
}

torch::Tensor sm70_f16_gemm(torch::Tensor _in_feats, torch::Tensor _kernel) {
  return vllm::awq_sm70::sm70_f16_gemm(_in_feats, _kernel);
}

bool sm70_fp8_moe_prepare_vec_enabled() {
  const char* raw = std::getenv("VLLM_SM70_FP8_MOE_PREPARE_VEC");
  return raw != nullptr && std::atoi(raw) != 0;
}

void awq_gemm_sm70_out(torch::Tensor out, torch::Tensor _in_feats,
                       torch::Tensor _kernel, torch::Tensor _scaling_factors,
                       int64_t group_size, int64_t k_ld, int64_t q_ld,
                       bool gated_silu) {
  vllm::awq_sm70::awq_gemm_sm70_out(out, _in_feats, _kernel, _scaling_factors,
                                    group_size, k_ld, q_ld, gated_silu);
}

void awq_gemm_sm70_out_tile_reduce(
    torch::Tensor out, torch::Tensor staging, torch::Tensor _in_feats,
    torch::Tensor _kernel, torch::Tensor _scaling_factors, int64_t group_size,
    int64_t k_ld, int64_t q_ld, int64_t fa_ptr, int64_t tile_numel,
    int64_t reducer_blocks, int64_t kernel_reducer_blocks, bool overlap) {
  vllm::awq_sm70::awq_gemm_sm70_out_tile_reduce(
      out, staging, _in_feats, _kernel, _scaling_factors, group_size, k_ld,
      q_ld, fa_ptr, tile_numel, reducer_blocks, kernel_reducer_blocks, overlap);
}

void fp8_gemm_sm70_out(torch::Tensor out, torch::Tensor _in_feats,
                       torch::Tensor _kernel, torch::Tensor _scaling_factors,
                       int64_t group_size, int64_t k_ld, int64_t q_ld,
                       bool gated_silu) {
  vllm::awq_sm70::fp8_gemm_sm70_out(out, _in_feats, _kernel, _scaling_factors,
                                    group_size, k_ld, q_ld, gated_silu);
}

void mxfp4_gemm_sm70_out(torch::Tensor out, torch::Tensor _in_feats,
                         torch::Tensor _kernel, torch::Tensor _scaling_factors,
                         int64_t group_size, int64_t k_ld, int64_t q_ld,
                         bool gated_silu) {
  vllm::awq_sm70::mxfp4_gemm_sm70_out(out, _in_feats, _kernel, _scaling_factors,
                                      group_size, k_ld, q_ld, gated_silu);
}

void nvfp4_gemm_sm70_out(torch::Tensor out, torch::Tensor _in_feats,
                         torch::Tensor _kernel, torch::Tensor _scaling_factors,
                         int64_t group_size, int64_t k_ld, int64_t q_ld,
                         bool gated_silu) {
  vllm::awq_sm70::nvfp4_gemm_sm70_out(out, _in_feats, _kernel, _scaling_factors,
                                      group_size, k_ld, q_ld, gated_silu);
}

void nvfp4_gemv_sm70_raw_out(torch::Tensor out, torch::Tensor _in_feats,
                             torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor partials, int64_t group_size,
                             int64_t split_k) {
  vllm::awq_sm70::nvfp4_gemv_sm70_raw_out(
      out, _in_feats, _kernel, _scaling_factors, partials, group_size, split_k);
}

void nvfp4_gemv_sm70_warp_out(torch::Tensor out, torch::Tensor _in_feats,
                              torch::Tensor _kernel,
                              torch::Tensor _scaling_factors,
                              int64_t group_size) {
  vllm::awq_sm70::nvfp4_gemv_sm70_warp_out(out, _in_feats, _kernel,
                                           _scaling_factors, group_size);
}

void nvfp4_gemv_sm70_h2_out(torch::Tensor out, torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors,
                            torch::Tensor partials, int64_t group_size,
                            int64_t split_k) {
  vllm::awq_sm70::nvfp4_gemv_sm70_h2_out(
      out, _in_feats, _kernel, _scaling_factors, partials, group_size, split_k);
}

void fp8_gemm_sm70_out_auto(torch::Tensor out, torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors) {
  vllm::awq_sm70::fp8_gemm_sm70_out_auto(out, _in_feats, _kernel,
                                         _scaling_factors);
}

void fp8_gemm_sm70_out_meta(torch::Tensor out, torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors, torch::Tensor _meta,
                            bool gated_silu) {
  vllm::awq_sm70::fp8_gemm_sm70_out_meta(out, _in_feats, _kernel,
                                         _scaling_factors, _meta, gated_silu);
}

void sm70_f16_gemm_out(torch::Tensor out, torch::Tensor _in_feats,
                       torch::Tensor _kernel, int64_t k_ld, bool gated_silu) {
  vllm::awq_sm70::sm70_f16_gemm_out(out, _in_feats, _kernel, k_ld, gated_silu);
}

void sm70_f16_lm_head_top1_out(torch::Tensor values_out,
                               torch::Tensor indices_out,
                               torch::Tensor _in_feats, torch::Tensor _kernel,
                               int64_t k_ld, int64_t vocab_start_index,
                               int64_t num_vocab_padding) {
  vllm::awq_sm70::sm70_f16_lm_head_top1_out(values_out, indices_out, _in_feats,
                                            _kernel, k_ld, vocab_start_index,
                                            num_vocab_padding);
}

void sm70_f16_lm_head_top1_tc_out(torch::Tensor values_out,
                                  torch::Tensor indices_out,
                                  torch::Tensor _in_feats,
                                  torch::Tensor _kernel, int64_t k_ld,
                                  int64_t vocab_start_index,
                                  int64_t num_vocab_padding) {
  vllm::awq_sm70::sm70_f16_lm_head_top1_tc_out(
      values_out, indices_out, _in_feats, _kernel, k_ld, vocab_start_index,
      num_vocab_padding);
}

void sm70_f16_lm_head_top20_tc_out(torch::Tensor values_out,
                                   torch::Tensor indices_out,
                                   torch::Tensor _in_feats,
                                   torch::Tensor _kernel, int64_t k_ld,
                                   int64_t vocab_start_index,
                                   int64_t num_vocab_padding) {
  vllm::awq_sm70::sm70_f16_lm_head_top20_tc_out(
      values_out, indices_out, _in_feats, _kernel, k_ld, vocab_start_index,
      num_vocab_padding);
}

void sm70_merge_tail_top20_pack_out(torch::Tensor pairs_out,
                                    torch::Tensor base_values,
                                    torch::Tensor base_indices,
                                    torch::Tensor base_token_id_map,
                                    torch::Tensor tail_logits,
                                    torch::Tensor tail_token_ids,
                                    int64_t tail_row_start) {
  vllm::awq_sm70::sm70_merge_tail_top20_pack_out(
      pairs_out, base_values, base_indices, base_token_id_map, tail_logits,
      tail_token_ids, tail_row_start);
}

void sm70_sample_packed_top20_out(torch::Tensor sampled_token_out,
                                  torch::Tensor sparse_ids_out,
                                  torch::Tensor sparse_probs_out,
                                  torch::Tensor gathered_pairs,
                                  torch::Tensor exponential, double top_p) {
  vllm::awq_sm70::sm70_sample_packed_top20_out(
      sampled_token_out, sparse_ids_out, sparse_probs_out, gathered_pairs,
      exponential, top_p);
}

void sm70_dynamic_draft_vocab_update_tail_out(
    torch::Tensor lru_token_ids, torch::Tensor local_tail_token_ids,
    torch::Tensor source_row_indices, torch::Tensor observed_output_ids,
    torch::Tensor target_candidate_ids, torch::Tensor base_token_mask,
    int64_t full_vocab_size, int64_t local_shard_start,
    int64_t local_shard_end) {
  vllm::awq_sm70::sm70_dynamic_draft_vocab_update_tail_out(
      lru_token_ids, local_tail_token_ids, source_row_indices,
      observed_output_ids, target_candidate_ids, base_token_mask,
      full_vocab_size, local_shard_start, local_shard_end);
}

void sm70_dynamic_draft_vocab_refresh_tail_weight_out(
    torch::Tensor local_tail_weight, torch::Tensor source_weight,
    torch::Tensor source_row_indices) {
  vllm::awq_sm70::sm70_dynamic_draft_vocab_refresh_tail_weight_out(
      local_tail_weight, source_weight, source_row_indices);
}

void sm70_f16_gate_mul_out(torch::Tensor out, torch::Tensor _in_feats,
                           torch::Tensor _gate_weight) {
  vllm::awq_sm70::sm70_f16_gate_mul_out(out, _in_feats, _gate_weight);
}

int64_t sm70_gemm_import_cache(torch::Tensor device_hint,
                               const std::string& path) {
  TORCH_CHECK(device_hint.is_cuda(),
              "sm70_gemm_import_cache: device_hint must be CUDA.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(device_hint));
  const int device = device_hint.get_device();

  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.good()) {
    return 0;
  }

  auto& gemm = vllm::awq_sm70::get_gemm(device);
  const int64_t imported = gemm.Import(ifs);
  if (imported > 0) {
    std::lock_guard<std::mutex> lock(vllm::awq_sm70::tune_mutex);
    vllm::awq_sm70::imported_cache_devices.insert(device);
  }
  return imported;
}

int64_t sm70_gemm_export_cache(torch::Tensor device_hint,
                               const std::string& path) {
  TORCH_CHECK(device_hint.is_cuda(),
              "sm70_gemm_export_cache: device_hint must be CUDA.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(device_hint));
  const int device = device_hint.get_device();

  try {
    const std::filesystem::path fs_path(path);
    if (fs_path.has_parent_path()) {
      std::filesystem::create_directories(fs_path.parent_path());
    }
  } catch (const std::exception& e) {
    TORCH_CHECK(false,
                "sm70_gemm_export_cache: failed to create parent "
                "directory for ",
                path, " (", e.what(), ").");
  }

  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  TORCH_CHECK(ofs.good(), "sm70_gemm_export_cache: failed to open ", path,
              " for writing.");

  auto& gemm = vllm::awq_sm70::get_gemm(device);
  return gemm.Export(ofs);
}

// ---------------------------------------------------------------------------
// MoE batched GEMM support
// ---------------------------------------------------------------------------

#if defined(ENABLE_SM70_TURBOMIND)

std::vector<torch::Tensor> awq_moe_build_strided_ptrs(
    torch::Tensor tm_weights,  // [E, ...]  stacked TM weights
    torch::Tensor tm_scales,   // [E, ...]  stacked TM scales
    int64_t k_ld, int64_t q_ld, int64_t num_experts) {
  TORCH_CHECK(tm_weights.is_cuda(),
              "awq_moe_build_strided_ptrs: weights must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(),
              "awq_moe_build_strided_ptrs: scales must be CUDA.");
  TORCH_CHECK(num_experts > 0,
              "awq_moe_build_strided_ptrs: num_experts must be > 0.");
  TORCH_CHECK(tm_weights.size(0) == num_experts,
              "awq_moe_build_strided_ptrs: weights dim0 != num_experts.");
  TORCH_CHECK(tm_scales.size(0) == num_experts,
              "awq_moe_build_strided_ptrs: scales dim0 != num_experts.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(tm_weights));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Build {ptr, stride} pairs for each expert
  std::vector<std::pair<void*, int>> w_ptrs;
  std::vector<std::pair<void*, int>> s_ptrs;
  w_ptrs.reserve(num_experts);
  s_ptrs.reserve(num_experts);

  const int64_t w_expert_stride =
      tm_weights.stride(0) * tm_weights.element_size();
  const int64_t s_expert_stride =
      tm_scales.stride(0) * tm_scales.element_size();
  char* w_base = static_cast<char*>(tm_weights.data_ptr());
  char* s_base = static_cast<char*>(tm_scales.data_ptr());

  for (int64_t e = 0; e < num_experts; ++e) {
    w_ptrs.emplace_back(w_base + e * w_expert_stride, static_cast<int>(k_ld));
    s_ptrs.emplace_back(s_base + e * s_expert_stride, static_cast<int>(q_ld));
  }

  // MakeStridedPtrs allocates GPU memory via cudaMallocAsync
  void* w_gpu = turbomind::gemm::MakeStridedPtrs(w_ptrs, stream);
  void* s_gpu = turbomind::gemm::MakeStridedPtrs(s_ptrs, stream);

  // Wrap in torch tensors for lifetime management.
  // StridedPtr is 16 bytes (__align__(16): void* ptr + int stride + padding).
  const int64_t buf_bytes = num_experts * 16;
  auto opts =
      torch::TensorOptions().device(tm_weights.device()).dtype(torch::kUInt8);

  // Copy into torch-managed tensors so cudaFree of the original is safe.
  auto w_tensor = torch::empty({buf_bytes}, opts);
  auto s_tensor = torch::empty({buf_bytes}, opts);
  cudaMemcpyAsync(w_tensor.data_ptr(), w_gpu, buf_bytes,
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(s_tensor.data_ptr(), s_gpu, buf_bytes,
                  cudaMemcpyDeviceToDevice, stream);
  cudaFreeAsync(w_gpu, stream);
  cudaFreeAsync(s_gpu, stream);

  return {w_tensor, s_tensor};
}

template <typename index_t>
__global__ void awq_moe_single_token_compact_prepare_kernel(
    const index_t* topk_ids, const uint8_t* src_w13_ptrs_w_rows,
    const uint8_t* src_w13_ptrs_s_rows, const uint8_t* src_w2_ptrs_w_rows,
    const uint8_t* src_w2_ptrs_s_rows, uint8_t* dst_w13_ptrs_w_rows,
    uint8_t* dst_w13_ptrs_s_rows, uint8_t* dst_w2_ptrs_w_rows,
    uint8_t* dst_w2_ptrs_s_rows, int* inv_permuted_idx, int top_k,
    int src_row_stride, int dst_row_stride, int row_bytes) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int sorted_ids[32];
  int sorted_src[32];

  for (int i = 0; i < top_k; ++i) {
    sorted_ids[i] = static_cast<int>(topk_ids[i]);
    sorted_src[i] = i;
  }

  // Stable insertion sort to match the native MoE single-token permutation.
  for (int i = 1; i < top_k; ++i) {
    const int expert_id = sorted_ids[i];
    const int src_idx = sorted_src[i];
    int j = i - 1;
    while (j >= 0 && sorted_ids[j] > expert_id) {
      sorted_ids[j + 1] = sorted_ids[j];
      sorted_src[j + 1] = sorted_src[j];
      --j;
    }
    sorted_ids[j + 1] = expert_id;
    sorted_src[j + 1] = src_idx;
  }

  for (int sorted_pos = 0; sorted_pos < top_k; ++sorted_pos) {
    const int expert_id = sorted_ids[sorted_pos];
    const int src_idx = sorted_src[sorted_pos];
    inv_permuted_idx[src_idx] = sorted_pos;

    const uint8_t* src_w13_w = src_w13_ptrs_w_rows + expert_id * src_row_stride;
    const uint8_t* src_w13_s = src_w13_ptrs_s_rows + expert_id * src_row_stride;
    const uint8_t* src_w2_w = src_w2_ptrs_w_rows + expert_id * src_row_stride;
    const uint8_t* src_w2_s = src_w2_ptrs_s_rows + expert_id * src_row_stride;

    uint8_t* dst_w13_w = dst_w13_ptrs_w_rows + sorted_pos * dst_row_stride;
    uint8_t* dst_w13_s = dst_w13_ptrs_s_rows + sorted_pos * dst_row_stride;
    uint8_t* dst_w2_w = dst_w2_ptrs_w_rows + sorted_pos * dst_row_stride;
    uint8_t* dst_w2_s = dst_w2_ptrs_s_rows + sorted_pos * dst_row_stride;

    for (int byte_idx = 0; byte_idx < row_bytes; ++byte_idx) {
      dst_w13_w[byte_idx] = src_w13_w[byte_idx];
      dst_w13_s[byte_idx] = src_w13_s[byte_idx];
      dst_w2_w[byte_idx] = src_w2_w[byte_idx];
      dst_w2_s[byte_idx] = src_w2_s[byte_idx];
    }
  }
}

void awq_moe_single_token_compact_prepare(
    torch::Tensor topk_ids, torch::Tensor src_w13_ptrs_w_rows,
    torch::Tensor src_w13_ptrs_s_rows, torch::Tensor src_w2_ptrs_w_rows,
    torch::Tensor src_w2_ptrs_s_rows, torch::Tensor dst_w13_ptrs_w_rows,
    torch::Tensor dst_w13_ptrs_s_rows, torch::Tensor dst_w2_ptrs_w_rows,
    torch::Tensor dst_w2_ptrs_s_rows, torch::Tensor inv_permuted_idx) {
  TORCH_CHECK(topk_ids.is_cuda(),
              "awq_moe_single_token_compact_prepare: topk_ids must be CUDA.");
  TORCH_CHECK(
      topk_ids.scalar_type() == torch::kInt32 ||
          topk_ids.scalar_type() == torch::kInt64,
      "awq_moe_single_token_compact_prepare: topk_ids must be int32/int64.");
  TORCH_CHECK(
      src_w13_ptrs_w_rows.is_cuda() && src_w13_ptrs_s_rows.is_cuda() &&
          src_w2_ptrs_w_rows.is_cuda() && src_w2_ptrs_s_rows.is_cuda(),
      "awq_moe_single_token_compact_prepare: source ptr rows must be CUDA.");
  TORCH_CHECK(dst_w13_ptrs_w_rows.is_cuda() && dst_w13_ptrs_s_rows.is_cuda() &&
                  dst_w2_ptrs_w_rows.is_cuda() && dst_w2_ptrs_s_rows.is_cuda(),
              "awq_moe_single_token_compact_prepare: destination ptr rows must "
              "be CUDA.");
  TORCH_CHECK(inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32,
              "awq_moe_single_token_compact_prepare: inv_permuted_idx "
              "must be CUDA int32.");

  topk_ids = topk_ids.contiguous().view({-1});
  TORCH_CHECK(
      topk_ids.numel() > 0,
      "awq_moe_single_token_compact_prepare: topk_ids must be non-empty.");
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(
      top_k <= 32,
      "awq_moe_single_token_compact_prepare: top_k > 32 is unsupported.");

  const int64_t row_bytes = src_w13_ptrs_w_rows.size(1);
  TORCH_CHECK(src_w13_ptrs_w_rows.scalar_type() == torch::kUInt8 &&
                  src_w13_ptrs_s_rows.scalar_type() == torch::kUInt8 &&
                  src_w2_ptrs_w_rows.scalar_type() == torch::kUInt8 &&
                  src_w2_ptrs_s_rows.scalar_type() == torch::kUInt8 &&
                  dst_w13_ptrs_w_rows.scalar_type() == torch::kUInt8 &&
                  dst_w13_ptrs_s_rows.scalar_type() == torch::kUInt8 &&
                  dst_w2_ptrs_w_rows.scalar_type() == torch::kUInt8 &&
                  dst_w2_ptrs_s_rows.scalar_type() == torch::kUInt8,
              "awq_moe_single_token_compact_prepare: ptr rows must be uint8.");
  TORCH_CHECK(
      src_w13_ptrs_w_rows.dim() == 2 && src_w13_ptrs_s_rows.dim() == 2 &&
          src_w2_ptrs_w_rows.dim() == 2 && src_w2_ptrs_s_rows.dim() == 2 &&
          dst_w13_ptrs_w_rows.dim() == 2 && dst_w13_ptrs_s_rows.dim() == 2 &&
          dst_w2_ptrs_w_rows.dim() == 2 && dst_w2_ptrs_s_rows.dim() == 2,
      "awq_moe_single_token_compact_prepare: ptr rows must be 2D.");
  TORCH_CHECK(src_w13_ptrs_w_rows.size(1) == row_bytes &&
                  src_w13_ptrs_s_rows.size(1) == row_bytes &&
                  src_w2_ptrs_w_rows.size(1) == row_bytes &&
                  src_w2_ptrs_s_rows.size(1) == row_bytes &&
                  dst_w13_ptrs_w_rows.size(0) == top_k &&
                  dst_w13_ptrs_s_rows.size(0) == top_k &&
                  dst_w2_ptrs_w_rows.size(0) == top_k &&
                  dst_w2_ptrs_s_rows.size(0) == top_k &&
                  dst_w13_ptrs_w_rows.size(1) == row_bytes &&
                  dst_w13_ptrs_s_rows.size(1) == row_bytes &&
                  dst_w2_ptrs_w_rows.size(1) == row_bytes &&
                  dst_w2_ptrs_s_rows.size(1) == row_bytes,
              "awq_moe_single_token_compact_prepare: ptr row shapes mismatch.");
  TORCH_CHECK(
      inv_permuted_idx.numel() == top_k,
      "awq_moe_single_token_compact_prepare: inv_permuted_idx size mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_ids));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int src_row_stride = static_cast<int>(src_w13_ptrs_w_rows.stride(0));
  const int dst_row_stride = static_cast<int>(dst_w13_ptrs_w_rows.stride(0));

  if (topk_ids.scalar_type() == torch::kInt32) {
    awq_moe_single_token_compact_prepare_kernel<int32_t><<<1, 1, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(), src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        inv_permuted_idx.data_ptr<int32_t>(), top_k, src_row_stride,
        dst_row_stride, static_cast<int>(row_bytes));
  } else {
    awq_moe_single_token_compact_prepare_kernel<int64_t><<<1, 1, 0, stream>>>(
        topk_ids.data_ptr<int64_t>(), src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        inv_permuted_idx.data_ptr<int32_t>(), top_k, src_row_stride,
        dst_row_stride, static_cast<int>(row_bytes));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename index_t>
__global__ void awq_moe_single_token_exact_layout_prepare_kernel(
    const index_t* topk_ids, const __half* x, __half* compact_input,
    int* expert_offsets, int64_t* expert_offsets64, int* inv_permuted_idx,
    int* sorted_expert_ids, int top_k, int hidden_size, int num_experts) {
  __shared__ int sorted_ids[32];
  __shared__ int sorted_src[32];

  if (threadIdx.x == 0) {
    for (int expert = 0; expert <= num_experts; ++expert) {
      expert_offsets[expert] = 0;
    }
    for (int i = 0; i < top_k; ++i) {
      sorted_ids[i] = static_cast<int>(topk_ids[i]);
      sorted_src[i] = i;
    }

    // Keep the same stable expert ordering as moe_permute for a single token.
    for (int i = 1; i < top_k; ++i) {
      const int expert_id = sorted_ids[i];
      const int src_idx = sorted_src[i];
      int j = i - 1;
      while (j >= 0 && sorted_ids[j] > expert_id) {
        sorted_ids[j + 1] = sorted_ids[j];
        sorted_src[j + 1] = sorted_src[j];
        --j;
      }
      sorted_ids[j + 1] = expert_id;
      sorted_src[j + 1] = src_idx;
    }

    for (int sorted_pos = 0; sorted_pos < top_k; ++sorted_pos) {
      const int expert_id = sorted_ids[sorted_pos];
      const int src_idx = sorted_src[sorted_pos];
      inv_permuted_idx[src_idx] = sorted_pos;
      if (sorted_expert_ids != nullptr) {
        sorted_expert_ids[sorted_pos] = expert_id;
      }
      expert_offsets[expert_id + 1] += 1;
    }
    for (int expert = 0; expert < num_experts; ++expert) {
      expert_offsets[expert + 1] += expert_offsets[expert];
    }
    if (expert_offsets64 != nullptr) {
      for (int expert = 0; expert <= num_experts; ++expert) {
        expert_offsets64[expert] = static_cast<int64_t>(expert_offsets[expert]);
      }
    }
  }
  __syncthreads();

  if ((hidden_size & 1) == 0) {
    const int hidden_pairs = hidden_size >> 1;
    const int total_pairs = top_k * hidden_pairs;
    const __half2* x_vec = reinterpret_cast<const __half2*>(x);
    __half2* compact_vec = reinterpret_cast<__half2*>(compact_input);
    for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
      const int col_pair = idx % hidden_pairs;
      compact_vec[idx] = x_vec[col_pair];
    }
  } else {
    const int total = top_k * hidden_size;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
      const int col = idx % hidden_size;
      compact_input[idx] = x[col];
    }
  }
}

void awq_moe_single_token_exact_layout_prepare(
    torch::Tensor topk_ids, torch::Tensor x, torch::Tensor compact_input,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, int64_t num_experts) {
  TORCH_CHECK(
      topk_ids.is_cuda(),
      "awq_moe_single_token_exact_layout_prepare: topk_ids must be CUDA.");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 ||
                  topk_ids.scalar_type() == torch::kInt64,
              "awq_moe_single_token_exact_layout_prepare: topk_ids must be "
              "int32/int64.");
  TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_exact_layout_prepare: x must be CUDA float16.");
  TORCH_CHECK(
      compact_input.is_cuda() && compact_input.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_exact_layout_prepare: compact_input must be CUDA "
      "float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32,
              "awq_moe_single_token_exact_layout_prepare: index buffers must "
              "be CUDA int32/int64.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "awq_moe_single_token_exact_layout_prepare: x must have shape "
              "[1, hidden].");

  topk_ids = topk_ids.contiguous().view({-1});
  inv_permuted_idx = inv_permuted_idx.contiguous().view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "awq_moe_single_token_exact_layout_prepare: top_k must be in [1, 32].");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) == top_k &&
                  compact_input.size(1) == x.size(1),
              "awq_moe_single_token_exact_layout_prepare: compact_input shape "
              "mismatch.");
  TORCH_CHECK(expert_offsets.numel() == num_experts + 1 &&
                  expert_offsets64.numel() == num_experts + 1,
              "awq_moe_single_token_exact_layout_prepare: expert_offsets size "
              "mismatch.");
  TORCH_CHECK(inv_permuted_idx.numel() == top_k,
              "awq_moe_single_token_exact_layout_prepare: inv_permuted_idx "
              "size mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  if (topk_ids.scalar_type() == torch::kInt32) {
    awq_moe_single_token_exact_layout_prepare_kernel<int32_t>
        <<<1, kThreads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
            expert_offsets.data_ptr<int32_t>(),
            expert_offsets64.data_ptr<int64_t>(),
            inv_permuted_idx.data_ptr<int32_t>(), nullptr, top_k,
            static_cast<int>(x.size(1)), static_cast<int>(num_experts));
  } else {
    awq_moe_single_token_exact_layout_prepare_kernel<int64_t>
        <<<1, kThreads, 0, stream>>>(
            topk_ids.data_ptr<int64_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
            expert_offsets.data_ptr<int32_t>(),
            expert_offsets64.data_ptr<int64_t>(),
            inv_permuted_idx.data_ptr<int32_t>(), nullptr, top_k,
            static_cast<int>(x.size(1)), static_cast<int>(num_experts));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void awq_moe_gemm_sm70_out(torch::Tensor out, torch::Tensor sorted_input,
                           torch::Tensor expert_offsets,
                           torch::Tensor strided_ptrs_w,
                           torch::Tensor strided_ptrs_s, int64_t num_experts,
                           int64_t k, int64_t n, int64_t group_size,
                           bool gated_silu);

void awq_moe_gemm_sm70_per_expert_dispatch_out(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w, torch::Tensor strided_ptrs_s,
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu);

void awq_moe_gemm_sm70_out_impl(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w, torch::Tensor strided_ptrs_s,
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu, torch::Tensor b_group_indices, bool per_expert_dispatch,
    torch::Tensor reduce_out, torch::Tensor sorted_weights,
    bool weighted_reduce);

template <typename index_t>
__global__ void awq_moe_single_token_prepare_kernel(
    const index_t* topk_ids, const float* topk_weights, const __half* x,
    const uint8_t* src_w13_ptrs_w_rows, const uint8_t* src_w13_ptrs_s_rows,
    const uint8_t* src_w2_ptrs_w_rows, const uint8_t* src_w2_ptrs_s_rows,
    uint8_t* dst_w13_ptrs_w_rows, uint8_t* dst_w13_ptrs_s_rows,
    uint8_t* dst_w2_ptrs_w_rows, uint8_t* dst_w2_ptrs_s_rows,
    __half* compact_input, float* sorted_weights, int* expert_offsets,
    int64_t* expert_offsets64, int* inv_permuted_idx, int* sorted_expert_ids,
    int top_k, int hidden_size, int src_row_stride, int dst_row_stride,
    int row_bytes, bool copy_expert_ptr_rows, bool vectorized_prepare) {
  __shared__ int sorted_ids[32];
  __shared__ int sorted_src[32];

  if (threadIdx.x == 0) {
    expert_offsets[0] = 0;
    if (expert_offsets64 != nullptr) {
      expert_offsets64[0] = 0;
    }
    for (int i = 0; i < top_k; ++i) {
      sorted_ids[i] = static_cast<int>(topk_ids[i]);
      sorted_src[i] = i;
      expert_offsets[i + 1] = i + 1;
      if (expert_offsets64 != nullptr) {
        expert_offsets64[i + 1] = static_cast<int64_t>(i + 1);
      }
    }

    // Stable insertion sort to match native single-token MoE permutation.
    for (int i = 1; i < top_k; ++i) {
      const int expert_id = sorted_ids[i];
      const int src_idx = sorted_src[i];
      int j = i - 1;
      while (j >= 0 && sorted_ids[j] > expert_id) {
        sorted_ids[j + 1] = sorted_ids[j];
        sorted_src[j + 1] = sorted_src[j];
        --j;
      }
      sorted_ids[j + 1] = expert_id;
      sorted_src[j + 1] = src_idx;
    }

    for (int sorted_pos = 0; sorted_pos < top_k; ++sorted_pos) {
      const int src_idx = sorted_src[sorted_pos];
      inv_permuted_idx[src_idx] = sorted_pos;
      if (sorted_expert_ids != nullptr) {
        sorted_expert_ids[sorted_pos] = sorted_ids[sorted_pos];
      }
      if (sorted_weights != nullptr && topk_weights != nullptr) {
        sorted_weights[sorted_pos] = topk_weights[src_idx];
      }
    }
  }
  __syncthreads();

  if (copy_expert_ptr_rows) {
    if (vectorized_prepare &&
        (row_bytes % static_cast<int>(sizeof(uint4)) == 0) &&
        (src_row_stride % static_cast<int>(sizeof(uint4)) == 0) &&
        (dst_row_stride % static_cast<int>(sizeof(uint4)) == 0)) {
      const int row_vecs = row_bytes / static_cast<int>(sizeof(uint4));
      const int src_row_vec_stride =
          src_row_stride / static_cast<int>(sizeof(uint4));
      const int dst_row_vec_stride =
          dst_row_stride / static_cast<int>(sizeof(uint4));
      const int total_row_vecs = top_k * row_vecs;
      const uint4* src_w13_w_vec =
          reinterpret_cast<const uint4*>(src_w13_ptrs_w_rows);
      const uint4* src_w13_s_vec =
          reinterpret_cast<const uint4*>(src_w13_ptrs_s_rows);
      const uint4* src_w2_w_vec =
          reinterpret_cast<const uint4*>(src_w2_ptrs_w_rows);
      const uint4* src_w2_s_vec =
          reinterpret_cast<const uint4*>(src_w2_ptrs_s_rows);
      uint4* dst_w13_w_vec = reinterpret_cast<uint4*>(dst_w13_ptrs_w_rows);
      uint4* dst_w13_s_vec = reinterpret_cast<uint4*>(dst_w13_ptrs_s_rows);
      uint4* dst_w2_w_vec = reinterpret_cast<uint4*>(dst_w2_ptrs_w_rows);
      uint4* dst_w2_s_vec = reinterpret_cast<uint4*>(dst_w2_ptrs_s_rows);

      for (int idx = threadIdx.x; idx < total_row_vecs; idx += blockDim.x) {
        const int sorted_pos = idx / row_vecs;
        const int vec_idx = idx - sorted_pos * row_vecs;
        const int expert_id = sorted_ids[sorted_pos];
        const int src_offset = expert_id * src_row_vec_stride + vec_idx;
        const int dst_offset = sorted_pos * dst_row_vec_stride + vec_idx;

        dst_w13_w_vec[dst_offset] = src_w13_w_vec[src_offset];
        dst_w13_s_vec[dst_offset] = src_w13_s_vec[src_offset];
        dst_w2_w_vec[dst_offset] = src_w2_w_vec[src_offset];
        dst_w2_s_vec[dst_offset] = src_w2_s_vec[src_offset];
      }
    } else {
      const int total_row_bytes = top_k * row_bytes;
      for (int idx = threadIdx.x; idx < total_row_bytes; idx += blockDim.x) {
        const int sorted_pos = idx / row_bytes;
        const int byte_idx = idx - sorted_pos * row_bytes;
        const int expert_id = sorted_ids[sorted_pos];

        const uint8_t* src_w13_w =
            src_w13_ptrs_w_rows + expert_id * src_row_stride;
        const uint8_t* src_w13_s =
            src_w13_ptrs_s_rows + expert_id * src_row_stride;
        const uint8_t* src_w2_w =
            src_w2_ptrs_w_rows + expert_id * src_row_stride;
        const uint8_t* src_w2_s =
            src_w2_ptrs_s_rows + expert_id * src_row_stride;

        uint8_t* dst_w13_w = dst_w13_ptrs_w_rows + sorted_pos * dst_row_stride;
        uint8_t* dst_w13_s = dst_w13_ptrs_s_rows + sorted_pos * dst_row_stride;
        uint8_t* dst_w2_w = dst_w2_ptrs_w_rows + sorted_pos * dst_row_stride;
        uint8_t* dst_w2_s = dst_w2_ptrs_s_rows + sorted_pos * dst_row_stride;

        dst_w13_w[byte_idx] = src_w13_w[byte_idx];
        dst_w13_s[byte_idx] = src_w13_s[byte_idx];
        dst_w2_w[byte_idx] = src_w2_w[byte_idx];
        dst_w2_s[byte_idx] = src_w2_s[byte_idx];
      }
    }
  }

  if (compact_input == nullptr) {
    return;
  }

  if (vectorized_prepare && (hidden_size & 1) == 0) {
    const int hidden_pairs = hidden_size >> 1;
    const int total_pairs = top_k * hidden_pairs;
    const __half2* x_vec = reinterpret_cast<const __half2*>(x);
    __half2* compact_vec = reinterpret_cast<__half2*>(compact_input);
    for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
      const int col_pair = idx % hidden_pairs;
      compact_vec[idx] = x_vec[col_pair];
    }
  } else {
    const int total_elems = top_k * hidden_size;
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
      const int col = idx % hidden_size;
      compact_input[idx] = x[col];
    }
  }
}

__global__ void fp8_moe_single_token_router_prepare_256_top8_kernel(
    const __half* router_logits, float* topk_weights, int32_t* topk_ids,
    const __half* x, const uint8_t* src_w13_ptrs_w_rows,
    const uint8_t* src_w13_ptrs_s_rows, const uint8_t* src_w2_ptrs_w_rows,
    const uint8_t* src_w2_ptrs_s_rows, uint8_t* dst_w13_ptrs_w_rows,
    uint8_t* dst_w13_ptrs_s_rows, uint8_t* dst_w2_ptrs_w_rows,
    uint8_t* dst_w2_ptrs_s_rows, __half* compact_input, float* sorted_weights,
    int* expert_offsets, int* inv_permuted_idx, int hidden_size,
    int src_row_stride, int dst_row_stride, int row_bytes, bool renormalize,
    bool vectorized_prepare) {
  constexpr int kNumExperts = 256;
  constexpr int kTopK = 8;
  constexpr int kValsPerLane = 8;
  __shared__ int selected_ids[kTopK];
  __shared__ int sorted_ids[kTopK];
  __shared__ int sorted_src[kTopK];
  __shared__ float selected_weights[kTopK];

  const int tid = threadIdx.x;
  const int lane = tid & 31;

  if (tid < 32) {
    float row_chunk[kValsPerLane];
    float row_choice[kValsPerLane];

  #pragma unroll
    for (int i = 0; i < kValsPerLane; ++i) {
      row_chunk[i] = __half2float(router_logits[lane * kValsPerLane + i]);
    }

    float thread_max = row_chunk[0];
  #pragma unroll
    for (int i = 1; i < kValsPerLane; ++i) {
      thread_max = max(thread_max, row_chunk[i]);
    }
  #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      thread_max =
          max(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask, 32));
    }

    float row_sum = 0.f;
  #pragma unroll
    for (int i = 0; i < kValsPerLane; ++i) {
      row_chunk[i] = expf(row_chunk[i] - thread_max);
      row_sum += row_chunk[i];
    }
  #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      row_sum += __shfl_xor_sync(0xffffffff, row_sum, mask, 32);
    }

    const float inv_row_sum = 1.f / row_sum;
  #pragma unroll
    for (int i = 0; i < kValsPerLane; ++i) {
      row_chunk[i] *= inv_row_sum;
      row_choice[i] = row_chunk[i];
    }

    float selected_sum = 0.f;
  #pragma unroll
    for (int k_idx = 0; k_idx < kTopK; ++k_idx) {
      float max_val = row_chunk[0];
      float max_choice = row_choice[0];
      int expert = lane * kValsPerLane;
  #pragma unroll
      for (int i = 1; i < kValsPerLane; ++i) {
        const float choice = row_choice[i];
        if (choice > max_choice) {
          max_choice = choice;
          max_val = row_chunk[i];
          expert = lane * kValsPerLane + i;
        }
      }
  #pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        const float other_choice =
            __shfl_xor_sync(0xffffffff, max_choice, mask, 32);
        const float other_val = __shfl_xor_sync(0xffffffff, max_val, mask, 32);
        const int other_expert = __shfl_xor_sync(0xffffffff, expert, mask, 32);
        if (other_choice > max_choice ||
            (other_choice == max_choice && other_expert < expert)) {
          max_choice = other_choice;
          max_val = other_val;
          expert = other_expert;
        }
      }

      if (lane == 0) {
        selected_ids[k_idx] = expert;
        selected_weights[k_idx] = max_val;
        selected_sum += max_val;
      }

      if (k_idx + 1 < kTopK) {
        const int owner_lane = expert / kValsPerLane;
        const int owner_offset = expert - owner_lane * kValsPerLane;
        if (lane == owner_lane) {
          row_choice[owner_offset] = -10000.f;
        }
      }
    }

    if (lane == 0) {
      const float denom =
          (renormalize && selected_sum > 0.f) ? selected_sum : 1.f;
      expert_offsets[0] = 0;
      for (int i = 0; i < kTopK; ++i) {
        const float weight = selected_weights[i] / denom;
        selected_weights[i] = weight;
        topk_ids[i] = selected_ids[i];
        topk_weights[i] = weight;
        sorted_ids[i] = selected_ids[i];
        sorted_src[i] = i;
        expert_offsets[i + 1] = i + 1;
      }

      for (int i = 1; i < kTopK; ++i) {
        const int expert_id = sorted_ids[i];
        const int src_idx = sorted_src[i];
        int j = i - 1;
        while (j >= 0 && sorted_ids[j] > expert_id) {
          sorted_ids[j + 1] = sorted_ids[j];
          sorted_src[j + 1] = sorted_src[j];
          --j;
        }
        sorted_ids[j + 1] = expert_id;
        sorted_src[j + 1] = src_idx;
      }

      for (int sorted_pos = 0; sorted_pos < kTopK; ++sorted_pos) {
        const int src_idx = sorted_src[sorted_pos];
        inv_permuted_idx[src_idx] = sorted_pos;
        sorted_weights[sorted_pos] = selected_weights[src_idx];
      }
    }
  }
  __syncthreads();

  if (vectorized_prepare &&
      (row_bytes % static_cast<int>(sizeof(uint4)) == 0) &&
      (src_row_stride % static_cast<int>(sizeof(uint4)) == 0) &&
      (dst_row_stride % static_cast<int>(sizeof(uint4)) == 0)) {
    const int row_vecs = row_bytes / static_cast<int>(sizeof(uint4));
    const int src_row_vec_stride =
        src_row_stride / static_cast<int>(sizeof(uint4));
    const int dst_row_vec_stride =
        dst_row_stride / static_cast<int>(sizeof(uint4));
    const int total_row_vecs = kTopK * row_vecs;
    const uint4* src_w13_w_vec =
        reinterpret_cast<const uint4*>(src_w13_ptrs_w_rows);
    const uint4* src_w13_s_vec =
        reinterpret_cast<const uint4*>(src_w13_ptrs_s_rows);
    const uint4* src_w2_w_vec =
        reinterpret_cast<const uint4*>(src_w2_ptrs_w_rows);
    const uint4* src_w2_s_vec =
        reinterpret_cast<const uint4*>(src_w2_ptrs_s_rows);
    uint4* dst_w13_w_vec = reinterpret_cast<uint4*>(dst_w13_ptrs_w_rows);
    uint4* dst_w13_s_vec = reinterpret_cast<uint4*>(dst_w13_ptrs_s_rows);
    uint4* dst_w2_w_vec = reinterpret_cast<uint4*>(dst_w2_ptrs_w_rows);
    uint4* dst_w2_s_vec = reinterpret_cast<uint4*>(dst_w2_ptrs_s_rows);

    for (int idx = tid; idx < total_row_vecs; idx += blockDim.x) {
      const int sorted_pos = idx / row_vecs;
      const int vec_idx = idx - sorted_pos * row_vecs;
      const int expert_id = sorted_ids[sorted_pos];
      const int src_offset = expert_id * src_row_vec_stride + vec_idx;
      const int dst_offset = sorted_pos * dst_row_vec_stride + vec_idx;

      dst_w13_w_vec[dst_offset] = src_w13_w_vec[src_offset];
      dst_w13_s_vec[dst_offset] = src_w13_s_vec[src_offset];
      dst_w2_w_vec[dst_offset] = src_w2_w_vec[src_offset];
      dst_w2_s_vec[dst_offset] = src_w2_s_vec[src_offset];
    }
  } else {
    const int total_row_bytes = kTopK * row_bytes;
    for (int idx = tid; idx < total_row_bytes; idx += blockDim.x) {
      const int sorted_pos = idx / row_bytes;
      const int byte_idx = idx - sorted_pos * row_bytes;
      const int expert_id = sorted_ids[sorted_pos];

      const uint8_t* src_w13_w =
          src_w13_ptrs_w_rows + expert_id * src_row_stride;
      const uint8_t* src_w13_s =
          src_w13_ptrs_s_rows + expert_id * src_row_stride;
      const uint8_t* src_w2_w = src_w2_ptrs_w_rows + expert_id * src_row_stride;
      const uint8_t* src_w2_s = src_w2_ptrs_s_rows + expert_id * src_row_stride;

      dst_w13_ptrs_w_rows[sorted_pos * dst_row_stride + byte_idx] =
          src_w13_w[byte_idx];
      dst_w13_ptrs_s_rows[sorted_pos * dst_row_stride + byte_idx] =
          src_w13_s[byte_idx];
      dst_w2_ptrs_w_rows[sorted_pos * dst_row_stride + byte_idx] =
          src_w2_w[byte_idx];
      dst_w2_ptrs_s_rows[sorted_pos * dst_row_stride + byte_idx] =
          src_w2_s[byte_idx];
    }
  }

  if (vectorized_prepare && (hidden_size & 1) == 0) {
    const int hidden_pairs = hidden_size >> 1;
    const int total_pairs = kTopK * hidden_pairs;
    const __half2* x_vec = reinterpret_cast<const __half2*>(x);
    __half2* compact_vec = reinterpret_cast<__half2*>(compact_input);
    for (int idx = tid; idx < total_pairs; idx += blockDim.x) {
      const int col_pair = idx % hidden_pairs;
      compact_vec[idx] = x_vec[col_pair];
    }
  } else {
    const int total_elems = kTopK * hidden_size;
    for (int idx = tid; idx < total_elems; idx += blockDim.x) {
      const int col = idx % hidden_size;
      compact_input[idx] = x[col];
    }
  }
}

__global__ void awq_moe_single_token_weighted_reduce_kernel(
    const __half* sorted_output, const float* topk_weights,
    const int* inv_permuted_idx, __half* out, int top_k,
    int hidden_logical_size, int sorted_output_row_stride) {
  for (int col = blockIdx.x * blockDim.x + threadIdx.x;
       col < hidden_logical_size; col += blockDim.x * gridDim.x) {
    float acc = 0.f;
    for (int route_idx = 0; route_idx < top_k; ++route_idx) {
      const int sorted_pos = inv_permuted_idx[route_idx];
      const __half value =
          sorted_output[sorted_pos * sorted_output_row_stride + col];
      acc = fmaf(topk_weights[route_idx], __half2float(value), acc);
    }
    out[col] = __float2half(acc);
  }
}

template <int TOPK>
__global__ void awq_moe_single_token_weighted_reduce_half2_kernel(
    const __half* sorted_output, const float* topk_weights,
    const int* inv_permuted_idx, __half* out, int hidden_logical_size,
    int sorted_output_row_stride) {
  __shared__ int sorted_pos_sh[TOPK];
  __shared__ float weight_sh[TOPK];

  if (threadIdx.x < TOPK) {
    const int route_idx = threadIdx.x;
    sorted_pos_sh[route_idx] = inv_permuted_idx[route_idx];
    weight_sh[route_idx] = topk_weights[route_idx];
  }
  __syncthreads();

  const int hidden_pairs = hidden_logical_size >> 1;
  for (int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pair_idx < hidden_pairs; pair_idx += blockDim.x * gridDim.x) {
    float2 acc = {0.f, 0.f};
  #pragma unroll
    for (int route_idx = 0; route_idx < TOPK; ++route_idx) {
      const __half2 value = reinterpret_cast<const __half2*>(
          sorted_output +
          sorted_pos_sh[route_idx] * sorted_output_row_stride)[pair_idx];
      const float2 value_f = __half22float2(value);
      const float weight = weight_sh[route_idx];
      acc.x = fmaf(weight, value_f.x, acc.x);
      acc.y = fmaf(weight, value_f.y, acc.y);
    }
    reinterpret_cast<__half2*>(out)[pair_idx] = __floats2half2_rn(acc.x, acc.y);
  }
}

void awq_moe_single_token_weighted_reduce_out(torch::Tensor sorted_output,
                                              torch::Tensor topk_weights,
                                              torch::Tensor inv_permuted_idx,
                                              torch::Tensor out, int64_t top_k,
                                              int64_t hidden_logical_size) {
  TORCH_CHECK(
      sorted_output.is_cuda() && sorted_output.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_weighted_reduce_out: sorted_output must be CUDA "
      "float16.");
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.scalar_type() == torch::kFloat32,
      "awq_moe_single_token_weighted_reduce_out: topk_weights must be CUDA "
      "float32.");
  TORCH_CHECK(inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32,
              "awq_moe_single_token_weighted_reduce_out: inv_permuted_idx must "
              "be CUDA int32.");
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_weighted_reduce_out: out must be CUDA float16.");
  TORCH_CHECK(sorted_output.dim() == 2 && out.dim() == 2 && out.size(0) == 1,
              "awq_moe_single_token_weighted_reduce_out: invalid tensor rank.");
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "awq_moe_single_token_weighted_reduce_out: top_k must be in [1, 32].");
  TORCH_CHECK(sorted_output.size(0) >= top_k && topk_weights.numel() >= top_k &&
                  inv_permuted_idx.numel() >= top_k,
              "awq_moe_single_token_weighted_reduce_out: top_k size mismatch.");
  TORCH_CHECK(
      out.size(1) >= hidden_logical_size &&
          sorted_output.size(1) >= hidden_logical_size,
      "awq_moe_single_token_weighted_reduce_out: hidden size mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  const int hidden_logical_size_i = static_cast<int>(hidden_logical_size);
  const int sorted_output_row_stride =
      static_cast<int>(sorted_output.stride(0));
  if (top_k == 8 && (hidden_logical_size_i % 2) == 0 &&
      (sorted_output_row_stride % 2) == 0) {
    const int blocks = std::max<int>(
        1, ((hidden_logical_size_i >> 1) + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_half2_kernel<8>
        <<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
            topk_weights.data_ptr<float>(),
            inv_permuted_idx.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            hidden_logical_size_i, sorted_output_row_stride);
  } else {
    const int blocks =
        std::max<int>(1, (hidden_logical_size_i + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_kernel<<<blocks, kThreads, 0,
                                                  stream>>>(
        reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        static_cast<int>(top_k), hidden_logical_size_i,
        sorted_output_row_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t x) {
  __nv_fp8_e4m3 v;
  v.__x = static_cast<__nv_fp8_storage_t>(x);
  return static_cast<float>(v);
}

template <typename TopKIdT>
__global__ void fp8_moe_w2_direct_reduce_kernel(
    const __half* __restrict__ intermediate,
    const float* __restrict__ topk_weights,
    const TopKIdT* __restrict__ topk_ids,
    const int* __restrict__ inv_permuted_idx,
    const uint8_t* __restrict__ w2_weight, const float* __restrict__ w2_scales,
    __half* __restrict__ out, int top_k, int k, int n, int scale_n_blocks,
    int scale_k_blocks) {
  constexpr int kWarpsPerBlock = 4;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int col = blockIdx.x * kWarpsPerBlock + warp_id;
  if (col >= n) {
    return;
  }

  float total = 0.f;
  for (int route_idx = 0; route_idx < top_k; ++route_idx) {
    const int sorted_pos = __ldg(inv_permuted_idx + route_idx);
    const int expert_id = static_cast<int>(__ldg(topk_ids + route_idx));
    const float route_weight = __ldg(topk_weights + route_idx);
    const uint8_t* weight_row =
        w2_weight + (static_cast<int64_t>(expert_id) * n + col) * k;
    const float* scale_base =
        w2_scales +
        (static_cast<int64_t>(expert_id) * scale_n_blocks + col / 128) *
            scale_k_blocks;
    const __half* input_row =
        intermediate + static_cast<int64_t>(sorted_pos) * k;

    float acc = 0.f;
    for (int kk = lane_id; kk < k; kk += WARP_SIZE) {
      const float a = __half2float(__ldg(input_row + kk));
      const float scale = __ldg(scale_base + kk / 128);
      const float b = fp8_e4m3_to_float(__ldg(weight_row + kk)) * scale;
      acc = fmaf(a, b, acc);
    }
  #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane_id == 0) {
      total = fmaf(route_weight, acc, total);
    }
  }
  if (lane_id == 0) {
    out[col] = __float2half_rn(total);
  }
}

void fp8_moe_w2_direct_reduce_sm70_out(
    torch::Tensor out, torch::Tensor intermediate, torch::Tensor topk_weights,
    torch::Tensor topk_ids, torch::Tensor inv_permuted_idx,
    torch::Tensor w2_weight, torch::Tensor w2_scales, int64_t top_k, int64_t k,
    int64_t n, int64_t group_size, cudaStream_t stream) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "fp8_moe_w2_direct_reduce_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(
      intermediate.is_cuda() && intermediate.scalar_type() == torch::kFloat16,
      "fp8_moe_w2_direct_reduce_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.scalar_type() == torch::kFloat32,
      "fp8_moe_w2_direct_reduce_sm70_out: weights must be CUDA float32.");
  TORCH_CHECK(
      topk_ids.is_cuda() && (topk_ids.scalar_type() == torch::kInt32 ||
                             topk_ids.scalar_type() == torch::kInt64),
      "fp8_moe_w2_direct_reduce_sm70_out: topk ids must be int32/int64.");
  TORCH_CHECK(
      inv_permuted_idx.is_cuda() &&
          inv_permuted_idx.scalar_type() == torch::kInt32,
      "fp8_moe_w2_direct_reduce_sm70_out: inverse indices must be int32.");
  TORCH_CHECK(w2_weight.is_cuda() &&
                  w2_weight.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_moe_w2_direct_reduce_sm70_out: W2 must be float8_e4m3fn.");
  TORCH_CHECK(w2_scales.is_cuda() && w2_scales.scalar_type() == torch::kFloat32,
              "fp8_moe_w2_direct_reduce_sm70_out: W2 scales must be float32.");
  TORCH_CHECK(
      group_size == 128,
      "fp8_moe_w2_direct_reduce_sm70_out: only group_size=128 is supported.");
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "fp8_moe_w2_direct_reduce_sm70_out: invalid top_k.");
  TORCH_CHECK(k > 0 && n > 0,
              "fp8_moe_w2_direct_reduce_sm70_out: invalid dimensions.");
  TORCH_CHECK(intermediate.dim() == 2 && intermediate.size(0) >= top_k &&
                  intermediate.size(1) == k,
              "fp8_moe_w2_direct_reduce_sm70_out: input shape mismatch.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == 1 && out.size(1) == n &&
                  out.stride(1) == 1,
              "fp8_moe_w2_direct_reduce_sm70_out: output shape mismatch.");
  TORCH_CHECK(topk_weights.numel() >= top_k && topk_ids.numel() >= top_k &&
                  inv_permuted_idx.numel() >= top_k,
              "fp8_moe_w2_direct_reduce_sm70_out: top-k buffers too small.");
  TORCH_CHECK(w2_weight.dim() == 3 && w2_weight.size(1) == n &&
                  w2_weight.size(2) == k && w2_weight.stride(2) == 1,
              "fp8_moe_w2_direct_reduce_sm70_out: W2 shape mismatch.");
  const int64_t scale_n_blocks = (n + group_size - 1) / group_size;
  const int64_t scale_k_blocks = (k + group_size - 1) / group_size;
  TORCH_CHECK(w2_scales.dim() == 3 && w2_scales.size(0) == w2_weight.size(0) &&
                  w2_scales.size(1) == scale_n_blocks &&
                  w2_scales.size(2) == scale_k_blocks &&
                  w2_scales.stride(2) == 1,
              "fp8_moe_w2_direct_reduce_sm70_out: W2 scale shape mismatch.");

  constexpr int kThreads = 128;
  constexpr int kWarpsPerBlock = kThreads / WARP_SIZE;
  const int blocks =
      static_cast<int>((n + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (topk_ids.scalar_type() == torch::kInt32) {
    fp8_moe_w2_direct_reduce_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(intermediate.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), topk_ids.data_ptr<int32_t>(),
        inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t*>(w2_weight.data_ptr()),
        w2_scales.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        static_cast<int>(top_k), static_cast<int>(k), static_cast<int>(n),
        static_cast<int>(scale_n_blocks), static_cast<int>(scale_k_blocks));
  } else {
    fp8_moe_w2_direct_reduce_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(intermediate.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), topk_ids.data_ptr<int64_t>(),
        inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t*>(w2_weight.data_ptr()),
        w2_scales.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        static_cast<int>(top_k), static_cast<int>(k), static_cast<int>(n),
        static_cast<int>(scale_n_blocks), static_cast<int>(scale_k_blocks));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void awq_moe_single_token_sm70_out(
    torch::Tensor out, torch::Tensor x, torch::Tensor topk_weights,
    torch::Tensor topk_ids, torch::Tensor src_w13_ptrs_w_rows,
    torch::Tensor src_w13_ptrs_s_rows, torch::Tensor src_w2_ptrs_w_rows,
    torch::Tensor src_w2_ptrs_s_rows, torch::Tensor compact_input,
    torch::Tensor intermediate, torch::Tensor sorted_output,
    torch::Tensor sorted_weights, torch::Tensor dst_w13_ptrs_w_rows,
    torch::Tensor dst_w13_ptrs_s_rows, torch::Tensor dst_w2_ptrs_w_rows,
    torch::Tensor dst_w2_ptrs_s_rows, torch::Tensor expert_offsets,
    torch::Tensor inv_permuted_idx, int64_t w13_k, int64_t w13_n, int64_t w2_k,
    int64_t w2_n, int64_t group_size, int64_t hidden_logical_size) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_sm70_out: x must be CUDA float16.");
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.scalar_type() == torch::kFloat32,
      "awq_moe_single_token_sm70_out: topk_weights must be CUDA float32.");
  TORCH_CHECK(
      topk_ids.is_cuda() && (topk_ids.scalar_type() == torch::kInt32 ||
                             topk_ids.scalar_type() == torch::kInt64),
      "awq_moe_single_token_sm70_out: topk_ids must be CUDA int32/int64.");
  TORCH_CHECK(
      compact_input.is_cuda() &&
          compact_input.scalar_type() == torch::kFloat16 &&
          intermediate.is_cuda() &&
          intermediate.scalar_type() == torch::kFloat16 &&
          sorted_output.is_cuda() &&
          sorted_output.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_sm70_out: scratch buffers must be CUDA float16.");
  TORCH_CHECK(
      sorted_weights.is_cuda() &&
          sorted_weights.scalar_type() == torch::kFloat32,
      "awq_moe_single_token_sm70_out: sorted_weights must be CUDA float32.");
  TORCH_CHECK(
      expert_offsets.is_cuda() &&
          expert_offsets.scalar_type() == torch::kInt32 &&
          inv_permuted_idx.is_cuda() &&
          inv_permuted_idx.scalar_type() == torch::kInt32,
      "awq_moe_single_token_sm70_out: index buffers must be CUDA int32.");
  TORCH_CHECK(
      src_w13_ptrs_w_rows.is_cuda() && src_w13_ptrs_s_rows.is_cuda() &&
          src_w2_ptrs_w_rows.is_cuda() && src_w2_ptrs_s_rows.is_cuda() &&
          dst_w13_ptrs_w_rows.is_cuda() && dst_w13_ptrs_s_rows.is_cuda() &&
          dst_w2_ptrs_w_rows.is_cuda() && dst_w2_ptrs_s_rows.is_cuda(),
      "awq_moe_single_token_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "awq_moe_single_token_sm70_out: x must have shape [1, hidden].");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == 1,
      "awq_moe_single_token_sm70_out: out must have shape [1, hidden].");
  TORCH_CHECK(out.size(1) == hidden_logical_size,
              "awq_moe_single_token_sm70_out: out cols must match "
              "hidden_logical_size.");

  topk_ids = topk_ids.contiguous().view({-1});
  topk_weights = topk_weights.contiguous().view({-1});
  inv_permuted_idx = inv_permuted_idx.contiguous().view({-1});

  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "awq_moe_single_token_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(static_cast<int>(topk_weights.numel()) == top_k,
              "awq_moe_single_token_sm70_out: topk_weights size mismatch.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) == top_k &&
                  compact_input.size(1) == x.size(1),
              "awq_moe_single_token_sm70_out: compact_input shape mismatch.");
  TORCH_CHECK(intermediate.dim() == 2 && intermediate.size(0) == top_k &&
                  intermediate.size(1) == w13_n / 2,
              "awq_moe_single_token_sm70_out: intermediate shape mismatch.");
  TORCH_CHECK(sorted_output.dim() == 2 && sorted_output.size(0) == top_k &&
                  sorted_output.size(1) == w2_n,
              "awq_moe_single_token_sm70_out: sorted_output shape mismatch.");
  TORCH_CHECK(sorted_weights.numel() >= top_k,
              "awq_moe_single_token_sm70_out: sorted_weights size mismatch.");
  TORCH_CHECK(expert_offsets.numel() == top_k + 1,
              "awq_moe_single_token_sm70_out: expert_offsets size mismatch.");
  TORCH_CHECK(inv_permuted_idx.numel() == top_k,
              "awq_moe_single_token_sm70_out: inv_permuted_idx size mismatch.");

  const int64_t row_bytes = src_w13_ptrs_w_rows.size(1);
  TORCH_CHECK(
      src_w13_ptrs_w_rows.dim() == 2 && src_w13_ptrs_s_rows.dim() == 2 &&
          src_w2_ptrs_w_rows.dim() == 2 && src_w2_ptrs_s_rows.dim() == 2 &&
          dst_w13_ptrs_w_rows.dim() == 2 && dst_w13_ptrs_s_rows.dim() == 2 &&
          dst_w2_ptrs_w_rows.dim() == 2 && dst_w2_ptrs_s_rows.dim() == 2,
      "awq_moe_single_token_sm70_out: ptr row tensors must be 2D.");
  TORCH_CHECK(
      dst_w13_ptrs_w_rows.size(0) == top_k &&
          dst_w13_ptrs_s_rows.size(0) == top_k &&
          dst_w2_ptrs_w_rows.size(0) == top_k &&
          dst_w2_ptrs_s_rows.size(0) == top_k &&
          dst_w13_ptrs_w_rows.size(1) == row_bytes &&
          dst_w13_ptrs_s_rows.size(1) == row_bytes &&
          dst_w2_ptrs_w_rows.size(1) == row_bytes &&
          dst_w2_ptrs_s_rows.size(1) == row_bytes,
      "awq_moe_single_token_sm70_out: destination ptr row shapes mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int src_row_stride = static_cast<int>(src_w13_ptrs_w_rows.stride(0));
  const int dst_row_stride = static_cast<int>(dst_w13_ptrs_w_rows.stride(0));
  const bool vectorized_prepare = false;
  constexpr int kThreads = 256;

  if (topk_ids.scalar_type() == torch::kInt32) {
    awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
        sorted_weights.data_ptr<float>(), expert_offsets.data_ptr<int32_t>(),
        nullptr, inv_permuted_idx.data_ptr<int32_t>(), nullptr, top_k,
        static_cast<int>(x.size(1)), src_row_stride, dst_row_stride,
        static_cast<int>(row_bytes), true, vectorized_prepare);
  } else {
    awq_moe_single_token_prepare_kernel<int64_t><<<1, kThreads, 0, stream>>>(
        topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
        sorted_weights.data_ptr<float>(), expert_offsets.data_ptr<int32_t>(),
        nullptr, inv_permuted_idx.data_ptr<int32_t>(), nullptr, top_k,
        static_cast<int>(x.size(1)), src_row_stride, dst_row_stride,
        static_cast<int>(row_bytes), true, vectorized_prepare);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  awq_moe_gemm_sm70_per_expert_dispatch_out(
      intermediate, compact_input, expert_offsets, dst_w13_ptrs_w_rows,
      dst_w13_ptrs_s_rows, top_k, w13_k, w13_n, group_size, true);
  const bool use_weighted_reduce_epilogue = out.size(1) == w2_n;
  if (use_weighted_reduce_epilogue) {
    awq_moe_gemm_sm70_out_impl(sorted_output, intermediate, expert_offsets,
                               dst_w2_ptrs_w_rows, dst_w2_ptrs_s_rows, top_k,
                               w2_k, w2_n, group_size, false, torch::Tensor(),
                               true, out, sorted_weights, true);
    return;
  }
  awq_moe_gemm_sm70_per_expert_dispatch_out(
      sorted_output, intermediate, expert_offsets, dst_w2_ptrs_w_rows,
      dst_w2_ptrs_s_rows, top_k, w2_k, w2_n, group_size, false);

  const int hidden_logical_size_i = static_cast<int>(hidden_logical_size);
  const int sorted_output_row_stride =
      static_cast<int>(sorted_output.stride(0));
  if (top_k == 8 && (hidden_logical_size_i % 2) == 0 &&
      (sorted_output_row_stride % 2) == 0) {
    const int blocks = std::max<int>(
        1, ((hidden_logical_size_i >> 1) + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_half2_kernel<8>
        <<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
            topk_weights.data_ptr<float>(),
            inv_permuted_idx.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            hidden_logical_size_i, sorted_output_row_stride);
  } else {
    const int blocks =
        std::max<int>(1, (hidden_logical_size_i + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_kernel<<<blocks, kThreads, 0,
                                                  stream>>>(
        reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), top_k,
        hidden_logical_size_i, sorted_output_row_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void awq_moe_gemm_sm70_out_impl(
    torch::Tensor out,
    torch::Tensor sorted_input,    // [total_tokens, K] float16
    torch::Tensor expert_offsets,  // [num_experts + 1] int32
    torch::Tensor
        strided_ptrs_w,  // [num_experts * 16] uint8 (StridedPtr array)
    torch::Tensor
        strided_ptrs_s,  // [num_experts * 16] uint8 (StridedPtr array)
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu, torch::Tensor b_group_indices = torch::Tensor(),
    bool per_expert_dispatch = false,
    torch::Tensor reduce_out = torch::Tensor(),
    torch::Tensor sorted_weights = torch::Tensor(),
    bool weighted_reduce = false) {
  TORCH_CHECK(
      sorted_input.is_cuda() && sorted_input.scalar_type() == torch::kFloat16,
      "awq_moe_gemm_sm70: input must be CUDA float16.");
  TORCH_CHECK(
      expert_offsets.is_cuda() && expert_offsets.scalar_type() == torch::kInt32,
      "awq_moe_gemm_sm70: expert_offsets must be CUDA int32.");
  TORCH_CHECK(strided_ptrs_w.is_cuda() && strided_ptrs_s.is_cuda(),
              "awq_moe_gemm_sm70: strided_ptrs must be CUDA.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_gemm_sm70: output must be CUDA float16.");
  if (weighted_reduce) {
    TORCH_CHECK(
        reduce_out.is_cuda() && reduce_out.scalar_type() == torch::kFloat16,
        "awq_moe_gemm_sm70: reduce_out must be CUDA float16.");
    TORCH_CHECK(sorted_weights.is_cuda() &&
                    sorted_weights.scalar_type() == torch::kFloat32,
                "awq_moe_gemm_sm70: sorted_weights must be CUDA float32.");
  }
  TORCH_CHECK(num_experts > 0 && k > 0 && n > 0,
              "awq_moe_gemm_sm70: invalid dimensions.");
  TORCH_CHECK(k % group_size == 0,
              "awq_moe_gemm_sm70: input dim must be divisible by group size.");
  torch::Tensor b_group_indices_flat = b_group_indices;
  const bool use_b_group_indices =
      b_group_indices.defined() && b_group_indices.numel() > 0;
  if (use_b_group_indices) {
    TORCH_CHECK(b_group_indices.is_cuda() &&
                    b_group_indices.scalar_type() == torch::kInt32,
                "awq_moe_gemm_sm70: B group indices must be CUDA int32.");
    b_group_indices_flat = b_group_indices.contiguous().view({-1});
    TORCH_CHECK(b_group_indices_flat.numel() >= num_experts,
                "awq_moe_gemm_sm70: B group indices size mismatch.");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_input));
  const int device = sorted_input.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t total_tokens = sorted_input.size(0);
  TORCH_CHECK(out.size(0) == total_tokens,
              "awq_moe_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "awq_moe_gemm_sm70: output must be row-major contiguous.");
  if (weighted_reduce) {
    TORCH_CHECK(
        !gated_silu,
        "awq_moe_gemm_sm70: weighted reduce cannot combine gated_silu.");
    TORCH_CHECK(out.size(1) == n,
                "awq_moe_gemm_sm70: output cols must match n.");
    TORCH_CHECK(reduce_out.dim() == 2 && reduce_out.size(0) == 1 &&
                    reduce_out.size(1) == n && reduce_out.stride(1) == 1,
                "awq_moe_gemm_sm70: reduce_out shape mismatch.");
    TORCH_CHECK(sorted_weights.numel() >= total_tokens,
                "awq_moe_gemm_sm70: sorted_weights size mismatch.");
  } else if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "awq_moe_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "awq_moe_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "awq_moe_gemm_sm70: output cols must match n.");
  }

  if (total_tokens == 0) return;

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_moe_gemm_sm70: no compatible TurboMind converters.");

  // desc_A: input activations with offsets (kBlocked mode)
  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(k),
      static_cast<int>(k),
  };
  desc_A.num = static_cast<int>(num_experts);
  desc_A.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::MatrixLayout desc_U{};

  // desc_B: weights via StridedPtr (ld=0 triggers StridedPtr resolution)
  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::data_type_v<turbomind::uint4_t>;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = 0;  // StridedPtr mode
  desc_B.num = static_cast<int>(num_experts);
  desc_B.group_idxs =
      use_b_group_indices ? b_group_indices_flat.data_ptr<int>() : nullptr;

  // desc_V: scales via StridedPtr
  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  const int64_t num_groups_raw = k / group_size;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups_raw),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = 0;  // StridedPtr mode
  desc_V.num = static_cast<int>(num_experts);
  desc_V.group_idxs =
      use_b_group_indices ? b_group_indices_flat.data_ptr<int>() : nullptr;

  // desc_D: output with offsets (same as A)
  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };
  desc_D.num = static_cast<int>(num_experts);
  desc_D.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::Operation op{};
  op.dispatch = vllm::awq_sm70::select_awq_moe_dispatch_policy(
      device, static_cast<int>(total_tokens), static_cast<int>(n),
      static_cast<int>(k), static_cast<int>(num_experts),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  turbomind::gemm::MoeWeightedReduceParam moe_reduce_param{};
  if (weighted_reduce) {
    op.epilogue = static_cast<turbomind::gemm::Epilogue>(
        static_cast<int>(op.epilogue) |
        static_cast<int>(turbomind::gemm::Epilogue::kMoeWeightedReduce));
    moe_reduce_param.out = reduce_out.data_ptr();
    moe_reduce_param.sorted_weights = sorted_weights.data_ptr<float>();
    moe_reduce_param.offsets = expert_offsets.data_ptr<int>();
    op.reserved = &moe_reduce_param;
  } else {
    op.reserved = nullptr;
  }
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;
  op.dispatch_num_override = per_expert_dispatch ? 1 : 0;

  auto& workspace_holder = vllm::awq_sm70::get_workspace(device, stream);
  auto& gemm = vllm::awq_sm70::get_gemm(device);

  const int ec =
      gemm.Run(op, 1.f, sorted_input.data_ptr(), desc_A, nullptr, desc_U,
               strided_ptrs_w.data_ptr(), desc_B, strided_ptrs_s.data_ptr(),
               desc_V, 0.f, out.data_ptr(), desc_D, out.data_ptr(), desc_D,
               workspace_holder.workspace, stream);

  TORCH_CHECK(ec == 0,
              "awq_moe_gemm_sm70: TurboMind batched GEMM failed (ec=", ec,
              ").");
}

void awq_moe_gemm_sm70_out(torch::Tensor out, torch::Tensor sorted_input,
                           torch::Tensor expert_offsets,
                           torch::Tensor strided_ptrs_w,
                           torch::Tensor strided_ptrs_s, int64_t num_experts,
                           int64_t k, int64_t n, int64_t group_size,
                           bool gated_silu) {
  awq_moe_gemm_sm70_out_impl(out, sorted_input, expert_offsets, strided_ptrs_w,
                             strided_ptrs_s, num_experts, k, n, group_size,
                             gated_silu);
}

void awq_moe_gemm_sm70_per_expert_dispatch_out(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w, torch::Tensor strided_ptrs_s,
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu) {
  awq_moe_gemm_sm70_out_impl(out, sorted_input, expert_offsets, strided_ptrs_w,
                             strided_ptrs_s, num_experts, k, n, group_size,
                             gated_silu, torch::Tensor(), true);
}

torch::Tensor awq_moe_gemm_sm70(torch::Tensor sorted_input,
                                torch::Tensor expert_offsets,
                                torch::Tensor strided_ptrs_w,
                                torch::Tensor strided_ptrs_s,
                                int64_t num_experts, int64_t k, int64_t n,
                                int64_t group_size) {
  auto out = torch::empty({sorted_input.size(0), n},
                          torch::TensorOptions()
                              .dtype(sorted_input.dtype())
                              .device(sorted_input.device()));
  awq_moe_gemm_sm70_out(out, sorted_input, expert_offsets, strided_ptrs_w,
                        strided_ptrs_s, num_experts, k, n, group_size, false);
  return out;
}

void awq_moe_dense_stage_sm70_out(torch::Tensor out, torch::Tensor input,
                                  torch::Tensor expert_offsets,
                                  torch::Tensor dense_expert_ids,
                                  torch::Tensor ptrs_w, torch::Tensor ptrs_s,
                                  int64_t num_experts, int64_t k, int64_t n,
                                  int64_t group_size) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16,
              "awq_moe_dense_stage_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_dense_stage_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous(),
              "awq_moe_dense_stage_sm70_out: expert_offsets must be contiguous "
              "CUDA int32.");
  TORCH_CHECK(dense_expert_ids.is_cuda() &&
                  dense_expert_ids.scalar_type() == torch::kInt32 &&
                  dense_expert_ids.is_contiguous(),
              "awq_moe_dense_stage_sm70_out: dense_expert_ids must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "awq_moe_dense_stage_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(num_experts > 0,
              "awq_moe_dense_stage_sm70_out: num_experts must be positive.");
  TORCH_CHECK(
      group_size == 32 || group_size == 64 || group_size == 128,
      "awq_moe_dense_stage_sm70_out: group_size must be 32, 64, or 128.");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == k,
              "awq_moe_dense_stage_sm70_out: input shape mismatch.");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == input.size(0) && out.size(1) == n,
      "awq_moe_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(expert_offsets.numel() >= num_experts + 1,
              "awq_moe_dense_stage_sm70_out: expert_offsets too small.");
  TORCH_CHECK(dense_expert_ids.numel() >= num_experts,
              "awq_moe_dense_stage_sm70_out: dense_expert_ids too small.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  static std::atomic<unsigned> logged_awq_dense_stage{0u};
  maybe_log_sm70_moe_route_once(
      logged_awq_dense_stage,
      "SM70 AWQ MoE CUDA-graph-safe dense-stage path enabled C++ op reached",
      input, input.size(0), num_experts);
  for (int expert = 0; expert < static_cast<int>(num_experts); ++expert) {
    torch::Tensor offsets = expert_offsets.narrow(0, expert, 2);
    torch::Tensor expert_idx = dense_expert_ids.narrow(0, expert, 1);
    awq_moe_gemm_sm70_out_impl(out, input, offsets, ptrs_w, ptrs_s, 1, k, n,
                               group_size, false, expert_idx);
  }
}

namespace {

__global__ void awq_moe_build_active_expert_segments_kernel(
    const int* __restrict__ permuted_experts_id,
    int* __restrict__ active_expert_offsets,
    int* __restrict__ active_expert_ids, int total_slots) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (total_slots <= 0) {
    active_expert_offsets[0] = 0;
    return;
  }

  int active_count = 0;
  int previous_expert = -1;
  for (int row = 0; row < total_slots; ++row) {
    const int expert = permuted_experts_id[row];
    if (row == 0 || expert != previous_expert) {
      active_expert_offsets[active_count] = row;
      active_expert_ids[active_count] = expert;
      ++active_count;
    }
    previous_expert = expert;
  }
  active_expert_offsets[active_count] = total_slots;
  for (int segment = active_count + 1; segment <= total_slots; ++segment) {
    active_expert_offsets[segment] = total_slots;
  }
  for (int segment = active_count; segment < total_slots; ++segment) {
    active_expert_ids[segment] = 0;
  }
}

}  // namespace

void awq_moe_active_dense_stage_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor permuted_experts_id,
    torch::Tensor active_expert_offsets, torch::Tensor active_expert_ids,
    torch::Tensor ptrs_w, torch::Tensor ptrs_s, int64_t total_slots, int64_t k,
    int64_t n, int64_t group_size) {
  TORCH_CHECK(
      input.is_cuda() && input.scalar_type() == torch::kFloat16,
      "awq_moe_active_dense_stage_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_active_dense_stage_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(permuted_experts_id.is_cuda() &&
                  permuted_experts_id.scalar_type() == torch::kInt32 &&
                  permuted_experts_id.is_contiguous() &&
                  active_expert_offsets.is_cuda() &&
                  active_expert_offsets.scalar_type() == torch::kInt32 &&
                  active_expert_offsets.is_contiguous() &&
                  active_expert_ids.is_cuda() &&
                  active_expert_ids.scalar_type() == torch::kInt32 &&
                  active_expert_ids.is_contiguous(),
              "awq_moe_active_dense_stage_sm70_out: index buffers must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "awq_moe_active_dense_stage_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(
      total_slots > 0,
      "awq_moe_active_dense_stage_sm70_out: total_slots must be positive.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_active_dense_stage_sm70_out: group_size must be 32, 64, "
              "or 128.");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= total_slots && input.size(1) == k,
      "awq_moe_active_dense_stage_sm70_out: input shape mismatch.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) >= total_slots && out.size(1) == n,
              "awq_moe_active_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(permuted_experts_id.numel() >= total_slots &&
                  active_expert_offsets.numel() >= total_slots + 1 &&
                  active_expert_ids.numel() >= total_slots,
              "awq_moe_active_dense_stage_sm70_out: index buffer too small.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  awq_moe_build_active_expert_segments_kernel<<<1, 1, 0, stream>>>(
      permuted_experts_id.data_ptr<int>(),
      active_expert_offsets.data_ptr<int>(), active_expert_ids.data_ptr<int>(),
      static_cast<int>(total_slots));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  static std::atomic<unsigned> logged_awq_active_dense_stage{0u};
  maybe_log_sm70_moe_route_once(
      logged_awq_active_dense_stage,
      "SM70 AWQ MoE active dense-stage path enabled C++ op reached", input,
      total_slots, total_slots);

  for (int segment = 0; segment < static_cast<int>(total_slots); ++segment) {
    torch::Tensor offsets = active_expert_offsets.narrow(0, segment, 2);
    torch::Tensor expert_idx = active_expert_ids.narrow(0, segment, 1);
    awq_moe_gemm_sm70_out_impl(out, input, offsets, ptrs_w, ptrs_s, 1, k, n,
                               group_size, false, expert_idx);
  }
}

void awq_moe_single_token_dense_stage_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids, torch::Tensor ptrs_w, torch::Tensor ptrs_s,
    int64_t top_k, int64_t k, int64_t n, int64_t group_size) {
  TORCH_CHECK(
      input.is_cuda() && input.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_dense_stage_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_dense_stage_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous() &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_contiguous(),
              "awq_moe_single_token_dense_stage_sm70_out: index buffers must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(
      ptrs_w.is_cuda() && ptrs_s.is_cuda(),
      "awq_moe_single_token_dense_stage_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "awq_moe_single_token_dense_stage_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= top_k && input.size(1) == k,
      "awq_moe_single_token_dense_stage_sm70_out: input shape mismatch.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) >= top_k && out.size(1) == n,
              "awq_moe_single_token_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 && sorted_expert_ids.numel() >= top_k,
      "awq_moe_single_token_dense_stage_sm70_out: index buffer too small.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_single_token_dense_stage_sm70_out: group_size must be "
              "32, 64, or 128.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  for (int route = 0; route < static_cast<int>(top_k); ++route) {
    torch::Tensor offsets = expert_offsets.narrow(0, route, 2);
    torch::Tensor expert_idx = sorted_expert_ids.narrow(0, route, 1);
    awq_moe_gemm_sm70_out_impl(out, input, offsets, ptrs_w, ptrs_s, 1, k, n,
                               group_size, false, expert_idx);
  }
}

void awq_moe_single_token_indexed_dense_stage_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids, torch::Tensor ptrs_w, torch::Tensor ptrs_s,
    int64_t top_k, int64_t k, int64_t n, int64_t group_size) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_indexed_dense_stage_sm70_out: input must "
              "be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_indexed_dense_stage_sm70_out: out must be "
              "CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous() &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_contiguous(),
              "awq_moe_single_token_indexed_dense_stage_sm70_out: index "
              "buffers must be contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "awq_moe_single_token_indexed_dense_stage_sm70_out: ptr rows "
              "must be CUDA.");
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "awq_moe_single_token_indexed_dense_stage_sm70_out: top_k must "
              "be in [1, 32].");
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= top_k && input.size(1) == k,
              "awq_moe_single_token_indexed_dense_stage_sm70_out: input shape "
              "mismatch.");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) >= top_k && out.size(1) == n,
      "awq_moe_single_token_indexed_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 && sorted_expert_ids.numel() >= top_k,
      "awq_moe_single_token_indexed_dense_stage_sm70_out: index buffer too "
      "small.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_single_token_indexed_dense_stage_sm70_out: group_size "
              "must be 32, 64, or 128.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  awq_moe_gemm_sm70_out_impl(out, input, expert_offsets, ptrs_w, ptrs_s, top_k,
                             k, n, group_size, false, sorted_expert_ids, true);
}

void awq_moe_single_token_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_dense_w13_sm70_out: x must be CUDA float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_dense_w13_sm70_out: scratch buffers must "
              "be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "awq_moe_single_token_dense_w13_sm70_out: topk_ids must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "awq_moe_single_token_dense_w13_sm70_out: index buffers must be "
              "CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "awq_moe_single_token_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(
      w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda(),
      "awq_moe_single_token_dense_w13_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "awq_moe_single_token_dense_w13_sm70_out: x must have shape [1, "
              "hidden].");
  TORCH_CHECK(x.size(1) == hidden_logical_size,
              "awq_moe_single_token_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "awq_moe_single_token_dense_w13_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 &&
          expert_offsets64.numel() >= top_k + 1 &&
          inv_permuted_idx.numel() >= top_k &&
          sorted_expert_ids.numel() >= top_k,
      "awq_moe_single_token_dense_w13_sm70_out: index buffer too small.");
  TORCH_CHECK(
      compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
          compact_input.size(1) == hidden_logical_size,
      "awq_moe_single_token_dense_w13_sm70_out: compact_input shape mismatch.");
  TORCH_CHECK(
      gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
          gate_up.size(1) == w13_n,
      "awq_moe_single_token_dense_w13_sm70_out: gate_up shape mismatch.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_single_token_dense_w13_sm70_out: group_size must be 32, "
              "64, or 128.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static std::atomic<unsigned> logged_awq_single_token{0u};
  maybe_log_sm70_moe_route_once(logged_awq_single_token,
                                "SM70 AWQ MoE single-token active-expert dense "
                                "path enabled C++ op reached",
                                x, x.size(0), top_k);
  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, false, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  awq_moe_single_token_dense_stage_sm70_out(
      gate_up, compact_input, expert_offsets, sorted_expert_ids, w13_ptrs_w,
      w13_ptrs_s, top_k, w13_k, w13_n, group_size);
}

void awq_moe_single_token_indexed_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: x must be CUDA "
              "float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: scratch "
              "buffers must be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "awq_moe_single_token_indexed_dense_w13_sm70_out: topk_ids must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: index buffers "
              "must be CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "awq_moe_single_token_indexed_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda(),
              "awq_moe_single_token_indexed_dense_w13_sm70_out: ptr rows must "
              "be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: x must have "
              "shape [1, hidden].");
  TORCH_CHECK(
      x.size(1) == hidden_logical_size,
      "awq_moe_single_token_indexed_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: top_k must be "
              "in [1, 32].");
  TORCH_CHECK(expert_offsets.numel() >= top_k + 1 &&
                  expert_offsets64.numel() >= top_k + 1 &&
                  inv_permuted_idx.numel() >= top_k &&
                  sorted_expert_ids.numel() >= top_k,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: index buffer "
              "too small.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
                  compact_input.size(1) == hidden_logical_size,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: compact_input "
              "shape mismatch.");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
                  gate_up.size(1) == w13_n,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: gate_up shape "
              "mismatch.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: group_size "
              "must be 32, 64, or 128.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static std::atomic<unsigned> logged_awq_single_token_indexed{0u};
  maybe_log_sm70_moe_route_once(
      logged_awq_single_token_indexed,
      "SM70 AWQ MoE single-token indexed dense path enabled C++ op reached", x,
      x.size(0), top_k);
  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, false, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  awq_moe_single_token_indexed_dense_stage_sm70_out(
      gate_up, compact_input, expert_offsets, sorted_expert_ids, w13_ptrs_w,
      w13_ptrs_s, top_k, w13_k, w13_n, group_size);
}

void awq_moe_single_token_compact_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor compact_w13_ptrs_w, torch::Tensor compact_w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_compact_dense_w13_sm70_out: x must be CUDA "
              "float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_compact_dense_w13_sm70_out: scratch "
              "buffers must be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "awq_moe_single_token_compact_dense_w13_sm70_out: topk_ids must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "awq_moe_single_token_compact_dense_w13_sm70_out: index buffers "
              "must be CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "awq_moe_single_token_compact_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda() &&
                  compact_w13_ptrs_w.is_cuda() && compact_w13_ptrs_s.is_cuda(),
              "awq_moe_single_token_compact_dense_w13_sm70_out: ptr rows must "
              "be CUDA.");
  TORCH_CHECK(compact_w13_ptrs_w.scalar_type() == torch::kUInt8 &&
                  compact_w13_ptrs_s.scalar_type() == torch::kUInt8,
              "awq_moe_single_token_compact_dense_w13_sm70_out: compact ptr "
              "rows must be uint8.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "awq_moe_single_token_compact_dense_w13_sm70_out: x must have "
              "shape [1, hidden].");
  TORCH_CHECK(
      x.size(1) == hidden_logical_size,
      "awq_moe_single_token_compact_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "awq_moe_single_token_compact_dense_w13_sm70_out: top_k must be "
              "in [1, 32].");
  TORCH_CHECK(expert_offsets.numel() >= top_k + 1 &&
                  expert_offsets64.numel() >= top_k + 1 &&
                  inv_permuted_idx.numel() >= top_k &&
                  sorted_expert_ids.numel() >= top_k,
              "awq_moe_single_token_compact_dense_w13_sm70_out: index buffer "
              "too small.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
                  compact_input.size(1) == hidden_logical_size,
              "awq_moe_single_token_compact_dense_w13_sm70_out: compact_input "
              "shape mismatch.");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
                  gate_up.size(1) == w13_n,
              "awq_moe_single_token_compact_dense_w13_sm70_out: gate_up shape "
              "mismatch.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_single_token_compact_dense_w13_sm70_out: group_size "
              "must be 32, 64, or 128.");

  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;
  TORCH_CHECK(compact_w13_ptrs_w.numel() >= top_k * kPtrRowBytes &&
                  compact_w13_ptrs_s.numel() >= top_k * kPtrRowBytes,
              "awq_moe_single_token_compact_dense_w13_sm70_out: compact ptr "
              "buffer too small.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static std::atomic<unsigned> logged_awq_single_token_compact{0u};
  maybe_log_sm70_moe_route_once(logged_awq_single_token_compact,
                                "SM70 AWQ MoE single-token compact grouped W13 "
                                "path enabled C++ op reached",
                                x, x.size(0), top_k);

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      compact_w13_ptrs_w.data_ptr<uint8_t>(),
      compact_w13_ptrs_s.data_ptr<uint8_t>(),
      compact_w13_ptrs_w.data_ptr<uint8_t>(),
      compact_w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, true, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  awq_moe_gemm_sm70_per_expert_dispatch_out(
      gate_up, compact_input, expert_offsets, compact_w13_ptrs_w,
      compact_w13_ptrs_s, top_k, w13_k, w13_n, group_size, false);
}

void fp8_moe_gemm_sm70_out_impl(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w, torch::Tensor strided_ptrs_s,
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu, torch::Tensor reduce_out, torch::Tensor sorted_weights,
    bool weighted_reduce, int64_t logical_total_tokens = -1,
    int64_t a_ld_override = -1, torch::Tensor a_indices = torch::Tensor(),
    torch::Tensor b_group_indices = torch::Tensor(),
    bool per_expert_dispatch = false, int64_t dispatch_num_experts = -1,
    torch::Tensor active_group_indices = torch::Tensor(),
    int64_t active_group_count = -1) {
  TORCH_CHECK(
      sorted_input.is_cuda() && sorted_input.scalar_type() == torch::kFloat16,
      "fp8_moe_gemm_sm70: input must be CUDA float16.");
  TORCH_CHECK(
      expert_offsets.is_cuda() && expert_offsets.scalar_type() == torch::kInt32,
      "fp8_moe_gemm_sm70: expert_offsets must be CUDA int32.");
  TORCH_CHECK(strided_ptrs_w.is_cuda() && strided_ptrs_s.is_cuda(),
              "fp8_moe_gemm_sm70: strided_ptrs must be CUDA.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "fp8_moe_gemm_sm70: output must be CUDA float16.");
  if (weighted_reduce) {
    TORCH_CHECK(
        reduce_out.is_cuda() && reduce_out.scalar_type() == torch::kFloat16,
        "fp8_moe_gemm_sm70: reduce_out must be CUDA float16.");
    TORCH_CHECK(sorted_weights.is_cuda() &&
                    sorted_weights.scalar_type() == torch::kFloat32,
                "fp8_moe_gemm_sm70: sorted_weights must be CUDA float32.");
  }
  TORCH_CHECK(num_experts > 0 && k > 0 && n > 0,
              "fp8_moe_gemm_sm70: invalid dimensions.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_gemm_sm70: only group_size=128 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_input));
  const int device = sorted_input.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(sorted_input.dim() == 2 && sorted_input.size(1) == k,
              "fp8_moe_gemm_sm70: input shape mismatch.");
  TORCH_CHECK(a_ld_override >= -1,
              "fp8_moe_gemm_sm70: invalid A leading dimension override.");
  const int64_t input_tokens = sorted_input.size(0);
  const int64_t total_tokens =
      logical_total_tokens > 0 ? logical_total_tokens : input_tokens;
  const bool broadcast_a = input_tokens != total_tokens;
  TORCH_CHECK(a_ld_override != 0,
              "fp8_moe_gemm_sm70: A ld=0 broadcast is unsupported.");
  TORCH_CHECK(total_tokens >= 0,
              "fp8_moe_gemm_sm70: invalid logical token count.");
  if (broadcast_a) {
    TORCH_CHECK(input_tokens == 1,
                "fp8_moe_gemm_sm70: broadcast input must have one row.");
  } else {
    TORCH_CHECK(input_tokens == total_tokens,
                "fp8_moe_gemm_sm70: input rows must match logical rows.");
  }
  torch::Tensor a_indices_flat = a_indices;
  const bool use_a_indices = a_indices.defined() && a_indices.numel() > 0;
  if (use_a_indices) {
    TORCH_CHECK(a_indices.is_cuda() && a_indices.scalar_type() == torch::kInt32,
                "fp8_moe_gemm_sm70: A row indices must be CUDA int32.");
    a_indices_flat = a_indices.contiguous().view({-1});
    TORCH_CHECK(a_indices_flat.numel() >= total_tokens,
                "fp8_moe_gemm_sm70: A row indices size mismatch.");
  }
  torch::Tensor b_group_indices_flat = b_group_indices;
  const bool use_b_group_indices =
      b_group_indices.defined() && b_group_indices.numel() > 0;
  if (use_b_group_indices) {
    TORCH_CHECK(b_group_indices.is_cuda() &&
                    b_group_indices.scalar_type() == torch::kInt32,
                "fp8_moe_gemm_sm70: B group indices must be CUDA int32.");
    b_group_indices_flat = b_group_indices.contiguous().view({-1});
    TORCH_CHECK(b_group_indices_flat.numel() >= num_experts,
                "fp8_moe_gemm_sm70: B group indices size mismatch.");
  }
  torch::Tensor active_group_indices_flat = active_group_indices;
  const bool use_active_group_indices = active_group_indices.defined() &&
                                        active_group_indices.numel() > 0 &&
                                        active_group_count > 0;
  if (use_active_group_indices) {
    TORCH_CHECK(active_group_indices.is_cuda() &&
                    active_group_indices.scalar_type() == torch::kInt32,
                "fp8_moe_gemm_sm70: active group indices must be CUDA int32.");
    active_group_indices_flat = active_group_indices.contiguous().view({-1});
    TORCH_CHECK(active_group_indices_flat.numel() >= active_group_count,
                "fp8_moe_gemm_sm70: active group indices size mismatch.");
    TORCH_CHECK(
        active_group_count <= num_experts,
        "fp8_moe_gemm_sm70: active group count exceeds source experts.");
  }
  const int64_t selected_dispatch_num_experts =
      dispatch_num_experts > 0 ? dispatch_num_experts : num_experts;
  TORCH_CHECK(out.size(0) == total_tokens,
              "fp8_moe_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "fp8_moe_gemm_sm70: output must be row-major contiguous.");
  if (weighted_reduce) {
    TORCH_CHECK(
        !gated_silu,
        "fp8_moe_gemm_sm70: weighted reduce cannot combine gated_silu.");
    TORCH_CHECK(out.size(1) == n,
                "fp8_moe_gemm_sm70: output cols must match n.");
    TORCH_CHECK(reduce_out.dim() == 2 && reduce_out.size(0) == 1 &&
                    reduce_out.size(1) == n && reduce_out.stride(1) == 1,
                "fp8_moe_gemm_sm70: reduce_out shape mismatch.");
    TORCH_CHECK(sorted_weights.numel() == total_tokens,
                "fp8_moe_gemm_sm70: sorted_weights size mismatch.");
  } else if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "fp8_moe_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "fp8_moe_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "fp8_moe_gemm_sm70: output cols must match n.");
  }
  if (total_tokens == 0) return;

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "fp8_moe_gemm_sm70: no compatible TurboMind converters.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(k),
      static_cast<int>(a_ld_override >= 0 ? a_ld_override : k),
  };
  desc_A.num = static_cast<int>(num_experts);
  desc_A.offsets = (broadcast_a && !use_a_indices)
                       ? nullptr
                       : expert_offsets.data_ptr<int>();
  desc_A.idxs = use_a_indices ? a_indices_flat.data_ptr<int>() : nullptr;
  turbomind::gemm::MatrixLayout desc_U{};

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;
  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::kFloat8_e4m3;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = 0;
  desc_B.num = static_cast<int>(num_experts);
  desc_B.group_idxs =
      use_b_group_indices ? b_group_indices_flat.data_ptr<int>() : nullptr;

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  const int64_t num_groups = (k + group_size - 1) / group_size;
  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = 0;
  desc_V.num = static_cast<int>(num_experts);
  desc_V.group_idxs =
      use_b_group_indices ? b_group_indices_flat.data_ptr<int>() : nullptr;

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };
  desc_D.num = static_cast<int>(num_experts);
  desc_D.offsets = expert_offsets.data_ptr<int>();
  desc_D.group_idxs = use_active_group_indices
                          ? active_group_indices_flat.data_ptr<int>()
                          : nullptr;

  turbomind::gemm::Operation op{};
  op.dispatch = vllm::awq_sm70::select_fp8_moe_dispatch_policy(
      device, static_cast<int>(total_tokens), static_cast<int>(n),
      static_cast<int>(k), static_cast<int>(selected_dispatch_num_experts),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  turbomind::gemm::MoeWeightedReduceParam moe_reduce_param{};
  if (weighted_reduce) {
    op.epilogue = static_cast<turbomind::gemm::Epilogue>(
        static_cast<int>(op.epilogue) |
        static_cast<int>(turbomind::gemm::Epilogue::kMoeWeightedReduce));
    moe_reduce_param.out = reduce_out.data_ptr();
    moe_reduce_param.sorted_weights = sorted_weights.data_ptr<float>();
    moe_reduce_param.offsets = expert_offsets.data_ptr<int>();
    op.reserved = &moe_reduce_param;
  } else {
    op.reserved = nullptr;
  }
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;
  op.dispatch_num_override = per_expert_dispatch ? 1 : 0;
  op.active_group_count =
      use_active_group_indices ? static_cast<int>(active_group_count) : 0;

  auto& workspace_holder = vllm::awq_sm70::get_workspace(device, stream);
  auto& gemm = vllm::awq_sm70::get_gemm(device);
  const int ec =
      gemm.Run(op, 1.f, sorted_input.data_ptr(), desc_A, nullptr, desc_U,
               strided_ptrs_w.data_ptr(), desc_B, strided_ptrs_s.data_ptr(),
               desc_V, 0.f, out.data_ptr(), desc_D, out.data_ptr(), desc_D,
               workspace_holder.workspace, stream);
  TORCH_CHECK(ec == 0,
              "fp8_moe_gemm_sm70: TurboMind batched GEMM failed (ec=", ec,
              ").");
}

void fp8_moe_gemm_sm70_out(torch::Tensor out, torch::Tensor sorted_input,
                           torch::Tensor expert_offsets,
                           torch::Tensor strided_ptrs_w,
                           torch::Tensor strided_ptrs_s, int64_t num_experts,
                           int64_t k, int64_t n, int64_t group_size,
                           bool gated_silu) {
  fp8_moe_gemm_sm70_out_impl(out, sorted_input, expert_offsets, strided_ptrs_w,
                             strided_ptrs_s, num_experts, k, n, group_size,
                             gated_silu, torch::Tensor(), torch::Tensor(),
                             false, -1, -1);
}

void fp8_moe_gemm_sm70_per_expert_dispatch_out(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w, torch::Tensor strided_ptrs_s,
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu) {
  fp8_moe_gemm_sm70_out_impl(
      out, sorted_input, expert_offsets, strided_ptrs_w, strided_ptrs_s,
      num_experts, k, n, group_size, gated_silu, torch::Tensor(),
      torch::Tensor(), false, -1, -1, torch::Tensor(), torch::Tensor(), true);
}

void fp8_moe_dense_stage_sm70_out(torch::Tensor out, torch::Tensor input,
                                  torch::Tensor expert_offsets,
                                  torch::Tensor dense_expert_ids,
                                  torch::Tensor ptrs_w, torch::Tensor ptrs_s,
                                  int64_t num_experts, int64_t k, int64_t n,
                                  int64_t group_size) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16,
              "fp8_moe_dense_stage_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "fp8_moe_dense_stage_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous(),
              "fp8_moe_dense_stage_sm70_out: expert_offsets must be contiguous "
              "CUDA int32.");
  TORCH_CHECK(dense_expert_ids.is_cuda() &&
                  dense_expert_ids.scalar_type() == torch::kInt32 &&
                  dense_expert_ids.is_contiguous(),
              "fp8_moe_dense_stage_sm70_out: dense_expert_ids must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "fp8_moe_dense_stage_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(num_experts > 0,
              "fp8_moe_dense_stage_sm70_out: num_experts must be positive.");
  TORCH_CHECK(
      group_size == 128,
      "fp8_moe_dense_stage_sm70_out: only group_size=128 is supported.");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == k,
              "fp8_moe_dense_stage_sm70_out: input shape mismatch.");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == input.size(0) && out.size(1) == n,
      "fp8_moe_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(expert_offsets.numel() >= num_experts + 1,
              "fp8_moe_dense_stage_sm70_out: expert_offsets too small.");
  TORCH_CHECK(dense_expert_ids.numel() >= num_experts,
              "fp8_moe_dense_stage_sm70_out: dense_expert_ids too small.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  static std::atomic<unsigned> logged_fp8_dense_stage{0u};
  maybe_log_sm70_moe_route_once(
      logged_fp8_dense_stage,
      "SM70 FP8 MoE CUDA-graph-safe dense-stage path enabled C++ op reached",
      input, input.size(0), num_experts);
  for (int expert = 0; expert < static_cast<int>(num_experts); ++expert) {
    torch::Tensor offsets = expert_offsets.narrow(0, expert, 2);
    torch::Tensor expert_idx = dense_expert_ids.narrow(0, expert, 1);
    fp8_moe_gemm_sm70_out_impl(out, input, offsets, ptrs_w, ptrs_s, 1, k, n,
                               group_size, false, torch::Tensor(),
                               torch::Tensor(), false, -1, -1, torch::Tensor(),
                               expert_idx);
  }
}

void fp8_moe_single_token_dense_stage_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids, torch::Tensor ptrs_w, torch::Tensor ptrs_s,
    int64_t top_k, int64_t k, int64_t n, int64_t group_size) {
  TORCH_CHECK(
      input.is_cuda() && input.scalar_type() == torch::kFloat16,
      "fp8_moe_single_token_dense_stage_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kFloat16,
      "fp8_moe_single_token_dense_stage_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous() &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_contiguous(),
              "fp8_moe_single_token_dense_stage_sm70_out: index buffers must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(
      ptrs_w.is_cuda() && ptrs_s.is_cuda(),
      "fp8_moe_single_token_dense_stage_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "fp8_moe_single_token_dense_stage_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= top_k && input.size(1) == k,
      "fp8_moe_single_token_dense_stage_sm70_out: input shape mismatch.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) >= top_k && out.size(1) == n,
              "fp8_moe_single_token_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 && sorted_expert_ids.numel() >= top_k,
      "fp8_moe_single_token_dense_stage_sm70_out: index buffer too small.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_single_token_dense_stage_sm70_out: only group_size=128 "
              "is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  for (int route = 0; route < static_cast<int>(top_k); ++route) {
    torch::Tensor offsets = expert_offsets.narrow(0, route, 2);
    torch::Tensor expert_idx = sorted_expert_ids.narrow(0, route, 1);
    fp8_moe_gemm_sm70_out_impl(out, input, offsets, ptrs_w, ptrs_s, 1, k, n,
                               group_size, false, torch::Tensor(),
                               torch::Tensor(), false, -1, -1, torch::Tensor(),
                               expert_idx);
  }
}

void fp8_moe_single_token_indexed_dense_stage_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids, torch::Tensor ptrs_w, torch::Tensor ptrs_s,
    int64_t top_k, int64_t k, int64_t n, int64_t group_size) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: input must "
              "be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: out must be "
              "CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous() &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_contiguous(),
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: index "
              "buffers must be contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: ptr rows "
              "must be CUDA.");
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: top_k must "
              "be in [1, 32].");
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= top_k && input.size(1) == k,
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: input shape "
              "mismatch.");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) >= top_k && out.size(1) == n,
      "fp8_moe_single_token_indexed_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 && sorted_expert_ids.numel() >= top_k,
      "fp8_moe_single_token_indexed_dense_stage_sm70_out: index buffer too "
      "small.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: only "
              "group_size=128 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  fp8_moe_gemm_sm70_out_impl(out, input, expert_offsets, ptrs_w, ptrs_s, top_k,
                             k, n, group_size, false, torch::Tensor(),
                             torch::Tensor(), false, -1, -1, torch::Tensor(),
                             sorted_expert_ids);
}

void fp8_moe_single_token_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == torch::kFloat16,
      "fp8_moe_single_token_dense_w13_sm70_out: x must be CUDA float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_dense_w13_sm70_out: scratch buffers must "
              "be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "fp8_moe_single_token_dense_w13_sm70_out: topk_ids must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "fp8_moe_single_token_dense_w13_sm70_out: index buffers must be "
              "CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "fp8_moe_single_token_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(
      w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda(),
      "fp8_moe_single_token_dense_w13_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "fp8_moe_single_token_dense_w13_sm70_out: x must have shape [1, "
              "hidden].");
  TORCH_CHECK(x.size(1) == hidden_logical_size,
              "fp8_moe_single_token_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "fp8_moe_single_token_dense_w13_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 &&
          expert_offsets64.numel() >= top_k + 1 &&
          inv_permuted_idx.numel() >= top_k &&
          sorted_expert_ids.numel() >= top_k,
      "fp8_moe_single_token_dense_w13_sm70_out: index buffer too small.");
  TORCH_CHECK(
      compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
          compact_input.size(1) == hidden_logical_size,
      "fp8_moe_single_token_dense_w13_sm70_out: compact_input shape mismatch.");
  TORCH_CHECK(
      gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
          gate_up.size(1) == w13_n,
      "fp8_moe_single_token_dense_w13_sm70_out: gate_up shape mismatch.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_single_token_dense_w13_sm70_out: only group_size=128 is "
              "supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static std::atomic<unsigned> logged_fp8_single_token{0u};
  maybe_log_sm70_moe_route_once(logged_fp8_single_token,
                                "SM70 FP8 MoE single-token active-expert dense "
                                "path enabled C++ op reached",
                                x, x.size(0), top_k);
  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;
  const bool vectorized_prepare = sm70_fp8_moe_prepare_vec_enabled();

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, false, vectorized_prepare);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fp8_moe_single_token_dense_stage_sm70_out(
      gate_up, compact_input, expert_offsets, sorted_expert_ids, w13_ptrs_w,
      w13_ptrs_s, top_k, w13_k, w13_n, group_size);
}

void fp8_moe_single_token_indexed_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: x must be CUDA "
              "float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: scratch "
              "buffers must be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: topk_ids must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: index buffers "
              "must be CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "fp8_moe_single_token_indexed_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda(),
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: ptr rows must "
              "be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: x must have "
              "shape [1, hidden].");
  TORCH_CHECK(
      x.size(1) == hidden_logical_size,
      "fp8_moe_single_token_indexed_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: top_k must be "
              "in [1, 32].");
  TORCH_CHECK(expert_offsets.numel() >= top_k + 1 &&
                  expert_offsets64.numel() >= top_k + 1 &&
                  inv_permuted_idx.numel() >= top_k &&
                  sorted_expert_ids.numel() >= top_k,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: index buffer "
              "too small.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
                  compact_input.size(1) == hidden_logical_size,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: compact_input "
              "shape mismatch.");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
                  gate_up.size(1) == w13_n,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: gate_up shape "
              "mismatch.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: only "
              "group_size=128 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static std::atomic<unsigned> logged_fp8_single_token_indexed{0u};
  maybe_log_sm70_moe_route_once(
      logged_fp8_single_token_indexed,
      "SM70 FP8 MoE single-token indexed dense path enabled C++ op reached", x,
      x.size(0), top_k);
  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;
  const bool vectorized_prepare = sm70_fp8_moe_prepare_vec_enabled();

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, false, vectorized_prepare);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fp8_moe_single_token_indexed_dense_stage_sm70_out(
      gate_up, compact_input, expert_offsets, sorted_expert_ids, w13_ptrs_w,
      w13_ptrs_s, top_k, w13_k, w13_n, group_size);
}

void fp8_moe_single_token_compact_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor compact_w13_ptrs_w, torch::Tensor compact_w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: x must be CUDA "
              "float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: scratch "
              "buffers must be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "fp8_moe_single_token_compact_dense_w13_sm70_out: topk_ids must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: index buffers "
              "must be CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "fp8_moe_single_token_compact_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda() &&
                  compact_w13_ptrs_w.is_cuda() && compact_w13_ptrs_s.is_cuda(),
              "fp8_moe_single_token_compact_dense_w13_sm70_out: ptr rows must "
              "be CUDA.");
  TORCH_CHECK(compact_w13_ptrs_w.scalar_type() == torch::kUInt8 &&
                  compact_w13_ptrs_s.scalar_type() == torch::kUInt8,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: compact ptr "
              "rows must be uint8.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: x must have "
              "shape [1, hidden].");
  TORCH_CHECK(
      x.size(1) == hidden_logical_size,
      "fp8_moe_single_token_compact_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: top_k must be "
              "in [1, 32].");
  TORCH_CHECK(expert_offsets.numel() >= top_k + 1 &&
                  expert_offsets64.numel() >= top_k + 1 &&
                  inv_permuted_idx.numel() >= top_k &&
                  sorted_expert_ids.numel() >= top_k,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: index buffer "
              "too small.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
                  compact_input.size(1) == hidden_logical_size,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: compact_input "
              "shape mismatch.");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
                  gate_up.size(1) == w13_n,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: gate_up shape "
              "mismatch.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: only "
              "group_size=128 is supported.");

  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;
  TORCH_CHECK(compact_w13_ptrs_w.numel() >= top_k * kPtrRowBytes &&
                  compact_w13_ptrs_s.numel() >= top_k * kPtrRowBytes,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: compact ptr "
              "buffer too small.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static std::atomic<unsigned> logged_fp8_single_token_compact{0u};
  maybe_log_sm70_moe_route_once(logged_fp8_single_token_compact,
                                "SM70 FP8 MoE single-token compact grouped W13 "
                                "path enabled C++ op reached",
                                x, x.size(0), top_k);
  const bool vectorized_prepare = sm70_fp8_moe_prepare_vec_enabled();

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      compact_w13_ptrs_w.data_ptr<uint8_t>(),
      compact_w13_ptrs_s.data_ptr<uint8_t>(),
      compact_w13_ptrs_w.data_ptr<uint8_t>(),
      compact_w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, true, vectorized_prepare);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fp8_moe_gemm_sm70_out_impl(gate_up, compact_input, expert_offsets,
                             compact_w13_ptrs_w, compact_w13_ptrs_s, top_k,
                             w13_k, w13_n, group_size, false, torch::Tensor(),
                             torch::Tensor(), false, -1, -1);
}

void fp8_moe_single_token_sm70_out(
    torch::Tensor out, torch::Tensor x, torch::Tensor topk_weights,
    torch::Tensor topk_ids, torch::Tensor src_w13_ptrs_w_rows,
    torch::Tensor src_w13_ptrs_s_rows, torch::Tensor src_w2_ptrs_w_rows,
    torch::Tensor src_w2_ptrs_s_rows, torch::Tensor compact_input,
    torch::Tensor gate_up, torch::Tensor intermediate,
    torch::Tensor sorted_output, torch::Tensor sorted_weights,
    torch::Tensor dst_w13_ptrs_w_rows, torch::Tensor dst_w13_ptrs_s_rows,
    torch::Tensor dst_w2_ptrs_w_rows, torch::Tensor dst_w2_ptrs_s_rows,
    torch::Tensor expert_offsets, torch::Tensor inv_permuted_idx,
    torch::Tensor sorted_expert_ids, torch::Tensor broadcast_input_indices,
    torch::Tensor w2_raw_weight, torch::Tensor w2_raw_scale_inv, int64_t w13_k,
    int64_t w13_n, int64_t w2_k, int64_t w2_n, int64_t group_size,
    int64_t hidden_logical_size, bool fused_gated_silu,
    bool fused_weighted_reduce, bool broadcast_input, bool w2_direct_reduce,
    bool indexed_expert_ptrs, bool exact_per_route) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_sm70_out: x must be CUDA float16.");
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.scalar_type() == torch::kFloat32,
      "fp8_moe_single_token_sm70_out: topk_weights must be CUDA float32.");
  TORCH_CHECK(
      topk_ids.is_cuda() && (topk_ids.scalar_type() == torch::kInt32 ||
                             topk_ids.scalar_type() == torch::kInt64),
      "fp8_moe_single_token_sm70_out: topk_ids must be CUDA int32/int64.");
  TORCH_CHECK(
      compact_input.is_cuda() &&
          compact_input.scalar_type() == torch::kFloat16 && gate_up.is_cuda() &&
          gate_up.scalar_type() == torch::kFloat16 && intermediate.is_cuda() &&
          intermediate.scalar_type() == torch::kFloat16 &&
          sorted_output.is_cuda() &&
          sorted_output.scalar_type() == torch::kFloat16,
      "fp8_moe_single_token_sm70_out: scratch buffers must be CUDA float16.");
  TORCH_CHECK(
      sorted_weights.is_cuda() &&
          sorted_weights.scalar_type() == torch::kFloat32,
      "fp8_moe_single_token_sm70_out: sorted_weights must be CUDA float32.");
  TORCH_CHECK(
      expert_offsets.is_cuda() &&
          expert_offsets.scalar_type() == torch::kInt32 &&
          inv_permuted_idx.is_cuda() &&
          inv_permuted_idx.scalar_type() == torch::kInt32 &&
          sorted_expert_ids.is_cuda() &&
          sorted_expert_ids.scalar_type() == torch::kInt32 &&
          broadcast_input_indices.is_cuda() &&
          broadcast_input_indices.scalar_type() == torch::kInt32,
      "fp8_moe_single_token_sm70_out: index buffers must be CUDA int32.");
  TORCH_CHECK(
      src_w13_ptrs_w_rows.is_cuda() && src_w13_ptrs_s_rows.is_cuda() &&
          src_w2_ptrs_w_rows.is_cuda() && src_w2_ptrs_s_rows.is_cuda() &&
          dst_w13_ptrs_w_rows.is_cuda() && dst_w13_ptrs_s_rows.is_cuda() &&
          dst_w2_ptrs_w_rows.is_cuda() && dst_w2_ptrs_s_rows.is_cuda(),
      "fp8_moe_single_token_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "fp8_moe_single_token_sm70_out: x must have shape [1, hidden].");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == 1,
      "fp8_moe_single_token_sm70_out: out must have shape [1, hidden].");
  TORCH_CHECK(out.size(1) == hidden_logical_size,
              "fp8_moe_single_token_sm70_out: out cols must match "
              "hidden_logical_size.");

  topk_ids = topk_ids.contiguous().view({-1});
  topk_weights = topk_weights.contiguous().view({-1});
  inv_permuted_idx = inv_permuted_idx.contiguous().view({-1});
  broadcast_input_indices = broadcast_input_indices.contiguous().view({-1});

  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "fp8_moe_single_token_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(static_cast<int>(topk_weights.numel()) == top_k,
              "fp8_moe_single_token_sm70_out: topk_weights size mismatch.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) == top_k &&
                  compact_input.size(1) == x.size(1),
              "fp8_moe_single_token_sm70_out: compact_input shape mismatch.");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) == top_k &&
                  gate_up.size(1) == w13_n,
              "fp8_moe_single_token_sm70_out: gate_up shape mismatch.");
  TORCH_CHECK(intermediate.dim() == 2 && intermediate.size(0) == top_k &&
                  intermediate.size(1) == w13_n / 2,
              "fp8_moe_single_token_sm70_out: intermediate shape mismatch.");
  TORCH_CHECK(sorted_output.dim() == 2 && sorted_output.size(0) == top_k &&
                  sorted_output.size(1) == w2_n,
              "fp8_moe_single_token_sm70_out: sorted_output shape mismatch.");
  TORCH_CHECK(sorted_weights.numel() == top_k,
              "fp8_moe_single_token_sm70_out: sorted_weights size mismatch.");
  const int64_t source_num_experts = src_w13_ptrs_w_rows.size(0);
  TORCH_CHECK(
      source_num_experts > 0,
      "fp8_moe_single_token_sm70_out: source expert count must be positive.");
  const int64_t expected_offsets =
      exact_per_route ? source_num_experts + 1 : top_k + 1;
  TORCH_CHECK(expert_offsets.numel() == expected_offsets,
              "fp8_moe_single_token_sm70_out: expert_offsets size mismatch.");
  TORCH_CHECK(inv_permuted_idx.numel() == top_k,
              "fp8_moe_single_token_sm70_out: inv_permuted_idx size mismatch.");
  TORCH_CHECK(
      sorted_expert_ids.numel() >= top_k && sorted_expert_ids.is_contiguous(),
      "fp8_moe_single_token_sm70_out: sorted expert ids must be contiguous and "
      "large enough.");
  if (broadcast_input && broadcast_input_indices.numel() > 0) {
    TORCH_CHECK(broadcast_input_indices.numel() >= top_k,
                "fp8_moe_single_token_sm70_out: broadcast input indices size "
                "mismatch.");
  }
  if (w2_direct_reduce) {
    TORCH_CHECK(!fused_weighted_reduce,
                "fp8_moe_single_token_sm70_out: W2 direct reduce conflicts "
                "with weighted-reduce epilogue.");
    TORCH_CHECK(w2_raw_weight.defined() && w2_raw_scale_inv.defined(),
                "fp8_moe_single_token_sm70_out: W2 direct reduce requires raw "
                "W2 tensors.");
  }
  if (indexed_expert_ptrs) {
    TORCH_CHECK(src_w13_ptrs_w_rows.is_contiguous() &&
                    src_w13_ptrs_s_rows.is_contiguous() &&
                    src_w2_ptrs_w_rows.is_contiguous() &&
                    src_w2_ptrs_s_rows.is_contiguous(),
                "fp8_moe_single_token_sm70_out: indexed expert ptrs require "
                "contiguous source ptr rows.");
  }
  if (exact_per_route) {
    TORCH_CHECK(!fused_gated_silu && !fused_weighted_reduce &&
                    !broadcast_input && !w2_direct_reduce,
                "fp8_moe_single_token_sm70_out: exact per-route mode only "
                "supports the unfused compact path.");
    TORCH_CHECK(src_w13_ptrs_w_rows.is_contiguous() &&
                    src_w13_ptrs_s_rows.is_contiguous() &&
                    src_w2_ptrs_w_rows.is_contiguous() &&
                    src_w2_ptrs_s_rows.is_contiguous(),
                "fp8_moe_single_token_sm70_out: exact per-route mode requires "
                "contiguous source ptr rows.");
  }
  const int64_t row_bytes = src_w13_ptrs_w_rows.size(1);
  TORCH_CHECK(
      src_w13_ptrs_w_rows.dim() == 2 && src_w13_ptrs_s_rows.dim() == 2 &&
          src_w2_ptrs_w_rows.dim() == 2 && src_w2_ptrs_s_rows.dim() == 2 &&
          dst_w13_ptrs_w_rows.dim() == 2 && dst_w13_ptrs_s_rows.dim() == 2 &&
          dst_w2_ptrs_w_rows.dim() == 2 && dst_w2_ptrs_s_rows.dim() == 2,
      "fp8_moe_single_token_sm70_out: ptr row tensors must be 2D.");
  TORCH_CHECK(
      dst_w13_ptrs_w_rows.size(0) == top_k &&
          dst_w13_ptrs_s_rows.size(0) == top_k &&
          dst_w2_ptrs_w_rows.size(0) == top_k &&
          dst_w2_ptrs_s_rows.size(0) == top_k &&
          dst_w13_ptrs_w_rows.size(1) == row_bytes &&
          dst_w13_ptrs_s_rows.size(1) == row_bytes &&
          dst_w2_ptrs_w_rows.size(1) == row_bytes &&
          dst_w2_ptrs_s_rows.size(1) == row_bytes,
      "fp8_moe_single_token_sm70_out: destination ptr row shapes mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int src_row_stride = static_cast<int>(src_w13_ptrs_w_rows.stride(0));
  const int dst_row_stride = static_cast<int>(dst_w13_ptrs_w_rows.stride(0));
  const bool vectorized_prepare = sm70_fp8_moe_prepare_vec_enabled();
  const bool prepare_sorted_weight_copy =
      fused_weighted_reduce || !vectorized_prepare;
  const float* prepare_topk_weights =
      prepare_sorted_weight_copy ? topk_weights.data_ptr<float>() : nullptr;
  float* prepare_sorted_weights =
      prepare_sorted_weight_copy ? sorted_weights.data_ptr<float>() : nullptr;
  constexpr int kThreads = 256;
  const bool copy_expert_ptr_rows = !indexed_expert_ptrs && !exact_per_route;

  if (exact_per_route) {
    if (topk_ids.scalar_type() == torch::kInt32) {
      awq_moe_single_token_exact_layout_prepare_kernel<int32_t>
          <<<1, kThreads, 0, stream>>>(
              topk_ids.data_ptr<int32_t>(),
              reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
              reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
              expert_offsets.data_ptr<int32_t>(), nullptr,
              inv_permuted_idx.data_ptr<int32_t>(),
              sorted_expert_ids.data_ptr<int32_t>(), top_k,
              static_cast<int>(x.size(1)),
              static_cast<int>(source_num_experts));
    } else {
      awq_moe_single_token_exact_layout_prepare_kernel<int64_t>
          <<<1, kThreads, 0, stream>>>(
              topk_ids.data_ptr<int64_t>(),
              reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
              reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
              expert_offsets.data_ptr<int32_t>(), nullptr,
              inv_permuted_idx.data_ptr<int32_t>(),
              sorted_expert_ids.data_ptr<int32_t>(), top_k,
              static_cast<int>(x.size(1)),
              static_cast<int>(source_num_experts));
    }
  } else if (topk_ids.scalar_type() == torch::kInt32) {
    awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(), prepare_topk_weights,
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        broadcast_input
            ? nullptr
            : reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
        prepare_sorted_weights, expert_offsets.data_ptr<int32_t>(), nullptr,
        inv_permuted_idx.data_ptr<int32_t>(),
        indexed_expert_ptrs ? sorted_expert_ids.data_ptr<int32_t>() : nullptr,
        top_k, static_cast<int>(x.size(1)), src_row_stride, dst_row_stride,
        static_cast<int>(row_bytes), copy_expert_ptr_rows, vectorized_prepare);
  } else {
    awq_moe_single_token_prepare_kernel<int64_t><<<1, kThreads, 0, stream>>>(
        topk_ids.data_ptr<int64_t>(), prepare_topk_weights,
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        broadcast_input
            ? nullptr
            : reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
        prepare_sorted_weights, expert_offsets.data_ptr<int32_t>(), nullptr,
        inv_permuted_idx.data_ptr<int32_t>(),
        indexed_expert_ptrs ? sorted_expert_ids.data_ptr<int32_t>() : nullptr,
        top_k, static_cast<int>(x.size(1)), src_row_stride, dst_row_stride,
        static_cast<int>(row_bytes), copy_expert_ptr_rows, vectorized_prepare);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  torch::Tensor w13_input = broadcast_input ? x : compact_input;
  const int64_t w13_logical_total_tokens = broadcast_input ? top_k : -1;
  const int64_t w13_a_ld_override = -1;
  torch::Tensor w13_input_indices =
      (broadcast_input && broadcast_input_indices.numel() > 0)
          ? broadcast_input_indices
          : torch::Tensor();
  torch::Tensor w13_ptrs_w =
      indexed_expert_ptrs ? src_w13_ptrs_w_rows : dst_w13_ptrs_w_rows;
  torch::Tensor w13_ptrs_s =
      indexed_expert_ptrs ? src_w13_ptrs_s_rows : dst_w13_ptrs_s_rows;
  torch::Tensor w2_ptrs_w =
      indexed_expert_ptrs ? src_w2_ptrs_w_rows : dst_w2_ptrs_w_rows;
  torch::Tensor w2_ptrs_s =
      indexed_expert_ptrs ? src_w2_ptrs_s_rows : dst_w2_ptrs_s_rows;
  torch::Tensor expert_ptr_indices =
      indexed_expert_ptrs ? sorted_expert_ids : torch::Tensor();
  const bool single_token_per_expert_dispatch =
      vllm::awq_sm70::fp8_moe_single_token_per_expert_dispatch_enabled();
  if (exact_per_route) {
    fp8_moe_gemm_sm70_out_impl(
        gate_up, compact_input, expert_offsets, src_w13_ptrs_w_rows,
        src_w13_ptrs_s_rows, source_num_experts, w13_k, w13_n, group_size,
        false, torch::Tensor(), torch::Tensor(), false, -1, -1, torch::Tensor(),
        torch::Tensor(), false, -1, sorted_expert_ids, top_k);
    sm70_silu_and_mul_fp16_out(intermediate, gate_up);
    fp8_moe_gemm_sm70_out_impl(
        sorted_output, intermediate, expert_offsets, src_w2_ptrs_w_rows,
        src_w2_ptrs_s_rows, source_num_experts, w2_k, w2_n, group_size, false,
        torch::Tensor(), torch::Tensor(), false, -1, -1, torch::Tensor(),
        torch::Tensor(), false, -1, sorted_expert_ids, top_k);
  } else {
    if (fused_gated_silu) {
      fp8_moe_gemm_sm70_out_impl(
          intermediate, w13_input, expert_offsets, w13_ptrs_w, w13_ptrs_s,
          top_k, w13_k, w13_n, group_size, true, torch::Tensor(),
          torch::Tensor(), false, w13_logical_total_tokens, w13_a_ld_override,
          w13_input_indices, expert_ptr_indices,
          single_token_per_expert_dispatch, -1);
    } else {
      fp8_moe_gemm_sm70_out_impl(
          gate_up, w13_input, expert_offsets, w13_ptrs_w, w13_ptrs_s, top_k,
          w13_k, w13_n, group_size, false, torch::Tensor(), torch::Tensor(),
          false, w13_logical_total_tokens, w13_a_ld_override, w13_input_indices,
          expert_ptr_indices, single_token_per_expert_dispatch, -1);
      sm70_silu_and_mul_fp16_out(intermediate, gate_up);
    }
    if (fused_weighted_reduce) {
      fp8_moe_gemm_sm70_out_impl(sorted_output, intermediate, expert_offsets,
                                 w2_ptrs_w, w2_ptrs_s, top_k, w2_k, w2_n,
                                 group_size, false, out, sorted_weights, true,
                                 -1, -1, torch::Tensor(), expert_ptr_indices,
                                 single_token_per_expert_dispatch, -1);
      return;
    }
    if (w2_direct_reduce) {
      fp8_moe_w2_direct_reduce_sm70_out(out, intermediate, topk_weights,
                                        topk_ids, inv_permuted_idx,
                                        w2_raw_weight, w2_raw_scale_inv, top_k,
                                        w2_k, w2_n, group_size, stream);
      return;
    }

    fp8_moe_gemm_sm70_out_impl(
        sorted_output, intermediate, expert_offsets, w2_ptrs_w, w2_ptrs_s,
        top_k, w2_k, w2_n, group_size, false, torch::Tensor(), torch::Tensor(),
        false, -1, -1, torch::Tensor(), expert_ptr_indices,
        single_token_per_expert_dispatch, -1);
  }

  const int hidden_logical_size_i = static_cast<int>(hidden_logical_size);
  const int sorted_output_row_stride =
      static_cast<int>(sorted_output.stride(0));
  if (top_k == 8 && (hidden_logical_size_i % 2) == 0 &&
      (sorted_output_row_stride % 2) == 0) {
    const int blocks = std::max<int>(
        1, ((hidden_logical_size_i >> 1) + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_half2_kernel<8>
        <<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
            topk_weights.data_ptr<float>(),
            inv_permuted_idx.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            hidden_logical_size_i, sorted_output_row_stride);
  } else {
    const int blocks =
        std::max<int>(1, (hidden_logical_size_i + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_kernel<<<blocks, kThreads, 0,
                                                  stream>>>(
        reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), top_k,
        hidden_logical_size_i, sorted_output_row_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

  #if defined(VLLM_ENABLE_SM70_TURBOMIND_EXPERIMENTAL_MOE)

void fp8_moe_single_token_router_sm70_out(
    torch::Tensor out, torch::Tensor x, torch::Tensor router_logits,
    torch::Tensor topk_weights, torch::Tensor topk_ids,
    torch::Tensor src_w13_ptrs_w_rows, torch::Tensor src_w13_ptrs_s_rows,
    torch::Tensor src_w2_ptrs_w_rows, torch::Tensor src_w2_ptrs_s_rows,
    torch::Tensor compact_input, torch::Tensor gate_up,
    torch::Tensor intermediate, torch::Tensor sorted_output,
    torch::Tensor sorted_weights, torch::Tensor dst_w13_ptrs_w_rows,
    torch::Tensor dst_w13_ptrs_s_rows, torch::Tensor dst_w2_ptrs_w_rows,
    torch::Tensor dst_w2_ptrs_s_rows, torch::Tensor expert_offsets,
    torch::Tensor inv_permuted_idx, int64_t w13_k, int64_t w13_n, int64_t w2_k,
    int64_t w2_n, int64_t group_size, int64_t hidden_logical_size,
    bool renormalize, bool fused_gated_silu, bool fused_weighted_reduce) {
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kFloat16,
      "fp8_moe_single_token_router_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_router_sm70_out: x must be CUDA float16.");
  TORCH_CHECK(
      router_logits.is_cuda() && router_logits.scalar_type() == torch::kFloat16,
      "fp8_moe_single_token_router_sm70_out: router_logits must be CUDA "
      "float16.");
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.scalar_type() == torch::kFloat32,
      "fp8_moe_single_token_router_sm70_out: topk_weights must be CUDA "
      "float32.");
  TORCH_CHECK(
      topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32,
      "fp8_moe_single_token_router_sm70_out: topk_ids must be CUDA int32.");
  TORCH_CHECK(
      compact_input.is_cuda() &&
          compact_input.scalar_type() == torch::kFloat16 && gate_up.is_cuda() &&
          gate_up.scalar_type() == torch::kFloat16 && intermediate.is_cuda() &&
          intermediate.scalar_type() == torch::kFloat16 &&
          sorted_output.is_cuda() &&
          sorted_output.scalar_type() == torch::kFloat16,
      "fp8_moe_single_token_router_sm70_out: scratch buffers must be CUDA "
      "float16.");
  TORCH_CHECK(sorted_weights.is_cuda() &&
                  sorted_weights.scalar_type() == torch::kFloat32,
              "fp8_moe_single_token_router_sm70_out: sorted_weights must be "
              "CUDA float32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32,
              "fp8_moe_single_token_router_sm70_out: index buffers must be "
              "CUDA int32.");
  TORCH_CHECK(
      src_w13_ptrs_w_rows.is_cuda() && src_w13_ptrs_s_rows.is_cuda() &&
          src_w2_ptrs_w_rows.is_cuda() && src_w2_ptrs_s_rows.is_cuda() &&
          dst_w13_ptrs_w_rows.is_cuda() && dst_w13_ptrs_s_rows.is_cuda() &&
          dst_w2_ptrs_w_rows.is_cuda() && dst_w2_ptrs_s_rows.is_cuda(),
      "fp8_moe_single_token_router_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) == 1,
      "fp8_moe_single_token_router_sm70_out: x must have shape [1, hidden].");
  TORCH_CHECK(router_logits.dim() == 2 && router_logits.size(0) == 1 &&
                  router_logits.size(1) == 256,
              "fp8_moe_single_token_router_sm70_out: only [1, 256] router "
              "logits are supported.");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == 1,
      "fp8_moe_single_token_router_sm70_out: out must have shape [1, hidden].");
  TORCH_CHECK(out.size(1) == hidden_logical_size,
              "fp8_moe_single_token_router_sm70_out: out cols must match "
              "hidden_logical_size.");

  topk_ids = topk_ids.contiguous().view({-1});
  topk_weights = topk_weights.contiguous().view({-1});
  inv_permuted_idx = inv_permuted_idx.contiguous().view({-1});
  sorted_weights = sorted_weights.contiguous().view({-1});

  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(
      top_k == 8,
      "fp8_moe_single_token_router_sm70_out: only top_k=8 is supported.");
  TORCH_CHECK(
      static_cast<int>(topk_weights.numel()) == top_k,
      "fp8_moe_single_token_router_sm70_out: topk_weights size mismatch.");
  TORCH_CHECK(
      sorted_weights.numel() == top_k,
      "fp8_moe_single_token_router_sm70_out: sorted_weights size mismatch.");
  TORCH_CHECK(
      compact_input.dim() == 2 && compact_input.size(0) == top_k &&
          compact_input.size(1) == x.size(1),
      "fp8_moe_single_token_router_sm70_out: compact_input shape mismatch.");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) == top_k &&
                  gate_up.size(1) == w13_n,
              "fp8_moe_single_token_router_sm70_out: gate_up shape mismatch.");
  TORCH_CHECK(
      intermediate.dim() == 2 && intermediate.size(0) == top_k &&
          intermediate.size(1) == w13_n / 2,
      "fp8_moe_single_token_router_sm70_out: intermediate shape mismatch.");
  TORCH_CHECK(
      sorted_output.dim() == 2 && sorted_output.size(0) == top_k &&
          sorted_output.size(1) == w2_n,
      "fp8_moe_single_token_router_sm70_out: sorted_output shape mismatch.");
  TORCH_CHECK(
      expert_offsets.numel() == top_k + 1,
      "fp8_moe_single_token_router_sm70_out: expert_offsets size mismatch.");
  TORCH_CHECK(
      inv_permuted_idx.numel() == top_k,
      "fp8_moe_single_token_router_sm70_out: inv_permuted_idx size mismatch.");

  const int64_t row_bytes = src_w13_ptrs_w_rows.size(1);
  TORCH_CHECK(
      src_w13_ptrs_w_rows.dim() == 2 && src_w13_ptrs_s_rows.dim() == 2 &&
          src_w2_ptrs_w_rows.dim() == 2 && src_w2_ptrs_s_rows.dim() == 2 &&
          dst_w13_ptrs_w_rows.dim() == 2 && dst_w13_ptrs_s_rows.dim() == 2 &&
          dst_w2_ptrs_w_rows.dim() == 2 && dst_w2_ptrs_s_rows.dim() == 2,
      "fp8_moe_single_token_router_sm70_out: ptr row tensors must be 2D.");
  TORCH_CHECK(dst_w13_ptrs_w_rows.size(0) == top_k &&
                  dst_w13_ptrs_s_rows.size(0) == top_k &&
                  dst_w2_ptrs_w_rows.size(0) == top_k &&
                  dst_w2_ptrs_s_rows.size(0) == top_k &&
                  dst_w13_ptrs_w_rows.size(1) == row_bytes &&
                  dst_w13_ptrs_s_rows.size(1) == row_bytes &&
                  dst_w2_ptrs_w_rows.size(1) == row_bytes &&
                  dst_w2_ptrs_s_rows.size(1) == row_bytes,
              "fp8_moe_single_token_router_sm70_out: destination ptr row "
              "shapes mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int src_row_stride = static_cast<int>(src_w13_ptrs_w_rows.stride(0));
  const int dst_row_stride = static_cast<int>(dst_w13_ptrs_w_rows.stride(0));
  const bool vectorized_prepare = sm70_fp8_moe_prepare_vec_enabled();
  constexpr int kThreads = 256;

  fp8_moe_single_token_router_prepare_256_top8_kernel<<<1, kThreads, 0,
                                                        stream>>>(
      reinterpret_cast<const __half*>(router_logits.data_ptr<at::Half>()),
      topk_weights.data_ptr<float>(), topk_ids.data_ptr<int32_t>(),
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
      src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
      src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
      src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
      dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
      dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
      dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
      dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
      sorted_weights.data_ptr<float>(), expert_offsets.data_ptr<int32_t>(),
      inv_permuted_idx.data_ptr<int32_t>(), static_cast<int>(x.size(1)),
      src_row_stride, dst_row_stride, static_cast<int>(row_bytes), renormalize,
      vectorized_prepare);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (fused_gated_silu) {
    fp8_moe_gemm_sm70_out_impl(intermediate, compact_input, expert_offsets,
                               dst_w13_ptrs_w_rows, dst_w13_ptrs_s_rows, top_k,
                               w13_k, w13_n, group_size, true, torch::Tensor(),
                               torch::Tensor(), false, -1, -1);
  } else {
    fp8_moe_gemm_sm70_out_impl(gate_up, compact_input, expert_offsets,
                               dst_w13_ptrs_w_rows, dst_w13_ptrs_s_rows, top_k,
                               w13_k, w13_n, group_size, false, torch::Tensor(),
                               torch::Tensor(), false, -1, -1);
    sm70_silu_and_mul_fp16_out(intermediate, gate_up);
  }
  if (fused_weighted_reduce) {
    fp8_moe_gemm_sm70_out_impl(sorted_output, intermediate, expert_offsets,
                               dst_w2_ptrs_w_rows, dst_w2_ptrs_s_rows, top_k,
                               w2_k, w2_n, group_size, false, out,
                               sorted_weights, true, -1, -1);
    return;
  }

  fp8_moe_gemm_sm70_out(sorted_output, intermediate, expert_offsets,
                        dst_w2_ptrs_w_rows, dst_w2_ptrs_s_rows, top_k, w2_k,
                        w2_n, group_size, false);

  const int hidden_logical_size_i = static_cast<int>(hidden_logical_size);
  const int sorted_output_row_stride =
      static_cast<int>(sorted_output.stride(0));
  if ((hidden_logical_size_i % 2) == 0 && (sorted_output_row_stride % 2) == 0) {
    const int blocks = std::max<int>(
        1, ((hidden_logical_size_i >> 1) + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_half2_kernel<8>
        <<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
            topk_weights.data_ptr<float>(),
            inv_permuted_idx.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            hidden_logical_size_i, sorted_output_row_stride);
  } else {
    const int blocks =
        std::max<int>(1, (hidden_logical_size_i + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_kernel<<<blocks, kThreads, 0,
                                                  stream>>>(
        reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), top_k,
        hidden_logical_size_i, sorted_output_row_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

  #endif  // defined(VLLM_ENABLE_SM70_TURBOMIND_EXPERIMENTAL_MOE)

#endif  // defined(ENABLE_SM70_TURBOMIND)
