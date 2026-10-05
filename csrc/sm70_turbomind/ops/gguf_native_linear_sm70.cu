// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Single projection using the measured native-pair operand and activation path.
#include <climits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_pair_shared_a_sm70.cuh"

namespace vllm::sm70_gguf {
// Consume the already prepared canonical stream. No source superblock or
// nested scale unpacking is repeated in the main loop. Operand conversion is
// the same transform used by canonical GEMM/dequantization.
template <int Bits, int Group>
struct CanonicalAffineReader {
  static constexpr int kBookBytes = 0;
  using Packet = std::conditional_t<Bits == 2, uint16_t, uint32_t>;
  using Code =
      std::conditional_t<Bits == 2, turbomind::uint2_t, turbomind::uint4_t>;
  struct Record {
    Packet packets[16];
    uint32_t coefficients[128 / Group];
  };
  const Packet* packets;
  const uint32_t* coefficients;
  int stat_stride;
  __device__ CanonicalAffineReader(const uint8_t* source, const uint32_t* stats,
                                   int n, int k, int tile, int first, int col) {
    packets = reinterpret_cast<const Packet*>(source) +
              int64_t{tile} * (k / 8) * 32 + first * 16 * 32 + col;
    coefficients = stats + int64_t{first} * (128 / Group) * n + tile * 32 + col;
    stat_stride = n;
  }
  __device__ static void initialize(uint8_t*) {}
  __device__ Record load() {
    Record record;
#pragma unroll
    for (int i = 0; i < 16; ++i) record.packets[i] = packets[i * 32];
#pragma unroll
    for (int i = 0; i < 128 / Group; ++i)
      record.coefficients[i] = coefficients[i * stat_stride];
    packets += 16 * 32;
    coefficients += (128 / Group) * stat_stride;
    return record;
  }
  template <int Segment, int Fragment>
  __device__ static turbomind::Array<half, 8> fragment(const Record& record,
                                                       const uint8_t*) {
    constexpr int octet = Segment * 2 + Fragment;
    turbomind::Array<Code, 8> data[1][1];
    reinterpret_cast<Packet&>(data[0][0]) = record.packets[octet];
    turbomind::Array<uint32_t, 1> stats[1][1];
    stats[0][0][0] = record.coefficients[(octet * 8) / Group];
    turbomind::Array<half, 8> decoded[1][1];
    turbomind::gemm::Transform_HMMA_SIMT_B::apply(decoded, 0, data, stats, 1);
    return decoded[0][0];
  }
};

// Two N32 subtiles share the pair's activation staging and codebook. Two
// global K partitions expose 160 CTAs for N5120, with no separate reduction
// launch. Partial sums remain FP32 until the last CTA writes the output.
template <class Reader, bool Canonical = false>
__global__ __launch_bounds__(512, 2) void native_linear_n64_kernel(
    half* __restrict__ output, const half* __restrict__ input,
    const uint8_t* __restrict__ weight, const uint32_t* __restrict__ stats,
    float* __restrict__ partials, int* __restrict__ counters, int n, int k) {
  constexpr int SplitK = 8, GlobalSplitK = 2;
  union alignas(16) Storage {
    uint8_t book[Reader::kBookBytes > 0 ? Reader::kBookBytes : 1];
    float partials[2][SplitK][256];
  };
  __shared__ Storage storage;
  __shared__ half staged_a[SplitK][8][136];
  __shared__ bool last_cta;
  Reader::initialize(storage.book);
  const int lane = threadIdx.x & 31;
  const int subtile = threadIdx.x >> 8;
  const int warp = (threadIdx.x >> 5) & 7;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = quadpair * 8 + row;
  const int parts = k / 128;
  const int global_first = blockIdx.y * parts / GlobalSplitK;
  const int local_parts = parts / GlobalSplitK;
  const int first = global_first + warp * local_parts / SplitK;
  const int last = global_first + (warp + 1) * local_parts / SplitK;
  Reader reader = [&]() {
    if constexpr (Canonical)
      return Reader(weight, stats, n, k, blockIdx.x * 2 + subtile, first, col);
    else
      return Reader(weight, blockIdx.x * 2 + subtile, k / 256, first, col);
  }();
  float accum[8] = {};
  for (int part = 0; part < (local_parts + SplitK - 1) / SplitK; ++part) {
    for (int vector = threadIdx.x; vector < SplitK * 8 * 16;
         vector += blockDim.x) {
      const int k_warp = vector / 128;
      const int input_row = (vector / 16) & 7;
      const int k_vector = vector & 15;
      const int input_part =
          global_first + k_warp * local_parts / SplitK + part;
      const int end_part = global_first + (k_warp + 1) * local_parts / SplitK;
      if (input_part < end_part) {
        const uint4 value = *reinterpret_cast<const uint4*>(
            input + int64_t{input_row} * k + input_part * 128 + k_vector * 8);
        *reinterpret_cast<uint4*>(&staged_a[k_warp][input_row][k_vector * 8]) =
            value;
      }
    }
    __syncthreads();
    if (first + part < last) {
      const auto record = reader.load();
      native_pair_segments<0, Reader>(accum, record, &staged_a[warp][row][0],
                                      storage.book);
    }
    __syncthreads();
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int output_row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int output_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    storage
        .partials[subtile][warp][output_row * 32 + quadpair * 8 + output_col] =
        accum[i];
  }
  __syncthreads();
  const int element = threadIdx.x & 255;
  float sum = 0.f;
#pragma unroll
  for (int part = 0; part < SplitK; ++part)
    sum += storage.partials[subtile][part][element];
  const int partial_base = (blockIdx.x * GlobalSplitK + blockIdx.y) * 512;
  volatile float* published = partials;
  published[partial_base + threadIdx.x] = sum;
  // All writers publish their own words before one thread publishes the
  // completion ticket. The last CTA reads each partition in fixed order.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    last_cta = atomicAdd(counters + blockIdx.x, 1) == GlobalSplitK - 1;
  __syncthreads();
  if (last_cta) {
    sum = 0.f;
#pragma unroll
    for (int part = 0; part < GlobalSplitK; ++part)
      sum += published[(blockIdx.x * GlobalSplitK + part) * 512 + threadIdx.x];
    output[int64_t{element / 32} * n + blockIdx.x * 64 + subtile * 32 +
           element % 32] = __float2half_rn(sum);
    __syncthreads();
    if (threadIdx.x == 0) atomicExch(counters + blockIdx.x, 0);
  }
}

template <class Reader>
__global__ __launch_bounds__(256, 2) void native_linear_shared_a_kernel(
    half* __restrict__ output, const half* __restrict__ input,
    const uint8_t* __restrict__ weight, int n, int k) {
  constexpr int SplitK = 8;
  union alignas(16) Storage {
    uint8_t book[Reader::kBookBytes > 0 ? Reader::kBookBytes : 1];
    float partials[SplitK][256];
  };
  __shared__ Storage storage;
  __shared__ half staged_a[SplitK][8][136];
  Reader::initialize(storage.book);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = quadpair * 8 + row;
  const int parts = k / 128;
  const int first = warp * parts / SplitK;
  const int last = (warp + 1) * parts / SplitK;
  Reader reader(weight, blockIdx.x, k / 256, first, col);
  float accum[8] = {};
  // Unequal K128 partitions cover TP shards such as K4352 without padding
  // the checkpoint or reducing its scale precision. All CTA barriers remain
  // uniform even when one warp has fewer records than the others.
  for (int part = 0; part < (parts + SplitK - 1) / SplitK; ++part) {
    for (int vector = threadIdx.x; vector < SplitK * 8 * 16;
         vector += blockDim.x) {
      const int k_warp = vector / 128;
      const int input_row = (vector / 16) & 7;
      const int k_vector = vector & 15;
      const int input_part = k_warp * parts / SplitK + part;
      const int end_part = (k_warp + 1) * parts / SplitK;
      if (input_part < end_part) {
        const uint4 value = *reinterpret_cast<const uint4*>(
            input + int64_t{input_row} * k + input_part * 128 + k_vector * 8);
        *reinterpret_cast<uint4*>(&staged_a[k_warp][input_row][k_vector * 8]) =
            value;
      }
    }
    __syncthreads();
    if (first + part < last) {
      const auto record = reader.load();
      native_pair_segments<0, Reader>(accum, record, &staged_a[warp][row][0],
                                      storage.book);
    }
    __syncthreads();
  }
  // Every codebook read is complete before shared storage becomes partials.
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int output_row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int output_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    storage.partials[warp][output_row * 32 + quadpair * 8 + output_col] =
        accum[i];
  }
  __syncthreads();
  float sum = 0.f;
#pragma unroll
  for (int part = 0; part < SplitK; ++part)
    sum += storage.partials[part][threadIdx.x];
  output[int64_t{threadIdx.x / 32} * n + blockIdx.x * 32 + threadIdx.x % 32] =
      __float2half_rn(sum);
}
}  // namespace vllm::sm70_gguf

namespace {
template <int Type>
void launch_linear(torch::Tensor output, torch::Tensor input,
                   torch::Tensor weight, int n, int k, cudaStream_t stream) {
  using namespace vllm::sm70_gguf;
  using Reader = NativePairReader<Type>;
  TORCH_CHECK(weight.numel() == int64_t{n} * (k / 256) * Reader::kBlockBytes,
              "GGUF native linear record byte count mismatch");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      native_linear_shared_a_kernel<Reader>,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  native_linear_shared_a_kernel<Reader><<<n / 32, 256, 0, stream>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      weight.data_ptr<uint8_t>(), n, k);
}

template <int Type>
void launch_linear_n64(torch::Tensor output, torch::Tensor input,
                       torch::Tensor weight, torch::Tensor partials,
                       torch::Tensor counters, int n, int k,
                       cudaStream_t stream) {
  using namespace vllm::sm70_gguf;
  using Reader = NativePairReader<Type>;
  TORCH_CHECK(weight.numel() == int64_t{n} * (k / 256) * Reader::kBlockBytes,
              "GGUF native linear record byte count mismatch");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      native_linear_n64_kernel<Reader>,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  native_linear_n64_kernel<Reader><<<dim3(n / 64, 2), 512, 0, stream>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      weight.data_ptr<uint8_t>(), nullptr, partials.data_ptr<float>(),
      counters.data_ptr<int>(), n, k);
}

template <int Bits, int Group>
void launch_canonical_linear(torch::Tensor output, torch::Tensor input,
                             torch::Tensor weight, torch::Tensor stats,
                             torch::Tensor partials, torch::Tensor counters,
                             int n, int k, cudaStream_t stream) {
  using namespace vllm::sm70_gguf;
  using Reader = CanonicalAffineReader<Bits, Group>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      native_linear_n64_kernel<Reader, true>,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  native_linear_n64_kernel<Reader, true><<<dim3(n / 64, 2), 512, 0, stream>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(weight.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(stats.data_ptr<int32_t>()),
      partials.data_ptr<float>(), counters.data_ptr<int>(), n, k);
}
}  // namespace

void gguf_native_linear_n64_sm70_out(torch::Tensor output, torch::Tensor input,
                                     torch::Tensor weight,
                                     torch::Tensor partials,
                                     torch::Tensor counters,
                                     int64_t source_type) {
  TORCH_CHECK(output.is_cuda() && input.is_cuda() && weight.is_cuda() &&
                  partials.is_cuda() && counters.is_cuda() &&
                  output.device() == input.device() &&
                  weight.device() == input.device() &&
                  partials.device() == input.device() &&
                  counters.device() == input.device() &&
                  output.is_contiguous() && input.is_contiguous() &&
                  weight.is_contiguous() && partials.is_contiguous() &&
                  counters.is_contiguous() && output.dim() == 2 &&
                  input.dim() == 2 && weight.dim() == 1 &&
                  output.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weight.scalar_type() == torch::kUInt8 &&
                  partials.scalar_type() == torch::kFloat32 &&
                  counters.scalar_type() == torch::kInt32,
              "GGUF N64 linear requires contiguous CUDA FP16 matrices, "
              "original records and FP32 reduction workspace");
  const int64_t m = input.size(0), k = input.size(1), n = output.size(1);
  TORCH_CHECK(m == 8 && output.size(0) == m && n > 0 && n % 64 == 0 && k > 0 &&
                  k % 256 == 0 && n <= INT_MAX && k <= INT_MAX &&
                  partials.sizes() == c10::IntArrayRef({n / 64, 2, 512}) &&
                  counters.sizes() == c10::IntArrayRef({n / 64}),
              "GGUF N64 linear requires M8/N64/K256 and matching workspace");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "GGUF native linear requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();
  switch (source_type) {
    case 10:
      launch_linear_n64<10>(output, input, weight, partials, counters, n, k,
                            stream);
      break;
    case 12:
      launch_linear_n64<12>(output, input, weight, partials, counters, n, k,
                            stream);
      break;
    case 16:
      launch_linear_n64<16>(output, input, weight, partials, counters, n, k,
                            stream);
      break;
    case 17:
      launch_linear_n64<17>(output, input, weight, partials, counters, n, k,
                            stream);
      break;
    case 18:
      launch_linear_n64<18>(output, input, weight, partials, counters, n, k,
                            stream);
      break;
    case 21:
      launch_linear_n64<21>(output, input, weight, partials, counters, n, k,
                            stream);
      break;
    case 22:
      launch_linear_n64<22>(output, input, weight, partials, counters, n, k,
                            stream);
      break;
    case 23:
      launch_linear_n64<23>(output, input, weight, partials, counters, n, k,
                            stream);
      break;
    case 29:
      launch_linear_n64<29>(output, input, weight, partials, counters, n, k,
                            stream);
      break;
    default:
      TORCH_CHECK(false, "Unsupported GGUF native linear format");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_native_linear_sm70_out(torch::Tensor output, torch::Tensor input,
                                 torch::Tensor weight, int64_t source_type) {
  TORCH_CHECK(output.is_cuda() && input.is_cuda() && weight.is_cuda() &&
                  output.device() == input.device() &&
                  weight.device() == input.device() && output.is_contiguous() &&
                  input.is_contiguous() && weight.is_contiguous() &&
                  output.dim() == 2 && input.dim() == 2 && weight.dim() == 1 &&
                  output.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weight.scalar_type() == torch::kUInt8,
              "GGUF native linear requires contiguous CUDA FP16 matrices and "
              "original records");
  const int64_t m = input.size(0), k = input.size(1), n = output.size(1);
  TORCH_CHECK(m == 8 && output.size(0) == m && n > 0 && n % 32 == 0 && k > 0 &&
                  k % 256 == 0 && n <= INT_MAX && k <= INT_MAX,
              "GGUF native linear requires M8/N32/K256");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "GGUF native linear requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();
  switch (source_type) {
    case 10:
      launch_linear<10>(output, input, weight, n, k, stream);
      break;
    case 12:
      launch_linear<12>(output, input, weight, n, k, stream);
      break;
    case 16:
      launch_linear<16>(output, input, weight, n, k, stream);
      break;
    case 17:
      launch_linear<17>(output, input, weight, n, k, stream);
      break;
    case 18:
      launch_linear<18>(output, input, weight, n, k, stream);
      break;
    case 21:
      launch_linear<21>(output, input, weight, n, k, stream);
      break;
    case 22:
      launch_linear<22>(output, input, weight, n, k, stream);
      break;
    case 23:
      launch_linear<23>(output, input, weight, n, k, stream);
      break;
    case 29:
      launch_linear<29>(output, input, weight, n, k, stream);
      break;
    default:
      TORCH_CHECK(false, "Unsupported GGUF native linear format");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_canonical_linear_n64_sm70_out(
    torch::Tensor output, torch::Tensor input, torch::Tensor weight,
    torch::Tensor stats, torch::Tensor partials, torch::Tensor counters,
    int64_t bits, int64_t group_size) {
  TORCH_CHECK(output.is_cuda() && input.is_cuda() && weight.is_cuda() &&
                  stats.is_cuda() && partials.is_cuda() && counters.is_cuda() &&
                  output.device() == input.device() &&
                  weight.device() == input.device() &&
                  stats.device() == input.device() &&
                  partials.device() == input.device() &&
                  counters.device() == input.device() &&
                  output.is_contiguous() && input.is_contiguous() &&
                  weight.is_contiguous() && stats.is_contiguous() &&
                  partials.is_contiguous() && counters.is_contiguous() &&
                  output.dim() == 2 && input.dim() == 2 && weight.dim() == 2 &&
                  stats.dim() == 2 && output.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weight.scalar_type() == torch::kInt32 &&
                  stats.scalar_type() == torch::kInt32 &&
                  partials.scalar_type() == torch::kFloat32 &&
                  counters.scalar_type() == torch::kInt32,
              "GGUF canonical linear requires FP16 CUDA matrices, prepared "
              "int32 code/stat streams and FP32 reduction workspace");
  const int64_t m = input.size(0), k = input.size(1), n = output.size(1);
  TORCH_CHECK(m == 8 && output.size(0) == m && n > 0 && n % 64 == 0 && k > 0 &&
                  k % 256 == 0 && n <= INT_MAX && k <= INT_MAX &&
                  ((bits == 2 && group_size == 16) ||
                   (bits == 4 && group_size == 32)) &&
                  weight.sizes() == c10::IntArrayRef({k, n * bits / 32}) &&
                  stats.sizes() == c10::IntArrayRef({k / group_size, n}) &&
                  partials.sizes() == c10::IntArrayRef({n / 64, 2, 512}) &&
                  counters.sizes() == c10::IntArrayRef({n / 64}),
              "GGUF canonical linear requires M8/N64/K256 and U2G16/U4G32");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "GGUF canonical linear requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (bits == 2)
    launch_canonical_linear<2, 16>(output, input, weight, stats, partials,
                                   counters, n, k, stream);
  else
    launch_canonical_linear<4, 32>(output, input, weight, stats, partials,
                                   counters, n, k, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
