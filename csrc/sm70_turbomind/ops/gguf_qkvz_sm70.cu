// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// QKVZ and floating B/A share a launch, with independent operand readers.
#include <climits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_linear_shared_a_sm70.cuh"

namespace vllm::sm70_gguf {
struct PackedFp16Reader {
  static constexpr int kBookBytes = 0;
  struct Record {
    const half* values;
  };
  const half* values;
  __device__ PackedFp16Reader(const uint8_t* weight, int tile, int blocks,
                              int first, int col) {
    values = reinterpret_cast<const half*>(weight) +
             int64_t{tile} * blocks * 8192 + first * 4096 + col * 8;
  }
  __device__ static void initialize(uint8_t*) {}
  __device__ Record load() {
    Record result{values};
    values += 4096;
    return result;
  }
  template <int Segment, int Fragment>
  __device__ static turbomind::Array<half, 8> fragment(const Record& record,
                                                       const uint8_t*) {
    turbomind::Array<half, 8> result;
    *reinterpret_cast<uint4*>(&result) = *reinterpret_cast<const uint4*>(
        record.values + (Segment * 2 + Fragment) * 256);
    return result;
  }
};
struct QkvzSource {
  const uint8_t* weight;
  const uint32_t* stats;
  int type;
  int columns;
  int first_tile;
  int output_offset;
  int stats_stride;
};
struct QkvzSources {
  QkvzSource items[5];
};

__global__ __launch_bounds__(512, 2) void qkvz_shared_a_kernel(
    half* output, const half* input, QkvzSources sources, float* partials,
    int* counters) {
  __shared__ alignas(16) uint8_t shared[33793];
  int source = 4;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    if (blockIdx.x >= sources.items[i].first_tile &&
        blockIdx.x < sources.items[i + 1].first_tile)
      source = i;
  const QkvzSource descriptor = sources.items[source];
  const int tile = blockIdx.x - descriptor.first_tile;
#define RUN_READER(Reader, Canonical)                                         \
  native_linear_n64_body<Reader, Canonical>(                                  \
      output, input, descriptor.weight, descriptor.stats, partials, counters, \
      descriptor.columns, 5120, tile, blockIdx.x, 4120,                       \
      descriptor.output_offset, descriptor.stats_stride, shared)
  switch (descriptor.type) {
    case 10:
      RUN_READER(NativePairReader<10>, false);
      break;
    case 12:
      RUN_READER(NativePairReader<12>, false);
      break;
    case 16:
      RUN_READER(NativePairReader<16>, false);
      break;
    case 17:
      RUN_READER(NativePairReader<17>, false);
      break;
    case 18:
      RUN_READER(NativePairReader<18>, false);
      break;
    case 21:
      RUN_READER(NativePairReader<21>, false);
      break;
    case 22:
      RUN_READER(NativePairReader<22>, false);
      break;
    case 23:
      RUN_READER(NativePairReader<23>, false);
      break;
    case 29:
      RUN_READER(NativePairReader<29>, false);
      break;
    case 102: {
      using Reader = CanonicalAffineReader<2, 16>;
      RUN_READER(Reader, true);
      break;
    }
    case 104: {
      using Reader = CanonicalAffineReader<4, 32>;
      RUN_READER(Reader, true);
      break;
    }
    case 1:
      RUN_READER(PackedFp16Reader, false);
      break;
  }
#undef RUN_READER
}
}  // namespace vllm::sm70_gguf

void gguf_qkvz_sm70_out(torch::Tensor output, torch::Tensor input,
                        const std::vector<torch::Tensor>& weights,
                        const std::vector<torch::Tensor>& stats,
                        const std::vector<int64_t>& types,
                        torch::Tensor partials, torch::Tensor counters) {
  TORCH_CHECK(weights.size() == 5 && stats.size() == 5 && types.size() == 5,
              "GGUF QKVZ requires four quantized sources and packed FP16 B/A");
  const auto device = input.device();
  auto check = [&](const torch::Tensor& tensor, at::ScalarType dtype,
                   bool contiguous = true) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device &&
                    (!contiguous || tensor.is_contiguous()) &&
                    tensor.scalar_type() == dtype,
                "GGUF QKVZ operand device, dtype or contiguity mismatch");
  };
  check(input, torch::kFloat16);
  check(output, torch::kFloat16);
  check(partials, torch::kFloat32);
  check(counters, torch::kInt32);
  TORCH_CHECK(input.sizes() == c10::IntArrayRef({8, 5120}) &&
                  output.sizes() == c10::IntArrayRef({8, 4120}) &&
                  partials.sizes() == c10::IntArrayRef({65, 2, 512}) &&
                  counters.sizes() == c10::IntArrayRef({65}),
              "GGUF QKVZ requires M8/K5120/N4120 and matching workspace");
  constexpr int widths[5] = {512, 512, 1536, 1536, 24};
  vllm::sm70_gguf::QkvzSources descriptors{};
  int first_tile = 0, offset = 0;
  for (int i = 0; i < 5; ++i) {
    const int type = types[i], n = widths[i];
    if (i == 4) {
      TORCH_CHECK(type == 1, "GGUF QKVZ requires FP16 B/A");
      check(weights[i], torch::kFloat16);
      TORCH_CHECK(weights[i].numel() == 64 * 5120,
                  "Packed B/A requires padded N64/K5120");
    } else if (type == 102 || type == 104) {
      const int bits = type == 102 ? 2 : 4, group = type == 102 ? 16 : 32;
      check(weights[i], torch::kInt32);
      check(stats[i], torch::kInt32, false);
      TORCH_CHECK(
          weights[i].sizes() == c10::IntArrayRef({5120, n * bits / 32}) &&
              stats[i].sizes() == c10::IntArrayRef({5120 / group, n}) &&
              stats[i].stride(1) == 1 && stats[i].stride(0) >= n,
          "GGUF QKVZ canonical stream geometry mismatch");
    } else {
      int bytes = 0;
      switch (type) {
        case 10:
          bytes = vllm::sm70_gguf::NativePairReader<10>::kBlockBytes;
          break;
        case 12:
          bytes = vllm::sm70_gguf::NativePairReader<12>::kBlockBytes;
          break;
        case 16:
          bytes = vllm::sm70_gguf::NativePairReader<16>::kBlockBytes;
          break;
        case 17:
          bytes = vllm::sm70_gguf::NativePairReader<17>::kBlockBytes;
          break;
        case 18:
          bytes = vllm::sm70_gguf::NativePairReader<18>::kBlockBytes;
          break;
        case 21:
          bytes = vllm::sm70_gguf::NativePairReader<21>::kBlockBytes;
          break;
        case 22:
          bytes = vllm::sm70_gguf::NativePairReader<22>::kBlockBytes;
          break;
        case 23:
          bytes = vllm::sm70_gguf::NativePairReader<23>::kBlockBytes;
          break;
        case 29:
          bytes = vllm::sm70_gguf::NativePairReader<29>::kBlockBytes;
          break;
        default:
          TORCH_CHECK(false, "Unsupported GGUF QKVZ source format");
      }
      check(weights[i], torch::kUInt8);
      TORCH_CHECK(weights[i].numel() == int64_t{n} * 20 * bytes,
                  "GGUF QKVZ native record geometry mismatch");
    }
    descriptors.items[i] = {
        static_cast<const uint8_t*>(weights[i].data_ptr()),
        (type == 102 || type == 104)
            ? reinterpret_cast<const uint32_t*>(stats[i].data_ptr<int>())
            : nullptr,
        type,
        n,
        first_tile,
        offset,
        (type == 102 || type == 104) ? static_cast<int>(stats[i].stride(0))
                                     : n};
    first_tile += (n + 63) / 64;
    offset += n;
  }
  const c10::cuda::CUDAGuard guard(device);
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "GGUF QKVZ requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      vllm::sm70_gguf::qkvz_shared_a_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  vllm::sm70_gguf::qkvz_shared_a_kernel<<<dim3(65, 2), 512, 0, stream>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()), descriptors,
      partials.data_ptr<float>(), counters.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
