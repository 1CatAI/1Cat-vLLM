// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only: warm the first groups consumed by each existing K partition.
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
__global__ void warm_qpn_sectors(const uint8_t* codes, unsigned* sink,
                                 int groups, int group_bytes, int splits,
                                 int prefix, int sectors) {
  unsigned value = 0;
  const int sectors_per_group = group_bytes / 32;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < sectors;
       i += blockDim.x * gridDim.x) {
    const int sector = i % sectors_per_group;
    const int group = i / sectors_per_group;
    const int offset = group % prefix;
    const int partition = (group / prefix) % splits;
    const int tile = group / (prefix * splits);
    const size_t address = (static_cast<size_t>(tile) * groups +
                            partition * (groups / splits) + offset) *
                               group_bytes +
                           sector * 32;
    unsigned loaded;
    // One word fetches its 32-byte L2 sector. Keep the result observable so
    // joining this kernel also waits for the reads to complete.
    asm volatile("ld.global.cg.u32 %0, [%1];"
                 : "=r"(loaded)
                 : "l"(codes + address)
                 : "memory");
    value ^= loaded;
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

void warm(torch::Tensor codes, torch::Tensor sink, int64_t tiles,
          int64_t groups, int64_t group_bytes, int64_t splits, int64_t prefix,
          int64_t blocks) {
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == torch::kUInt8 &&
              codes.is_contiguous());
  TORCH_CHECK(tiles > 0 && groups > 0 && splits > 0 && groups % splits == 0 &&
              prefix > 0 && prefix <= groups / splits && blocks > 0 &&
              blocks <= 160 && (group_bytes == 256 || group_bytes == 512));
  TORCH_CHECK(codes.numel() == tiles * groups * group_bytes);
  TORCH_CHECK(sink.device() == codes.device() && sink.is_contiguous() &&
              sink.scalar_type() == torch::kInt32 &&
              sink.numel() >= blocks * 128);
  c10::cuda::CUDAGuard guard(codes.device());
  warm_qpn_sectors<<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<unsigned*>(sink.data_ptr<int>()), groups, group_bytes,
      splits, prefix, tiles * splits * prefix * (group_bytes / 32));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY(_qpn_overlap700, library) {
  library.def(
      "warm(Tensor codes, Tensor(a!) sink, int tiles, int groups, "
      "int group_bytes, int splits, int prefix, int blocks) -> ()");
}
TORCH_LIBRARY_IMPL(_qpn_overlap700, CUDA, library) {
  library.impl("warm", &warm);
}
