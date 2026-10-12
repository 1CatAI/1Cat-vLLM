# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exhaustive packed INT8 reader check and isolated conversion cost experiment.

The benchmark-only extension is never loaded by an engine. Its timings decide
whether to implement/qualify a shared reader change; they are not an engine
performance result or a clean-package qualification.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "kv_codec_traits.cuh"

// Frozen original reader: each signed integer is exactly representable in half.
__device__ uint4 original(uint64_t raw) {
  uint4 result;
  auto* words = reinterpret_cast<uint32_t*>(&result);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    int8_t lo = static_cast<int8_t>(raw >> (16 * j));
    int8_t hi = static_cast<int8_t>(raw >> (16 * j + 8));
    union { __half2 half; uint32_t bits; } pair;
    pair.half = __floats2half2_rn(float(lo), float(hi));
    words[j] = pair.bits;
  }
  return result;
}

__device__ uint4 prototype(uint64_t raw) {
  return flash_v100::KVCodecTraits<3>::half8_from_packed(raw);
}

template<bool Fast>
__global__ void check(const uint64_t* input, uint4* output, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = Fast ? prototype(input[index]) : original(input[index]);
}

template<bool Fast>
__global__ void cost(const uint64_t* input, uint32_t* output, int count,
                    int iterations) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  uint64_t raw = input[index];
  uint32_t sum = 0;
  for (int i = 0; i < iterations; ++i) {
    uint4 value = Fast ? prototype(raw) : original(raw);
    sum += value.x ^ value.y ^ value.z ^ value.w;
    // Change the live input and retain the result; never time a hoisted
    // conversion or an eliminated loop.
    raw ^= raw << 13;
    raw ^= raw >> 7;
    raw ^= raw << 17;
  }
  output[index] = sum;
}

void run(torch::Tensor input, torch::Tensor output, bool fast, int iterations) {
  const c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kLong &&
              input.is_contiguous());
  TORCH_CHECK(output.device()==input.device() && output.is_contiguous());
  const int count = input.numel();
  const int blocks = (count + 255) / 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto* data = reinterpret_cast<const uint64_t*>(input.data_ptr());
  if (iterations == 0) {
    TORCH_CHECK(output.scalar_type()==at::kHalf && output.numel()==count*8);
    auto* dst = reinterpret_cast<uint4*>(output.data_ptr());
    if (fast) check<true><<<blocks,256,0,stream>>>(data,dst,count);
    else check<false><<<blocks,256,0,stream>>>(data,dst,count);
  } else {
    TORCH_CHECK(iterations>0 && output.scalar_type()==at::kInt &&
                output.numel()==count);
    auto* dst = reinterpret_cast<uint32_t*>(output.data_ptr());
    if (fast) cost<true><<<blocks,256,0,stream>>>(data,dst,count,iterations);
    else cost<false><<<blocks,256,0,stream>>>(data,dst,count,iterations);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=256)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This bounded conversion experiment targets SM70")
    library = load_inline(
        name="sm70_kv_reader_experiment",
        cpp_sources="void run(torch::Tensor,torch::Tensor,bool,int);",
        cuda_sources=CUDA_SOURCE,
        functions=["run"],
        extra_include_paths=[
            str(Path(__file__).resolve().parents[2] / "flash-attention-v100/kernel")
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )
    pairs = torch.arange(65536, device="cuda", dtype=torch.int64)
    raw = torch.zeros_like(pairs)
    for position in range(4):
        raw |= ((pairs ^ (position * 0x9E37)) & 65535) << (position * 16)
    reference = raw.view(torch.int8).view(-1, 8).to(torch.float16)
    original, candidate = torch.empty_like(reference), torch.empty_like(reference)
    library.run(raw, original, False, 0)
    library.run(raw, candidate, True, 0)
    torch.accelerator.synchronize()
    assert torch.equal(original.view(torch.int16), reference.view(torch.int16))
    assert torch.equal(candidate.view(torch.int16), reference.view(torch.int16))
    checksums = [torch.empty(65536, device="cuda", dtype=torch.int32) for _ in range(2)]
    samples = {False: [], True: []}
    for repeat in range(8):
        for fast in (False, True) if repeat % 2 == 0 else (True, False):
            start, stop = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            library.run(raw, checksums[int(fast)], fast, args.iterations)
            stop.record()
            stop.synchronize()
            if repeat >= 2:
                samples[fast].append(start.elapsed_time(stop))
    assert torch.equal(*checksums)
    baseline = statistics.median(samples[False])
    candidate = statistics.median(samples[True])
    report = dict(
        scope="Exhaustive reader bits and conversion cost; not engine acceptance",
        pairs_per_position=65536,
        byte_positions=8,
        iterations=args.iterations,
        original_ms=samples[False],
        prototype_ms=samples[True],
        original_median_ms=baseline,
        prototype_median_ms=candidate,
        candidate_over_baseline=candidate / baseline,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
