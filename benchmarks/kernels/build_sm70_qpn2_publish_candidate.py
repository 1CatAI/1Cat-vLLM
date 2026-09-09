# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Literal CUDA templates and source anchors retain their original spelling.
# ruff: noqa: E501
"""Generate a private SM70 QPN2 publisher and matching consumer for experiments.

The generator extracts production arithmetic and the existing packet protocol,
then changes only publication placement and parameter passing. It deliberately
fails if its source anchors change. Its generated source/DSO is a benchmark
candidate, not an installed vLLM operator. Use one candidate library per process.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--packed-input",
        action="store_true",
        help="Read private [K/16, 8, 16] input in the q8 publisher only",
    )
    parser.add_argument(
        "--use-fast-math",
        action="store_true",
        help="Reproduce historical experiments; changes gated SiLU math",
    )
    args = parser.parse_args()
    cuda_flags = [
        "-O3",
        "-lineinfo",
        "-gencode=arch=compute_70,code=sm_70",
        "-DVLLM_NVFP4_QPN2_STANDALONE",
        "-DVLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE",
        "-Xptxas=-v",
    ]
    if args.use_fast_math:
        cuda_flags.append("--use_fast_math")
    A = args.output_dir.resolve()
    D = A / "sources"
    D.mkdir(parents=True, exist_ok=True)
    W = Path(__file__).resolve().parents[2]
    for name in (
        "custom_all_reduce.cuh",
        "cub_helpers.h",
        "sm70_tile_runtime_signal.cuh",
    ):
        shutil.copy2(W / "csrc" / name, D / name)
    source = (W / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu").read_text()
    start = source.index(
        "template <int SplitK, int NAcc, int RowTiles = 1>\n__global__ void nvfp4_qpn2_sm70_kernel"
    )
    end = source.index(
        "template <int SplitK, int NAcc, int RowTiles = 1>\n__global__ void nvfp4_qpn2_gated_sm70_kernel",
        start,
    )
    producer = source[start:end].replace(
        "nvfp4_qpn2_sm70_kernel", "nvfp4_qpn2_publish_sm70_kernel"
    )
    if args.packed_input:
        old = """const half* input_row = input + static_cast<size_t>(row) * k;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);"""
        assert producer.count(old) == 1
        producer = producer.replace(
            old,
            """const half* input_row = input + static_cast<size_t>(group) * 128 + row * 16;
        input01 = *reinterpret_cast<const uint4*>(input_row);
        input23 = *reinterpret_cast<const uint4*>(input_row + 8);""",
        )
    old = "int m, float global_scale) {"
    assert producer.count(old) == 1
    producer = producer.replace(
        old, "int m, float global_scale, vllm::RankData peers, int rank) {"
    )
    old = """      output[static_cast<size_t>(output_row) * n + tile * 32 + output_col] =
          __float2half(value);"""
    assert producer.count(old) == 1
    producer = producer.replace(
        old,
        """      const int output_index = output_row * n + tile * 32 + output_col;
      const half rounded = __float2half(value);
      output[output_index] = rounded;
      // Every active warp owns one complete 32-column row fragment. Pack
      // eight adjacent FP16 results without modifying their payload bits.
      const unsigned bits = __half_as_ushort(rounded);
      const unsigned pair = bits | (__shfl_down_sync(0xffffffffu, bits, 1) << 16);
      const uint4 packet = make_uint4(
          pair, __shfl_down_sync(0xffffffffu, pair, 2),
          __shfl_down_sync(0xffffffffu, pair, 4),
          __shfl_down_sync(0xffffffffu, pair, 6));
      if ((lane & 7) == 0) {
        using P = vllm::packed_t<half>::P;
        P payload = *reinterpret_cast<const P*>(&packet);
#pragma unroll
        for (int i = 0; i < P::size; ++i)
          vllm::sm70_push_escape_sentinel(payload.data[i]);
        const int packed_index = output_index / P::size;
        const int consumer_block = packed_index / vllm::kSm70Tp4PushAllreduceThreads;
        const auto* epochs = reinterpret_cast<const uint32_t*>(peers.ptrs[rank]);
        const uint32_t epoch = epochs[consumer_block];
        constexpr int stride = vllm::kSm70Tp4PushAllreduceMaxBytes / sizeof(P);
        const int epoch_offset = (epoch * 4 + rank) * stride;
#pragma unroll
        for (int peer = 0; peer < 4; ++peer) {
          auto* base = const_cast<char*>(reinterpret_cast<const char*>(peers.ptrs[peer]));
          void* destination = base + vllm::kSm70Tp4PushAllreduceSignalBytes + epoch_offset * sizeof(P);
          vllm::sm70_push_store_volatile_16b(payload, destination, packed_index);
        }
      }""",
    )
    header = (D / "custom_all_reduce.cuh").read_text()
    start_c = header.index(
        "template <int ngpus>\n__global__ void __launch_bounds__(1024, 1)\n    sm70_cross_device_reduce_1stage_push("
    )
    end_c = header.index(
        "template <int ngpus>\n__global__ void __launch_bounds__(1024, 1)\n    sm70_cross_device_reduce_sum2_1stage_push(",
        start_c,
    )
    consumer = header[start_c:end_c].replace(
        "sm70_cross_device_reduce_1stage_push", "qpn2_consume_published"
    )
    start_publish = consumer.index(
        "    P value = reinterpret_cast<const P*>(input)[offset];"
    )
    end_publish = consumer.index("    P peer_values[ngpus];", start_publish)
    consumer = consumer[:start_publish] + consumer[end_publish:]
    source = '#include "custom_all_reduce.cuh"\n' + source
    # Original producer arithmetic and standalone namespace stay available as
    # independent controls. The new code uses the identical header/constants as
    # the isolated communicator that owns the IPC buffers.
    source += "\nnamespace {\n" + producer + "\n}\n"
    source += "\nnamespace vllm {\n" + consumer + "\n}\n"
    source += r"""
vllm::RankData qpn2_peer_pointers(const std::vector<int64_t>& pointers, int64_t rank) {
  TORCH_CHECK(pointers.size() == 4 && rank >= 0 && rank < 4, "TP4 pointers/rank required");
  vllm::RankData peers{};
  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(pointers[i] != 0 && pointers[i] % 16 == 0, "invalid IPC pointer");
    peers.ptrs[i] = reinterpret_cast<void*>(pointers[i]);
  }
  return peers;
}

void qpn2_publish(torch::Tensor out, torch::Tensor input, torch::Tensor codes,
                  torch::Tensor scales, double global_scale, int64_t split_k,
                  int64_t nacc, std::vector<int64_t> pointers, int64_t rank) {
  check_qpn2_tensors(out, input, codes, scales, false);
  const int k = input.size(1);
  TORCH_CHECK(input.size(0) == 8 && out.size(1) == 5120 && nacc == 2 &&
      ((k == 1536 && split_k == 8) || (k == 4352 && split_k == 16)), "exact TP4 q8 row projection required");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto peers = qpn2_peer_pointers(pointers, rank);
  const auto* in = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* dst = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  if (split_k == 8)
    nvfp4_qpn2_publish_sm70_kernel<8, 2, 1><<<160, 256, 0, stream>>>(codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), in, dst, 5120, k, 8, global_scale, peers, rank);
  else
    nvfp4_qpn2_publish_sm70_kernel<16, 2, 1><<<160, 512, 0, stream>>>(codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), in, dst, 5120, k, 8, global_scale, peers, rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qpn2_consume(torch::Tensor projected, torch::Tensor out,
                  std::vector<int64_t> pointers, int64_t rank) {
  TORCH_CHECK(projected.is_cuda() && out.is_cuda() && projected.device() == out.device() &&
      projected.scalar_type() == torch::kFloat16 && out.scalar_type() == torch::kFloat16 &&
      projected.is_contiguous() && out.is_contiguous() && projected.sizes() == out.sizes() &&
      out.dim() == 2 && out.size(0) == 8 && out.size(1) == 5120, "exact q8 FP16 output required");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(projected));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto peers = qpn2_peer_pointers(pointers, rank);
  vllm::qpn2_consume_published<4><<<80, 128, 0, stream>>>(peers, nullptr,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), rank, 5120);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY_FRAGMENT(_qpn2_candidate, ops) {
  ops.def("publish(Tensor(a!) out, Tensor input, Tensor codes, Tensor scales, float global_scale, int split_k, int nacc, int[] pointers, int rank) -> ()");
  ops.impl("publish", torch::kCUDA, &qpn2_publish);
  ops.def("consume(Tensor projected, Tensor(a!) out, int[] pointers, int rank) -> ()");
  ops.impl("consume", torch::kCUDA, &qpn2_consume);
}
"""
    # Keep the local pointer out of dynamically indexed by-value arrays.
    # Otherwise ptxas materializes a 64-byte local stack in both kernels.
    old = "int m, float global_scale, vllm::RankData peers, int rank) {"
    assert source.count(old) == 1
    source = source.replace(
        old,
        "int m, float global_scale, vllm::RankData peers, int rank, const uint32_t* epochs) {",
    )
    old = "        const auto* epochs = reinterpret_cast<const uint32_t*>(peers.ptrs[rank]);\n"
    assert source.count(old) == 1
    source = source.replace(old, "")
    old = "8, global_scale, peers, rank);"
    assert source.count(old) == 2
    source = source.replace(
        old,
        "8, global_scale, peers, rank, reinterpret_cast<const uint32_t*>(peers.ptrs[rank]));",
    )
    source = source.replace(
        "qpn2_consume_published(RankData push_buffers,",
        "qpn2_consume_published(const void* local_pointer,",
    )
    source = source.replace(
        "reinterpret_cast<const char*>(push_buffers.ptrs[rank])",
        "reinterpret_cast<const char*>(local_pointer)",
    )
    source = source.replace(
        "qpn2_consume_published<4><<<80, 128, 0, stream>>>(peers, nullptr,",
        "qpn2_consume_published<4><<<80, 128, 0, stream>>>(peers.ptrs[rank], nullptr,",
    )
    if args.packed_input:
        source = source.replace("_qpn2_candidate", "_qpn2_packed_row")
        source = source.replace("nvfp4_qpn2_", "packedrow_nvfp4_qpn2_")
        source = source.replace("qpn2_publish", "qpn2_packedrow_publish")
        source = source.replace("qpn2_consume", "qpn2_packedrow_consume")
        source = source.replace("qpn2_peer_pointers", "qpn2_packedrow_peer_pointers")
    shutil.copy2(
        W / "csrc/sm70_turbomind/ops/LICENSE.v100-skinny", D / "LICENSE.v100-skinny"
    )
    p = D / "qpn2-publish.cu"
    p.write_text(source)
    manifest = {
        str(f.name): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in D.iterdir()
        if f.is_file()
    }
    (A / "manifest.json").write_text(
        json.dumps(
            dict(
                sources=manifest,
                extra_cuda_cflags=cuda_flags,
                math_mode="fast" if args.use_fast_math else "default",
                publisher_input_layout="[K/16, 8, 16]"
                if args.packed_input
                else "[8, K]",
                input_sources={
                    name: hashlib.sha256((W / name).read_bytes()).hexdigest()
                    for name in (
                        "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu",
                        "csrc/custom_all_reduce.cuh",
                    )
                },
                hypothesis="Publish completed FP16 output fragments during QPN2 epilogue, without polling or waiting in producer CTAs. The separate consumer keeps the established FP32 rank order, two epochs and sentinel cleanup. This differs from the previously rejected producer-poll fusion.",
            ),
            indent=2,
        )
    )
    print(p)

    if args.build:
        from torch.utils.cpp_extension import load

        build_dir = A / "build"
        build_dir.mkdir(exist_ok=True)
        library = load(
            name="qpn2_publish_candidate",
            sources=[str(p)],
            build_directory=str(build_dir),
            extra_cuda_cflags=cuda_flags,
            is_python_module=False,
            verbose=True,
        )
        result = json.loads((A / "manifest.json").read_text())
        result["library"] = str(library)
        result["library_sha256"] = hashlib.sha256(
            Path(library).read_bytes()
        ).hexdigest()
        (A / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
