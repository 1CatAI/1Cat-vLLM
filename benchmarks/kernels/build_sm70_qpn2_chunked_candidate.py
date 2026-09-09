# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# CUDA source anchors retain their original spelling.
# ruff: noqa: E501
"""Build a private two-chunk QPN2 publisher with independent packet channels.

The existing generator supplies the unchanged dot-product arithmetic and
packet protocol. Only column indexing and the consumer's output addressing
change. A caller must complete each local publisher before its consumer and
join both consumers before the next dependent projection. This does not
install a model route or change the communicator's registered storage.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    directory = args.output_dir.resolve()
    template_dir = directory / "template"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("build_sm70_qpn2_publish_candidate.py")),
            "--output-dir",
            str(template_dir),
        ],
        check=True,
    )
    template = (template_dir / "sources/qpn2-publish.cu").read_text()
    producer_start = template.index(
        "template <int SplitK, int NAcc, int RowTiles = 1>\n"
        "__global__ void nvfp4_qpn2_publish_sm70_kernel"
    )
    producer_end = template.index("\n}\n\nnamespace vllm {", producer_start)
    producer = template[producer_start:producer_end]
    producer = producer.replace(
        "nvfp4_qpn2_publish_sm70_kernel", "nvfp4_qpn2_chunk_publish_sm70_kernel"
    )
    producer = replace_once(
        producer,
        "const uint32_t* epochs) {",
        "const uint32_t* epochs, int column_start, int chunk_columns) {",
    )
    producer = replace_once(
        producer,
        "const int tile = blockIdx.x;",
        "const int tile = blockIdx.x + column_start / 32;",
    )
    producer = replace_once(
        producer,
        "const int packed_index = output_index / P::size;",
        "const int packed_index = (output_row * chunk_columns + tile * 32 + output_col - column_start) / P::size;",
    )
    consumer_start = template.index("template <int ngpus>\n__global__", producer_end)
    consumer_end = template.index(
        "\n}\n\nvllm::RankData qpn2_peer_pointers", consumer_start
    )
    consumer = template[consumer_start:consumer_end].replace(
        "qpn2_consume_published", "qpn2_consume_chunk"
    )
    consumer = replace_once(
        consumer,
        "int packed_size) {",
        "int packed_size, int column_start, int chunk_columns) {",
    )
    consumer = replace_once(
        consumer,
        "reinterpret_cast<P*>(output)[offset] =",
        "reinterpret_cast<P*>(output)[(offset / (chunk_columns / P::size)) * (5120 / P::size) + column_start / P::size + offset % (chunk_columns / P::size)] =",
    )
    # Keep separately named controls available without colliding with the
    # frozen production publisher loaded in the same benchmark process.
    source = template.replace("_qpn2_candidate", "_qpn2_chunked_base")
    source += "\nnamespace {\n" + producer + "\n}\n"
    source += "\nnamespace vllm {\n" + consumer + "\n}\n"
    source += r"""
void qpn2_chunk_initialize(torch::Tensor anchor, int64_t pointer) {
  TORCH_CHECK(anchor.is_cuda() && pointer != 0 && pointer % 16 == 0,
              "same-device anchor and aligned local channel required");
  const at::cuda::OptionalCUDAGuard guard(device_of(anchor));
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto* local = reinterpret_cast<char*>(pointer);
  C10_CUDA_CHECK(cudaMemsetAsync(local, 0,
      vllm::kSm70Tp4PushAllreduceSignalBytes, stream));
  C10_CUDA_CHECK(cudaMemsetAsync(local + vllm::kSm70Tp4PushAllreduceSignalBytes,
      vllm::kSm70Tp4PushAllreduceSentinelByte,
      vllm::kSm70Tp4PushAllreduceGenericBufferBytes -
          vllm::kSm70Tp4PushAllreduceSignalBytes, stream));
}

void qpn2_chunk_publish(torch::Tensor out, torch::Tensor input,
    torch::Tensor codes, torch::Tensor scales, double global_scale,
    int64_t split_k, int64_t nacc, std::vector<int64_t> pointers,
    int64_t rank, int64_t column_start) {
  check_qpn2_tensors(out, input, codes, scales, false);
  const int k = input.size(1);
  TORCH_CHECK(input.size(0) == 8 && out.size(1) == 5120 && nacc == 2 &&
      ((k == 1536 && split_k == 8) || (k == 4352 && split_k == 16)),
      "exact TP4 q8 row projection required");
  TORCH_CHECK(column_start == 0 || column_start == 2560,
              "two complete 2560-column chunks required");
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto peers = qpn2_peer_pointers(pointers, rank);
  const auto* in = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* dst = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const auto* epochs = reinterpret_cast<const uint32_t*>(peers.ptrs[rank]);
  if (split_k == 8)
    nvfp4_qpn2_chunk_publish_sm70_kernel<8, 2, 1><<<80, 256, 0, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), in, dst,
        5120, k, 8, global_scale, peers, rank, epochs, column_start, 2560);
  else
    nvfp4_qpn2_chunk_publish_sm70_kernel<16, 2, 1><<<80, 512, 0, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), in, dst,
        5120, k, 8, global_scale, peers, rank, epochs, column_start, 2560);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qpn2_chunk_consume(torch::Tensor out, std::vector<int64_t> pointers,
                       int64_t rank, int64_t column_start) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16 &&
      out.is_contiguous() && out.dim() == 2 && out.size(0) == 8 &&
      out.size(1) == 5120, "exact q8 FP16 output required");
  TORCH_CHECK(column_start == 0 || column_start == 2560,
              "two complete 2560-column chunks required");
  const at::cuda::OptionalCUDAGuard guard(device_of(out));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto peers = qpn2_peer_pointers(pointers, rank);
  vllm::qpn2_consume_chunk<4><<<20, 128, 0, stream>>>(
      peers.ptrs[rank], nullptr,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), rank,
      2560, column_start, 2560);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY_FRAGMENT(_qpn2_chunked, ops) {
  ops.def("initialize(Tensor anchor, int pointer) -> ()");
  ops.impl("initialize", torch::kCUDA, &qpn2_chunk_initialize);
  ops.def("publish(Tensor(a!) out, Tensor input, Tensor codes, Tensor scales, float global_scale, int split_k, int nacc, int[] pointers, int rank, int column_start) -> ()");
  ops.impl("publish", torch::kCUDA, &qpn2_chunk_publish);
  ops.def("consume(Tensor(a!) out, int[] pointers, int rank, int column_start) -> ()");
  ops.impl("consume", torch::kCUDA, &qpn2_chunk_consume);
}
"""
    path = template_dir / "sources/qpn2-chunked.cu"
    path.write_text(source)
    flags = [
        "-O3",
        "-lineinfo",
        "-gencode=arch=compute_70,code=sm_70",
        "-DVLLM_NVFP4_QPN2_STANDALONE",
        "-DVLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE",
        "-Xptxas=-v",
    ]
    manifest = dict(
        source=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        template_manifest=json.loads((template_dir / "manifest.json").read_text()),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        repository=str(root),
        chunks=2,
        columns_per_chunk=2560,
        separate_channels=True,
        producer_waits=False,
        local_producer_event_required=True,
        flags=flags,
        installed=False,
    )
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir(exist_ok=True)
        library = load(
            name="qpn2_chunked_candidate",
            sources=[str(path)],
            build_directory=str(build),
            extra_cuda_cflags=flags,
            is_python_module=False,
            verbose=True,
        )
        manifest.update(
            library=str(library),
            library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
