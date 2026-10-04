# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a research D128 draft sliding-window split-KV operator."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def partition_ranges(length: int, splits: int = 40) -> list[tuple[int, int]]:
    """Partition the union of eight inclusive 2048-wide query windows."""
    first_tile = max(0, length - 8 - 2047) // 32
    tiles = (length + 31) // 32 - first_tile
    active = 1 if length <= 128 else min(splits, (length + 63) // 64)
    base, extra = divmod(tiles, active)
    return [
        (
            (first_tile + i * base + min(i, extra)) * 32,
            min(
                length,
                (first_tile + i * base + min(i, extra) + base + (i < extra)) * 32,
            ),
        )
        for i in range(active)
    ]


def check_partition_contract() -> None:
    for length in (
        8,
        127,
        128,
        129,
        1024,
        2047,
        2048,
        2049,
        2055,
        4095,
        4096,
        4097,
        131071,
        131072,
        262143,
        262144,
    ):
        ranges = partition_ranges(length)
        assert all(a < b for a, b in ranges)
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
        scanned = set(range(ranges[0][0], ranges[-1][1]))
        for row in range(8):
            position = length - 8 + row
            expected = set(range(max(0, position - 2047), length))
            actual = {k for k in scanned if k >= position - 2047}
            assert actual == expected, (length, row)


def draft_source(source: str) -> str:
    end = source.index("at::Tensor private_grouped_e4m3_fp32_paged(")
    source = source[:end]
    constants = {
        "kGroupedVerifyHeads = 6": "kGroupedVerifyHeads = 8",
        "kGroupedVerifyHeadDim = 256": "kGroupedVerifyHeadDim = 128",
        "kGroupedVerifyRows = 48": "kGroupedVerifyRows = 32",
        "kGroupedVerifyQStride = 264": "kGroupedVerifyQStride = 136",
        "kGroupedVerifyKVStride = 264": "kGroupedVerifyKVStride = 136",
        "kGroupedVerifyThreads = 512": "kGroupedVerifyThreads = 256",
        "kGroupedVerifyQ8Splits = 80": "kGroupedVerifyQ8Splits = 40",
    }
    for old, new in constants.items():
        if source.count(old) != 1:
            raise ValueError(f"Expected one declaration: {old}")
        source = source.replace(old, new)
    source = source.replace(
        "return kv_idx <= prefix_kv_len + token_idx;",
        "return kv_idx >= max(0, prefix_kv_len + token_idx - 2047) &&\n"
        "         kv_idx < prefix_kv_len + query_len;",
    )
    start = source.index(
        "__launch_bounds__(kGroupedVerifyThreads, 1) void "
        "flash_attention_grouped_verify_e5m2_partial_kernel("
    )
    full_q8 = source.index("void flash_attention_grouped_verify_e4m3_full_q8_kernel(")
    end = source.rfind("template <int MAX_QUERY_TOKENS, bool TWO_PASS", start, full_q8)
    body = source[start:end]
    old = (
        "  const int total_tiles =\n"
        "      (total_kv + kGroupedVerifyBlockN - 1) / kGroupedVerifyBlockN;"
    )
    if body.count(old) != 1:
        raise ValueError("Expected one generic split range")
    body = body.replace(
        old,
        "  const int first_tile = max(0, total_kv - query_len - 2047) / 32;\n"
        "  const int total_tiles =\n"
        "      (total_kv + kGroupedVerifyBlockN - 1) / "
        "kGroupedVerifyBlockN - first_tile;",
    ).replace(
        "const int split_start = split_tile_start * kGroupedVerifyBlockN;",
        "const int split_start = (first_tile + split_tile_start) * "
        "kGroupedVerifyBlockN;",
    )
    body = body.replace(
        "tile_page_offset, 0, page_block_size, 0,",
        "tile_page_offset, 0, page_block_size, head_group,",
    )
    # FP16 needs no conversion lookup table.
    old = (
        "  __shared__ uint16_t e4m3_lut[256];\n"
        "  if (tid < 256)\n"
        "    e4m3_lut[tid] = fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(tid));"
    )
    body = body.replace(old, "  const uint16_t* e4m3_lut = nullptr;")
    first = body.index("    static_assert(COMPENSATE_P && kGroupedVerifyWarps == 16,")
    last = body.index("    __syncthreads();\n  }", first)
    # The non-compensated online update already rescales every output tile.
    # Consume the prefetched V panel while sharing each B fragment across GQA.
    body = (
        body[:first]
        + """
#pragma unroll
    for (int k_offset = 0; k_offset < kGroupedVerifyBlockN; k_offset += 16) {
      volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::row_major> value;
      volta::load_matrix_sync(value,
          shared_values + k_offset * kGroupedVerifyKVStride + warp_id * 16,
          kGroupedVerifyKVStride);
#pragma unroll
      for (int fragment_idx = 0;
           fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
        volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
            probability;
        load_grouped_a_swizzled(probability,
            shared_probs + fragment_idx * 16 * kGroupedVerifyProbStride, k_offset);
        volta::mma_sync(output_fragments[fragment_idx], probability, value,
                        output_fragments[fragment_idx]);
      }
    }
"""
        + body[last:]
    )
    source = source[:start] + body + source[end:]
    first = source.index("void flash_attention_grouped_verify_e5m2_combine_kernel(")
    before, combine = source[:first], source[first:]
    old = "  int total_kv = seq_lens[0];"
    if combine.count(old) != 1:
        raise ValueError("Expected one combine sequence length")
    combine = combine.replace(
        old,
        old
        + """
  // A padded graph request leaves every partial unwritten. Do not read it.
  if (total_kv <= 0) {
    for (int d = threadIdx.x; d < kGroupedVerifyHeadDim; d += blockDim.x)
      out[(token_idx * kGroupedVerifyHeads + head_idx) *
          kGroupedVerifyHeadDim + d] = __float2half_rn(0.0f);
    return;
  }
""",
    )
    source = before + combine
    source = "#include <torch/extension.h>\n" + source
    source += """
at::Tensor private_dflash_sliding_fp16_paged(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, const at::Tensor& blocks, const at::Tensor& lengths,
    at::Tensor& partial, at::Tensor& lse, float scale) {
  const int64_t batch = blocks.size(0);
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.is_contiguous()
      && q.sizes() == at::IntArrayRef({batch * 8, 8, 128}), "Expected q8/H8/D128");
  TORCH_CHECK(k.dim() == 4 && k.size(1) > 0 && k.size(1) % 16 == 0
      && k.size(2) == 2 && k.size(3) == 128 && k.scalar_type() == at::kHalf
      && v.scalar_type() == at::kHalf && v.sizes() == k.sizes(), "Expected FP16 Hkv2");
  TORCH_CHECK(blocks.dim() == 2 && batch >= 1 && batch <= 4
      && blocks.scalar_type() == at::kInt && blocks.is_contiguous()
      && lengths.sizes() == at::IntArrayRef({batch})
      && lengths.scalar_type() == at::kInt && lengths.is_contiguous(),
      "Expected lengths");
  TORCH_CHECK(out.sizes() == q.sizes() && out.scalar_type() == at::kHalf
      && out.is_contiguous() && partial.scalar_type() == at::kFloat
      && lse.scalar_type() == at::kFloat && partial.is_contiguous()
      && lse.is_contiguous()
      && partial.sizes() == at::IntArrayRef({batch, 40, 8, 8, 128})
      && lse.sizes() == at::IntArrayRef({batch, 40, 8, 8}), "Expected split buffers");
  for (auto tensor : {&k, &v, static_cast<const at::Tensor*>(&out), &blocks,
                     &lengths, static_cast<const at::Tensor*>(&partial),
                     static_cast<const at::Tensor*>(&lse)})
    TORCH_CHECK(tensor->device() == q.device(), "Expected one device");
  for (auto tensor : {&k, &v}) {
    TORCH_CHECK(tensor->stride(3) == 1, "Expected contiguous head dimension");
    for (int dim = 0; dim < 3; ++dim)
      TORCH_CHECK(tensor->stride(dim) % 8 == 0, "Expected aligned KV strides");
  }
  c10::cuda::CUDAGuard guard(q.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0, "SM70 only");
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto kernel = flash_attention_grouped_verify_e5m2_partial_kernel<
      8, false, 0, false, false, false, flash_v100::KV_CACHE_DTYPE_FP16,
      false, float, false, false, false>;
  constexpr int smem = sizeof(GroupedVerifySmem)
      + kGroupedVerifyRows * kGroupedVerifyProbStride * sizeof(__half)
      + kGroupedVerifyBlockN * kGroupedVerifyKVStride * sizeof(__half);
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  kernel<<<dim3(2, 40, batch), 256, smem, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()), k.data_ptr(), v.data_ptr(),
      blocks.data_ptr<int>(), lengths.data_ptr<int>(), partial.data_ptr<float>(),
      lse.data_ptr<float>(), 8, blocks.size(1), k.size(1),
      k.stride(0), k.stride(1), k.stride(2),
      v.stride(0), v.stride(1), v.stride(2), scale, 1.0f, nullptr, batch, nullptr);
  flash_attention_grouped_verify_e5m2_combine_kernel<8, false, float, false>
      <<<dim3(8, 8, batch), 128, 0, stream>>>(
          partial.data_ptr<float>(), lse.data_ptr<float>(), lengths.data_ptr<int>(),
          reinterpret_cast<__half*>(out.data_ptr()), 8, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &private_dflash_sliding_fp16_paged);
}
"""
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    check_partition_contract()
    repo = Path(__file__).resolve().parents[2]
    root = repo / "csrc/attention/sm70_grouped_long"
    original = root / "kernel/grouped-attention.cu"
    source = draft_source(original.read_text())
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    for name in ("include", "kernel"):
        (sources / name).mkdir(parents=True, exist_ok=True)
        for pattern in ("*.h", "*.cuh"):
            for header in (root / name).glob(pattern):
                shutil.copy2(header, sources / name)
    path = sources / "kernel/draft-sliding.cu"
    path.write_text(source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = "sm70_dflash_sliding_" + digest[:12]
    manifest = {
        "source_sha": args.source_sha,
        "module_name": module_name,
        "source_sha256": digest,
        "input_source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "shape": "q8/H8/Hkv2/D128",
        "window_left": 2047,
        "window_right": 2047,
        "splits": 40,
        "dtype": "FP16 KV with FP32 partial outputs",
        "mask_partition_contract": "Passed CPU boundary enumeration",
        "scope": "Independent research operator; GPU and model admission pending",
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir(exist_ok=True)
        module = load(
            name=module_name,
            sources=[str(path)],
            build_directory=str(build),
            extra_include_paths=[str(sources / "include"), str(sources / "kernel")],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "-gencode=arch=compute_70,code=sm_70",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-lineinfo",
                "-Xptxas=-v",
            ],
            verbose=True,
        )
        library = Path(module.__file__)
        manifest.update(
            library=str(library),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
