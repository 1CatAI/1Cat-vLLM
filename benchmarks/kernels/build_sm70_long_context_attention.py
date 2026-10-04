# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build isolated long-context q8 attention candidates from current source.

No installed operator or production route is replaced. Each library has a
source-derived module name and exports the existing paged-attention interface.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import regex as re


def candidate_source(source: str, variant: str) -> str:
    start = source.index(
        "__launch_bounds__(kGroupedVerifyThreads, 1) void "
        "flash_attention_grouped_verify_e4m3_full_q8_kernel("
    )
    end = source.index(
        "void flash_attention_grouped_verify_e5m2_combine_kernel(", start
    )
    body = source[start:end]
    swizzled_helper = None
    if variant == "swizzled-q":
        helper_start = source.index(
            "template <bool COMPENSATE = false>\n"
            "__device__ __forceinline__ void grouped_verify_qk("
        )
        helper_end = source.index(
            "__device__ __forceinline__ void grouped_verify_scale_output_fragment(",
            helper_start,
        )
        swizzled_helper = source[helper_start:helper_end].replace(
            "void grouped_verify_qk(", "void grouped_verify_qk_swizzled("
        )
        old = (
            "    volta::load_matrix_sync(\n"
            "        q_fragment, shared_q + m_tile * 16 * "
            "kGroupedVerifyQStride + k_offset,\n"
            "        kGroupedVerifyQStride);"
        )
        if swizzled_helper.count(old) != 1:
            raise ValueError("Expected one Q fragment load")
        swizzled_helper = swizzled_helper.replace(
            old,
            "    load_grouped_a_swizzled(\n"
            "        q_fragment, shared_q + m_tile * 16 * "
            "kGroupedVerifyHeadDim, k_offset);",
        )
        old = "shared_q_vec[row * kSharedQVecsPerRow + vec_col]"
        if body.count(old) != 2:
            raise ValueError("Expected two Q panel stores")
        body = body.replace(
            old, "shared_q_vec[grouped_a_offset(row, vec_col * 8, 256) / 8]"
        ).replace("grouped_verify_qk<", "grouped_verify_qk_swizzled<")
    if variant == "single-width-kv":
        body = body.replace("(!ROW_SEQLENS || PAIR_E4M3)", "true")
    if variant in ("bit-decode", "bit-decode-single-pv"):
        old = (
            "  __shared__ uint16_t e4m3_lut[256];\n"
            "  if (tid < 256)\n"
            "    e4m3_lut[tid] = "
            "fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(tid));"
        )
        if body.count(old) != 1:
            raise ValueError("Expected one full-q8 lookup table")
        body = body.replace(old, "  const uint16_t* e4m3_lut = nullptr;")
        body, count = re.subn(r"(PAIR_E4M3\)\)),\s*true>\(", r"\1, false>(", body)
        if count != 3:
            raise ValueError(f"Expected three full-q8 panel loads, found {count}")
        for half in ("lo", "hi"):
            old = f"fp8_e4m3fn_vector_to_half8_lut({half}, e4m3_lut)"
            if body.count(old) != 1:
                raise ValueError("Expected one paired key conversion")
            body = body.replace(old, f"fp8_e4m3fn_vector_to_half8_fast({half})")
    if variant == "uncompensated-qk":
        old = "grouped_verify_qk<COMPENSATE_P>(shared_q, shared_kv, shared_scores,"
        if body.count(old) != 1:
            raise ValueError("Expected one full-q8 QK product")
        body = body.replace(
            old, "grouped_verify_qk<false>(shared_q, shared_kv, shared_scores,"
        )
    if variant in ("single-pv", "bit-decode-single-pv"):
        old = (
            "        load_grouped_a_swizzled(\n"
            "            probability_fragment,\n"
            "            shared_prob_residual + m_tile * 16 * kResidualStride, "
            "k_offset);\n"
            "        volta::mma_sync(tile_fragments[fragment_idx], "
            "probability_fragment,\n"
            "                        residual_value_fragment, "
            "tile_fragments[fragment_idx]);"
        )
        if body.count(old) != 1:
            raise ValueError("Expected one compensated PV residual product")
        body = body.replace(old, "")
        # Keep the shared layout unchanged to isolate the extra PV product.
        # Dead residual-value conversion is removed by the CUDA compiler.
    result = source[:start] + body + source[end:]
    if variant in ("compact-q8", "shared-dual-q8"):
        result = compact_q8_source(result)
    if variant == "shared-dual-q8":
        result = shared_dual_q8_source(result)
    if swizzled_helper is not None:
        anchor = "__device__ __forceinline__ void grouped_verify_scale_output_fragment("
        result = result.replace(anchor, swizzled_helper + anchor, 1)
    if variant == "single-width-kv":
        start = result.index("at::Tensor private_grouped_e4m3_fp32_paged(")
        end = result.index("at::Tensor private_grouped_fp16_fp32_paged(", start)
        entry = result[start:end]
        anchor = "  auto kernel = paired\n"
        if entry.count(anchor) != 1:
            raise ValueError("Expected one paired-KV dispatch")
        entry = entry.replace(anchor, "  paired = false;\n" + anchor)
        result = result[:start] + entry + result[end:]
    return result


def compact_q8_source(source: str) -> str:
    """Screen two resident CTAs with global Q and direct register output."""
    helper_start = source.index(
        "template <bool COMPENSATE = false>\n"
        "__device__ __forceinline__ void grouped_verify_qk("
    )
    anchor = "__device__ __forceinline__ void grouped_verify_scale_output_fragment("
    helper_end = source.index(anchor, helper_start)
    helper = source[helper_start:helper_end].replace(
        "void grouped_verify_qk(", "void grouped_verify_qk_global("
    )
    helper = helper.replace("kGroupedVerifyQStride", "kGroupedVerifyHeadDim")
    source = source.replace(anchor, helper + anchor, 1)
    declaration = """
struct alignas(256) CompactQ8Smem {
  struct {
    struct {
      alignas(16) __half kv[kGroupedVerifyBlockN * kGroupedVerifyKVStride];
      alignas(16) float scores[kGroupedVerifyRows * kGroupedVerifyScoreStride];
      alignas(16) __half probs[kGroupedVerifyRows * kGroupedVerifyProbStride];
    } compute;
  } storage;
  alignas(16) float row_max[kGroupedVerifyRows];
  alignas(16) float row_sum[kGroupedVerifyRows];
  alignas(16) float row_scale[kGroupedVerifyRows];
  alignas(16) int page_ids[kGroupedVerifyPageIdsCapacity];
  alignas(16) uint32_t sparse_token_masks[kGroupedVerifyBlockN / 4];
};
"""
    source = source.replace(anchor, declaration + anchor, 1)
    start = source.index(
        "__launch_bounds__(kGroupedVerifyThreads, 1) void "
        "flash_attention_grouped_verify_e4m3_full_q8_kernel("
    )
    end = source.index("// FP16 KV verifier: one context read", start)
    body = source[start:end]
    body = body.replace(
        "__launch_bounds__(kGroupedVerifyThreads, 1)", "__launch_bounds__(512, 2)"
    )
    body = body.replace("GroupedVerifySmem", "CompactQ8Smem")
    body = body.replace(
        "  __half* shared_q = smem.storage.compute.q;",
        "  const __half* shared_q = q + static_cast<int64_t>(group_idx) * 48 * 256;",
    )
    first = body.index("  constexpr int kVecsPerRow = kGroupedVerifyHeadDim / 8;")
    last = body.index("  if (tid < kGroupedVerifyRows) {", first)
    body = body[:first] + body[last:]
    body = body.replace(
        "grouped_verify_qk<COMPENSATE_P>", "grouped_verify_qk_global<false>"
    )
    first = body.index(
        "    volta::fragment<volta::accumulator, 16, 16, 16, float>\n"
        "        tile_fragments[kGroupedVerifyOutputTilesPerWarp];"
    )
    last = body.index("    __syncthreads();\n  }", first)
    body = (
        body[:first]
        + """
#pragma unroll
    for (int fragment_idx = 0;
         fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
      const int m_tile = fragment_idx;
      volta::fragment<volta::accumulator, 16, 16, 16, float> tile;
      volta::fill_fragment(tile, 0.0f);
#pragma unroll
      for (int k_offset = 0; k_offset < kGroupedVerifyBlockN; k_offset += 16) {
        volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::row_major> value;
        volta::load_matrix_sync(value,
            shared_values + k_offset * kGroupedVerifyKVStride + warp_id * 16,
            kGroupedVerifyKVStride);
        volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
            probability;
        load_grouped_a_swizzled(probability,
            shared_probs + m_tile * 16 * kGroupedVerifyProbStride, k_offset);
        volta::mma_sync(tile, probability, value, tile);
      }
      grouped_verify_add_output_tile(output_fragments[fragment_idx], tile,
                                     smem.row_scale, m_tile * 16);
    }
"""
        + body[last:]
    )
    first = body.index("  // The compute buffers are dead.")
    last = body.index("  if (tid < kGroupedVerifyRows) {", first)
    body = (
        body[:first]
        + """
  const int lane = threadIdx.x & 31;
  const int row_base = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const int col_base = ((lane >> 1) & 1) * 2 + ((lane >> 3) & 1) * 8;
#pragma unroll
  for (int fragment_idx = 0;
       fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = fragment_idx * 16 + row_base + ((i >> 1) & 1) * 2;
      const int col = warp_id * 16 + col_base + (i & 1) + ((i >> 2) & 1) * 4;
      const float scale = smem.row_sum[row] > 0.0f ? v_scale : 0.0f;
      partial_out[(static_cast<int64_t>(split_id) * 48 + row) * 256 + col] =
          output_fragments[fragment_idx].x[i] * scale;
    }
  }
"""
        + body[last:]
    )
    source = source[:start] + body + source[end:]
    source = source.replace(
        "constexpr int kGroupedVerifyQ8Splits = 80;",
        "constexpr int kGroupedVerifyQ8Splits = 160;",
    )
    start = source.index("at::Tensor private_grouped_e4m3_fp32_paged(")
    end = source.index("at::Tensor private_grouped_fp16_fp32_paged(", start)
    entry = (
        source[start:end]
        .replace("{80,", "{160,")
        .replace("{batch_size, 80,", "{batch_size, 160,")
    )
    entry = entry.replace("dim3(1, 80,", "dim3(1, 160,")
    first = entry.index("  C10_CUDA_CHECK(\n      cudaFuncSetAttribute(kernel,")
    entry = (
        entry[:first]
        + """
  const int launch_smem = query_len == 8
      ? sizeof(CompactQ8Smem)
          + kGroupedVerifyRows * kGroupedVerifyProbStride * sizeof(__half)
          + kGroupedVerifyBlockN * kGroupedVerifyKVStride * sizeof(__half)
      : kCompensatedSmemBytes;
"""
        + entry[first:]
    )
    entry = entry.replace(
        "                           kCompensatedSmemBytes));",
        "                           launch_smem));",
    )
    entry = entry.replace(
        "kGroupedVerifyThreads, kCompensatedSmemBytes, stream",
        "kGroupedVerifyThreads, launch_smem, stream",
    )
    entry = entry.replace(
        "reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 ? q : q.clone()",
        "reinterpret_cast<uintptr_t>(q.data_ptr()) % 32 == 0 ? q : q.clone()",
    )
    return source[:start] + entry + source[end:]


def shared_dual_q8_source(source: str) -> str:
    """Keep Q resident; alias dead K/V and score/probability storage."""
    start = source.index("struct alignas(256) CompactQ8Smem {")
    end = source.index("__device__ __forceinline__ void grouped_verify_scale", start)
    declaration = source[start:end]
    declaration = declaration.replace(
        "      alignas(16) __half kv[",
        "      alignas(16) __half q[kGroupedVerifyRows * kGroupedVerifyHeadDim];\n"
        "      alignas(16) __half kv[",
    ).replace(
        "      alignas(16) float scores[kGroupedVerifyRows * "
        "kGroupedVerifyScoreStride];\n"
        "      alignas(16) __half probs[kGroupedVerifyRows * "
        "kGroupedVerifyProbStride];",
        "      union {\n"
        "        alignas(16) float scores[kGroupedVerifyRows * "
        "kGroupedVerifyScoreStride];\n"
        "        alignas(16) __half probs[kGroupedVerifyRows * "
        "kGroupedVerifyProbStride];\n"
        "      };",
    )
    source = source[:start] + declaration + source[end:]
    start = source.index(
        "template <bool COMPENSATE = false>\n"
        "__device__ __forceinline__ void grouped_verify_qk_global("
    )
    end = source.index("struct alignas(256) CompactQ8Smem", start)
    helper = source[start:end].replace(
        "grouped_verify_qk_global(", "grouped_verify_qk_resident("
    )
    old = (
        "    volta::load_matrix_sync(\n"
        "        q_fragment, shared_q + m_tile * 16 * "
        "kGroupedVerifyHeadDim + k_offset,\n"
        "        kGroupedVerifyHeadDim);"
    )
    if helper.count(old) != 1:
        raise ValueError("Expected one global Q loader")
    helper = helper.replace(
        old,
        "    load_grouped_a_swizzled(q_fragment,\n"
        "        shared_q + m_tile * 16 * kGroupedVerifyHeadDim, k_offset);",
    )
    source = source[:end] + helper + source[end:]
    start = source.index(
        "__launch_bounds__(512, 2) void "
        "flash_attention_grouped_verify_e4m3_full_q8_kernel("
    )
    end = source.index("// FP16 KV verifier: one context read", start)
    body = source[start:end]
    body = body.replace(
        "  const __half* shared_q = q + static_cast<int64_t>(group_idx) * 48 * 256;",
        "  __half* shared_q = smem.storage.compute.q;",
    )
    body = body.replace(
        "  __half* shared_values = shared_prob_residual + "
        "kGroupedVerifyRows * kResidualStride;",
        "  __half* shared_values = shared_kv;",
    )
    first = body.index("  if (tid < kGroupedVerifyRows) {")
    body = (
        body[:first]
        + """
  for (int idx = tid; idx < 48 * 32; idx += 512) {
    const int row = idx / 32, vec = idx % 32;
    const int64_t query_row = static_cast<int64_t>(group_idx) * 48 + row;
    reinterpret_cast<uint4*>(shared_q)[grouped_a_offset(row, vec * 8, 256) / 8] =
        __ldg(reinterpret_cast<const uint4*>(q) + query_row * 32 + vec);
  }
"""
        + body[first:]
    )
    body = body.replace(
        "grouped_verify_qk_global<false>", "grouped_verify_qk_resident<false>"
    )
    first = body.index(
        "    if (warp_id >= kGroupedVerifyQKWarps) {",
        body.index("grouped_verify_qk_resident<false>"),
    )
    last = body.index("\n\n    if constexpr (TWO_PASS)", first)
    body = (
        body[:first]
        + """
    __syncthreads();
    load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                         kGroupedVerifyThreads, KV_DTYPE, PAIR_E4M3, true>(
        shared_values, v_cache, page_ids, valid_k_rows, kPanelStrideVec,
        kSharedStrideVec, tile_page_offset, 0, page_block_size, 0,
        v_block_stride, v_token_stride, v_head_stride, 0, tid, e4m3_lut);
    for (int idx = tid + valid_k_rows * kSharedStrideVec;
         idx < kGroupedVerifyBlockN * kSharedStrideVec; idx += 512)
      reinterpret_cast<uint4*>(shared_values)[idx] = make_uint4(0, 0, 0, 0);
    __syncthreads();
"""
        + body[last:]
    )
    first = body.index("        for (int row = warp_id * Values + lane_id / Width;")
    last = body.index("          float maxima[Values];", first)
    block = body[first:last]
    reads = block[block.index("#pragma unroll") :]
    body = (
        body[:first]
        + """
        const int row = warp_id * Values + lane_id / Width;
        float scores[Values];
        bool visible[Values];
        if (row < kGroupedVerifyRows) {
"""
        + reads
        + """
        }
        // Every score must be in registers before any aliased P store.
        __syncthreads();
        if (row < kGroupedVerifyRows) {
"""
        + body[last:]
    )
    old = (
        "            shared_prob_residual[grouped_a_offset(row, col, 32)] =\n"
        "                __float2half_rn((probability - "
        "__half2float(rounded)) * 2048.f);"
    )
    if body.count(old) != 1:
        raise ValueError("Expected one residual probability store")
    body = body.replace(old, "")
    source = source[:start] + body + source[end:]
    old = (
        "      ? sizeof(CompactQ8Smem)\n"
        "          + kGroupedVerifyRows * kGroupedVerifyProbStride * "
        "sizeof(__half)\n"
        "          + kGroupedVerifyBlockN * kGroupedVerifyKVStride * sizeof(__half)"
    )
    if source.count(old) != 1:
        raise ValueError("Expected one compact shared-memory budget")
    return source.replace(old, "      ? sizeof(CompactQ8Smem)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(
            "reference",
            "bit-decode",
            "uncompensated-qk",
            "single-pv",
            "bit-decode-single-pv",
            "swizzled-q",
            "single-width-kv",
            "compact-q8",
            "shared-dual-q8",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha", help="Revision of a transferred source archive")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = repo / "csrc/attention/sm70_grouped_long"
    original = root / "kernel/grouped-attention.cu"
    source = candidate_source(original.read_text(), args.variant)
    # Avoid duplicate registration when reference/candidates share a process.
    source = source[: source.index("// Registered into the shipped FA2 extension")]
    source = "#include <torch/extension.h>\n" + source
    source += r"""
__global__ void private_check_e4m3_decoders(uint32_t* out) {
  const unsigned int i = threadIdx.x;
  const uint16_t pair = static_cast<uint16_t>(i | ((255u - i) << 8));
  out[i] = fp8_e4m3fn_pair_to_half2_bits(pair);
  out[256 + i] = fp8_e4m3fn_pair_to_half2_bits_fast(pair);
}

at::Tensor private_e4m3_decoder_check() {
  auto out = at::empty({2, 256}, at::TensorOptions().device(at::kCUDA).dtype(at::kInt));
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  private_check_e4m3_decoders<<<1, 256, 0, stream>>>(
      reinterpret_cast<uint32_t*>(out.data_ptr<int>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""
    source += (
        "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        '  m.def("run", &private_grouped_e4m3_fp32_paged);\n'
        '  m.def("decoder_check", &private_e4m3_decoder_check);\n}\n'
    )
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    for name in ("include", "kernel"):
        (sources / name).mkdir(parents=True, exist_ok=True)
        for pattern in ("*.h", "*.cuh"):
            for header in (root / name).glob(pattern):
                shutil.copy2(header, sources / name)
    path = sources / "kernel/grouped-attention.cu"
    path.write_text(source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = "sm70_long_attention_" + digest[:12]
    flags = [
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
    ]
    manifest = {
        "source_sha": args.source_sha
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "variant": args.variant,
        "input_source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "source_sha256": digest,
        "module_name": module_name,
        "entrypoint": "run",
        "splits": 160 if args.variant in ("compact-q8", "shared-dual-q8") else 80,
        "arithmetic_change": args.variant
        not in ("reference", "bit-decode", "swizzled-q", "single-width-kv"),
        "source_files": {
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*"))
            if p.is_file()
        },
        "extra_cuda_cflags": flags,
        "scope": "Independent operator screen; model admission pending",
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir(exist_ok=True)
        module = load(
            name=module_name,
            sources=[str(path)],
            build_directory=str(build),
            extra_cuda_cflags=flags,
            extra_include_paths=[str(sources / "include"), str(sources / "kernel")],
            verbose=True,
        )
        library = Path(module.__file__)
        manifest["library"] = str(library)
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
