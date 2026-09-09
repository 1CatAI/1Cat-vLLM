# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a private E4M3 grouped-attention scheduling candidate.

The original per-head arithmetic, FP32 numerator/max/sum workspace and
native input validation remain intact. Candidates change CTA grouping,
loop scheduling or address-equivalent loads. The builder installs no route.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Expected one source anchor: {old}")
    return source.replace(old, new)


def prefetch_values(source: str) -> str:
    """Fill a disjoint V panel with idle QK warps before the existing barrier."""
    start = source.index("__launch_bounds__(kGroupedVerifyThreads, 1) void ")
    end = source.index(
        "void flash_attention_grouped_verify_e5m2_combine_kernel(", start
    )
    partial = source[start:end]
    value_argument = partial.index("        shared_kv, v_cache, page_ids,")
    load_start = partial.rfind("    load_xqa_tc_kv_panel<", 0, value_argument)
    barrier = "    __syncthreads();"
    load_end = partial.index(barrier, value_argument) + len(barrier)
    load = partial[load_start:load_end]
    load = load[: load.rfind(barrier)]
    load = load.replace("shared_kv", "shared_values")
    load = load.replace("kGroupedVerifyThreads", "kValueLoadThreads")
    load = load.replace("idx = tid +", "idx = value_load_tid +")
    load = replace_once(
        load,
        "v_block_stride, v_token_stride, v_head_stride, 0);",
        "v_block_stride, v_token_stride, v_head_stride, 0, value_load_tid);",
    )
    partial = partial[:load_start] + partial[load_end:]
    marker = "  constexpr int kResidualStride = kGroupedVerifyProbStride;"
    partial = replace_once(
        partial,
        marker,
        marker + "\n  __half* shared_values = shared_prob_residual + "
        "kGroupedVerifyRows * kResidualStride;",
    )
    qk = (
        "    grouped_verify_qk<COMPENSATE_P>(shared_q, shared_kv, shared_scores,\n"
        "                                    qk_scale, active_m_tiles);"
    )
    partial = replace_once(
        partial,
        qk,
        qk + "\n    if (warp_id >= kGroupedVerifyQKWarps) {\n"
        "      constexpr int kValueLoadThreads = kGroupedVerifyThreads - "
        "kGroupedVerifyQKWarps * kWarpSize;\n"
        "      const int value_load_tid = tid - kGroupedVerifyQKWarps * kWarpSize;\n"
        + load
        + "    }\n",
    )
    partial = replace_once(
        partial,
        "shared_kv + k_offset * kGroupedVerifyKVStride + d_tile * 16,",
        "shared_values + k_offset * kGroupedVerifyKVStride + d_tile * 16,",
    )
    source = source[:start] + partial + source[end:]
    source = replace_once(
        source,
        "kGroupedVerifyRows * kGroupedVerifyProbStride * sizeof(__half);",
        "kGroupedVerifyRows * kGroupedVerifyProbStride * sizeof(__half) +\n"
        "      kGroupedVerifyBlockN * kGroupedVerifyKVStride * sizeof(__half);",
    )
    source = replace_once(
        source,
        "static_assert(kCompensatedSmemBytes <= 64 * 1024,",
        "static_assert(kCompensatedSmemBytes <= 96 * 1024,",
    )
    source = replace_once(
        source,
        '"compensated P must fit the SM70 shared-memory budget");',
        '"compensated P and prefetched V must fit the SM70 budget");\n'
        "  TORCH_CHECK(properties->sharedMemPerBlockOptin >= kCompensatedSmemBytes,\n"
        '              "V prefetch exceeds device opt-in shared memory");',
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--head-groups", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--qk-unroll", type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--vector-load", action="store_true")
    parser.add_argument("--page-specialize", action="store_true")
    parser.add_argument("--prefetch-v", action="store_true")
    parser.add_argument("--grouped-only", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.prefetch_v and args.head_groups != 1:
        parser.error("--prefetch-v currently requires --head-groups 1")
    root = Path(__file__).resolve().parents[2] / "flash-attention-v100"
    original = root / "kernel/flash_decode_paged.cu"
    source = original.read_text()
    if args.grouped_only:
        prefix_end = source.index(
            "at::Tensor flash_attention_grouped_sparse_page4_plan("
        )
        entry_start = source.index(
            "at::Tensor flash_attention_grouped_e4m3_fp32_paged("
        )
        entry_end = source.index(
            "int64_t flash_attention_grouped_e4m3_fp32_precision_version()"
        )
        source = source[:prefix_end] + source[entry_start:entry_end]
    if args.qk_unroll is not None:
        start = source.index("__device__ __forceinline__ void grouped_verify_qk(")
        end = source.index(
            "__device__ __forceinline__ void grouped_verify_scale_", start
        )
        qk = source[start:end]
        qk = replace_once(
            qk,
            "#pragma unroll\n  for (int k_offset = 0; "
            "k_offset < kGroupedVerifyHeadDim; k_offset += 16)",
            f"#pragma unroll {args.qk_unroll}\n  for (int k_offset = 0; "
            "k_offset < kGroupedVerifyHeadDim; k_offset += 16)",
        )
        source = source[:start] + qk + source[end:]
    if args.vector_load:
        source = replace_once(
            source,
            "bool ROW_SEQLENS = false, bool COMPENSATE_P = false>",
            "bool ROW_SEQLENS = false, bool COMPENSATE_P = false, "
            "bool PAIR_E4M3 = false>",
        )
        start = source.index("__launch_bounds__(kGroupedVerifyThreads, 1) void ")
        end = source.index(
            "void flash_attention_grouped_verify_e5m2_combine_kernel(", start
        )
        partial = source[start:end]
        old = "!SPARSE_PAGE4 && !ROW_SEQLENS"
        if partial.count(old) != 3:
            raise ValueError("Expected three grouped KV panel loads")
        partial = partial.replace(old, "!SPARSE_PAGE4 && (!ROW_SEQLENS || PAIR_E4M3)")
        source = source[:start] + partial + source[end:]
        source = replace_once(
            source,
            "  auto kernel = flash_attention_grouped_verify_e5m2_partial_kernel<\n"
            "      8, false, 0, false, false, false, "
            "flash_v100::KV_CACHE_DTYPE_FP8_E4M3,\n"
            "      false, float, true, true>;",
            "  bool paired = true;\n"
            "  for (const auto* tensor : {&k, &v}) {\n"
            "    for (int dim = 0; dim < 3; ++dim)\n"
            "      paired = paired && tensor->stride(dim) % 16 == 0;\n"
            "  }\n"
            "  auto kernel = paired\n"
            "      ? flash_attention_grouped_verify_e5m2_partial_kernel<\n"
            "          8, false, 0, false, false, false, "
            "flash_v100::KV_CACHE_DTYPE_FP8_E4M3,\n"
            "          false, float, true, true, true>\n"
            "      : flash_attention_grouped_verify_e5m2_partial_kernel<\n"
            "          8, false, 0, false, false, false, "
            "flash_v100::KV_CACHE_DTYPE_FP8_E4M3,\n"
            "          false, float, true, true, false>;",
        )
    if args.page_specialize:
        statements = []
        for page in (1648, 3296):
            prefix = (
                "flash_attention_grouped_verify_e5m2_partial_kernel<"
                f"8, false, {page}, false, false, false, "
                "flash_v100::KV_CACHE_DTYPE_FP8_E4M3, false, float, true, true"
            )
            expression = (
                f"paired ? {prefix}, true> : {prefix}, false>"
                if args.vector_load
                else prefix + ">"
            )
            statements.append(f"  if (k.size(1) == {page}) kernel = {expression};\n")
        marker = "  constexpr int kCompensatedSmemBytes ="
        source = replace_once(source, marker, "".join(statements) + marker)
    if args.prefetch_v:
        source = prefetch_values(source)
    barrier = """        __syncwarp();
        if (lane_id == 0) {
          if (tile_sum > 0.0f) {"""
    if source.count(barrier) != 1:
        raise ValueError("The grouped online-softmax warp-state fix is required")
    if args.head_groups in (2, 3):
        source = replace_once(
            source,
            "constexpr int kGroupedVerifyRows = 48;",
            "constexpr int kGroupedVerifyRows = "
            f"{32 if args.head_groups == 2 else 16};",
        )
        source = replace_once(
            source,
            "constexpr int kGroupedVerifyThreads = 512;",
            "constexpr int kGroupedVerifyThreads = 256;",
        )
        source = replace_once(
            source,
            "kernel<<<dim3(1, 80), kGroupedVerifyThreads, "
            "kCompensatedSmemBytes, stream>>>",
            f"kernel<<<dim3({args.head_groups}, 80), kGroupedVerifyThreads, "
            "kCompensatedSmemBytes, stream>>>",
        )
    if args.head_groups == 2:
        source = replace_once(
            source,
            "static constexpr int kHeadsPerCta = "
            "kGroupedVerifyRows / MAX_QUERY_TOKENS;",
            "static constexpr int kHeadsPerCta = "
            "MAX_QUERY_TOKENS == kGroupedVerifyQ8MaxQ "
            "? 3 : kGroupedVerifyRows / MAX_QUERY_TOKENS;",
        )
        source = replace_once(
            source,
            "MAX_QUERY_TOKENS * kHeadsPerCta == kGroupedVerifyRows,",
            "MAX_QUERY_TOKENS * kHeadsPerCta <= kGroupedVerifyRows,",
        )
    source = replace_once(
        source,
        "flash_attention_grouped_e4m3_fp32_paged(",
        "private_grouped_e4m3_fp32_paged(",
    )
    source += (
        "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        '  m.def("run", &private_grouped_e4m3_fp32_paged);\n}\n'
    )
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    for name in ("include", "kernel"):
        target = sources / name
        target.mkdir(exist_ok=True)
        for pattern in ("*.h", "*.cuh"):
            for header in (root / name).glob(pattern):
                shutil.copy2(header, target)
    shutil.copy2(root / "LICENSE", sources)
    path = sources / "kernel/grouped-attention.cu"
    path.write_text(source)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = "sm70_grouped_attention_" + source_sha256[:12]
    # Retain Flash-V100's existing math flags; this is a scheduling candidate.
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
        "input_source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "source_sha256": source_sha256,
        "module_name": module_name,
        "source_files": {
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*"))
            if p.is_file()
        },
        "head_groups": args.head_groups,
        "qk_unroll": args.qk_unroll,
        "vector_load": args.vector_load,
        "page_specialize": args.page_specialize,
        "prefetch_v": args.prefetch_v,
        "grouped_only": args.grouped_only,
        "extra_cuda_cflags": flags,
        "scope": "Private operator candidate; full-model admission required",
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir(exist_ok=True)
        library = Path(
            load(
                name=module_name,
                sources=[str(path)],
                build_directory=str(build),
                extra_cuda_cflags=flags,
                extra_include_paths=[str(sources / "kernel"), str(sources / "include")],
                verbose=True,
            ).__file__
        )
        manifest["library"] = str(library)
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
