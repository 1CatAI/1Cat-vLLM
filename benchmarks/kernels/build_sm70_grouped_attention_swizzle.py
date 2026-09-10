# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen compact SM70 matrix-A layouts in compensated q8 attention.

Reuse the fragment map established by the existing SM70 prefill probes.
Q, K and P layout changes are independent; every mathematical operation and
logical split stays in place. Wider PV tiles and register-held QK products are
separate screens. No service route is installed by this builder.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.kernels.build_sm70_grouped_attention_candidate import replace_once

SWIZZLE = r"""
// Same SM70 matrix-A word map as the validated native prefill layout.
__device__ __forceinline__ int grouped_a_offset(int row, int col, int width) {
  const int local_row = row & 15;
  const int slot = (local_row & 3) | ((local_row & 8) >> 1) |
                   ((local_row & 4) << 1);
  return (row / 16) * 16 * width + (col / 16) * 256 +
         ((col & 15) / 8) * 128 + slot * 8 + (col & 7);
}

__device__ __forceinline__ void load_grouped_a_swizzled(
    volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>& frag,
    const __half* matrix_tile, const int k_offset) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 3) + ((lane >> 4) & 1) * 4 + ((lane >> 2) & 1) * 8;
  const int slot = (row & 3) | ((row & 8) >> 1) | ((row & 4) << 1);
  uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
      matrix_tile + (k_offset / 16) * 256 + slot * 8));
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3])
      : "r"(address) : "memory");
  address += 128 * sizeof(__half);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
      : "r"(address) : "memory");
}

"""


def swizzle_source(
    source: str, query: bool, probabilities: bool, early_store: bool = False
) -> str:
    marker = "template <bool COMPENSATE = false>"
    source = replace_once(source, marker, SWIZZLE + marker)
    if query:
        source = replace_once(
            source,
            "constexpr int kGroupedVerifyQStride = 264;",
            "constexpr int kGroupedVerifyQStride = 256;",
        )
        source = replace_once(
            source,
            """    volta::load_matrix_sync(
        q_fragment, shared_q + m_tile * 16 * kGroupedVerifyQStride + k_offset,
        kGroupedVerifyQStride);""",
            """    load_grouped_a_swizzled(
        q_fragment, shared_q + m_tile * 16 * kGroupedVerifyQStride, k_offset);""",
        )
    if probabilities:
        source = replace_once(
            source,
            "constexpr int kGroupedVerifyProbStride = 40;",
            "constexpr int kGroupedVerifyProbStride = 32;",
        )
    start = source.index("__launch_bounds__(kGroupedVerifyThreads, 1) void ")
    end = source.index(
        "void flash_attention_grouped_verify_e5m2_combine_kernel(", start
    )
    partials = source[start:end]
    if query:
        old = "shared_q_vec[row * kSharedQVecsPerRow + vec_col]"
        if partials.count(old) != 4:
            raise ValueError("Expected two query stores in each partial kernel")
        partials = partials.replace(
            old,
            "shared_q_vec[grouped_a_offset(row, vec_col * 8, 256) / 8]",
        )
    if probabilities:
        for old, new, count in (
            (
                "shared_probs[row * kGroupedVerifyProbStride + col]",
                "shared_probs[grouped_a_offset(row, col, 32)]",
                2,
            ),
            (
                "shared_probs[row * kGroupedVerifyProbStride + lane_id]",
                "shared_probs[grouped_a_offset(row, lane_id, 32)]",
                3,
            ),
            (
                "shared_prob_residual[row * kResidualStride + lane_id]",
                "shared_prob_residual[grouped_a_offset(row, lane_id, 32)]",
                3,
            ),
        ):
            if partials.count(old) != count:
                raise ValueError(f"Unexpected probability stores: {old}")
            partials = partials.replace(old, new)
        for name, stride in (
            ("shared_probs", "kGroupedVerifyProbStride"),
            ("shared_prob_residual", "kResidualStride"),
        ):
            old = (
                "        volta::load_matrix_sync(\n"
                "            probability_fragment,\n"
                f"            {name} + m_tile * 16 * {stride} + k_offset,\n"
                f"            {stride});"
            )
            if partials.count(old) != 2:
                raise ValueError(f"Expected two probability fragment loads: {name}")
            partials = partials.replace(
                old,
                "        load_grouped_a_swizzled(\n"
                "            probability_fragment,\n"
                f"            {name} + m_tile * 16 * {stride}, k_offset);",
            )
    if early_store:
        stores = """        shared_probs[grouped_a_offset(row, lane_id, 32)] =
            __float2half_rn(probability);
        if constexpr (COMPENSATE_P) {
          const float rounded = __half2float(__float2half_rn(probability));
          shared_prob_residual[grouped_a_offset(row, lane_id, 32)] =
              __float2half_rn((probability - rounded) * 2048.0f);
        }
"""
        reduction = """        const float tile_sum_lane = warp_reduce_sum(probability);
        const float tile_sum = __shfl_sync(0xffffffffu, tile_sum_lane, 0);
        const float exp_diff =
            tile_sum > 0.0f ? __expf(fmaxf(old_max - new_max, -80.0f)) : 1.0f;
"""
        # TWO_PASS has neither exp_diff nor compensated P/residual stores.
        if partials.count(reduction + stores) != 3:
            raise ValueError("Expected generic and full-q8 online probability stores")
        partials = partials.replace(reduction + stores, stores + reduction)
    return source[:start] + partials + source[end:]


def swizzle_key(source: str) -> str:
    """Apply the compact layout to K, leaving the separate V tile unchanged."""
    begin = source.index("__device__ __forceinline__ void load_xqa_tc_kv_panel(")
    begin = source.rindex("template <int BLOCK_SIZE", 0, begin)
    end = source.index("template <int BLOCK_SIZE", begin + 1)
    loader = source[begin:end].replace(
        "load_xqa_tc_kv_panel(", "load_grouped_key_swizzled("
    )
    loader = replace_once(
        loader,
        "row * kv_smem_stride_uint4 + vec_pair * 2",
        "grouped_a_offset(row, vec_pair * 16, 256) / 8",
    )
    loader = replace_once(loader, "shared_offset + 1", "shared_offset + 16")
    loader = replace_once(
        loader,
        "row * kv_smem_stride_uint4 + vec_col",
        "grouped_a_offset(row, vec_col * 8, 256) / 8",
    )
    fragment_load = SWIZZLE[SWIZZLE.index("__device__ __forceinline__ void load") :]
    fragment_load = fragment_load.replace(
        "load_grouped_a_swizzled", "load_grouped_b_col_swizzled"
    ).replace("volta::matrix_a", "volta::matrix_b")
    fragment_load = fragment_load.replace("volta::row_major", "volta::col_major")
    fragment_load = fragment_load.replace(
        "((lane >> 2) & 1) * 8", "((lane >> 3) & 1) * 8"
    )
    marker = "template <bool COMPENSATE = false>"
    source = replace_once(source, marker, loader + fragment_load + marker)
    source = replace_once(
        source,
        """    volta::load_matrix_sync(
        k_fragment, shared_k + n_tile * 16 * kGroupedVerifyKVStride + k_offset,
        kGroupedVerifyKVStride);""",
        """    load_grouped_b_col_swizzled(
        k_fragment, shared_k + n_tile * 16 * 256, k_offset);""",
    )
    begin = source.index("__launch_bounds__(kGroupedVerifyThreads, 1) void ")
    end = source.index("void flash_attention_grouped_verify_e5m2_combine_kernel(")
    partials = source[begin:end]
    # Four K calls and two V calls occur in the two partial kernels.
    pieces = partials.split("load_xqa_tc_kv_panel<")
    if len(pieces) != 7:
        raise ValueError("Expected six grouped K/V loader calls")
    partials = pieces[0]
    for piece in pieces[1:]:
        name = (
            "load_grouped_key_swizzled<"
            if "shared_kv, k_cache" in piece.split(";", 1)[0]
            else "load_xqa_tc_kv_panel<"
        )
        partials += name + piece
    old = """for (int idx = tid + valid_k_rows * kSharedStrideVec;
"""
    if partials.count(old) != 4:
        raise ValueError("Expected four K zero-padding loops")
    pieces = partials.split(old)
    partials = pieces[0]
    for piece in pieces[1:]:
        body, following = piece.split("}", 1)
        body = body.replace("kSharedStrideVec", "kPanelStrideVec")
        body = replace_once(
            body,
            "reinterpret_cast<uint4*>(shared_kv)[idx]",
            "reinterpret_cast<uint4*>(shared_kv)[grouped_a_offset(\n"
            "          idx / kPanelStrideVec, (idx % kPanelStrideVec) * 8, 256) / 8]",
        )
        partials += old.replace("kSharedStrideVec", "kPanelStrideVec")
        partials += body + "}" + following
    return source[:begin] + partials + source[end:]


def retile_pv_m8n32(source: str) -> str:
    """Screen wider PV tiles without changing any per-element K16 update."""
    begin = source.index(
        "__device__ __forceinline__ void grouped_verify_scale_output_fragment("
    )
    end = source.index("template <bool SPARSE_PAGE4", begin)
    helpers = (
        source[begin:end]
        .replace("16, 16, 16, float>", "8, 32, 16, float>")
        .replace(" + ((lane >> 2) & 1) * 8", "")
    )
    load = SWIZZLE[SWIZZLE.index("__device__ __forceinline__ void load_grouped") :]
    load = load.replace("16, 16, 16, half", "8, 32, 16, half")
    load = load.replace(
        "const int k_offset)", "const int k_offset, const int row_start)"
    )
    load = load.replace(
        "(lane & 3) + ((lane >> 4) & 1) * 4 + ((lane >> 2) & 1) * 8",
        "row_start + (lane & 3) + ((lane >> 4) & 1) * 4",
    )
    load = load.replace(
        "  const int slot = (row & 3) | ((row & 8) >> 1) | ((row & 4) << 1);\n",
        "",
    )
    load = load.replace(
        "(k_offset / 16) * 256 + slot * 8", "grouped_a_offset(row, k_offset, 32)"
    )
    source = source[:end] + helpers + load + source[end:]
    begin = source.index("__launch_bounds__(kGroupedVerifyThreads, 1) void ")
    end = source.index("void flash_attention_grouped_verify_e5m2_combine_kernel(")
    partials = source[begin:end]
    for old, new, count in (
        ("16, 16, 16, float>", "8, 32, 16, float>", 4),
        (
            "16, 16, 16, half, volta::row_major>",
            "8, 32, 16, half, volta::row_major>",
            4,
        ),
        ("const int d_tile = warp_id;", "const int d_tile = warp_id % 8;", 2),
        (
            "const int m_tile = fragment_idx;",
            "const int m_tile = warp_id / 8 + 2 * fragment_idx;",
            4,
        ),
        ("(1 << m_tile)", "(1 << (m_tile / 2))", 4),
        ("(kGroupedVerifyHeadDim / 16)", "(kGroupedVerifyHeadDim / 32)", 6),
        ("smem.row_scale, m_tile * 16", "smem.row_scale, m_tile * 8", 4),
        ("d_tile * 16", "d_tile * 32", 4),
        (
            "shared_output + m_tile * 16 * kGroupedVerifyHeadDim",
            "shared_output + m_tile * 8 * kGroupedVerifyHeadDim",
            2,
        ),
        (
            "shared_probs + m_tile * 16 * kGroupedVerifyProbStride, k_offset",
            "shared_probs, k_offset, m_tile * 8",
            2,
        ),
        (
            "shared_prob_residual + m_tile * 16 * kResidualStride, k_offset",
            "shared_prob_residual, k_offset, m_tile * 8",
            2,
        ),
    ):
        if partials.count(old) != count:
            raise ValueError(f"Unexpected PV retile anchor: {old}")
        partials = partials.replace(old, new)
    return source[:begin] + partials + source[end:]


def pipeline_qk_products(source: str) -> str:
    """Form two K16 products in one warp before ordered compensated sums."""
    begin = source.index("__device__ __forceinline__ void grouped_verify_qk(")
    end = source.index("__device__ __forceinline__ void grouped_verify_scale", begin)
    qk = source[begin:end]
    start = qk.index("#pragma unroll 2\n")
    stop = qk.index("#pragma unroll\n  for (int i = 0;", start)
    old_loop = qk[start:stop]
    loads = old_loop[
        old_loop.index("{\n") + 2 : old_loop.index("    if constexpr (COMPENSATE)")
    ]
    # All fragment loads remain scoped to their original K16 product.
    # Only the two independent products move ahead of their FP32 sums.
    loop = (
        """  if constexpr (COMPENSATE) {
#pragma unroll 1
    for (int pair_offset = 0; pair_offset < kGroupedVerifyHeadDim;
         pair_offset += 32) {
      volta::fragment<volta::accumulator, 16, 16, 16, float> products[2];
#pragma unroll
      for (int stage = 0; stage < 2; ++stage) {
        const int k_offset = pair_offset + stage * 16;
"""
        + loads
        + """
        volta::fill_fragment(products[stage], 0.0f);
        volta::mma_sync(products[stage], q_fragment, k_fragment, products[stage]);
      }
#pragma unroll
      for (int stage = 0; stage < 2; ++stage) {
#pragma unroll
        for (int i = 0; i < score_fragment.num_elements; ++i) {
          const float y = __fsub_rn(products[stage].x[i], correction[i]);
          const float sum = __fadd_rn(score_fragment.x[i], y);
          correction[i] = __fsub_rn(__fsub_rn(sum, score_fragment.x[i]), y);
          score_fragment.x[i] = sum;
        }
      }
    }
  } else {
"""
        + old_loop
        + "  }\n"
    )
    qk = qk[:start] + loop + qk[stop:]
    return source[:begin] + qk + source[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--query", action="store_true")
    parser.add_argument("--key", action="store_true")
    parser.add_argument("--probabilities", action="store_true")
    parser.add_argument("--early-store", action="store_true")
    parser.add_argument("--pv-m8n32", action="store_true")
    parser.add_argument("--qk-register-pipeline", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not (args.query or args.key or args.probabilities):
        parser.error("Choose --query, --key and/or --probabilities")
    if args.early_store and not args.probabilities:
        parser.error("--early-store requires the probability-layout screen")
    if args.pv_m8n32 and not args.probabilities:
        parser.error("--pv-m8n32 requires the probability-layout screen")
    base = json.loads(args.base_manifest.read_text())
    if base["source_sha256"] not in (
        "3b0c9688ce17e1870408ef81fb5cd9b63a677b7cfd7d4777b8df77dd0fc24132",
        "e78e922b17ff44262c49e6664871d0692e0ecd048b676061516d0419d3819f92",
    ):
        parser.error("Use a frozen visible-q8 or exact-LUT parent")
    source_dir = args.base_manifest.parent / "sources"
    for relative, digest in base["source_files"].items():
        if hashlib.sha256((source_dir / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Parent source hash mismatch: {relative}")
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    shutil.copytree(source_dir, sources)
    path = sources / "kernel/grouped-attention.cu"
    path.write_text(
        swizzle_source(
            path.read_text(), args.query, args.probabilities, args.early_store
        )
    )
    if args.pv_m8n32:
        path.write_text(retile_pv_m8n32(path.read_text()))
    if args.key:
        path.write_text(swizzle_key(path.read_text()))
    if args.qk_register_pipeline:
        path.write_text(pipeline_qk_products(path.read_text()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = "sm70_grouped_swizzle_" + digest[:12]
    manifest = {
        **{k: v for k, v in base.items() if k not in ("library", "library_sha256")},
        "input_source_sha256": base["source_sha256"],
        "source_sha256": digest,
        "module_name": module_name,
        "query_swizzle": args.query,
        "key_swizzle": args.key,
        "probability_swizzle": args.probabilities,
        "early_probability_store": args.early_store,
        "pv_m8n32": args.pv_m8n32,
        "qk_register_pipeline": args.qk_register_pipeline,
        "scope": "Private layout screen; byte and full-model admission required",
        "source_files": {
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*"))
            if p.is_file()
        },
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir()
        module = load(
            name=module_name,
            sources=[str(path)],
            extra_include_paths=[str(sources / "include"), str(sources / "kernel")],
            extra_cuda_cflags=base["extra_cuda_cflags"],
            build_directory=str(build),
            verbose=True,
        )
        library = Path(module.__file__).resolve()
        manifest["library"] = str(library)
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
