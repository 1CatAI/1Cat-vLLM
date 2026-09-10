# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen parallel K16 products with the original ordered compensation.

Twelve warps compute independent products, while four load V. Six warps then
consume each group of four K16 products in the original order. Extra shared
storage and CTA barriers are included in timing; this installs no service route.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.kernels.build_sm70_grouped_attention_candidate import replace_once

QK_PARALLEL = r"""
__device__ __forceinline__ void grouped_verify_qk_parallel(
    const __half* shared_q, const __half* shared_k, float* shared_scores,
    float* products, const float qk_scale) {
  const int warp = threadIdx.x / 32;
  volta::fragment<volta::accumulator, 16, 16, 16, float> score;
  volta::fill_fragment(score, 0.0f);
  float correction[8] = {};
#pragma unroll 1
  for (int base_k = 0; base_k < 256; base_k += 64) {
    if (warp < 12) {
#pragma unroll
      for (int task = warp; task < 24; task += 12) {
        const int slot = task / 6;
        const int tile = task % 6;
        const int m_tile = tile / 2;
        const int n_tile = tile % 2;
        const int k_offset = base_k + slot * 16;
        volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major> qf;
        volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::col_major> kf;
        volta::fragment<volta::accumulator, 16, 16, 16, float> product;
        volta::load_matrix_sync(qf, shared_q + m_tile * 16 * 264 + k_offset, 264);
        volta::load_matrix_sync(kf, shared_k + n_tile * 16 * 264 + k_offset, 264);
        volta::fill_fragment(product, 0.0f);
        volta::mma_sync(product, qf, kf, product);
        volta::store_matrix_sync(products + slot * 48 * 32 +
            m_tile * 16 * 32 + n_tile * 16, product, 32, volta::mem_row_major);
      }
    }
    __syncthreads();
    if (warp < 6) {
#pragma unroll
      for (int slot = 0; slot < 4; ++slot) {
        volta::fragment<volta::accumulator, 16, 16, 16, float> product;
        volta::load_matrix_sync(product, products + slot * 48 * 32 +
            (warp / 2) * 16 * 32 + (warp % 2) * 16, 32, volta::mem_row_major);
#pragma unroll
        for (int i = 0; i < score.num_elements; ++i) {
          const float y = __fsub_rn(product.x[i], correction[i]);
          const float sum = __fadd_rn(score.x[i], y);
          correction[i] = __fsub_rn(__fsub_rn(sum, score.x[i]), y);
          score.x[i] = sum;
        }
      }
    }
    // Complete all reads before any warp overwrites a product slot.
    __syncthreads();
  }
  if (warp < 6) {
#pragma unroll
    for (int i = 0; i < score.num_elements; ++i) score.x[i] *= qk_scale;
    volta::store_matrix_sync(shared_scores + (warp / 2) * 16 * 32 +
        (warp % 2) * 16, score, 32, volta::mem_row_major);
  }
}

"""


def parallel_source(source: str) -> str:
    symbol = source.index("void flash_attention_grouped_verify_e4m3_full_q8_kernel(")
    start = source.rfind("template <", 0, symbol)
    end = source.index("template <", symbol)
    partial = source[start:end]
    qk_call = (
        "    grouped_verify_qk<COMPENSATE_P>(shared_q, shared_kv, shared_scores,\n"
        "                                    qk_scale, active_m_tiles);"
    )
    partial = replace_once(partial, qk_call, "")
    partial = replace_once(
        partial,
        "if (warp_id >= kGroupedVerifyQKWarps)",
        "if (warp_id >= 12)",
    )
    partial = partial.replace("kGroupedVerifyQKWarps * kWarpSize", "12 * kWarpSize")
    marker = "    __syncthreads();\n\n    if constexpr (TWO_PASS)"
    partial = replace_once(
        partial,
        marker,
        "    grouped_verify_qk_parallel(shared_q, shared_kv, shared_scores,\n"
        "        reinterpret_cast<float*>(shared_values + 32 * 264), qk_scale);\n"
        + marker,
    )
    source = source[:start] + QK_PARALLEL + partial + source[end:]
    return replace_once(
        source,
        "kGroupedVerifyBlockN * kGroupedVerifyKVStride * sizeof(__half);",
        "kGroupedVerifyBlockN * kGroupedVerifyKVStride * sizeof(__half) +\n"
        "      4 * 48 * 32 * sizeof(float);",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    base = json.loads(args.base_manifest.read_text())
    # Freeze the measured parent so another scheduling experiment cannot
    # silently change this screen's numerical or shared-memory contract.
    if base["source_sha256"] != (
        "3b0c9688ce17e1870408ef81fb5cd9b63a677b7cfd7d4777b8df77dd0fc24132"
    ):
        parser.error("This screen requires the frozen visible-q8 parent")
    source_dir = args.base_manifest.parent / "sources"
    for relative, digest in base["source_files"].items():
        if hashlib.sha256((source_dir / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Parent source hash mismatch: {relative}")
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    shutil.copytree(source_dir, sources)
    path = sources / "kernel/grouped-attention.cu"
    path.write_text(parallel_source(path.read_text()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = "sm70_grouped_qk_parallel_" + digest[:12]
    manifest = {
        **{k: v for k, v in base.items() if k not in ("library", "library_sha256")},
        "input_source_sha256": base["source_sha256"],
        "source_sha256": digest,
        "module_name": module_name,
        "qk_parallel_products": True,
        "scope": "Private operator screen; no model or complete-round admission",
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
