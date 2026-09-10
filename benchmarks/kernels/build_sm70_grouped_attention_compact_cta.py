# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen unpadded two-head q8 CTAs with the exact N32 numerical contract.

Three head groups each retain sixteen real rows. Shared storage can fit two
CTAs per SM; actual occupancy and speed must be measured. This trades additional
KV reads for independent CTAs without changing any logical split or row sum.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import regex as re

from benchmarks.kernels.build_sm70_grouped_attention_candidate import replace_once

PV = r"""
    volta::fragment<volta::accumulator,16,16,16,float> tile_fragments[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) volta::fill_fragment(tile_fragments[i], 0.0f);
#pragma unroll
    for (int k_offset = 0; k_offset < 32; k_offset += 16) {
      volta::fragment<volta::matrix_a,16,16,16,half,volta::row_major> p, residual;
      load_grouped_a_swizzled(p, shared_probs, k_offset);
      load_grouped_a_swizzled(residual, shared_prob_residual, k_offset);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        volta::fragment<volta::matrix_b,16,16,16,half,volta::row_major> v;
        volta::load_matrix_sync(v,
            shared_values + k_offset * kGroupedVerifyKVStride + (warp_id + i * 8) * 16,
            kGroupedVerifyKVStride);
        auto residual_v = v;
#pragma unroll
        for (int j = 0; j < v.num_elements / 2; ++j) {
          union { uint32_t bits; __half2 pair; } packed;
          packed.bits = v.x[j];
          packed.pair = __hmul2(packed.pair, __float2half2_rn(1.0f / 2048.0f));
          residual_v.x[j] = packed.bits;
        }
        volta::mma_sync(tile_fragments[i], p, v, tile_fragments[i]);
        volta::mma_sync(tile_fragments[i], residual, residual_v, tile_fragments[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
      grouped_verify_add_output_tile(output_fragments[i], tile_fragments[i],
                                     smem.row_scale, 0);
"""


def compact_cta(source: str) -> str:
    constants = {
        "kGroupedVerifyRows": "16",
        "kGroupedVerifyThreads": "256",
        "kGroupedVerifyWarps": "8",
        "kGroupedVerifyQKWarps": "2",
        "kGroupedVerifyOutputTilesPerWarp": "2",
        "GroupedVerifySmem": "CompactVerifySmem",
        "grouped_verify_qk": "compact_verify_qk",
    }

    def substitute(text: str) -> str:
        for old, new in constants.items():
            text = re.sub(r"\b" + old + r"\b", new, text)
        return text

    start = source.index("struct alignas(256) GroupedVerifySmem {")
    end = source.index("\nstatic_assert(sizeof(GroupedVerifySmem)", start)
    storage = substitute(source[start:end])
    start = source.index("template <bool COMPENSATE = false>")
    end = source.index("__device__ __forceinline__ void grouped_verify_scale_", start)
    qk = substitute(source[start:end])
    start = source.index("void flash_attention_grouped_verify_e4m3_full_q8_kernel(")
    start = source.rindex("template <int MAX_QUERY_TOKENS", 0, start)
    end = source.index(
        "template <int MAX_QUERY_TOKENS, bool SINGLE_QUERY, typename PARTIAL_T", start
    )
    partial = substitute(source[start:end])
    partial = replace_once(
        partial,
        "  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;",
        "  struct Traits { enum { "
        "kHeadsPerCta = 2, kHeadGroups = 3, kSplits = 80 }; };",
    )
    a = partial.index("    static_assert(COMPENSATE_P && 8 == 16,")
    b = partial.index("    __syncthreads();\n  }", a)
    partial = partial[:a] + PV + partial[b:]
    source = source[:start] + storage + qk + partial + source[end:]
    marker = "  constexpr int kCompensatedSmemBytes =\n"
    begin = source.rindex(marker)
    end = source.index("  TORCH_CHECK(properties->sharedMemPerBlockOptin", begin)
    source = (
        source[:begin]
        + r"""  constexpr int kOriginalSmemBytes =
      sizeof(GroupedVerifySmem) + 48 * 32 * sizeof(__half) + 32 * 264 * sizeof(__half);
  constexpr int kCompactSmemBytes =
      sizeof(CompactVerifySmem) + 16 * 32 * sizeof(__half) + 32 * 264 * sizeof(__half);
"""
        + "  static_assert(kCompactSmemBytes + 512 <= 48 * 1024, "
        '"Two compact CTAs fit SM70 shared memory");\n'
        "  const int kCompensatedSmemBytes = q.size(0) == 8 ? "
        "kCompactSmemBytes : kOriginalSmemBytes;\n" + source[end:]
    )
    return replace_once(
        source,
        "kernel<<<dim3(1, 80), kGroupedVerifyThreads, kCompensatedSmemBytes, stream>>>",
        "kernel<<<dim3(q.size(0) == 8 ? 3 : 1, 80), q.size(0) == 8 ? 256 : 512, "
        "kCompensatedSmemBytes, stream>>>",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    base = json.loads(args.base_manifest.read_text())
    if (
        base["source_sha256"]
        != "eb7a85511f581fcd22cf13619c85ed2f42a8cbc8b216bb3e632bf448f6b820e1"
    ):
        parser.error("Use the frozen service-checked P-layout source")
    parent = args.base_manifest.parent / "sources"
    for rel, digest in base["source_files"].items():
        if hashlib.sha256((parent / rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Parent source hash mismatch: {rel}")
    output = args.output_dir.resolve()
    sources = output / "sources"
    shutil.copytree(parent, sources)
    path = sources / "kernel/grouped-attention.cu"
    path.write_text(compact_cta(path.read_text()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = "sm70_compact_cta_" + digest[:12]
    manifest = {
        **{k: v for k, v in base.items() if k not in ("library", "library_sha256")},
        "input_source_sha256": base["source_sha256"],
        "source_sha256": digest,
        "module_name": name,
        "q8_head_groups": 3,
        "q8_threads": 256,
        "scope": "Private exact compact-CTA attention; no service admission",
        "source_files": {
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*"))
            if p.is_file()
        },
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = output / "build"
        build.mkdir()
        library = Path(
            load(
                name=name,
                sources=[str(path)],
                extra_include_paths=[str(sources / "include"), str(sources / "kernel")],
                extra_cuda_cflags=base["extra_cuda_cflags"],
                build_directory=str(build),
                verbose=True,
            ).__file__
        ).resolve()
        manifest["library"] = str(library)
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
