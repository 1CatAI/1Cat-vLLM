# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build an operator-only clock64 probe of the ordered N32 attention loop.

The explicit ``profile`` entrypoint writes five timestamps per visible tile.
Its timings include probe overhead and synchronization, and are neither achieved
occupancy nor uninstrumented latency. No serving route imports this module.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.kernels.build_sm70_grouped_attention_candidate import replace_once


def phase_source(source: str) -> str:
    start = source.index(
        "template <int MAX_QUERY_TOKENS, bool TWO_PASS, int PAGE_BLOCK_SIZE"
    )
    end = source.index("void flash_attention_grouped_verify_e5m2_combine_kernel(")
    partial = source[start:end]
    partial = replace_once(
        partial,
        "const int* row_lengths = nullptr) {",
        "const int* row_lengths = nullptr,\n"
        "    unsigned long long* clock_stamps = nullptr) {",
    )
    loop_start = partial.index("  // Recompute QK for the conservative path")
    loop_end = partial.index("    __syncthreads();\n  }", loop_start)
    loop_end += len("    __syncthreads();\n  }")
    loop = partial[loop_start:loop_end]

    def stamp(index: int) -> str:
        return (
            "    if (clock_stamps != nullptr && tid == 0)\n"
            f"      clock_stamps[(tile_start / 32) * 5 + {index}] = clock64();\n"
        )

    loop = replace_once(
        loop,
        "    const int valid_k_rows =",
        stamp(0) + "    const int valid_k_rows =",
    )
    loop = replace_once(
        loop,
        "    __syncthreads();\n\n    int active_m_tiles",
        "    __syncthreads();\n" + stamp(1) + "\n    int active_m_tiles",
    )
    loop = replace_once(
        loop,
        "    __syncthreads();\n\n    if constexpr (TWO_PASS)",
        "    __syncthreads();\n" + stamp(2) + "\n    if constexpr (TWO_PASS)",
    )
    loop = replace_once(
        loop,
        "    static_assert(COMPENSATE_P && kGroupedVerifyWarps == 16,",
        stamp(3) + "    static_assert(COMPENSATE_P && kGroupedVerifyWarps == 16,",
    )
    loop = replace_once(
        loop,
        "    __syncthreads();\n  }",
        "    __syncthreads();\n" + stamp(4) + "  }",
    )
    partial = partial[:loop_start] + loop + partial[loop_end:]
    source = source[:start] + partial + source[end:]
    host_start = source.index("at::Tensor private_grouped_e4m3_fp32_paged(")
    host_end = source.index("PYBIND11_MODULE(", host_start)
    host = source[host_start:host_end]
    host = replace_once(
        host,
        "      1, row_lengths.data_ptr<int>());",
        "      1, row_lengths.data_ptr<int>(), nullptr);",
    )
    profile = host.replace(
        "private_grouped_e4m3_fp32_paged", "profile_grouped_e4m3_fp32_paged"
    )
    profile = replace_once(
        profile,
        "float scale, float k_scale, float v_scale) {",
        "float scale, float k_scale, float v_scale, at::Tensor& stamps) {\n"
        "  TORCH_CHECK(stamps.device() == q.device() &&\n"
        "      stamps.scalar_type() == at::kLong && stamps.is_contiguous() &&\n"
        "      stamps.dim() == 2 && stamps.size(1) == 5 &&\n"
        "      stamps.size(0) >= (block_table.numel() * k.size(1) + 31) / 32,\n"
        '      "The diagnostic needs CUDA int64 timestamps [capacity_tiles,5]");',
    )
    profile = replace_once(
        profile,
        "      1, row_lengths.data_ptr<int>(), nullptr);",
        "      1, row_lengths.data_ptr<int>(),\n"
        "      reinterpret_cast<unsigned long long*>(stamps.data_ptr<int64_t>()));",
    )
    source = source[:host_start] + host + profile + source[host_end:]
    return replace_once(
        source,
        '  m.def("run", &private_grouped_e4m3_fp32_paged);',
        '  m.def("run", &private_grouped_e4m3_fp32_paged);\n'
        '  m.def("profile", &profile_grouped_e4m3_fp32_paged);',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    base = json.loads(args.base_manifest.read_text())
    assert base["head_groups"] == 1 and base["splits"] == 80
    assert base["prefetch_v"] and base["reuse_pv_values"]
    source_dir = args.base_manifest.parent / "sources"
    for relative, digest in base["source_files"].items():
        assert (
            hashlib.sha256((source_dir / relative).read_bytes()).hexdigest() == digest
        )
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    shutil.copytree(source_dir, sources)
    path = sources / "kernel/grouped-attention.cu"
    path.write_text(phase_source(path.read_text()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = "sm70_grouped_phase_" + digest[:12]
    manifest = {
        **{k: v for k, v in base.items() if k not in ("library", "library_sha256")},
        "input_source_sha256": base["source_sha256"],
        "source_sha256": digest,
        "module_name": module_name,
        "scope": "Instrumented phase attribution only, never performance admission",
        "phase_names": ["K load", "QK and V load", "online softmax", "ordered PV"],
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
        manifest["library"] = str(Path(module.__file__).resolve())
        manifest["library_sha256"] = hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
