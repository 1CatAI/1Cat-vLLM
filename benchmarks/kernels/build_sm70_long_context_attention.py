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
    return source[:start] + body + source[end:]


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
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = repo / "csrc/attention/sm70_grouped_long"
    original = root / "kernel/grouped-attention.cu"
    source = candidate_source(original.read_text(), args.variant)
    # Avoid duplicate registration when reference/candidates share a process.
    source = source[: source.index("// Registered into the shipped FA2 extension")]
    source = "#include <torch/extension.h>\n" + source
    source += (
        "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        '  m.def("run", &private_grouped_e4m3_fp32_paged);\n}\n'
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
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "variant": args.variant,
        "input_source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "source_sha256": digest,
        "module_name": module_name,
        "entrypoint": "run",
        "splits": 80,
        "arithmetic_change": args.variant not in ("reference", "bit-decode"),
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
