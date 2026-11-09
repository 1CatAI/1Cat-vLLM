# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the admitted Qwen3.8 single-request overlays from this checkout.

Requires the project's CUDA-enabled Torch/build environment and a compatible
native vLLM installation. This is an explicit source-overlay build, not a full
vLLM wheel build. No model or GPU context is initialized by this script.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.sm70_qwen38_baseline import production_fingerprint  # noqa: E402


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(output: Path) -> dict:
    import torch
    from torch.utils.cpp_extension import load

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_root": str(ROOT),
        "production_fingerprint": production_fingerprint(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "nvcc": subprocess.check_output(
            [str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"), "--version"],
            text=True,
        ),
        "libraries": {},
    }
    specifications = {
        "hc": (
            "vllm_qwen38_hc_sm70",
            ["benchmarks/kernels/sm70_qwen38_custom_ar_sidecar.cu"],
            ["-O3", "-DNDEBUG", "-std=c++17"],
            ["csrc/custom_all_reduce.cu", "csrc/custom_all_reduce.cuh"],
        ),
        "qpn": (
            "vllm_qwen38_nvfp4_sm70",
            [
                "csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu",
                "benchmarks/kernels/sm70_qwen38_nvfp4_sidecar.cpp",
            ],
            ["-O3", "-lineinfo"],
            ["csrc/ops.h"],
        ),
        "qsa": (
            "vllm_qsa_decode_topk_sm70",
            ["benchmarks/kernels/sm70_qsa_topk_sidecar.cu"],
            ["-O3", "-lineinfo"],
            ["csrc/qsa_lexicographic_topk.cuh"],
        ),
        "flashqla": (
            "flash_qla_sm70_gdn_strided",
            ["flash_qla/ops/gated_delta_rule/chunk/sm70/csrc/gdn_forward.cu"],
            ["-O3"],
            [],
        ),
    }
    for component, (name, sources, flags, headers) in specifications.items():
        destination = output / component
        destination.mkdir(exist_ok=True)
        print(f"Building {component} from {ROOT}", flush=True)
        library = load(
            name=name,
            sources=[str(ROOT / s) for s in sources],
            extra_cflags=["-O3", "-DNDEBUG"] if component == "hc" else ["-O3"],
            extra_cuda_cflags=flags,
            extra_include_paths=[str(ROOT / "csrc")] if component == "hc" else [],
            extra_ldflags=["-lcuda"] if component == "hc" else [],
            build_directory=str(destination),
            is_python_module=component == "flashqla",
            verbose=True,
        )
        path = Path(library.__file__ if component == "flashqla" else library)
        manifest["libraries"][component] = {
            "path": str(path),
            "sha256": digest(path),
            "sources": {s: digest(ROOT / s) for s in sources + headers},
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    destination = output / "flashv100"
    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "build_ext",
            "--build-lib",
            str(destination),
            "--build-temp",
            str(output / "flashv100_build"),
            "-j1",
        ],
        cwd=ROOT / "flash-attention-v100",
        check=True,
    )
    for component, pattern in (
        ("flashv100", "flash_attn_v100_cuda*.so"),
        ("paged_kv", "paged_kv_utils*.so"),
    ):
        libraries = list(destination.glob(pattern))
        if len(libraries) != 1:
            raise RuntimeError(f"Expected one {pattern} in {destination}")
        path = libraries[0]
        manifest["libraries"][component] = {"path": str(path), "sha256": digest(path)}
    manifest["flashv100_sources"] = {
        str(p.relative_to(ROOT)): digest(p)
        for folder in ("kernel", "include")
        for p in sorted((ROOT / "flash-attention-v100" / folder).rglob("*"))
        if p.suffix in (".h", ".cuh", ".cu", ".cpp")
    }
    if manifest["production_fingerprint"] != production_fingerprint():
        raise RuntimeError("Production sources changed during compilation")
    manifest["complete"] = True
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("CUDA_HOME"):
        parser.error("Set CUDA_HOME to the compatible CUDA toolkit")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MAX_JOBS", "2")
    build(args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
