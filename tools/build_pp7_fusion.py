# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the optional additive SM70 gated RMSNorm library without a GPU."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Build; otherwise dry-run"
    )
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--cuda-home", type=Path, help="Override CUDA_HOME for nvcc")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "csrc/sm70_turbomind/ops/rmsnorm_gated_exact_pp7.cu"
    output = root / "vllm/_pp7_fusion.so"
    source_hash = sha256(source)
    work = args.build_dir or (
        Path(tempfile.gettempdir()) / "vllm-pp7-fusion" / source_hash[:16]
    )
    work = work.resolve()
    plan = {
        "source": str(source),
        "source_sha256": source_hash,
        "build_directory": str(work),
        "output": str(output),
        "cuda_home": str(args.cuda_home) if args.cuda_home else os.getenv("CUDA_HOME"),
        "cuda_arch": "7.0",
        "namespace": "vllm_pp7",
        "math_flags": ["-O3", "--fmad=false"],
        "no_fast_math": True,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return

    # This library is additive. Never replace a shared or symlinked extension.
    if output.is_symlink():
        raise FileExistsError(f"Refusing symlink output: {output}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    os.environ["MAX_JOBS"] = "1"
    if args.cuda_home:
        os.environ["CUDA_HOME"] = str(args.cuda_home.resolve())

    import torch
    from torch.utils.cpp_extension import load

    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA context before build")
    work.mkdir(parents=True, exist_ok=True)
    library = Path(
        load(
            name="qwen_pp7_exact_norm",
            sources=[str(source)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            with_cuda=True,
            is_python_module=False,
            build_directory=str(work),
            verbose=True,
        )
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA context during compile/load")
    if not hasattr(torch.ops.vllm_pp7, "rmsnorm_gated_exact_out"):
        raise RuntimeError("Library did not register the expected operator")
    if output.is_symlink():
        raise FileExistsError(f"Refusing symlink output: {output}")
    if output.exists():
        if sha256(output) != sha256(library):
            raise FileExistsError(
                f"Different additive output exists: {output}. "
                "Build in a separate checkout to preserve the installed library."
            )
    else:
        # Hard-link publication is atomic and fails if another process has
        # created the destination; it never overwrites an existing library.
        with tempfile.NamedTemporaryFile(
            prefix="_pp7_fusion.", suffix=".tmp", dir=output.parent, delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
            try:
                with library.open("rb") as reader:
                    shutil.copyfileobj(reader, temporary)
            except BaseException:
                temp_path.unlink(missing_ok=True)
                raise
        try:
            os.chmod(temp_path, 0o755)
            os.link(temp_path, output)
        finally:
            temp_path.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "plan": plan,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "library_sha256": sha256(output),
                "library_bytes": output.stat().st_size,
                "operator_registered": True,
                "cuda_initialized": False,
                "gpu_numerical_tests_run": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
