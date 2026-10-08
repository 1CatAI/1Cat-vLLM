# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run existing GPU metadata oracles against the extracted module.

This isolated gate uses installed Torch/Triton and the installed Triton metadata
type. It does not qualify engine imports, dispatch, or the final vLLM wheel.
Run with GPU ownership locks and a CUDA-enabled virtual environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=True)
    module = root / "vllm/v1/attention/backends/flash_v100/metadata.py"
    test = root / "tests/kernels/attention/test_sm70_flash_v100_smallq_metadata.py"
    # Only the import is redirected. All numerical and graph assertions remain
    # the repository's existing tests; no duplicated oracle is introduced.
    source = test.read_text().replace(
        "from vllm.v1.attention.backends.flash_attn_v100 import (",
        "from onecat_kv_metadata_gate import (",
    )
    assert source != test.read_text(), "Original metadata import was not found"
    (args.out / "onecat_kv_metadata_gate.py").write_bytes(module.read_bytes())
    test_path = args.out / test.name
    test_path.write_text(source)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.out.resolve())
    for key, name in (
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
    ):
        environment[key] = str(args.out.resolve() / name)
    command = [
        str(args.runtime_python),
        "-m",
        "pytest",
        str(test_path),
        "-q",
        f"--confcutdir={args.out.resolve()}",
        f"--junitxml={args.out.resolve() / 'result.xml'}",
    ]
    result = subprocess.run(command, env=environment, cwd=args.out, check=False)
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": result.returncode,
                "metadata_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                "original_tests_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
                "scope": (
                    "isolated metadata numerical/graph oracle, not engine admission"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
