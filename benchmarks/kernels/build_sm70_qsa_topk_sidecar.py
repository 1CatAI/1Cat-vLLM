# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build current QSA decode specialization for source-overlay validation."""

import argparse
import hashlib
import json
from pathlib import Path

from torch.utils.cpp_extension import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("sm70_qsa_topk_sidecar.cu")
    header = source.parents[2] / "csrc/qsa_lexicographic_topk.cuh"
    library = load(
        name="vllm_qsa_decode_topk_sm70",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        build_directory=str(args.build_dir.resolve()),
        is_python_module=False,
        verbose=True,
    )
    print(
        json.dumps(
            {
                "library": library,
                "library_sha256": hashlib.sha256(
                    Path(library).read_bytes()
                ).hexdigest(),
                "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
            }
        )
    )
