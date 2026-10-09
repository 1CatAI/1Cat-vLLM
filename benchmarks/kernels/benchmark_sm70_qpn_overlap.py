# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a research-only cross-operator L2 warming extension, without GPU use."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "qpn_overlap.cu"
    source.write_text(Path(__file__).with_suffix(".cu").read_text())
    from torch.utils.cpp_extension import load

    os.environ.setdefault("MAX_JOBS", "1")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    load(
        name="qpn_overlap700",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
        extra_ldflags=["-Wl,-Bsymbolic"],
        build_directory=str(args.output),
        is_python_module=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
