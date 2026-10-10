# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the publication screen with a separate local-buffer pointer.

This isolates the thread-local RankData copy observed in the rejected
publication candidate. Peer stores, rounding and twenty-part norm stay intact.
"""

import argparse
from pathlib import Path

from sm70_tp4_local_buffer_pointer import generate
from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "local_publish.cu"
    text = generate(args.source_root, publication=True)
    if not source.exists() or source.read_text() != text:
        source.write_text(text)
    load(
        name="qpn2_local_publish_norm_screen",
        sources=[str(source)],
        extra_include_paths=[
            str(args.source_root / "csrc"),
            str(args.source_root / "csrc/sm70_turbomind/ops"),
        ],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
