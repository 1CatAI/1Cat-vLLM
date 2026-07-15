# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recover exact SM70 WMMA matrix-A/B lane/register mappings."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _load_extension() -> object:
    source = (
        Path(__file__).resolve().parent
        / "csrc"
        / "sm70_wmma_fragment_probe.cu"
    )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    return load(
        name="sm70_wmma_fragment_probe_v3",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=os.getenv("VLLM_SM70_WMMA_PROBE_VERBOSE") == "1",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, choices=(16, 24, 32), default=16)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(
        device
    ) != (7, 0):
        raise RuntimeError("This probe requires an SM70 CUDA device.")

    extension = _load_extension()
    tile = torch.arange(256, device=device, dtype=torch.float16).reshape(16, 16)
    words = extension.dump_matrix_a_fragment(tile, args.stride)
    reference_words, swizzled_words = (
        extension.compare_swizzled_matrix_a_fragment(tile)
    )
    matrix_b_row_words = extension.dump_matrix_b_fragment(tile, False)
    matrix_b_col_storage = tile.t().contiguous()
    matrix_b_col_words = extension.dump_matrix_b_fragment(
        matrix_b_col_storage,
        True,
    )
    matrix_b_reference_words, matrix_b_compact_words = (
        extension.compare_compact_matrix_b_col_fragment(
            matrix_b_col_storage
        )
    )
    torch.cuda.synchronize(device)
    elements = words.cpu().view(torch.float16).to(torch.int32).reshape(32, 16)
    mapping = elements.tolist()
    counts = Counter(value for lane in mapping for value in lane)
    multiplicities = Counter(counts.values())
    matrix_b_elements = (
        matrix_b_col_words.cpu()
        .view(torch.float16)
        .to(torch.int32)
        .reshape(32, 16)
    )
    matrix_b_mapping = matrix_b_elements.tolist()
    matrix_b_counts = Counter(
        value for lane in matrix_b_mapping for value in lane
    )
    matrix_b_multiplicities = Counter(matrix_b_counts.values())
    payload = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "stride": args.stride,
        "lane_elements": mapping,
        "unique_elements": len(counts),
        "element_multiplicity_histogram": {
            str(key): value for key, value in sorted(multiplicities.items())
        },
        "all_elements_covered": sorted(counts) == list(range(256)),
        "swizzled_fragment_words_equal": bool(
            torch.equal(reference_words, swizzled_words)
        ),
        "swizzled_fragment_max_word_xor": int(
            torch.bitwise_xor(reference_words, swizzled_words)
            .abs()
            .max()
            .item()
        ),
        "matrix_b": {
            "lane_elements": matrix_b_mapping,
            "unique_elements": len(matrix_b_counts),
            "element_multiplicity_histogram": {
                str(key): value
                for key, value in sorted(matrix_b_multiplicities.items())
            },
            "all_elements_covered": sorted(matrix_b_counts)
            == list(range(256)),
            "row_col_fragment_words_equal": bool(
                torch.equal(matrix_b_row_words, matrix_b_col_words)
            ),
            "compact_col_fragment_words_equal": bool(
                torch.equal(
                    matrix_b_reference_words,
                    matrix_b_compact_words,
                )
            ),
            "compact_col_fragment_max_word_xor": int(
                torch.bitwise_xor(
                    matrix_b_reference_words,
                    matrix_b_compact_words,
                )
                .abs()
                .max()
                .item()
            ),
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
