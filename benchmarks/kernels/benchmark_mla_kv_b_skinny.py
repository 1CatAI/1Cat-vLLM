# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cuBLAS bmm against the skinny kernel for the two MLA kv_b products.

Each case runs inside a CUDA graph of ``--calls`` back-to-back launches, so
launch gaps do not count. The weight set is ``--layers`` distinct copies,
cycled, so every call streams the weights from DRAM (one 16.8 MB matrix would
sit in the 6 MB L2 only partly, but several copies make it certain). Reports
microseconds per call and weight bytes / time in GB/s. ``--sweep`` times the
kernel for a grid of tiles to tune ``TILE_*`` in mla_kv_b_skinny.py.

Run on one idle card (GPU-Orc hold first):
    CUDA_VISIBLE_DEVICES=<card> python benchmarks/kernels/benchmark_mla_kv_b_skinny.py
"""

import argparse
import itertools

import torch

from vllm.model_executor.layers.attention.mla_kv_b_skinny import (
    KvBTile,
    kv_b_bmm_skinny,
    kv_b_skinny_tile,
)
from vllm.triton_utils import triton

HEADS, NOPE, VALUE, LATENT = 64, 256, 256, 512
PEAK_GBPS = {"V100": 900.0, "RTX 8000": 672.0}


def _weights(layers: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(W_UK_T, W_UV) per layer, as process_weights_after_loading builds them."""
    sets = []
    for _ in range(layers):
        weight = torch.randn(HEADS * (NOPE + VALUE), LATENT, device="cuda").half()
        per_head = weight.T.view(LATENT, HEADS, NOPE + VALUE)
        w_uk, w_uv = per_head.split([NOPE, VALUE], dim=-1)
        sets.append((w_uk.permute(1, 2, 0), w_uv.transpose(0, 1)))
    return sets


def _graph_time_us(launch, calls: int, repeats: int) -> float:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for index in range(calls):  # compile and warm up outside the capture
            launch(index)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for index in range(calls):
            launch(index)
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (repeats * calls)


def _case(name, rows, sets, which, args, tile=None):
    weights = [pair[which] for pair in sets]
    n = weights[0].shape[2]
    k = weights[0].shape[1]
    x = torch.randn(HEADS, rows, k, device="cuda").half()
    out = torch.empty(HEADS, rows, n, dtype=torch.half, device="cuda")
    reference = torch.bmm(x, weights[0])

    def cublas(index):
        torch.bmm(x, weights[index % len(weights)], out=out)

    def skinny(index):
        kv_b_bmm_skinny(x, weights[index % len(weights)], out, shown_tile)

    weight_bytes = weights[0].numel() * weights[0].element_size()
    major, minor = torch.cuda.get_device_capability()
    shown_tile = tile or kv_b_skinny_tile(weights[0], major * 10 + minor)
    results = {}
    for label, launch in (("cublas", cublas), ("skinny", skinny)):
        if label == "cublas" and tile is not None:
            continue
        microseconds = _graph_time_us(launch, args.calls, args.repeats)
        results[label] = microseconds
        launch(0)
        torch.accelerator.synchronize()
        error = (out.float() - reference.float()).abs().max().item()
        print(
            f"{name:18s} M={rows} {label:6s} tile={shown_tile} "
            f"{microseconds:8.2f} us {weight_bytes / microseconds / 1000:7.1f} GB/s "
            f"max|out-cublas|={error:.3g}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--sweep", action="store_true", help="tile grid, skinny only")
    args = parser.parse_args()

    device = torch.cuda.get_device_name()
    peak = next((v for key, v in PEAK_GBPS.items() if key in device), None)
    print(f"device {device}, peak {peak} GB/s, weight 16.78 MB per call")
    sets = _weights(args.layers)
    cases = (("W_UK_T (nn)", 0), ("W_UV (tn)", 1))
    for name, which in cases:
        for rows in args.rows:
            if not args.sweep:
                _case(name, rows, sets, which, args)
                continue
            for block_n, block_k, warps in itertools.product(
                (16, 32, 64, 128), (32, 64, 128, 256), (2, 4, 8)
            ):
                n, k = sets[0][which].shape[2], sets[0][which].shape[1]
                if n % block_n or k % block_k:
                    continue
                tile = KvBTile(block_n, block_k, warps)
                try:
                    _case(name, rows, sets, which, args, tile)
                except triton.runtime.errors.OutOfResources as error:
                    # The sweep grid includes tiles larger than a card's shared
                    # memory (Turing: 64 KiB); they are not candidates there.
                    print(f"{name:18s} M={rows} skinny tile={tile} skipped: {error}")


if __name__ == "__main__":
    main()
