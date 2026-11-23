# SPDX-License-Identifier: Apache-2.0
"""Exact top-k for a six-row graph whose KV-length bound covers at most k keys."""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _all_positions(positions, valid, first, second, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    length = tl.where(tl.load(valid + row), tl.load(positions + row) + 1, 0)
    lane = tl.arange(0, WIDTH)
    tl.store(first + row * WIDTH + lane, tl.where(lane < length, lane, -1))
    half = tl.arange(0, WIDTH // 2)
    tl.store(
        second + row * (WIDTH // 2) + half,
        tl.where(half < length // 2, half, -1),
    )


def all_compressed_positions(positions, valid, upper_bound):
    # The indexer returns live indices in ascending order, followed by -1.
    # A shorter trailing padding region is legal for attn_paged_dec.
    assert upper_bound in (128, 512)
    rows = positions.numel()
    options = dict(device=positions.device, dtype=torch.int32)
    first = torch.empty((rows, upper_bound), **options)
    second = torch.empty((rows, upper_bound // 2), **options)
    _all_positions[(rows,)](positions, valid, first, second, WIDTH=upper_bound)
    return {1: first, 2: second}
