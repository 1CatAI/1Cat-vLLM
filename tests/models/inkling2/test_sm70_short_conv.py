# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 decode tile of short_conv vs the generic decode tile (V100 only)."""

import pytest
import torch

from vllm.models.inkling2.nvidia.ops import is_sm70
from vllm.models.inkling2.nvidia.ops import sconv as sconv_ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm70(), reason="SM70 GPU"
)

H, WS, OFF, STATE, PAGE, W = 18, 128, 128, 256, 4, 4  # 2304 channels, K/V-like
MAX_BLOCKS = 11  # positions up to 43


def _decode_batch(tokens: int):
    """One decode token per request at mixed positions (incl. sequence starts)."""
    g = torch.Generator(device="cpu").manual_seed(tokens)
    pos = torch.randint(0, MAX_BLOCKS * PAGE, (tokens,), generator=g)
    pos[: min(tokens, 3)] = torch.tensor([0, 1, 2])[: min(tokens, 3)]
    block_table = torch.arange(tokens * MAX_BLOCKS, dtype=torch.int32).view(
        tokens, MAX_BLOCKS
    )
    rows = torch.arange(tokens)
    slots = block_table[rows, pos // PAGE].long() * PAGE + pos % PAGE
    cache = torch.randn(tokens * MAX_BLOCKS, H, PAGE, STATE, generator=g).half()
    x = torch.randn(tokens, H * WS, generator=g).half()
    weight = (torch.randn(H * WS, W, generator=g) * 0.5).half()
    addend = torch.randn(tokens, H * WS, generator=g).half()
    residual = torch.randn(tokens, H * WS, generator=g).half()
    cuda = {
        "x": x,
        "weight": weight,
        "cache": cache,
        "positions": pos.int(),
        "block_table": block_table,
        "seq_idx": rows.int(),
        "slot_mapping": slots,
        "query_start": rows.int(),
        "addend": addend,
        "residual": residual,
    }
    return {k: v.cuda() for k, v in cuda.items()}


def _run(batch, with_addend: bool):
    cache = batch["cache"].clone()
    out = sconv_ops.short_conv(
        batch["x"],
        batch["weight"],
        cache,
        batch["positions"],
        batch["block_table"],
        batch["seq_idx"],
        batch["slot_mapping"],
        batch["query_start"],
        OFF,
        WS,
        batch["addend"] if with_addend else None,
        batch["residual"] if with_addend else None,
    )
    return out, cache


@pytest.mark.parametrize("tokens", [1, 3, 8])
@pytest.mark.parametrize("with_addend", [False, True])
def test_decode_tile_matches_generic_tile(tokens, with_addend, monkeypatch):
    batch = _decode_batch(tokens)
    out, cache = _run(batch, with_addend)  # SM70: 128-channel decode tiles
    monkeypatch.setattr(sconv_ops, "is_sm70", lambda: False)  # 1024-channel tiles
    ref_out, ref_cache = _run(batch, with_addend)
    # A pure tiling change: each channel's convolution is computed alike.
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(cache, ref_cache, rtol=0, atol=0)
    # The current inputs were written to their slots of the conv-state cache.
    # cache: [block, head, slot, state] -> rows by flat slot
    flat = cache.permute(0, 2, 1, 3).reshape(-1, H, STATE)[batch["slot_mapping"]]
    torch.testing.assert_close(
        flat[:, :, OFF : OFF + WS].reshape(tokens, H * WS), batch["x"], rtol=0, atol=0
    )
