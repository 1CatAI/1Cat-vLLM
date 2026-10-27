# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from benchmarks.deepseek_v41.oracle_primitives import (
    EngramHashState,
    GGUFWeights,
    engram_update,
    rms_norm,
    route,
)


def hash_state():
    # One layer, two heads per n-gram; small disjoint prime-sized buckets.
    return EngramHashState(
        torch.arange(10),
        torch.tensor([[11, 13, 17, 19, 23, 29]]),
        torch.tensor([[0, 11, 24, 41, 60, 83]]),
        torch.tensor([[3, 5, 7, 9]]),
        pad_token_id=2,
    )


def test_hash_dead_token_prevents_lookback_across_it():
    state = hash_state()
    ids = torch.tensor([[7, 8, 9]])
    actual = state(ids, 0, torch.tensor([[True, False, True]]))
    fresh = hash_state()(torch.tensor([[9]]), 0)
    assert torch.equal(actual[:, 2:], fresh)
    # The first token's 2-gram is (7*3) XOR (pad*5), then modulo 11.
    assert int(actual[0, 0, 0, 0]) == (7 * 3 ^ 2 * 5) % 11


def test_decode_uses_previous_chunk_but_reset_discards_it():
    state = hash_state()
    first = state(torch.tensor([[7, 8, 9]]), 0)
    next_token = state(torch.tensor([[3]]), 3)
    full = hash_state()(torch.tensor([[7, 8, 9, 3]]), 0)
    assert torch.equal(torch.cat((first, next_token), dim=1), full)
    assert torch.equal(
        state(torch.tensor([[3]]), 0), hash_state()(torch.tensor([[3]]), 0)
    )


@pytest.mark.parametrize("position", [-1, 1, 4])
def test_hash_rejects_missing_or_stale_context(position):
    with pytest.raises(ValueError, match="contiguous"):
        hash_state()(torch.tensor([[7]]), position)


def test_fp32_norm_epsilon_survives_zero_fp16_input():
    actual = rms_norm(torch.zeros(2, 4, dtype=torch.float16), torch.ones(4), 1e-20)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, torch.zeros(2, 4))


def test_routing_bias_selects_but_does_not_rescale_experts():
    weights, indices = route(
        torch.ones(1, 2),
        torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        torch.tensor([100.0, 0.0, 0.0]),
        topk=2,
    )
    assert indices.tolist() == [[0, 2]]
    raw = F.softplus(torch.tensor([2.0, 6.0])).sqrt()
    torch.testing.assert_close(weights[0], raw / raw.sum() * 1.5)


def test_engram_mask_is_exact_passthrough_even_with_nonzero_value():
    hidden = torch.arange(24).reshape(1, 3, 2, 4).float()
    kv = torch.ones(1, 3, 12)
    result = engram_update(
        hidden, kv, torch.ones(2, 4), torch.ones(2, 4), torch.zeros(1, 3).bool()
    )
    assert torch.equal(result, hidden)


@pytest.mark.parametrize(
    "kind", [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.IQ4_NL]
)
def test_mapped_rows_preserve_order_duplicates_and_linear_chunks(tmp_path, kind):
    source = np.arange(5 * 256, dtype=np.float32).reshape(5, 256) / 128 - 5
    if kind == gguf.GGMLQuantizationType.F32:
        packed = source
    else:
        # gguf-py decodes IQ4_NL but does not implement its encoder.
        blocks = np.random.default_rng(41).integers(
            0, 256, size=(5, 8, 18), dtype=np.uint8
        )
        scales = np.full((5, 8), 0.03125, dtype=np.float16)
        blocks[..., :2] = scales.view(np.uint8).reshape(5, 8, 2)
        packed = blocks.reshape(5, -1)
    path = tmp_path / "rows.gguf"
    writer = gguf.GGUFWriter(path, "test")
    writer.add_tensor("w", packed, raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    store = GGUFWeights([path])
    dense = torch.from_numpy(gguf.quants.dequantize(packed, kind).copy())
    ids = torch.tensor([[4, 1, 1], [0, 4, 3]])
    torch.testing.assert_close(store.rows("w", ids), dense[ids], rtol=0, atol=0)
    x = torch.ones(2, 256)
    torch.testing.assert_close(store.linear("w", x, 2), F.linear(x, dense))
    with pytest.raises(ValueError, match="Row outside"):
        store.rows("w", torch.tensor([-1]))
