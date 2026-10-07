# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.sm70_dmv13_projection import _fp16_weight


def _layer(parts):
    tensors = [part.contiguous().view(torch.uint8) for part, _ in parts]
    raw = tensors[0]
    raw.data_container = tensors
    raw.shard_id = list(range(len(tensors)))
    raw.shard_id_map = {i: i for i in raw.shard_id}
    types = {i: kind for i, (_, kind) in enumerate(parts)}
    return SimpleNamespace(
        qweight=raw,
        qweight_type=SimpleNamespace(shard_weight_type=types),
    )


@pytest.mark.parametrize("kinds", [(1,), (30,), (30, 30), (1, 30)])
def test_merged_dense_rows_use_existing_fp16_weight_contract(kinds):
    parts = []
    for i, kind in enumerate(kinds):
        dtype = torch.bfloat16 if kind == 30 else torch.float16
        values = (torch.arange(24).reshape(3, 8).float() / 37 + i).to(dtype)
        parts.append((values, kind))
    got = _fp16_weight(_layer(parts))
    expected = torch.cat([part.half() for part, _ in parts])
    assert got is not None and got.dtype == torch.float16
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("value", [65536.0, float("inf"), float("nan")])
def test_unrepresentable_bf16_weight_declines_fp16_fusion(value):
    parts = [(torch.full((3, 8), value, dtype=torch.bfloat16), 30)]
    assert _fp16_weight(_layer(parts)) is None


def test_mixed_width_rows_decline_fusion():
    assert (
        _fp16_weight(
            _layer([(torch.ones(3, 8).half(), 1), (torch.ones(3, 16).half(), 1)])
        )
        is None
    )
