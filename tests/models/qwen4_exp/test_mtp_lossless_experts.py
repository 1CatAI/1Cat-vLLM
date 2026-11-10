# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.models.qwen4_exp.nvidia.mtp_lossless_experts import (
    MTPLosslessConfig,
    MTPLosslessMoEMethod,
    pack_lossless_fp16,
    unpack_lossless_fp16,
)


def test_lossless_expert_config_keeps_dense_linears_unquantized():
    linear = object.__new__(LinearBase)
    assert isinstance(
        MTPLosslessConfig().get_quant_method(linear, "mtp.fc_embedding"),
        UnquantizedLinearMethod,
    )


def test_every_admitted_fp16_pattern_is_bit_exact_including_signed_subnormals():
    magnitudes = torch.cat((torch.arange(1024), torch.arange(1024, 0x6000, 8)))
    bits = torch.cat((magnitudes, magnitudes | 0x8000)).to(torch.uint16)
    source = bits.view(torch.float16).reshape(-1, 32)
    packed = pack_lossless_fp16(source)
    assert packed.numel() * 16 == source.numel() * 2 * 13
    assert torch.equal(
        unpack_lossless_fp16(packed).view(torch.uint16), source.view(torch.uint16)
    )


@pytest.mark.parametrize("value", [512.0, float("inf"), float("nan"), 0.10004])
def test_unrepresentable_weights_fail_without_requantizing(value):
    with pytest.raises(ValueError, match="outside the exact"):
        pack_lossless_fp16(torch.full((1, 32), value, dtype=torch.float16))


@pytest.mark.parametrize("rank", range(4))
def test_loader_slices_in_element_units_before_packing(rank):
    method = object.__new__(MTPLosslessMoEMethod)
    method.loaded = {shard: set() for shard in ("w1", "w3", "w2")}
    layer = SimpleNamespace(tp_rank=rank)
    w13 = torch.empty((1, 320, 4160), dtype=torch.uint8)
    w2 = torch.empty((1, 2560, 260), dtype=torch.uint8)
    for shard, bank, shape in (
        ("w1", w13, (640, 2560)),
        ("w3", w13, (640, 2560)),
        ("w2", w2, (2560, 640)),
    ):
        source = (torch.arange(shape[0] * shape[1]).reshape(shape) % 151 - 75).to(
            torch.bfloat16
        ) / 128
        method.load_expert(layer, bank, source, shard, 0)
        expected = (
            source[:, rank * 160 : (rank + 1) * 160]
            if shard == "w2"
            else source[rank * 160 : (rank + 1) * 160]
        )
        packed = (
            bank[0]
            if shard == "w2"
            else bank[0, :160]
            if shard == "w1"
            else bank[0, 160:]
        )
        assert torch.equal(
            unpack_lossless_fp16(packed).view(torch.uint16),
            expected.half().contiguous().view(torch.uint16),
        )
        with pytest.raises(ValueError, match="Duplicate"):
            method.load_expert(layer, bank, source, shard, 0)
    with pytest.raises(ValueError, match="Incomplete"):
        method.process_weights_after_loading(layer)
