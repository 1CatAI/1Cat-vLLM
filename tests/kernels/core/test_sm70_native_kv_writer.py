# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native publication agrees with reference encoding and preserves untouched bytes."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    INT8,
    INT8_PER_TOKEN_HEAD,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_per_token_head_quant,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 native cache publication",
)


@pytest.mark.parametrize(
    "codec", (FP16, FP8_E4M3, FP8_E5M2, INT8, INT8_PER_TOKEN_HEAD), ids=lambda c: c.name
)
@pytest.mark.parametrize("dim,heads", [(64, 1), (80, 2), (128, 2), (256, 1)])
def test_common_native_writer_codec_bytes_and_replay(codec, dim, heads):
    torch.manual_seed(197)
    rows = 7
    source = torch.randn((rows, 2, heads, dim), device="cuda", dtype=torch.float16)
    source[0].zero_()
    source[1].fill_(0.5)
    source[1, :, :, -1] = 127
    source[2] *= 200
    key, value = source.unbind(1)
    carrier = torch.full(
        (4, 2, 32, heads, dim + (4 if codec.per_token_head else 0)),
        57,
        dtype=codec.storage_dtype,
        device="cuda",
    )
    view = KVCacheView.bind(carrier, codec, dim)
    slots = torch.tensor([65, 31, -1, 32, 127, 7, 98], device="cuda")
    ks, vs = (torch.tensor([s], device="cuda") for s in (0.7, 0.13))

    def publish():
        ops.reshape_and_cache_flash(
            key,
            value,
            view.key,
            view.value,
            slots,
            codec.name,
            ks,
            vs,
            view.key_scale,
            view.value_scale,
        )

    publish()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        publish()
    for repeat in range(2):
        source.mul_(1.5)
        source[repeat].zero_()
        slots.copy_(slots.flip(0))
        expected = carrier.clone()
        other = KVCacheView.bind(expected, codec, dim)
        if codec.per_token_head:
            triton_reshape_and_cache_flash_per_token_head_quant(
                key,
                value,
                other.key,
                other.value,
                other.key_scale,
                other.value_scale,
                slots,
                round_int8_even=codec is INT8,
            )
        else:
            for row, slot in enumerate(slots.tolist()):
                if slot < 0:
                    continue
                block, token = divmod(slot, 32)
                if codec is FP16:
                    other.key[block, token] = key[row]
                    other.value[block, token] = value[row]
                else:
                    limit = torch.finfo(codec.element_dtype).max
                    other.key[block, token] = (
                        (key[row].float() / ks)
                        .clamp(-limit, limit)
                        .to(codec.element_dtype)
                        .view(torch.uint8)
                    )
                    other.value[block, token] = (
                        (value[row].float() / vs)
                        .clamp(-limit, limit)
                        .to(codec.element_dtype)
                        .view(torch.uint8)
                    )
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(carrier.view(torch.uint8), expected.view(torch.uint8))


def test_native_int8_invalid_rows_propagate_without_touching_masked_slots():
    carrier = torch.zeros((2, 2, 16, 1, 68), dtype=torch.int8, device="cuda")
    view = KVCacheView.bind(carrier, INT8, 64)
    key = torch.zeros((4, 1, 64), dtype=torch.float32, device="cuda")
    value = key.clone()
    key[0, 0, 0] = float("nan")
    value[1, 0, 3] = float("inf")
    key[2, 0, 7] = torch.finfo(torch.float32).max
    key[3].fill_(1.0)
    slots = torch.tensor([1, 17, -1, 32], device="cuda")
    scale = torch.ones(1, device="cuda")
    ops.reshape_and_cache_flash(
        key,
        value,
        view.key,
        view.value,
        slots,
        INT8.name,
        scale,
        scale,
        view.key_scale,
        view.value_scale,
    )
    assert torch.isnan(view.key_scale[0, 1, 0])
    assert torch.isnan(view.value_scale[1, 1, 0])
    assert torch.count_nonzero(view.key[1]) == 0
    assert torch.isfinite(view.key_scale[1]).all()
