# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused publishers must produce the ordinary writer's exact cache bytes."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention.sm70_dflash2_qk_rope import qk_norm_rope_cache
from vllm.model_executor.layers.attention.sm70_qwen38_qk_rope import qk_norm_rope
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
    reason="SM70 fused publication",
)


@pytest.mark.parametrize(
    "codec", (FP16, FP8_E4M3, FP8_E5M2, INT8, INT8_PER_TOKEN_HEAD), ids=lambda c: c.name
)
@pytest.mark.parametrize(
    "gated", (False, True), ids=("full_rope", "partial_gated_rope")
)
def test_fused_encoder_matches_writer_and_replays_slots(codec, gated):
    torch.manual_seed(814)
    rows, dim, heads = 8, (256 if gated else 128), (1 if gated else 2)
    raw = torch.randn(
        (rows, 3584 if gated else 1536), device="cuda", dtype=torch.float16
    )
    qw = torch.randn(dim, device="cuda", dtype=torch.float16) * 0.125
    kw = torch.randn_like(qw) * 0.125
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) + 33
    rope = torch.randn((128, 64 if gated else 128), device="cuda", dtype=torch.float16)
    carrier = torch.full(
        (4, 2, 32, heads, dim + (4 if codec.per_token_head else 0)),
        57,
        device="cuda",
        dtype=codec.storage_dtype,
    )
    view = KVCacheView.bind(carrier, codec, dim)
    slots = torch.tensor(
        [65, 31, -1, 32, 127, 7, 98, 49], device="cuda", dtype=torch.int64
    )
    ks, vs = (torch.tensor([s], device="cuda") for s in (0.7, 0.13))
    q = raw.new_empty((rows, 1536 if gated else 1024))
    k = raw.new_empty((rows, heads * dim))
    arguments = dict(
        key_cache=view.key,
        value_cache=view.value,
        slots=slots,
        q_out=q,
        k_out=k,
        kv_cache_dtype=codec.name,
        k_scale=ks,
        v_scale=vs,
        **view.scale_kwargs,
    )

    def fused():
        operation = qk_norm_rope if gated else qk_norm_rope_cache
        operation(raw, qw, kw, positions, rope, **arguments)

    fused()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused()
    for repeat in range(2):
        raw.mul_(1.5)
        raw[repeat].zero_()
        slots.copy_(slots.flip(0))
        expected = carrier.clone()
        graph.replay()
        expected_view = KVCacheView.bind(expected, codec, dim)
        value = raw[:, 3328 if gated else 1280 :].reshape(rows, heads, dim)
        if codec.per_token_head:
            triton_reshape_and_cache_flash_per_token_head_quant(
                k.view(rows, heads, dim),
                value,
                expected_view.key,
                expected_view.value,
                expected_view.key_scale,
                expected_view.value_scale,
                slots,
                round_int8_even=codec is INT8,
            )
        else:
            ops.reshape_and_cache_flash(
                k.view(rows, heads, dim),
                value,
                expected_view.key,
                expected_view.value,
                slots,
                codec.name,
                ks,
                vs,
            )
        torch.accelerator.synchronize()
        assert torch.equal(carrier.view(torch.uint8), expected.view(torch.uint8))
