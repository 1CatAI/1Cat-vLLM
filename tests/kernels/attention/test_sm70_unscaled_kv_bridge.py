# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact payload expansion and live FP32 scale planes for dense consumers."""

import pytest
import torch

from tests.kernels.attention.test_sm70_kv_engine_bindings import cache
from vllm.v1.attention.kv_codecs import KV_CODECS

CODECS = tuple(c for c in KV_CODECS if c.native_id is not None)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("page", [32, 1648])
def test_unscaled_bridge_payload_scales_and_graph(codec, page):
    from flash_attn_v100 import paged_kv_to_fp16

    assert codec.name in paged_kv_to_fp16.unscaled_kv_cache_codecs
    torch.manual_seed(943)
    _, view = cache(codec, pages=4, page=page, heads=2)
    table = torch.tensor([[3, 1], [0, 2]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([2 * page - 1, page + 3], device="cuda", dtype=torch.int32)
    output_blocks = (2 * page + 31) // 32
    payload = torch.empty(
        (2 * output_blocks, 2, 32, 2, 256), device="cuda", dtype=torch.float16
    )
    key, value = payload.unbind(1)
    # Both source and destination scale addresses are independent strided views.
    scale_storage = torch.empty(
        (2 * output_blocks, 32, 2, 4), device="cuda", dtype=torch.float32
    )
    key_scale, value_scale = scale_storage[..., 1], scale_storage[..., 3]

    def launch():
        paged_kv_to_fp16(
            view.key,
            view.value,
            table,
            lengths,
            key,
            value,
            0.003,
            3.14159,
            kv_cache_dtype=codec.name,
            **view.scale_kwargs,
            key_scale_out=key_scale,
            value_scale_out=value_scale,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for lens in ((2 * page - 1, page + 3), (0, 1), (17, 2 * page - 5)):
        lengths.copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
        table.copy_(table.flip(1))
        if codec.per_token_head:
            view.key_scale.mul_(1.125)
            view.value_scale.mul_(0.875)
        payload.fill_(float("nan"))
        scale_storage.fill_(float("nan"))
        graph.replay()
        for batch, length in enumerate(lens):
            for src, dst, src_scale, dst_scale, tensor_scale in (
                (view.key, key, view.key_scale, key_scale, 0.003),
                (view.value, value, view.value_scale, value_scale, 3.14159),
            ):
                source = src[table[batch].long()].flatten(0, 1)
                if codec.quantized and not codec.per_token_head:
                    dtype = (
                        torch.float8_e4m3fn
                        if codec.native_id == 1
                        else torch.float8_e5m2
                    )
                    source = source.view(dtype)
                expected = source.half()
                actual = dst[
                    batch * output_blocks : (batch + 1) * output_blocks
                ].flatten(0, 1)
                scales = dst_scale[
                    batch * output_blocks : (batch + 1) * output_blocks
                ].flatten(0, 1)
                torch.testing.assert_close(
                    actual[:length], expected[:length], rtol=0, atol=0
                )
                expected_scale = (
                    src_scale[table[batch].long()].flatten(0, 1)
                    if codec.per_token_head
                    else torch.full_like(
                        scales, tensor_scale if codec.quantized else 1.0
                    )
                )
                torch.testing.assert_close(
                    scales[:length], expected_scale[:length], rtol=0, atol=0
                )
                padded = (length + 15) // 16 * 16
                assert bool((actual[length:padded] == 0).all())
                assert bool((scales[length:padded] == 1).all())
                assert bool(torch.isnan(actual[padded:]).all())
                assert bool(torch.isnan(scales[padded:]).all())
