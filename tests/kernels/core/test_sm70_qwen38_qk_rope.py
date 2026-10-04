# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention.sm70_qwen38_qk_rope import (
    _e4m3_satfinite,
    qk_norm_rope,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="SM70 Qwen preparation"
)


@triton.jit
def _encode_test(In, Out, Scale):
    col = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.load(In + col).to(tl.float32)
    tl.store(Out + col, _e4m3_satfinite(tl.div_rn(value, tl.load(Scale))))


@pytest.mark.parametrize("divisor", [0.125, 0.7, 1.0, 2.0, 0.01])
def test_all_half_codes_match_cache_writer(divisor):
    source = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.float16)
        .view(256, 1, 256)
    )
    key_cache = torch.empty((2, 128, 1, 256), device="cuda", dtype=torch.uint8)
    value_cache = torch.empty_like(key_cache)
    output = torch.empty(65536, device="cuda", dtype=torch.uint8)
    slots = torch.arange(256, device="cuda", dtype=torch.int64)
    scale = torch.tensor([divisor], device="cuda", dtype=torch.float32)
    ops.reshape_and_cache_flash(
        source, source, key_cache, value_cache, slots, "fp8_e4m3", scale, scale
    )
    _encode_test[(256,)](source, output, scale, num_warps=4, enable_fp_fusion=False)
    torch.accelerator.synchronize()
    assert torch.equal(key_cache.flatten(), output)


@pytest.mark.parametrize("tokens", [1, 8, 32])
def test_strided_positions_and_paged_cache(tokens):
    torch.manual_seed(22)
    qkv = torch.randn(tokens, 3584, device="cuda", dtype=torch.float16)
    qw = torch.randn(256, device="cuda", dtype=torch.float16) * 0.125
    kw = torch.randn_like(qw) * 0.125
    cache = torch.randn(10000, 64, device="cuda", dtype=torch.float16)
    backing = torch.arange(3 * (tokens + 5), device="cuda", dtype=torch.int64)
    # Model position buffers retain a padded row stride.
    backing = backing.view(3, tokens + 5) + 1024
    positions = backing[:, :tokens]
    positions[0, 0] = -1
    slots = torch.arange(2047, 2047 + tokens, device="cuda", dtype=torch.int64)
    if tokens > 1:
        slots[1] = -1
    key_cache = torch.full((3, 2048, 1, 256), 0x55, device="cuda", dtype=torch.uint8)
    value_cache = torch.full_like(key_cache, 0x55)
    expected_k = key_cache.clone()
    expected_v = value_cache.clone()
    scale = torch.tensor([0.7], device="cuda", dtype=torch.float32)
    q_dest = torch.empty(tokens, 1536, device="cuda", dtype=torch.float16)
    k_dest = torch.empty(tokens, 256, device="cuda", dtype=torch.float16)
    q, k = qk_norm_rope(
        qkv,
        qw,
        kw,
        positions,
        cache,
        key_cache=key_cache,
        value_cache=value_cache,
        slots=slots,
        k_scale=scale,
        v_scale=scale,
        q_out=q_dest,
        k_out=k_dest,
    )
    assert q.data_ptr() == q_dest.data_ptr() and k.data_ptr() == k_dest.data_ptr()
    for raw, weight, actual in [
        (qkv[:, :3072].view(tokens, 6, 512)[:, :, :256], qw, q),
        (qkv[:, 3072:3328].view(tokens, 1, 256), kw, k),
    ]:
        values = raw.float()
        normalized = (
            (
                values
                * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
                * (1.0 + weight.float())
            )
            .half()
            .float()
        )
        frequencies = torch.arange(32, device="cuda")
        planes = torch.zeros(32, device="cuda", dtype=torch.int64)
        planes[(frequencies % 3 == 1) & (frequencies < 33)] = 1
        planes[(frequencies % 3 == 2) & (frequencies < 30)] = 2
        angle = cache[positions[planes].t(), frequencies].float()
        sine = cache[positions[planes].t(), frequencies + 32].float()
        left, right = normalized[:, :, :32], normalized[:, :, 32:64]
        result = normalized.clone()
        result[:, :, :32] = left * angle[:, None] - right * sine[:, None]
        result[:, :, 32:64] = right * angle[:, None] + left * sine[:, None]
        assert (actual - result.half().reshape(tokens, -1)).abs().max() <= 0.02
    ops.reshape_and_cache_flash(
        k.view(tokens, 1, 256),
        qkv[:, 3328:].view(tokens, 1, 256),
        expected_k,
        expected_v,
        slots,
        "fp8_e4m3",
        scale,
        scale,
    )
    torch.accelerator.synchronize()
    assert torch.equal(key_cache, expected_k) and torch.equal(value_cache, expected_v)
