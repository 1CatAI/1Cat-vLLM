# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention.sm70_qwen38_qk_rope import (
    _e4m3_satfinite,
    _qk_norm_rope,
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


@pytest.mark.parametrize("tokens", [1, 5, 8, 9, 16, 17, 32, 33, 64, 65])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_strided_positions_and_paged_cache(tokens, kv_heads):
    torch.manual_seed(22)
    q_heads = kv_heads * 6
    q_width, kv_width = q_heads * 256, kv_heads * 256
    qkv_width = q_width * 2 + kv_width * 2
    qkv = torch.randn(tokens, qkv_width + 32, device="cuda", dtype=torch.float16)
    qkv = qkv[:, :qkv_width]
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
    cache_backing = torch.full(
        (3, 2048, kv_heads, 264), 0x55, device="cuda", dtype=torch.uint8
    )
    value_backing = torch.full_like(cache_backing, 0x55)
    key_cache, value_cache = cache_backing[..., :256], value_backing[..., :256]
    expected_k = key_cache.clone()
    expected_v = value_cache.clone()
    scale = torch.tensor([0.7], device="cuda", dtype=torch.float32)
    q_dest = torch.empty(tokens, q_width, device="cuda", dtype=torch.float16)
    k_dest = torch.empty(tokens, kv_width, device="cuda", dtype=torch.float16)
    gate_dest = torch.empty_like(q_dest)
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
        gate_out=gate_dest,
    )
    expected_gate = (
        qkv[:, : q_width * 2]
        .view(tokens, q_heads, 512)[:, :, 256:]
        .reshape(tokens, q_width)
    )
    assert torch.equal(gate_dest.view(torch.int16), expected_gate.view(torch.int16))
    assert q.data_ptr() == q_dest.data_ptr() and k.data_ptr() == k_dest.data_ptr()
    for raw, weight, actual in [
        (qkv[:, : q_width * 2].view(tokens, q_heads, 512)[:, :, :256], qw, q),
        (
            qkv[:, q_width * 2 : q_width * 2 + kv_width].view(tokens, kv_heads, 256),
            kw,
            k,
        ),
    ]:
        values = raw.double()
        normalized = (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
            * (1.0 + weight.double())
        )
        frequencies = torch.arange(32, device="cuda")
        planes = torch.zeros(32, device="cuda", dtype=torch.int64)
        planes[(frequencies % 3 == 1) & (frequencies < 33)] = 1
        planes[(frequencies % 3 == 2) & (frequencies < 30)] = 2
        angle = cache[positions[planes].t(), frequencies].double()
        sine = cache[positions[planes].t(), frequencies + 32].double()
        left, right = normalized[:, :, :32], normalized[:, :, 32:64]
        result = normalized.clone()
        result[:, :, :32] = left * angle[:, None] - right * sine[:, None]
        result[:, :, 32:64] = right * angle[:, None] + left * sine[:, None]
        assert (actual - result.half().reshape(tokens, -1)).abs().max() <= 0.02
    ops.reshape_and_cache_flash(
        k.view(tokens, kv_heads, 256),
        qkv[:, q_width * 2 + kv_width :].view(tokens, kv_heads, 256),
        expected_k,
        expected_v,
        slots,
        "fp8_e4m3",
        scale,
        scale,
    )
    torch.accelerator.synchronize()
    assert torch.equal(key_cache, expected_k) and torch.equal(value_cache, expected_v)
    assert torch.all(cache_backing[..., 256:] == 0x55)
    assert torch.all(value_backing[..., 256:] == 0x55)


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_new_lengths_reuse_the_compiled_kernel(kv_heads):
    qkv = torch.randn(1403, kv_heads * 3584, device="cuda", dtype=torch.float16)
    weight = torch.randn(256, device="cuda", dtype=torch.float16) * 0.125
    cache = torch.randn(10000, 64, device="cuda", dtype=torch.float16)
    positions = torch.arange(1403, device="cuda", dtype=torch.int64)

    def compiled():
        return sum(len(kernels) for kernels, *_ in _qk_norm_rope.device_caches.values())

    qk_norm_rope(qkv[:5], weight, weight, positions[:5], cache)
    before = compiled()
    for tokens in (7, 16, 758, 1403):
        q, k = qk_norm_rope(qkv[:tokens], weight, weight, positions[:tokens], cache)
        assert q.shape == (tokens, kv_heads * 1536)
        assert k.shape == (tokens, kv_heads * 256)
    assert compiled() == before


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_qk_preparation_replays_live_positions_and_cache_slots(kv_heads):
    tokens = 17
    qkv = torch.randn(tokens, kv_heads * 3584, device="cuda", dtype=torch.float16)
    weight = torch.randn(256, device="cuda", dtype=torch.float16) * 0.125
    angles = torch.randn(10000, 64, device="cuda", dtype=torch.float16)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    key_cache = torch.full(
        (2, 128, kv_heads, 256), 0x55, device="cuda", dtype=torch.uint8
    )
    value_cache = torch.full_like(key_cache, 0x55)
    scale = torch.tensor([0.7], device="cuda", dtype=torch.float32)

    def run():
        return qk_norm_rope(
            qkv,
            weight,
            weight,
            positions,
            angles,
            key_cache=key_cache,
            value_cache=value_cache,
            slots=slots,
            k_scale=scale,
            v_scale=scale,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, k = run()
    for iteration in range(3):
        qkv.normal_()
        positions.add_(123)
        slots.copy_(torch.arange(tokens, device="cuda") + 127 + iteration * 17)
        slots[1] = -1
        key_cache.fill_(0x55)
        value_cache.fill_(0x55)
        graph.replay()
        replay_q, replay_k = q.clone(), k.clone()
        replay_cache = key_cache.clone(), value_cache.clone()
        key_cache.fill_(0x55)
        value_cache.fill_(0x55)
        eager_q, eager_k = run()
        assert torch.equal(replay_q, eager_q) and torch.equal(replay_k, eager_k)
        assert torch.equal(replay_cache[0], key_cache)
        assert torch.equal(replay_cache[1], value_cache)
