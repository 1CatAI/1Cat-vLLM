# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator

import pytest
import torch

import vllm.envs as envs
from vllm.models.deepseek_v4.sm70.gemv import (
    _has_sm70_dsv4_fp13_weight_contract,
    _pack_sm70_dsv4_fp13_weight,
    can_use_sm70_dsv4_fp13_gemv,
    can_use_sm70_dsv4_fp16_gemv,
    maybe_sm70_dsv4_fp16_gemv,
)


@pytest.fixture(autouse=True)
def reset_env_cache() -> Iterator[None]:
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def test_sm70_dsv4_fp13_gemv_is_default_on_with_rollback(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SM70_DSV4_FP16_GEMV", raising=False)
    monkeypatch.delenv("VLLM_SM70_DSV4_FP13_GEMV", raising=False)
    assert not envs.VLLM_SM70_DSV4_FP16_GEMV
    assert envs.VLLM_SM70_DSV4_FP13_GEMV

    monkeypatch.setenv("VLLM_SM70_DSV4_FP13_GEMV", "0")
    envs.disable_envs_cache()
    assert not envs.VLLM_SM70_DSV4_FP13_GEMV


def test_sm70_dsv4_fp13_enables_fp16_fallback(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_DSV4_FP16_GEMV", "0")
    monkeypatch.setenv("VLLM_SM70_DSV4_FP13_GEMV", "1")
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.sm70.gemv._has_sm70_dsv4_gemv_contract",
        lambda *args: True,
    )
    x = torch.empty((1, 4096), dtype=torch.float16)
    weight = torch.empty((256, 4096), dtype=torch.float16)
    assert can_use_sm70_dsv4_fp16_gemv(x, weight, torch.float32)


def test_sm70_dsv4_gemv_rejects_cpu_tensors(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_DSV4_FP13_GEMV", "1")
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.sm70.gemv.current_platform.is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.sm70.gemv.current_platform.is_device_capability",
        lambda capability: capability == (7, 0),
    )
    x = torch.empty((1, 4096), dtype=torch.float16)
    weight = torch.empty((256, 4096), dtype=torch.float16)
    assert not can_use_sm70_dsv4_fp16_gemv(x, weight, torch.float32)


def test_sm70_dsv4_fp13_weight_contract_is_tensor_based() -> None:
    compatible_bits = torch.tensor([0x0000, 0x0007, 0x3C00, -0x4400], dtype=torch.int16)
    assert _has_sm70_dsv4_fp13_weight_contract(compatible_bits.view(torch.float16))

    normal_with_discarded_bit = torch.tensor([0x3C01], dtype=torch.int16)
    assert not _has_sm70_dsv4_fp13_weight_contract(
        normal_with_discarded_bit.view(torch.float16)
    )
    nonfinite = torch.tensor([0x7C00], dtype=torch.int16)
    assert not _has_sm70_dsv4_fp13_weight_contract(nonfinite.view(torch.float16))


def test_sm70_dsv4_fp13_pack_preserves_upper_13_bits() -> None:
    raw = (torch.arange(64 * 4096, dtype=torch.int32).mul(40503).add(17) & 0xFFFF).to(
        torch.uint16
    )
    weight = raw.view(torch.float16).reshape(64, 4096)
    packed = _pack_sm70_dsv4_fp13_weight(weight)
    assert packed.shape == (64, 128, 13)

    words = packed.to(torch.int64) & 0xFFFFFFFF
    unpacked = []
    for value_index in range(32):
        bit_offset = value_index * 13
        word_index = bit_offset // 32
        shift = bit_offset % 32
        code = words[..., word_index] >> shift
        if shift > 19:
            code |= words[..., word_index + 1] << (32 - shift)
        unpacked.append(code & 0x1FFF)
    codes = torch.stack(unpacked, dim=-1).reshape(64, 4096)
    expected = (raw.to(torch.int32) >> 3).reshape(64, 4096)
    assert torch.equal(codes.to(torch.int32), expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires NVIDIA V100/SM70",
)
@pytest.mark.parametrize(
    ("n", "output_dtype"),
    [
        (64, torch.float16),
        (256, torch.float32),
        (512, torch.float32),
        (1024, torch.float32),
        (2048, torch.float32),
    ],
)
def test_sm70_dsv4_fp16_gemv_graph(
    monkeypatch, n: int, output_dtype: torch.dtype
) -> None:
    monkeypatch.setenv("VLLM_SM70_DSV4_FP16_GEMV", "1")
    torch.manual_seed(20260802 + n)
    x = torch.randn((1, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randn((n, 4096), device="cuda", dtype=torch.float16) * 0.01

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        candidate = maybe_sm70_dsv4_fp16_gemv(x, weight, output_dtype)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert candidate is not None

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = maybe_sm70_dsv4_fp16_gemv(x, weight, output_dtype)
    assert captured is not None

    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    if output_dtype == torch.float16:
        reference = torch.mm(x, weight.T)
        torch.testing.assert_close(captured, reference, rtol=0, atol=0)
    else:
        reference = torch.mm(x, weight.T, out_dtype=torch.float32)
        torch.testing.assert_close(captured, reference, rtol=2e-4, atol=5e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires NVIDIA V100/SM70",
)
@pytest.mark.parametrize(
    ("n", "output_dtype"),
    [
        (64, torch.float16),
        (256, torch.float32),
        (512, torch.float32),
        (1024, torch.float32),
        (2048, torch.float32),
    ],
)
def test_sm70_dsv4_fp13_gemv_graph(
    monkeypatch, n: int, output_dtype: torch.dtype
) -> None:
    monkeypatch.setenv("VLLM_SM70_DSV4_FP16_GEMV", "0")
    monkeypatch.setenv("VLLM_SM70_DSV4_FP13_GEMV", "1")
    torch.manual_seed(20260826 + n)
    x = torch.randn((1, 4096), device="cuda", dtype=torch.float16)
    source = torch.randn((n, 4096), device="cuda", dtype=torch.bfloat16) * 0.01
    weight = source.to(torch.float16)
    packed = _pack_sm70_dsv4_fp13_weight(weight)
    assert can_use_sm70_dsv4_fp13_gemv(x, weight, packed, output_dtype)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        candidate = maybe_sm70_dsv4_fp16_gemv(
            x, weight, output_dtype, packed_weight=packed
        )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert candidate is not None

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = maybe_sm70_dsv4_fp16_gemv(
            x, weight, output_dtype, packed_weight=packed
        )
    assert captured is not None

    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    reference = torch.mm(x, weight.T, out_dtype=torch.float32)
    if output_dtype == torch.float16:
        reference = reference.to(torch.float16)
    torch.testing.assert_close(captured, reference, rtol=3e-4, atol=5e-5)
