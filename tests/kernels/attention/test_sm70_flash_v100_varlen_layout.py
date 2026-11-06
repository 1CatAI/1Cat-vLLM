# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from vllm import _custom_ops as ops


def _load_exactness_benchmark():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "benchmarks" / "benchmark_sm70_attention_exactness.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_sm70_attention_exactness", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sm70_flash_v100_varlen_type_a_layout_exact():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    bench = _load_exactness_benchmark()
    diagnostics = bench.run_varlen_layout_diagnostic_cases(flash_attn_v100)

    failures: list[str] = []
    for case in diagnostics:
        type_a = set(case["type_a_must_equal"])
        for comparison in case["comparisons"]:
            label = f"{comparison['name']} vs {comparison['reference']}"
            if label not in type_a:
                continue
            if not comparison["equal"] or comparison["max_diff"] != 0.0:
                failures.append(
                    f"{case['name']} {label}: "
                    f"equal={comparison['equal']} "
                    f"max_diff={comparison['max_diff']}"
                )

    assert not failures, "\n".join(failures)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_sm70_flash_v100_fp8_e5m2_paged_kv_read_exact():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    device = "cuda:0"
    batch_size = 1
    block_size = 4
    num_heads = 2
    head_dim = 64
    kv_cache_dtype = "fp8_e5m2"

    q_decode = torch.zeros(
        batch_size, num_heads, head_dim, dtype=torch.float16, device=device
    )
    q_prefill = q_decode.unsqueeze(1)
    block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([1], dtype=torch.int32, device=device)

    base16 = torch.tensor(
        [
            -64.0,
            -32.0,
            -16.0,
            -8.0,
            -4.0,
            -2.0,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
            32.0,
        ],
        dtype=torch.float16,
        device=device,
    )
    base = base16.repeat(head_dim // base16.numel())
    k_fp16 = torch.zeros(
        1, block_size, num_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16 = torch.zeros_like(k_fp16)
    for head_idx in range(num_heads):
        k_fp16[0, 0, head_idx] = base.roll(head_idx)
        v_fp16[0, 0, head_idx] = base.flip(0).roll(head_idx)

    k_cache = torch.empty_like(k_fp16, dtype=torch.uint8)
    v_cache = torch.empty_like(v_fp16, dtype=torch.uint8)
    ops.convert_fp8(k_cache, k_fp16, 1.0, kv_dtype=kv_cache_dtype)
    ops.convert_fp8(v_cache, v_fp16, 1.0, kv_dtype=kv_cache_dtype)

    expected_cache = torch.empty_like(v_fp16)
    ops.convert_fp8(expected_cache, v_cache, 1.0, kv_dtype=kv_cache_dtype)
    expected_decode = expected_cache[0, 0]

    decode_out = flash_attn_v100.flash_attn_decode_paged(
        q_decode,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        softmax_scale=1.0,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=1.0,
        v_scale=1.0,
    )
    prefill_out = flash_attn_v100.flash_attn_prefill_paged(
        q_prefill,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        softmax_scale=1.0,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=1.0,
        v_scale=1.0,
        causal=True,
    )

    torch.accelerator.synchronize()
    assert torch.equal(decode_out[0], expected_decode)
    assert torch.equal(prefill_out[0, 0], expected_decode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_sm70_flash_v100_mla_context_lse_matches_fp64():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    pytest.importorskip("flash_attn_v100")
    from vllm.v1.attention.backends.flash_attn_v100 import (
        flash_v100_dense_prefill_lse,
    )

    torch.manual_seed(200001)
    query_lens = [3, 2, 4]
    key_lens = [5, 0, 3]
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 256
    softmax_scale = head_dim**-0.5
    query = torch.randn(
        sum(query_lens),
        num_q_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    key = torch.randn(
        sum(key_lens),
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    value = torch.randn_like(key)
    cu_q = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(key_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )

    def run():
        output = torch.empty_like(query)
        lse = torch.empty(
            (num_q_heads, query.shape[0]), dtype=torch.float32, device="cuda"
        )
        return flash_v100_dense_prefill_lse(
            query,
            key,
            value,
            output,
            lse,
            cu_q,
            cu_k,
            query.shape[0],
            softmax_scale,
        )

    output, lse = run()
    output_repeat, lse_repeat = run()

    kv_head_for_q = torch.arange(num_q_heads, device="cuda") // (
        num_q_heads // num_kv_heads
    )
    output_refs = []
    lse_refs = []
    query_offset = 0
    key_offset = 0
    for query_len, key_len in zip(query_lens, key_lens):
        if key_len == 0:
            output_refs.append(
                torch.zeros(
                    (query_len, num_q_heads, head_dim),
                    dtype=torch.float64,
                    device="cuda",
                )
            )
            lse_refs.append(
                torch.full(
                    (num_q_heads, query_len),
                    -torch.inf,
                    dtype=torch.float64,
                    device="cuda",
                )
            )
        else:
            q_seq = query[query_offset : query_offset + query_len].double()
            k_seq = key[key_offset : key_offset + key_len, kv_head_for_q].double()
            v_seq = value[key_offset : key_offset + key_len, kv_head_for_q].double()
            scores = torch.einsum("qhd,khd->hqk", q_seq, k_seq) * softmax_scale
            probabilities = torch.softmax(scores, dim=-1)
            output_refs.append(torch.einsum("hqk,khd->qhd", probabilities, v_seq))
            lse_refs.append(torch.logsumexp(scores, dim=-1))
        query_offset += query_len
        key_offset += key_len

    output_ref = torch.cat(output_refs)
    lse_ref = torch.cat(lse_refs, dim=1)
    torch.accelerator.synchronize()

    assert torch.equal(output, output_repeat)
    assert torch.equal(lse, lse_repeat)
    empty_start = query_lens[0]
    empty_end = empty_start + query_lens[1]
    assert torch.count_nonzero(output[empty_start:empty_end]) == 0
    assert torch.isneginf(lse[:, empty_start:empty_end]).all()
    output_error = output.double() - output_ref
    relative_rms = torch.sqrt(torch.mean(output_error.square())) / torch.sqrt(
        torch.mean(output_ref.square())
    )
    assert float(relative_rms) < 5e-4
    finite_lse = torch.isfinite(lse_ref)
    assert float((lse.double()[finite_lse] - lse_ref[finite_lse]).abs().max()) < 2e-5

    bad_cu_q = cu_q.clone()
    bad_cu_q[-1] -= 1
    with pytest.raises(ValueError, match="cover every actual query token"):
        flash_v100_dense_prefill_lse(
            query,
            key,
            value,
            torch.empty_like(query),
            torch.empty_like(lse),
            bad_cu_q,
            cu_k,
            query.shape[0],
            softmax_scale,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_sm70_flash_v100_mla_causal_lse_matches_fp64():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    pytest.importorskip("flash_attn_v100")
    from vllm.v1.attention.backends.flash_attn_v100 import (
        flash_v100_dense_prefill_lse,
    )

    torch.manual_seed(200002)
    seq_lens = [1, 3, 5]
    num_heads = 2
    head_dim = 256
    softmax_scale = head_dim**-0.5
    shape = (sum(seq_lens), num_heads, head_dim)
    query = torch.randn(shape, dtype=torch.float16, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seq_lens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    output = torch.empty_like(query)
    lse = torch.empty((num_heads, query.shape[0]), dtype=torch.float32, device="cuda")

    flash_v100_dense_prefill_lse(
        query,
        key,
        value,
        output,
        lse,
        cu_seq_lens,
        cu_seq_lens.clone(),
        query.shape[0],
        softmax_scale,
        causal=True,
    )

    output_refs = []
    lse_refs = []
    offset = 0
    for seq_len in seq_lens:
        q_seq = query[offset : offset + seq_len].double()
        k_seq = key[offset : offset + seq_len].double()
        v_seq = value[offset : offset + seq_len].double()
        scores = torch.einsum("qhd,khd->hqk", q_seq, k_seq) * softmax_scale
        causal_mask = torch.ones(
            (seq_len, seq_len), dtype=torch.bool, device="cuda"
        ).triu(1)
        scores.masked_fill_(causal_mask, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        output_refs.append(torch.einsum("hqk,khd->qhd", probabilities, v_seq))
        lse_refs.append(torch.logsumexp(scores, dim=-1))
        offset += seq_len

    output_ref = torch.cat(output_refs)
    lse_ref = torch.cat(lse_refs, dim=1)
    torch.accelerator.synchronize()
    output_error = output.double() - output_ref
    relative_rms = torch.sqrt(torch.mean(output_error.square())) / torch.sqrt(
        torch.mean(output_ref.square())
    )
    assert float(relative_rms) < 5e-4
    assert float((lse.double() - lse_ref).abs().max()) < 2e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_sm70_flash_v100_bm32_phase_multi_kv_stride_exact(monkeypatch):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    torch.manual_seed(20260715)
    page_size = 784
    seq_len = 913
    query_len = 32
    num_q_heads = 6
    num_kv_heads = 2
    head_dim = 256

    key_cache = torch.randn(
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(
        1,
        query_len,
        num_q_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    block_table = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE", "0")
    expected = flash_attn_v100.flash_attn_prefill_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        causal=True,
    )
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE", "1")
    actual = flash_attn_v100.flash_attn_prefill_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        causal=True,
    )

    torch.accelerator.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    (
        "input_page_size",
        "seq_len",
        "query_len",
        "num_q_heads",
        "num_kv_heads",
    ),
    (
        (1568, 1700, 32, 6, 2),
        (1568, 1700, 48, 6, 2),
        (1056, 4096, 2048, 4, 1),
        (1056, 16384, 2048, 4, 1),
        (1056, 16384, 1056, 4, 1),
        (1088, 16384, 2048, 4, 1),
        (1088, 16384, 1088, 4, 1),
    ),
)
@torch.inference_mode()
def test_sm70_flash_v100_fp8_e5m2_bridge_prefill_exact(
    monkeypatch,
    input_page_size,
    seq_len,
    query_len,
    num_q_heads,
    num_kv_heads,
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    torch.manual_seed(20260715)
    output_page_size = 784
    head_dim = 256
    k_scale = 0.75
    v_scale = 1.25

    input_blocks = math.ceil(seq_len / input_page_size)
    input_shape = (input_blocks, input_page_size, num_kv_heads, head_dim)
    key_cache = (
        torch.randn(input_shape, dtype=torch.float16, device="cuda")
        .to(torch.float8_e5m2)
        .view(torch.uint8)
    )
    value_cache = (
        torch.randn(input_shape, dtype=torch.float16, device="cuda")
        .to(torch.float8_e5m2)
        .view(torch.uint8)
    )
    query = torch.randn(
        1,
        query_len,
        num_q_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    block_table = torch.arange(
        input_blocks - 1,
        -1,
        -1,
        dtype=torch.int32,
        device="cuda",
    ).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

    direct = flash_attn_v100.flash_attn_prefill_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        kv_cache_dtype="fp8_e5m2",
        k_scale=k_scale,
        v_scale=v_scale,
        causal=True,
    )

    output_blocks = math.ceil(block_table.shape[1] * input_page_size / output_page_size)
    sentinel = float("nan")
    output_shape = (
        output_blocks,
        output_page_size,
        num_kv_heads,
        head_dim,
    )
    key_out = torch.full(output_shape, sentinel, dtype=torch.float16, device="cuda")
    value_out = torch.full_like(key_out, sentinel)
    flash_attn_v100.fp8_e5m2_paged_kv_to_fp16(
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        key_out,
        value_out,
        k_scale,
        v_scale,
    )

    logical_key = torch.cat(
        [key_cache[physical_block] for physical_block in block_table[0].tolist()]
    ).view(torch.float8_e5m2)
    logical_value = torch.cat(
        [value_cache[physical_block] for physical_block in block_table[0].tolist()]
    ).view(torch.float8_e5m2)
    flat_key_out = key_out.flatten(0, 1)
    flat_value_out = value_out.flatten(0, 1)
    expected_key = (logical_key.float() * k_scale).half()
    expected_value = (logical_value.float() * v_scale).half()
    torch.testing.assert_close(
        flat_key_out[:seq_len], expected_key[:seq_len], rtol=0, atol=0
    )
    torch.testing.assert_close(
        flat_value_out[:seq_len], expected_value[:seq_len], rtol=0, atol=0
    )
    padded_seq_len = min(flat_key_out.shape[0], math.ceil(seq_len / 16) * 16)
    assert torch.count_nonzero(flat_key_out[seq_len:padded_seq_len]) == 0
    assert torch.count_nonzero(flat_value_out[seq_len:padded_seq_len]) == 0
    assert torch.all(torch.isnan(flat_key_out[padded_seq_len:]))
    assert torch.all(torch.isnan(flat_value_out[padded_seq_len:]))

    output_block_table = torch.arange(
        output_blocks, dtype=torch.int32, device="cuda"
    ).unsqueeze(0)
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE", "1")
    bridged = flash_attn_v100.flash_attn_prefill_paged(
        query,
        key_out,
        value_out,
        output_block_table,
        seq_lens,
        causal=True,
    )

    torch.accelerator.synchronize()
    assert torch.equal(bridged, direct)


@pytest.mark.parametrize(("k_scale", "v_scale"), ((1.0, 1.0), (0.75, 1.25)))
@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads", "block_size"),
    (
        (12, 2, 1568),
        (6, 1, 1568),
        (4, 1, 1056),
        (4, 1, 1088),
    ),
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_sm70_flash_v100_fp8_e5m2_xqa_decode_matches_scalar(
    k_scale,
    v_scale,
    num_q_heads,
    num_kv_heads,
    block_size,
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    torch.manual_seed(20260715)
    seq_len = 2049
    head_dim = 256
    cache_shape = (3, block_size, num_kv_heads, head_dim)
    key_cache = (
        torch.randn(cache_shape, dtype=torch.float16, device="cuda")
        .to(torch.float8_e5m2)
        .view(torch.uint8)
    )
    value_cache = (
        torch.randn(cache_shape, dtype=torch.float16, device="cuda")
        .to(torch.float8_e5m2)
        .view(torch.uint8)
    )
    query = torch.randn(
        1,
        num_q_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    block_table = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    kwargs = {
        "kv_cache_dtype": "fp8_e5m2",
        "k_scale": k_scale,
        "v_scale": v_scale,
        "max_seq_len_hint": seq_len,
    }

    expected = flash_attn_v100.flash_attn_decode_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        **kwargs,
    )
    actual = flash_attn_v100.flash_attn_decode_paged_xqa(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        **kwargs,
    )

    torch.accelerator.synchronize()
    delta = (actual.float() - expected.float()).abs()
    # The two reductions may round to adjacent fp16 outputs. Preserve the
    # original absolute gate near zero, but do not set it below one output ULP.
    positive = torch.full_like(expected, torch.inf)
    negative = torch.full_like(expected, -torch.inf)
    one_ulp = torch.maximum(
        (torch.nextafter(expected, positive) - expected).abs(),
        (torch.nextafter(expected, negative) - expected).abs(),
    ).float()
    assert torch.all(delta <= torch.maximum(one_ulp, torch.full_like(delta, 1e-4)))
    assert float(delta.mean()) <= 1e-5


@pytest.mark.parametrize(("k_scale", "v_scale"), ((1.0, 1.0), (0.04, 0.25)))
@pytest.mark.parametrize("partition_size", (64, 128, 256))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_sm70_flash_v100_fp8_e4m3_g6_xqa_decode_matches_scalar(
    k_scale,
    v_scale,
    partition_size,
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    torch.manual_seed(20260823)
    seq_len = 2049
    block_size = 1568
    head_dim = 256
    cache_shape = (3, block_size, 1, head_dim)
    key_cache = (
        torch.randn(cache_shape, dtype=torch.float16, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    value_cache = (
        torch.randn(cache_shape, dtype=torch.float16, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    query = torch.randn(1, 6, head_dim, dtype=torch.float16, device="cuda")
    block_table = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    kwargs = {
        "kv_cache_dtype": "fp8_e4m3",
        "k_scale": k_scale,
        "v_scale": v_scale,
        "max_seq_len_hint": seq_len,
    }

    expected = flash_attn_v100.flash_attn_decode_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partition_size_hint=256,
        **kwargs,
    )
    actual = flash_attn_v100.flash_attn_decode_paged_xqa(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partition_size_hint=partition_size,
        **kwargs,
    )

    torch.accelerator.synchronize()
    delta = (actual.float() - expected.float()).abs()
    positive = torch.full_like(expected, torch.inf)
    negative = torch.full_like(expected, -torch.inf)
    one_ulp = torch.maximum(
        (torch.nextafter(expected, positive) - expected).abs(),
        (torch.nextafter(expected, negative) - expected).abs(),
    ).float()
    assert torch.all(delta <= torch.maximum(one_ulp, torch.full_like(delta, 1e-4)))
    assert float(delta.mean()) <= 1e-5


@pytest.mark.parametrize("batch_size", (2, 16, 32))
@pytest.mark.parametrize("partition_size", (64, 128, 256))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_sm70_flash_v100_fp8_e4m3_batched_g6_xqa_matches_scalar(
    monkeypatch,
    batch_size,
    partition_size,
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    monkeypatch.setenv("VLLM_FLASH_V100_E4M3_BATCH_XQA", "1")
    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    torch.manual_seed(20260824)
    block_size = 1568
    head_dim = 256
    blocks_per_seq = 2
    num_blocks = batch_size * blocks_per_seq
    cache_shape = (num_blocks, block_size, 1, head_dim)
    key_cache = (
        torch.randn(cache_shape, dtype=torch.float16, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    value_cache = (
        torch.randn(cache_shape, dtype=torch.float16, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    query = torch.randn(
        batch_size,
        6,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    block_table = (
        torch.arange(num_blocks, dtype=torch.int32, device="cuda")
        .view(batch_size, blocks_per_seq)
        .flip(1)
    )
    seq_lens = 1025 + (torch.arange(batch_size, device="cuda") * 61) % 900
    seq_lens = seq_lens.to(torch.int32)
    kwargs = {
        "kv_cache_dtype": "fp8_e4m3",
        "k_scale": 0.04,
        "v_scale": 0.25,
        "max_seq_len_hint": int(seq_lens.max().item()),
    }

    expected = flash_attn_v100.flash_attn_decode_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partition_size_hint=256,
        **kwargs,
    )
    actual = flash_attn_v100.flash_attn_decode_paged_xqa(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partition_size_hint=partition_size,
        **kwargs,
    )

    torch.accelerator.synchronize()
    delta = (actual.float() - expected.float()).abs()
    positive = torch.full_like(expected, torch.inf)
    negative = torch.full_like(expected, -torch.inf)
    one_ulp = torch.maximum(
        (torch.nextafter(expected, positive) - expected).abs(),
        (torch.nextafter(expected, negative) - expected).abs(),
    ).float()
    assert torch.all(delta <= torch.maximum(one_ulp, torch.full_like(delta, 1e-4)))
    assert float(delta.mean()) <= 1e-5


@pytest.mark.parametrize("batch_size", (8, 16, 32))
@pytest.mark.parametrize("partition_size", (64, 128, 256))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_sm70_flash_v100_fp8_e4m3_batched_long_context_route_matches_scalar(
    monkeypatch,
    batch_size,
    partition_size,
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    monkeypatch.setenv("VLLM_FLASH_V100_E4M3_BATCH_XQA", "1")
    monkeypatch.delenv("VLLM_FLASH_V100_E4M3_BATCH_XQA_OPTIMIZED", raising=False)
    monkeypatch.setenv("VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH_TRACE", "0")
    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    torch.manual_seed(20260824 + batch_size)
    block_size = 800
    head_dim = 256
    seq_lens_cpu = 15360 + torch.arange(batch_size, dtype=torch.int32) * 113
    max_seq_len = int(seq_lens_cpu.max().item())
    blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_seq
    cache_shape = (num_blocks, 2, block_size, 1, head_dim)
    kv_cache = (
        torch.randn(cache_shape, dtype=torch.float16, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    key_cache, value_cache = kv_cache.unbind(1)
    query = torch.randn(
        batch_size,
        6,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    block_table = (
        torch.arange(num_blocks, dtype=torch.int32, device="cuda")
        .view(batch_size, blocks_per_seq)
        .flip(1)
    )
    seq_lens = seq_lens_cpu.to(device="cuda")
    kwargs = {
        "kv_cache_dtype": "fp8_e4m3",
        "k_scale": 0.04,
        "v_scale": 0.25,
        "max_seq_len_hint": max_seq_len,
        "workspace_seq_capacity_hint": 32768,
    }

    expected = flash_attn_v100.flash_attn_decode_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partition_size_hint=256,
        **kwargs,
    )
    monkeypatch.setenv("VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH", "0")
    fallback = flash_attn_v100.flash_attn_decode_paged_xqa(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partition_size_hint=partition_size,
        batch_context_routing=True,
        **kwargs,
    )
    monkeypatch.setenv("VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH", "1")
    actual = flash_attn_v100.flash_attn_decode_paged_xqa(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partition_size_hint=partition_size,
        batch_context_routing=True,
        **kwargs,
    )

    torch.accelerator.synchronize()
    assert torch.equal(actual, fallback)
    delta = (actual.float() - expected.float()).abs()
    positive = torch.full_like(expected, torch.inf)
    negative = torch.full_like(expected, -torch.inf)
    one_ulp = torch.maximum(
        (torch.nextafter(expected, positive) - expected).abs(),
        (torch.nextafter(expected, negative) - expected).abs(),
    ).float()
    assert torch.all(delta <= torch.maximum(one_ulp, torch.full_like(delta, 1e-4)))
    assert float(delta.mean()) <= 1e-5


@pytest.mark.parametrize(
    ("seq_len", "expected_partition"),
    (
        (8191, 64),
        (8192, 256),
        (16383, 256),
        (16384, 512),
        (32767, 512),
        (32768, 896),
        (49151, 896),
        (49152, 1664),
    ),
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("block_size", (800, 1568))
@torch.inference_mode()
def test_sm70_flash_v100_fp8_e4m3_g6_routes_wave_partitions_exactly(
    monkeypatch,
    seq_len,
    expected_partition,
    block_size,
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")

    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_E4M3_G6_P256_BEGIN", "8192")
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_E4M3_G6_P512_BEGIN", "16384")
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_E4M3_G6_P896_BEGIN", "32768")
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_E4M3_G6_P1664_BEGIN", "49152")
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH", "1")

    torch.manual_seed(20260824)
    head_dim = 256
    num_blocks = (seq_len + block_size - 1) // block_size
    cache_shape = (num_blocks, 2, block_size, 1, head_dim)
    kv_cache = (
        torch.randn(cache_shape, dtype=torch.float16, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    key_cache, value_cache = kv_cache.unbind(1)
    assert not key_cache.is_contiguous()
    assert not value_cache.is_contiguous()
    query = torch.randn(1, 6, head_dim, dtype=torch.float16, device="cuda")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(1, -1)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    kwargs = {
        "kv_cache_dtype": "fp8_e4m3",
        "k_scale": 0.04,
        "v_scale": 0.25,
        "max_seq_len_hint": seq_len,
        "workspace_seq_capacity_hint": 262144,
    }

    monkeypatch.setenv("VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO", "0")
    expected = flash_attn_v100.flash_attn_decode_paged_xqa(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partition_size_hint=expected_partition,
        **kwargs,
    ).clone()
    scalar = flash_attn_v100.flash_attn_decode_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partition_size_hint=256,
        **kwargs,
    )

    monkeypatch.setenv("VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO", "1")
    actual = flash_attn_v100.flash_attn_decode_paged_xqa(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partition_size_hint=64,
        batch_context_routing=True,
        **kwargs,
    )

    torch.accelerator.synchronize()
    assert torch.equal(actual, expected)
    delta = (actual.float() - scalar.float()).abs()
    positive = torch.full_like(scalar, torch.inf)
    negative = torch.full_like(scalar, -torch.inf)
    one_ulp = torch.maximum(
        (torch.nextafter(scalar, positive) - scalar).abs(),
        (torch.nextafter(scalar, negative) - scalar).abs(),
    ).float()
    assert torch.all(delta <= torch.maximum(one_ulp, torch.full_like(delta, 1e-4)))
    assert float(delta.mean()) <= 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "cache_dtype,torch_dtype",
    [("fp8_e4m3", torch.float8_e4m3fn), ("fp8_e5m2", torch.float8_e5m2)],
)
@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (0.75, 1.25)])
@pytest.mark.parametrize(
    "head_groups,num_q_heads,num_kv_heads,input_page_size,seq_len,query_len",
    [
        (1, 12, 4, 1568, 1700, 32),
        (2, 24, 4, 1568, 1700, 32),
        (4, 4, 4, 1056, 2048, 48),
        (3, 12, 6, 1088, 4096, 1024),
        (2, 6, 1, 1056, 1700, 32),  # MQA cannot be split.
        (3, 12, 4, 1568, 2048, 48),  # Non-divisor stays unsplit.
        (8, 12, 4, 1088, 1700, 32),  # More groups than KV heads.
        (0, 12, 4, 1568, 1700, 32),
        (-2, 12, 4, 1056, 2048, 48),
        ("auto1", 12, 4, 1568, 1700, 32),
        ("auto2", 24, 4, 1568, 1700, 32),
        ("auto4", 4, 4, 1056, 2048, 48),
        ("auto3", 12, 6, 1088, 4096, 1024),
    ],
)
@torch.inference_mode()
def test_sm70_flash_v100_fp8_bridge_head_groups_exact(
    monkeypatch,
    cache_dtype,
    torch_dtype,
    k_scale,
    v_scale,
    head_groups,
    num_q_heads,
    num_kv_heads,
    input_page_size,
    seq_len,
    query_len,
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")
    pytest.importorskip("flash_attn_v100")

    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    automatic = isinstance(head_groups, str)
    monkeypatch.setenv(
        "VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS",
        "1" if automatic else str(head_groups),
    )
    monkeypatch.setenv("VLLM_FLASH_V100_FP8_PREFILL_BRIDGE", "1")
    impl = FlashAttnV100Impl(
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=256,
        scale=256**-0.5,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=cache_dtype,
    )
    # Match the production operator owner: E4M3 uses the FA2 bridge, not the
    # similarly named conversion wrapper exported by flash_attn_v100.
    if cache_dtype == "fp8_e4m3":
        if impl.fp8_e4m3_paged_kv_to_fp16 is None:
            pytest.skip(
                "installed FA2 extension lacks "
                "_vllm_fa2_C.sm70_v37_e4m3_bridge"
            )
    elif impl.fp8_e5m2_paged_kv_to_fp16 is None:
        pytest.skip("installed Flash-V100 extension lacks the E5M2 bridge")
    torch.manual_seed(20261026)
    input_blocks = math.ceil(seq_len / input_page_size)
    shape = (input_blocks, input_page_size, num_kv_heads, 256)
    key_cache = (
        torch.randn(shape, dtype=torch.float16, device="cuda")
        .to(torch_dtype)
        .view(torch.uint8)
    )
    value_cache = (
        torch.randn(shape, dtype=torch.float16, device="cuda")
        .to(torch_dtype)
        .view(torch.uint8)
    )
    query = torch.randn(
        1, query_len, num_q_heads, 256, dtype=torch.float16, device="cuda"
    )
    # Reverse physical pages and retain a partial final page to catch indexing
    # errors independently of the GQA head ranges.
    block_table = torch.arange(
        input_blocks - 1, -1, -1, dtype=torch.int32, device="cuda"
    ).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    assert impl._should_use_fp8_prefill_bridge(
        q_len=query_len,
        head_dim=256,
        key_cache=key_cache,
        value_cache=value_cache,
        causal=True,
        window_size=(-1, -1),
    )
    arguments = dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        seq_len=seq_len,
        k_scale=k_scale,
        v_scale=v_scale,
        causal=True,
        window_size=(-1, -1),
    )
    unsplit = impl._run_fp8_prefill_bridge(
        **arguments, out=torch.empty_like(query), head_groups=1
    )
    assert unsplit is not None
    # The grouped execution reuses workspace; preserve the reference output.
    expected = unsplit[0].clone()
    if automatic:
        from types import SimpleNamespace

        from vllm.config import set_current_vllm_config
        from vllm.config.execution_policy import FlashV100Policy, GraphPolicy
        from vllm.config.flash_v100 import FlashV100Diagnostics, FlashV100Options
        from vllm.config.sm70_runtime import RuntimeTraceConfig
        from vllm.v1.attention.backends.flash_v100.config import (
            FP8_PREFILL_BRIDGE_PAGE_SIZE,
        )

        head_groups = int(head_groups.removeprefix("auto"))
        page_size = FP8_PREFILL_BRIDGE_PAGE_SIZE
        kv_tokens = math.ceil(
            input_blocks * input_page_size / page_size
        ) * page_size
        projected = (
            4 * kv_tokens * (num_kv_heads // head_groups) * 256
            + kv_tokens // page_size * 4
            + 4 * math.ceil(query_len / 64) * 64
            * (num_q_heads // head_groups) * 256
            + (
                4 * query_len * (num_q_heads // head_groups) * 256
                if head_groups > 1 else 0
            )
        )
        monkeypatch.delenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS")
        policy = FlashV100Policy(
            options=FlashV100Options(fp8_bridge_workspace_bytes=projected)
        )
        policy.resolve()
        policy.options.finalize(
            GraphPolicy(), FlashV100Diagnostics(), cache_dtype=cache_dtype
        )
        engine = SimpleNamespace(
            attention_config=SimpleNamespace(flash_v100=policy),
            model_config=SimpleNamespace(max_model_len=seq_len),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=query_len),
            cache_config=SimpleNamespace(block_size=16),
            observability_config=SimpleNamespace(runtime_trace=RuntimeTraceConfig()),
        )
        with set_current_vllm_config(engine):
            impl._initialize_prefill_policy()
        from vllm.v1.attention.backends.flash_v100.config import (
            configure_fp8_bridge_workspace_plans,
        )
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

        configure_fp8_bridge_workspace_plans(
            engine,
            [KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=input_page_size, num_kv_heads=num_kv_heads,
                    head_size=256, dtype=torch.uint8,
                ),
            )],
            {"attention": SimpleNamespace(impl=impl)},
        )
        assert impl.fp8_bridge_head_groups == head_groups
    from contextlib import nullcontext

    from vllm.runtime_resources import release_runtime_resources

    context = set_current_vllm_config(engine) if automatic else nullcontext()
    try:
        with context:
            if automatic:
                from vllm.v1.attention.backends.flash_v100.runtime import (
                    prepare_prefill_runtime,
                )

                prepare_prefill_runtime(torch.ops._vllm_fa2_C)
            if automatic:
                from vllm.v1.attention.backends.flash_v100 import dense_prefill
                from vllm.v1.attention.ops.sm70_workspaces import workspace_cache

                dense_prefill.profile_fp8_prefill_bridge_workspace(
                    query[0], num_kv_heads, impl.fp8_bridge_workspace_plan
                )
            out = torch.full_like(query, float("nan"))
            grouped = impl._run_fp8_prefill_bridge(**arguments, out=out)
            assert grouped is not None
            if head_groups > 1 and num_kv_heads % head_groups == 0:
                assert grouped[0] is out
                assert grouped[1]
            torch.accelerator.synchronize()
            assert torch.equal(grouped[0], expected)
            if automatic:
                retained_bytes = sum(
                    tensor.untyped_storage().nbytes()
                    for name in (
                        "fp8_prefill_bridge_workspaces",
                        "fp8_prefill_bridge_tail_workspaces",
                        "fp8_prefill_bridge_group_workspaces",
                    )
                    for buffers in workspace_cache(name, {}).values()
                    for tensor in buffers
                )
                plan = impl.fp8_bridge_workspace_plan
                assert retained_bytes == plan.projected_bytes <= plan.budget_bytes
    finally:
        if automatic:
            release_runtime_resources(engine)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "failed_buffer", ["query", "output", "budget", "auto_query", "auto_output"]
)
@pytest.mark.parametrize(
    "cache_dtype,torch_dtype",
    [("fp8_e4m3", torch.float8_e4m3fn), ("fp8_e5m2", torch.float8_e5m2)],
)
@torch.inference_mode()
def test_sm70_flash_v100_grouped_bridge_oom_falls_back(
    monkeypatch, failed_buffer, cache_dtype, torch_dtype
):
    import weakref
    from types import SimpleNamespace

    from vllm.v1.attention.backends.flash_v100.impl import FlashAttnV100Impl
    from vllm.v1.attention.backends.flash_v100.prefill_candidates import (
        Fp8BridgePrefill,
    )

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 regression is SM70/V100 only")
    pytest.importorskip("flash_attn_v100")
    monkeypatch.setenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS", "2")
    impl = FlashAttnV100Impl(
        num_heads=24,
        num_kv_heads=4,
        head_size=256,
        scale=256**-0.5,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=cache_dtype,
    )
    automatic = failed_buffer.startswith("auto_")
    if failed_buffer == "budget" or automatic:
        from vllm.v1.attention.backends.flash_v100.config import (
            Fp8BridgeWorkspacePlan,
        )

        impl.fp8_bridge_workspace_plan = Fp8BridgeWorkspacePlan.build(
            budget_bytes=6 * 1024**2 if automatic else 0,
            max_model_len=1700, max_query_tokens=32,
            block_size=1056, num_kv_heads=4, num_heads=24, head_size=256,
        )
        impl.fp8_bridge_head_groups = impl.fp8_bridge_workspace_plan.head_groups
        if automatic:
            assert impl.fp8_bridge_head_groups == 2
            failed_buffer = failed_buffer.removeprefix("auto_")
    torch.manual_seed(127)
    query = torch.randn(1, 32, 24, 256, device="cuda", dtype=torch.float16)
    key = torch.randn(2, 1056, 4, 256, device="cuda").to(torch_dtype).view(torch.uint8)
    value = torch.randn_like(key, dtype=torch.float16).to(torch_dtype).view(torch.uint8)
    blocks = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([1700], device="cuda", dtype=torch.int32)
    expected = impl.flash_attn_prefill_paged(
        query,
        key,
        value,
        blocks,
        lengths,
        softmax_scale=256**-0.5,
        kv_cache_dtype=cache_dtype,
        k_scale=0.75,
        v_scale=1.25,
        causal=True,
        window_size=(-1, -1),
    ).clone()
    request = SimpleNamespace(
        q_seq=query,
        key_cache=key,
        value_cache=value,
        attn_metadata=SimpleNamespace(block_table=blocks, seq_lens=lengths),
        i=0,
        start=0,
        end=32,
        seq_len=1700,
        layer=SimpleNamespace(_k_scale_float=0.75, _v_scale_float=1.25),
        causal=True,
        window_size=(-1, -1),
        out_view=torch.full_like(query[0], float("nan")),
    )
    executor = SimpleNamespace(
        ops=SimpleNamespace(
            bridge=impl._run_fp8_prefill_bridge,
            paged=impl.flash_attn_prefill_paged,
        ),
        config=SimpleNamespace(scale=256**-0.5, kv_cache_dtype=cache_dtype),
    )
    group_shape = (1, 32, 12, 256)
    empty, empty_like = torch.empty, torch.empty_like
    scratch_refs = []

    def allocate_query(*args, **kwargs):
        if args and isinstance(args[0], tuple) and args[0] == group_shape:
            if failed_buffer == "query":
                raise torch.OutOfMemoryError("group query scratch")
            tensor = empty(*args, **kwargs)
            scratch_refs.append(weakref.ref(tensor))
            return tensor
        return empty(*args, **kwargs)

    def allocate_output(tensor, *args, **kwargs):
        if tuple(tensor.shape) == group_shape and failed_buffer == "output":
            raise torch.OutOfMemoryError("group output scratch")
        return empty_like(tensor, *args, **kwargs)

    with monkeypatch.context() as allocation_failure:
        allocation_failure.setattr(torch, "empty", allocate_query)
        allocation_failure.setattr(torch, "empty_like", allocate_output)
        result = Fp8BridgePrefill(executor).run(request, lambda *args, **kwargs: None)
    torch.accelerator.synchronize()
    assert torch.equal(result.output, expected)
    assert all(reference() is None for reference in scratch_refs)
