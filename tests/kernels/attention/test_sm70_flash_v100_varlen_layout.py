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
