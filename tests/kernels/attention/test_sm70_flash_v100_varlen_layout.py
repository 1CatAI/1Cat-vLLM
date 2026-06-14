# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
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

    torch.cuda.synchronize()
    assert torch.equal(decode_out[0], expected_decode)
    assert torch.equal(prefill_out[0, 0], expected_decode)
