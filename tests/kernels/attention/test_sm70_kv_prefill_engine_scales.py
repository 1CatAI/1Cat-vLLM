# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache view, bridge owner and native prefill through the attention implementation."""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.attention.test_sm70_kv_engine_bindings import cache, reference
from vllm.v1.attention.kv_codecs import INT8, INT8_PER_TOKEN_HEAD

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


def inputs(codec, q_len, kv_len):
    from vllm.v1.attention.backends.flash_v100.impl import FlashAttnV100Impl

    torch.manual_seed(946)
    page = 32
    pages = kv_len // page
    carrier, view = cache(codec, pages=pages + 2, page=page)
    instance = FlashAttnV100Impl(
        num_heads=6,
        head_size=256,
        scale=0.0625,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=codec.name,
    )
    assert instance._supports_flash_v100_path()
    assert codec.name in instance.paged_kv_to_fp16.unscaled_kv_cache_codecs
    query = torch.randn((q_len, 6, 256), device="cuda", dtype=torch.float16)
    raw = torch.zeros((q_len, 1, 256), device="cuda", dtype=torch.float16)
    table = (torch.randperm(pages, device="cuda") + 2).int()[None]
    lengths = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, q_len], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=q_len,
        max_query_len=q_len,
        max_seq_len=kv_len,
        num_reqs=1,
        max_model_len=262144,
        causal=True,
        query_start_loc=starts.cuda(),
        query_start_loc_cpu=starts,
        seq_lens=lengths,
        seq_lens_cpu=lengths.cpu(),
        block_table=table,
    )
    layer = SimpleNamespace(
        _k_scale_float=1.0, _v_scale_float=1.0, layer_name="codec_operator_contract"
    )
    return instance, carrier, view, query, raw, metadata, layer


def check(query, out, view, table, kv_len):
    q_len = query.shape[0]
    rows = (
        torch.arange(q_len, device=query.device)
        if q_len <= 384
        else torch.tensor([0, 1, q_len // 2, q_len - 1], device=query.device)
    )
    expected = reference(
        query[rows], view, table, kv_len - q_len + rows.int() + 1, grouped=True
    )
    torch.testing.assert_close(
        out[rows].cpu().double(), expected, atol=0.004, rtol=0.015
    )
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("codec", [INT8, INT8_PER_TOKEN_HEAD], ids=lambda c: c.name)
@pytest.mark.parametrize(
    "v37,q_len,kv_len",
    [
        (True, 384, 3136),
        (True, 1568, 8192),
        (True, 1600, 8192),
        (False, 1024, 3136),
        (False, 8000, 16416),
        (False, 8001, 16416),
        (False, 8192, 16416),
    ],
)
def test_engine_prefix_preserves_row_scales(monkeypatch, codec, v37, q_len, kv_len):
    from vllm import envs

    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_GQA_V37", str(int(v37)))
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_FP8_PREFILL_BRIDGE", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_FA2_D256_PREFILL", "1")
    envs.disable_envs_cache()
    instance, carrier, view, q, raw, metadata, layer = inputs(codec, q_len, kv_len)
    out = torch.full_like(q, float("nan"))
    instance.forward(layer, q, raw, raw, carrier, metadata, out)
    check(q, out, view, metadata.block_table, kv_len)
    assert instance._kv_cache_view.key_scale.data_ptr() == view.key_scale.data_ptr()


def test_engine_bridge_executes_the_native_scaled_core(monkeypatch, record_property):
    from vllm import envs

    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_GQA_V37", "0")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL", "0")
    monkeypatch.setenv("VLLM_FLASH_V100_FP8_PREFILL_BRIDGE", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_FA2_D256_PREFILL", "1")
    envs.disable_envs_cache()
    instance, _, view, query, _, metadata, _ = inputs(INT8, 1024, 3136)
    instance._kv_cache_view = view
    executor = instance._new_prefill_executor()
    kwargs = dict(
        query=query[None],
        key_cache=view.key,
        value_cache=view.value,
        block_table=metadata.block_table,
        seq_lens=metadata.seq_lens,
        seq_len=3136,
        k_scale=1.0,
        v_scale=1.0,
        causal=True,
        window_size=(-1, -1),
        out=torch.empty_like(query)[None],
    )
    # Prepare the bounded owner before profiling actual launches.
    assert executor._run_fp8_prefill_bridge(**kwargs)[1]
    torch.accelerator.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        result = executor._run_fp8_prefill_bridge(**kwargs)
        torch.accelerator.synchronize()
    assert result is not None and result[1]
    kernels = [
        e.name
        for e in profile.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    record_property("executed_cuda_kernels", kernels)
    assert any("paged_kv_to_fp16" in name for name in kernels), kernels
    assert any("sm70_d256_splitd_dense_kernel" in name for name in kernels), kernels
    check(query, result[0][0], view, metadata.block_table, 3136)
