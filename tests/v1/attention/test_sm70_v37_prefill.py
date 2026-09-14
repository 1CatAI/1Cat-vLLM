# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Version identity and explicit E4M3 prefill admission; no speculative path."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "q_len,kv_len,expected",
    [
        (64, 3136, True),
        (384, 8192, True),
        (1600, 3136, True),
        (1664, 4848, False),
        (8000, 128000, True),
        (8192, 262144, True),
        (32, 3136, False),
        (1568, 3136, False),
        (1600, 1600, False),
        (8192, 262145, False),
        (8192, 131071, False),
        (8256, 131072, False),
    ],
)
def test_v37_tile_aligned_shape_family(monkeypatch, q_len, kv_len, expected):
    from vllm.v1.attention.backends import flash_attn_v100 as mod

    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_GQA_V37", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL", "1")
    mod.envs.disable_envs_cache()
    q = torch.empty((1, q_len, 6, 256), dtype=torch.float16, device="meta")
    k = torch.empty((1, kv_len, 1, 256), dtype=torch.float16, device="meta")
    assert (
        mod._should_use_prefill_d256_gqa_architecture(
            q,
            k,
            k,
            max_seqlen_q=q_len,
            max_seqlen_k=kv_len,
            softmax_scale=0.0625,
            architecture_op=object(),
        )
        is expected
    )


def test_missing_v37_does_not_select_legacy(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as mod

    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_GQA_V37", "1")
    mod.envs.disable_envs_cache()
    monkeypatch.setattr(mod, "_sm70_d256_gqa_architecture_op_checked", False)
    monkeypatch.setattr(mod, "_sm70_d256_gqa_architecture_op", None)
    legacy = object()
    monkeypatch.setattr(mod, "_get_sm70_splitd_d256_ops", lambda: None)
    monkeypatch.setattr(
        mod,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(
                _vllm_fa2_C=SimpleNamespace(sm70_d256_gqa_architecture_fwd=legacy)
            )
        ),
    )
    assert mod._get_sm70_d256_gqa_architecture_op() is None


@pytest.mark.parametrize("v37", ["0", "1"])
def test_e4m3_bridge_independent_of_compute_kernel(monkeypatch, v37):
    from vllm.v1.attention.backends import flash_attn_v100 as mod

    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_D256_GQA_V37", v37)
    mod.envs.disable_envs_cache()
    bridge = object()
    monkeypatch.setattr(mod, "_get_sm70_splitd_d256_ops", lambda: None)
    monkeypatch.setattr(
        mod,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(
                _vllm_fa2_C=SimpleNamespace(sm70_v37_e4m3_bridge=bridge)
            )
        ),
    )
    assert mod._get_sm70_v37_e4m3_bridge_op() is bridge


@pytest.mark.parametrize("dtype", ["fp8_e4m3", "fp8_e5m2"])
def test_explicit_fp8_bridge_routes(dtype):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    impl = object.__new__(FlashAttnV100Impl)
    impl.use_fp8_prefill_bridge = True
    impl.use_flash_v100_prefill_paged = True
    impl.kv_cache_dtype = dtype
    backing = torch.empty((2, 2, 1616, 1, 256), dtype=torch.uint8)
    k, v = backing.unbind(1)
    assert impl._should_use_fp8_prefill_bridge(
        q_len=8000,
        head_dim=256,
        key_cache=k,
        value_cache=v,
        causal=True,
        window_size=(-1, -1),
    )
    assert not impl._should_use_fp8_prefill_bridge(
        q_len=1,
        head_dim=256,
        key_cache=k,
        value_cache=v,
        causal=True,
        window_size=(-1, -1),
    )


def test_e4m3_wide_bridge_rejects_unaligned_cache():
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    impl = object.__new__(FlashAttnV100Impl)
    impl.use_fp8_prefill_bridge = True
    impl.use_flash_v100_prefill_paged = True
    impl.kv_cache_dtype = "fp8_e4m3"
    storage = torch.empty(1616 * 256 + 1, dtype=torch.uint8)
    k = storage[1:].view(1, 1616, 1, 256)
    assert not impl._should_use_fp8_prefill_bridge(
        q_len=8000,
        head_dim=256,
        key_cache=k,
        value_cache=k,
        causal=True,
        window_size=(-1, -1),
    )
