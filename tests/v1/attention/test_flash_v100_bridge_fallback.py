# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit native capability gates bridge selection and fallback accounting."""

import importlib.util
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import impl, ops, routing
from vllm.v1.attention.kv_codecs import BF16, FP8_E4M3, FP8_E5M2, FP16

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("codec", [FP16, BF16, FP8_E4M3, FP8_E5M2, None])
@pytest.mark.parametrize("names", [(), ("auto", "fp8_e4m3", "fp8_e5m2"), "fp8_e4m3"])
def test_capability_requires_explicit_native_codec_declaration(codec, names):
    expected = isinstance(names, tuple) and codec in (FP16, FP8_E4M3, FP8_E5M2)
    expected = expected and codec is not None and codec.name in names
    assert (
        routing.native_prefill_supports_codec(
            SimpleNamespace(native_kv_codecs=names), codec
        )
        == expected
    )
    assert not routing.native_prefill_supports_codec(None, codec)


def _backend(monkeypatch, codec, native):
    operator = MagicMock(
        native_kv_codecs=(codec.name,) if native else (),
        _sm70_dflash2_direct_bmhd=False,
        _sm70_dflash2_split_pages=(),
    )
    monkeypatch.setattr(
        ops, "_get_flash_ops", lambda: (None,) * 5 + (operator,) + (None,) * 3
    )
    monkeypatch.setattr(ops, "_get_fp8_e5m2_paged_kv_bridge_op", lambda: MagicMock())
    monkeypatch.setattr(ops, "_get_sm70_v37_e4m3_bridge_op", lambda: MagicMock())
    monkeypatch.setattr(impl.current_platform, "is_device_capability", lambda *a: False)
    monkeypatch.setenv("VLLM_FLASH_V100_FP8_PREFILL_BRIDGE", "1")
    instance = impl.FlashAttnV100Impl(
        num_heads=6,
        head_size=256,
        scale=0.0625,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=codec.name,
    )
    assert instance.kv_codec is codec
    wrapped = instance.flash_attn_prefill_paged
    assert (wrapped.func if isinstance(wrapped, partial) else wrapped) is operator
    assert operator.native_kv_codecs == ((codec.name,) if native else ())
    return instance, operator


@pytest.mark.parametrize("codec", [FP8_E4M3, FP8_E5M2])
@pytest.mark.parametrize("native", [False, True])
def test_constructor_captures_native_support_once(monkeypatch, codec, native):
    instance, operator = _backend(monkeypatch, codec, native)
    cache = torch.empty((1, 64, 1, 256), dtype=torch.uint8)
    args = dict(
        q_len=32,
        head_dim=256,
        key_cache=cache,
        value_cache=cache,
        causal=True,
        window_size=(-1, -1),
    )
    assert instance._native_fp8_prefill_supported is native
    assert instance._should_use_fp8_prefill_bridge(**args) is not native
    operator.native_kv_codecs = () if native else (codec.name,)
    assert instance._should_use_fp8_prefill_bridge(**args) is not native
    if native:
        assert (
            instance._run_fp8_prefill_bridge(
                query=torch.empty((1, 32, 6, 256), dtype=torch.float16),
                key_cache=cache,
                value_cache=cache,
                block_table=torch.tensor([[0]], dtype=torch.int32),
                seq_lens=torch.tensor([64], dtype=torch.int32),
                seq_len=64,
                k_scale=1.0,
                v_scale=1.0,
                causal=True,
                window_size=(-1, -1),
                out=torch.empty((1, 32, 6, 256), dtype=torch.float16),
            )
            is None
        )
        bridge = (
            instance.fp8_e4m3_paged_kv_to_fp16
            if codec is FP8_E4M3
            else instance.fp8_e5m2_paged_kv_to_fp16
        )
        bridge.assert_not_called()


@pytest.mark.parametrize("codec", [FP8_E4M3, FP8_E5M2])
@pytest.mark.parametrize("native", [False, True])
def test_prefix_dispatch_preserves_storage_and_counts_one_bridge_fallback(
    monkeypatch, codec, native
):
    instance, operator = _backend(monkeypatch, codec, native)
    monkeypatch.setattr(routing, "_route_summary_enabled", lambda: False)
    monkeypatch.setattr(routing, "_route_summary_registered", True)
    monkeypatch.setattr(routing, "_fallback_counts", {})
    monkeypatch.setattr(routing, "_route_counts", {})
    monkeypatch.setenv("VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE", "0")
    query = torch.zeros((32, 6, 256), dtype=torch.float16)
    output = torch.empty_like(query)
    cache = torch.zeros((1, 2, 64, 1, 256), dtype=torch.uint8)
    operator.return_value = torch.full((1, 32, 6, 256), 2, dtype=torch.float16)
    bridge = MagicMock(
        return_value=(torch.full((1, 32, 6, 256), 3, dtype=torch.float16), False)
    )
    monkeypatch.setattr(instance, "_run_fp8_prefill_bridge", bridge)
    qsl = torch.tensor([0, 32], dtype=torch.int32)
    lengths = torch.tensor([64], dtype=torch.int32)
    metadata: Any = SimpleNamespace(
        causal=True,
        num_actual_tokens=32,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=lengths,
        seq_lens_cpu=lengths,
        block_table=torch.tensor([[0]], dtype=torch.int32),
    )
    layer: Any = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    result = instance._flash_v100_prefill_with_prefix(
        layer, query, None, None, cache, metadata, output
    )
    assert result is output
    if native:
        bridge.assert_not_called()
        operator.assert_called_once()
        assert operator.call_args.args[1].dtype == torch.uint8
        assert operator.call_args.kwargs["kv_cache_dtype"] == codec.name
        assert operator.call_args.kwargs["k_scale"] == 0.5
        assert operator.call_args.kwargs["v_scale"] == 0.25
        assert not routing._fallback_counts
        assert torch.all(output == 2)
    else:
        bridge.assert_called_once()
        operator.assert_not_called()
        assert routing._fallback_counts == {f"prefill_prefix_{codec.name}_bridge": 1}
        assert torch.all(output == 3)
    assert not routing._route_counts  # fallback telemetry works without debug


@pytest.mark.parametrize("advertised", [(), ("auto", "fp8_e4m3", "fp8_e5m2")])
def test_source_interface_copies_only_installed_native_capabilities(
    monkeypatch, advertised
):
    native = SimpleNamespace(prefill_paged_native_kv_codecs=advertised)
    monkeypatch.setitem(sys.modules, "flash_attn_v100_cuda", native)
    path = (
        Path(routing.__file__).parents[5]
        / "flash-attention-v100/flash_attn_v100/flash_attn_interface.py"
    )
    spec = importlib.util.spec_from_file_location("_bridge_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    operator: Any = module.flash_attn_prefill_paged
    assert operator.native_kv_codecs == advertised
    # No general-prefill capability leaks onto narrower specialized operators.
    assert not hasattr(module.flash_attn_prefill_paged_splitkv, "native_kv_codecs")
