# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature hooks preserve observable backend dispatch and compatibility."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends import flash_attn_v100 as legacy
from vllm.v1.attention.backends.flash_v100 import impl
from vllm.v1.attention.backends.flash_v100.spec import attention
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

pytestmark = pytest.mark.cpu_test


def _instance():
    instance = object.__new__(impl.FlashAttnV100Impl)
    instance.attn_type = AttentionType.DECODER
    instance.alibi_slopes = None
    instance.logits_soft_cap = 0
    instance.sinks = None
    instance.kv_cache_dtype = "fp16"
    instance.use_flash_v100 = True
    instance.allow_triton_fallback = False
    instance.use_decode_scalar_paged = True
    instance.use_decode_paged_prefill = False
    instance.use_flash_v100_prefill_paged = True
    instance.use_triton_prefill = False
    instance._validate_dflash_attention_contract = MagicMock()
    return instance


def test_registration_matches_mechanically_inlined_callbacks():
    tree = ast.parse(Path(attention.__file__).read_text())
    register = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "register_attention_hooks"
    )
    registration = register.body[0]
    assert isinstance(registration, ast.Return)
    assert isinstance(registration.value, ast.Call)
    for keyword in registration.value.keywords:
        assert keyword.arg == ast.unparse(keyword.value)
        assert getattr(attention.ATTENTION_HOOKS, keyword.arg) is getattr(
            attention, keyword.arg
        )
    for name in ("impl.py", "metadata.py", "backend.py"):
        text = Path(impl.__file__).with_name(name).read_text().lower()
        assert "dflash" not in text and "ddtree" not in text


@pytest.mark.parametrize("draft,allow", [(False, False), (False, True), (True, False)])
def test_forward_fallback_keeps_admission_route_and_base_call(
    monkeypatch, draft, allow
):
    instance = _instance()
    instance.allow_triton_fallback = allow
    instance._supports_flash_v100_path = lambda: False
    layer = SimpleNamespace(is_dflash_draft_attn=draft, layer_name="test")
    routes: list[str] = []
    monkeypatch.setattr(legacy, "_record_route", routes.append)
    monkeypatch.setattr(legacy, "_warned_feature_fallback", False)
    logger = MagicMock()
    monkeypatch.setattr(legacy, "logger", logger)
    base = MagicMock(side_effect=lambda *args: args[6].fill_(7))
    monkeypatch.setattr(TritonAttentionImpl, "forward", base)
    tensor = torch.zeros((1, 6, 256), dtype=torch.float16)
    metadata = SimpleNamespace()
    if not (draft or allow):
        with pytest.raises(RuntimeError, match="cannot run"):
            instance.forward(layer, tensor, tensor, tensor, tensor, metadata, tensor)
        assert routes == []
        base.assert_not_called()
    else:
        result = instance.forward(
            layer, tensor, tensor, tensor, tensor, metadata, tensor
        )
        assert result is tensor and tensor.eq(7).all()
        base.assert_called_once()
        assert routes == [
            "dflash_draft_triton_fallback" if draft else "unsupported_triton_fallback"
        ]
        logger.warning.assert_called_once()
    instance._validate_dflash_attention_contract.assert_called_once_with(
        layer, metadata
    )


def test_forward_noncausal_capture_keeps_paged_prefix_route(monkeypatch):
    instance = _instance()
    instance._supports_flash_v100_path = lambda: True
    layer = SimpleNamespace(is_dflash_draft_attn=True, layer_name="test")
    routes: list[str] = []
    monkeypatch.setattr(legacy, "_record_route", routes.append)
    monkeypatch.setattr(legacy, "_is_cuda_graph_capturing", lambda _: True)
    output = torch.zeros((3, 6, 256), dtype=torch.float16)
    instance._flash_v100_prefill_with_prefix = MagicMock(return_value=output)
    metadata = SimpleNamespace(
        max_query_len=3,
        max_seq_len=3,
        num_actual_tokens=3,
        causal=False,
        query_start_loc=torch.tensor([0, 3]),
    )
    assert (
        instance.forward(layer, output, output, output, output, metadata, output)
        is output
    )
    assert routes == ["prefill_capture_dflash_noncausal_paged"]
    instance._flash_v100_prefill_with_prefix.assert_called_once_with(
        layer, output, output, output, output, metadata, output
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_configuration_retains_legacy_split_attributes(monkeypatch, enabled):
    instance = _instance()
    calls = []

    def op(*, dflash2_window_split=True):
        calls.append(dflash2_window_split)

    operator: Any = op
    operator._sm70_dflash2_direct_bmhd = True
    operator._sm70_dflash2_split_pages = 128, 832
    instance.flash_attn_prefill_paged = operator
    monkeypatch.setattr(
        legacy,
        "capture_sm70_dflash2_config",
        lambda: SimpleNamespace(draft_window_split=enabled),
    )
    attention.ATTENTION_HOOKS.configure_prefill(instance)
    instance.flash_attn_prefill_paged()
    assert calls == [enabled]
    assert instance._flash_prefill_paged_supports_dflash2_bmhd
    assert instance._flash_prefill_paged_dflash2_split_pages == (
        (128, 832) if enabled else ()
    )
