# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared planning preserves E4M3 storage/FP32 partials and old-ABI rollback."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.kernel import KernelConfig
from vllm.v1.attention.backends.flash_v100 import routing
from vllm.v1.attention.kv_codecs import FP8_E4M3, FP8_E5M2, FP16

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "requested,revision,enabled,expected",
    [
        ("shared", 0, True, "legacy"),
        ("shared", 1, True, "shared"),
        ("shared", 2, True, "shared"),
        ("legacy", 1, True, "legacy"),
        ("shared", None, True, "legacy"),
        ("shared", 0, False, "legacy"),
    ],
)
def test_strategy_is_captured_from_config_and_native_revision(
    monkeypatch, requested, revision, enabled, expected
):
    cfg = SimpleNamespace(kernel_config=KernelConfig(sm70_decode_strategy=requested))
    monkeypatch.setattr(routing, "get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr(routing, "_record_route", MagicMock())
    operator = SimpleNamespace(shared_decode_strategy_revision=revision)
    assert (
        routing.resolve_decode_strategy(FP8_E4M3, operator, enabled=enabled) == expected
    )
    fallback = enabled and requested == "shared" and expected == "legacy"
    assert routing._record_route.call_count == int(fallback)
    if fallback:
        routing._record_route.assert_called_once_with("decode_strategy_legacy_revision")


def test_decode_strategy_participates_in_graph_config_hash():
    assert KernelConfig().sm70_decode_strategy == "shared"
    assert KernelConfig(sm70_decode_strategy="shared").compute_hash() != (
        KernelConfig(sm70_decode_strategy="legacy").compute_hash()
    )


@pytest.mark.parametrize("page", [256, 784, 800, 832, 1648, 3296])
def test_shared_e4m3_hint_matches_fp16_without_changing_storage(monkeypatch, page):
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", raising=False)
    q = torch.empty((1, 6, 256), dtype=torch.float16)
    k16 = torch.empty((1, page, 1, 256), dtype=torch.float16)
    k8 = torch.empty_like(k16, dtype=torch.uint8)
    expected = routing._g6_aligned_page_partition_size_hint(q, k16, k16, "auto")
    assert (
        routing._g6_aligned_page_partition_size_hint(
            q, k8, k8, "fp8_e4m3", strategy="shared"
        )
        == expected
    )
    assert (
        routing._g6_aligned_page_partition_size_hint(
            q, k8, k8, "fp8_e4m3", strategy="legacy"
        )
        == 64
    )
    assert k8.dtype == torch.uint8


@pytest.fixture
def source_interface(monkeypatch):
    path = (
        Path(routing.__file__).parents[5]
        / "flash-attention-v100/flash_attn_v100/flash_attn_interface.py"
    )
    native = SimpleNamespace(
        xqa_shared_decode_strategy_revision=1, decode_paged_xqa_fwd=MagicMock()
    )
    import sys

    monkeypatch.setitem(sys.modules, "flash_attn_v100_cuda", native)
    spec = importlib.util.spec_from_file_location("_shared_decode_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, native


@pytest.mark.parametrize("length", [4097, 32768, 131072, 262144])
def test_shared_decode_native_arguments_and_fp32_workspace(
    monkeypatch, source_interface, length
):
    module, native = source_interface
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    q = torch.zeros((1, 6, 256), dtype=torch.float16)
    k8 = torch.zeros((1, 832, 1, 256), dtype=torch.uint8)
    table = torch.zeros((1, 316), dtype=torch.int32)
    seq = torch.tensor([length], dtype=torch.int32)
    out = torch.empty_like(q)
    assert module.flash_attn_decode_paged_xqa.shared_decode_strategy_revision == 1
    module.flash_attn_decode_paged_xqa(
        q,
        k8,
        k8,
        table,
        seq,
        out=out,
        kv_cache_dtype="fp8_e4m3",
        max_seq_len_hint=length,
        workspace_seq_capacity_hint=length,
        partition_size_hint=None,
    )
    args = native.decode_paged_xqa_fwd.call_args.args
    assert args[1] is k8 and args[2] is k8
    assert args[6].dtype == torch.float32
    assert args[11] == (1024 if length >= 32768 else 256)
    assert args[13] == "fp8_e4m3"
    assert args[6].shape[2] >= args[12]


def test_non_e4_codecs_do_not_require_the_shared_revision(monkeypatch):
    monkeypatch.setattr(routing, "get_current_vllm_config_or_none", lambda: None)
    record = MagicMock()
    monkeypatch.setattr(routing, "_record_route", record)
    for codec in (FP16, FP8_E5M2):
        assert (
            routing.resolve_decode_strategy(codec, object(), enabled=True) == "shared"
        )
    record.assert_not_called()
