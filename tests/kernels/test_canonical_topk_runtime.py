# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise actual layer/config/op plumbing without launching GPU kernels."""

import importlib
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config import (
    KernelConfig,
    VllmConfig,
    get_current_vllm_config_or_none,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
)


@pytest.fixture
def module(monkeypatch):
    mod = importlib.import_module(
        "vllm.model_executor.layers.sparse_attn_indexer_kpool"
    )
    # Constructor hardware qualification is unrelated to config lifetime.
    monkeypatch.setattr(mod.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(mod.current_platform, "is_xpu", lambda: False)
    return mod


def make_layer(module: Any, enabled: bool):
    config = VllmConfig(
        kernel_config=KernelConfig(canonical_prefill_topk_order=enabled)
    )
    with set_current_vllm_config(config):
        layer = module.SparseAttnIndexerKpool(
            SimpleNamespace(prefix="indexer", kv_cache=torch.empty(1)),
            128,
            "ue8m0",
            4,
            128,
            40,
            40,
            torch.empty((4, 4), dtype=torch.int32),
            skip_k_cache_insert=True,
        )
    return config, layer


def test_constructed_flags_survive_context_exit_and_do_not_leak(module, monkeypatch):
    enabled_config, enabled = make_layer(module, True)
    disabled_config, disabled = make_layer(module, False)
    assert get_current_vllm_config_or_none() is None
    calls: list[bool] = []

    def spy(*args):
        calls.append(args[-1])
        return args[13]

    monkeypatch.setattr(torch.ops.vllm, "sparse_attn_indexer_kpool", spy)
    x = torch.zeros((4, 128))
    # Use the *opposite* forward config, as well as no current global config.
    for layer, forward_config in (
        (enabled, disabled_config),
        (disabled, enabled_config),
        (enabled, disabled_config),
    ):
        with set_forward_context({}, forward_config, num_tokens=4):
            assert get_current_vllm_config_or_none() is None
            assert layer.forward_cuda(x, x, x, x) is layer.topk_indices_buffer
    assert calls == [True, False, True]
    # A later mutable config change cannot silently change a captured graph.
    enabled_config.kernel_config.canonical_prefill_topk_order = False
    assert enabled.canonical_prefill_topk_order is True


def test_real_fake_and_registered_legacy_signatures_default_false(module):
    for function in (
        module.sparse_attn_indexer_kpool,
        module.sparse_attn_indexer_kpool_fake,
    ):
        parameter = inspect.signature(function).parameters["canonical_prefill_order"]
        assert parameter.default is False
        assert list(inspect.signature(function).parameters)[-1] == parameter.name
    schema = torch.ops.vllm.sparse_attn_indexer_kpool.default._schema
    assert schema.arguments[-1].name == "canonical_prefill_order"
    assert schema.arguments[-1].default_value is False
    x = torch.empty((4, 128))
    output = torch.empty((4, 4), dtype=torch.int32)
    assert (
        module.sparse_attn_indexer_kpool_fake(
            x,
            "indexer",
            x,
            x,
            None,
            x,
            x,
            128,
            "ue8m0",
            4,
            128,
            40,
            40,
            output,
            True,
        )
        is output
    )


@pytest.mark.parametrize("pool", [1, 4])
@pytest.mark.parametrize("enabled", [False, True])
def test_real_prefill_dispatch_uses_explicit_flag_without_current_config(
    module, monkeypatch, pool, enabled
):
    portable = importlib.import_module("vllm.models.deepseek_v4.common.portable_sparse")
    monkeypatch.setattr(portable, "use_portable_sparse", lambda: False)
    raw = torch.tensor([[9, -1, 2, 5], [8, 8, -3, 1]], dtype=torch.int32)
    seen: list[torch.Tensor] = []

    def expand(ids, valid, width, pool_size):
        seen.append(ids.clone())
        expanded = ids[:, :, None] * pool_size + torch.arange(pool_size)
        return torch.where(valid[:, :, None], expanded, -1).reshape(2, width)

    monkeypatch.setattr(
        module, "_get_kpool_ops", lambda: SimpleNamespace(expand_pools_to_tokens=expand)
    )
    workspace = SimpleNamespace(
        get_simultaneous=lambda *specs: (
            torch.empty((16, 128), dtype=torch.float8_e4m3fn),
            torch.empty((16, 4), dtype=torch.uint8),
        )
    )
    monkeypatch.setattr(module, "current_workspace_manager", lambda: workspace)
    monkeypatch.setattr(
        module, "fp8_fp4_mqa_logits", lambda *a, **kw: torch.zeros(2, 16)
    )

    def select(logits, starts, ends, destination, *rest):
        destination.copy_(raw)

    monkeypatch.setattr(torch.ops._C, "top_k_per_row_prefill", select)
    zeros = torch.zeros(2, dtype=torch.int32)
    chunk = DeepseekV32IndexerPrefillChunkMetadata(
        block_table=zeros,
        cu_seqlen_ks=zeros,
        cu_seqlen_ke=torch.full((2,), 16, dtype=torch.int32),
        cu_seq_lens=zeros,
        token_to_seq=zeros,
        total_seq_lens=16,
        token_start=1,
        token_end=3,
        num_reqs=1,
        skip_kv_gather=True,
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=zeros,
        max_seq_len=40,
        compressed_max_seq_len=40,
        slot_mapping=torch.zeros(4, dtype=torch.int32),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=4,
        prefill=DeepseekV32IndexerPrefillMetadata([chunk], 40),
    )
    config = VllmConfig(
        kernel_config=KernelConfig(canonical_prefill_topk_order=not enabled)
    )
    output = torch.full((4, 4 * pool), 777, dtype=torch.int32)
    x = torch.empty((4, 128))
    # False deliberately uses the legacy call with the new argument omitted.
    kwargs = {"canonical_prefill_order": True} if enabled else {}
    with set_forward_context({"indexer": metadata}, config, num_tokens=4):
        assert get_current_vllm_config_or_none() is None
        actual = module.sparse_attn_indexer_kpool(
            x,
            "indexer",
            x,
            x,
            None,
            x,
            x,
            128,
            "ue8m0",
            4 * pool,
            128,
            40,
            16,
            output,
            True,
            index_kpool=pool,
            **kwargs,
        )
    expected = torch.tensor([[2, -1, 5, 9], [1, 8, -3, 8]]) if enabled else raw
    selected = actual[1:3] if pool == 1 else seen[0]
    assert torch.equal(selected, expected)
    assert torch.equal(selected < 0, raw < 0)
    assert torch.equal(selected[selected < 0], raw[raw < 0])
    assert torch.equal(selected.sort(-1).values, raw.sort(-1).values)
    assert (actual[[0, 3]] == -1).all()
