# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine bindings must carry codec views and check the installed capability."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import routing
from vllm.v1.attention.backends.flash_v100.spec import verifier
from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    INT8,
    INT8_PER_TOKEN_HEAD,
)
from vllm.v1.attention.ops.sm70_grouped import grouped_fp32_reason

CODECS = (FP16, FP8_E4M3, FP8_E5M2, INT8, INT8_PER_TOKEN_HEAD)
pytestmark = pytest.mark.cpu_test


def view(codec):
    return KVCacheView.bind(
        torch.zeros(
            (4, 2, 32, 2, 260 if codec.per_token_head else 256),
            dtype=codec.storage_dtype,
        ),
        codec,
        256,
    )


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
def test_shared_grouped_engine_admits_each_installed_codec(codec):
    cache = view(codec)
    op = SimpleNamespace(kv_cache_codecs={c.name for c in CODECS})
    instance = SimpleNamespace(
        kv_cache_dtype=codec.name,
        use_smallq_decode_xqa=True,
        flash_attn_grouped_fp32_paged=op,
        _flash_v100_window_size=lambda causal: (-1, -1),
    )
    q = torch.zeros((8, 12, 256), dtype=torch.float16)
    rows = torch.arange(57, 65, dtype=torch.int32)
    table = torch.tensor([[3, 1]], dtype=torch.int32)
    args = (
        codec,
        instance,
        q,
        cache.key,
        cache.value,
        table.expand(8, -1).contiguous(),
        rows,
        SimpleNamespace(block_table=table, seq_lens=rows[-1:]),
    )
    assert grouped_fp32_reason(*args, out=torch.empty_like(q), shared=True) is None
    op.kv_cache_codecs = {c.name for c in CODECS if c is not codec}
    assert grouped_fp32_reason(*args, out=torch.empty_like(q), shared=True) == (
        "operator_codec_missing:grouped_fp32"
    )


def test_scalar_workspace_is_owned_by_engine_and_survives_inference_mode():
    from vllm.config import set_current_vllm_config
    from vllm.runtime_resources import release_runtime_resources
    from vllm.v1.attention.ops import sm70_grouped_scalar as scalar

    configs = (SimpleNamespace(), SimpleNamespace())
    buffers = []
    for config in configs:
        with set_current_vllm_config(config), torch.inference_mode():
            current = scalar._scalar_workspace(torch.device("cpu"), ("abi1", 262144))
            assert current is scalar._scalar_workspace(
                torch.device("cpu"), ("abi1", 262144)
            )
            assert all(not tensor.is_inference() for tensor in current)
            buffers.append(current)
    assert buffers[0][0].data_ptr() != buffers[1][0].data_ptr()
    release_runtime_resources(configs[0])
    with set_current_vllm_config(configs[1]):
        assert buffers[1] is scalar._scalar_workspace(
            torch.device("cpu"), ("abi1", 262144)
        )


@pytest.mark.parametrize("codec", (INT8, INT8_PER_TOKEN_HEAD), ids=lambda c: c.name)
@pytest.mark.parametrize("shared", [False, True])
def test_verifier_passes_explicit_scales_and_authoritative_row_lengths(codec, shared):
    cache = view(codec)
    calls = []

    def native(q, k, v, table, lengths, **kwargs):
        calls.append((k, v, table, lengths, kwargs))
        assert kwargs["k_scale_cache"] is cache.key_scale
        assert kwargs["v_scale_cache"] is cache.value_scale
        assert kwargs["kv_cache_dtype"] == codec.name
        kwargs["out"].fill_(17)

    native.kv_cache_codecs = {codec.name}  # type: ignore[attr-defined]

    def forbidden(*args, **kwargs):
        raise AssertionError("A bound grouped/XQA path must not reach scalar")

    config = verifier.VerificationConfig(
        SimpleNamespace(use_smallq_decode_xqa=True),
        0.0625,
        codec.name,
        None,
        0.0,
        False,
        False,
        8,
        1,
        1,
    )
    ops = verifier.VerificationOps(
        grouped=None,
        fp16_grouped=None,
        e4m3_grouped=None,
        xqa=native,
        window_size=lambda causal: (-1, -1),
        layer_info=lambda layer: {},
        xqa_codec=lambda *args: codec,
        decode=forbidden,
        cache_view=cache,
        shared_grouped=native if shared else None,
        partition_hint=lambda: None,
    )
    executor = verifier.VerificationExecutor(config, ops)
    q = torch.zeros((8, 12, 256), dtype=torch.float16)
    out = torch.empty_like(q)
    parent = torch.tensor([[3, 1]], dtype=torch.int32)
    rows = torch.tensor([58, 59, 60, 61, 62, 63, 64, 0], dtype=torch.int32)
    table = parent.expand(8, -1).contiguous()
    metadata = SimpleNamespace(
        block_table=parent, seq_lens=torch.tensor([64], dtype=torch.int32)
    )
    executor.call_smallq_decode_paged(
        SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0),
        q,
        cache.key,
        cache.value,
        table,
        rows,
        metadata,
        out=out,
        max_seq_len_hint=64,
        workspace_seq_capacity_hint=64,
        partition_size_hint=None,
    )
    assert len(calls) == 1 and calls[0][3] is rows
    assert calls[0][2] is (parent if shared else table)
    assert torch.all(out == 17)


def test_partial_policy_binding_preserves_native_capabilities(monkeypatch):
    from vllm.v1.attention.backends.flash_v100.spec import attention_policy as policy

    def operation(*args, dflash2_window_split=True, **kwargs):
        pass

    operation.kv_cache_codecs = {"int8", "auto"}  # type: ignore[attr-defined]
    monkeypatch.setattr(
        policy,
        "capture_sm70_dflash2_config",
        lambda: SimpleNamespace(draft_window_split=False),
    )
    state = SimpleNamespace(
        flash_attn_prefill_paged=operation, accepts_keyword=lambda fn, kw: True
    )
    policy.configure_prefill(state)
    assert routing.native_codec_supported(state.flash_attn_prefill_paged, INT8)
    assert state.flash_attn_prefill_paged.keywords == {"dflash2_window_split": False}
