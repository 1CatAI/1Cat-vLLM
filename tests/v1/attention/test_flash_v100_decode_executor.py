# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode can execute with explicit inputs and independently injected operators."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tools.sm70.flash_v100_trace import Recorder, cpu_cuda, install_ops, strict_shim
from vllm.v1.attention.backends.flash_v100.decode import (
    DecodeConfig,
    DecodeExecutor,
    DecodeOps,
    DecodeRequest,
)
from vllm.v1.attention.backends.flash_v100.workspace import V100Workspace

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("codec", ["auto", "int8", "int8_per_token_head"])
@pytest.mark.parametrize(
    "use_xqa,use_scalar",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
@pytest.mark.parametrize(
    "available_xqa,available_scalar",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_decode_executor_runs_with_independent_operators(
    monkeypatch, codec, use_xqa, use_scalar, available_xqa, available_scalar
):
    expected = (
        "xqa"
        if use_xqa and available_xqa
        else "scalar"
        if use_scalar and available_scalar
        else "triton"
    )
    with cpu_cuda(monkeypatch, False), strict_shim() as legacy:
        unavailable = [
            name
            for name, available in (
                ("xqa", available_xqa),
                ("decode", available_scalar),
            )
            if not available
        ]
        install_ops(monkeypatch, Recorder(), legacy, {"unavailable": unavailable})
        backend = legacy.FlashAttnV100Impl(
            num_heads=6,
            num_kv_heads=1,
            head_size=256,
            scale=0.0625,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )
        config = DecodeConfig(
            replace(
                backend.config,
                use_decode_xqa=use_xqa,
                use_decode_scalar_paged=use_scalar,
                allow_triton_fallback=True,
            ),
            backend.scale,
            codec,
            backend.attn_type,
            backend.sliding_window,
        )
        calls = []

        def scalar(*args, **kwargs):
            calls.append(("scalar", kwargs))
            kwargs["out"].fill_(2)

        def xqa(*args, **kwargs):
            calls.append(("xqa", kwargs))
            kwargs["out"].fill_(3)

        def triton(*args):
            calls.append(("triton", {}))
            args[6].fill_(4)
            return args[6]

        for operation in (scalar, xqa):
            operation.kv_cache_codecs = frozenset([codec])  # type: ignore[attr-defined]

        def unused(*args, **kwargs):
            raise AssertionError("unexpected diagnostic/native route")

        executor = DecodeExecutor(
            config,
            DecodeOps(
                dense=unused,
                paged=scalar if available_scalar else None,
                xqa=xqa if available_xqa else None,
                wmma=unused,
                prefill=unused,
                prefill_bhmd=unused,
                paged_keywords={"max_seq_len_hint", "k_scale_cache", "v_scale_cache"},
                scalar_tail=None,
                reject_xqa=lambda codec, metadata: False,
                reserve_bhmd_compare=unused,
                write_bhmd_compare=unused,
                compare_bhmd=unused,
                compare_triton=lambda *args: None,
                triton_forward=triton,
                profile_trace=lambda *args: None,
                draft_debug_enabled=lambda: False,
                draft_debug_log=unused,
                format_debug=unused,
            ),
            V100Workspace(),
        )
        # No executor field can fall back to the backend or its native functions.
        assert set(vars(executor)) == {"config", "ops", "workspace"}
        backend.flash_attn_decode_paged = unused
        backend.flash_attn_decode_paged_xqa = unused
        query = torch.zeros((1, 6, 256), dtype=torch.float16)
        output = torch.empty_like(query)
        cache = (
            torch.zeros((2, 1, 16, 1, 256), dtype=torch.float16)
            if codec == "auto"
            else torch.zeros((1, 2, 16, 1, 260), dtype=torch.int8)
        )
        metadata = SimpleNamespace(
            num_actual_tokens=1,
            seq_lens=torch.tensor([16], dtype=torch.int32),
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            flash_v100_decode_max_seq_len_hint=16,
        )
        layer = SimpleNamespace(_k_scale_float=1.25, _v_scale_float=0.75)
        assert (
            executor.forward(
                DecodeRequest(
                    layer,
                    query,
                    query,
                    query,
                    cache,
                    metadata,
                    output,
                    None,
                    None,
                    False,
                    "test_layer",
                )
            )
            is output
        )
        assert len(calls) == 1
        route, kwargs = calls[0]
        assert route == expected
        if expected != "triton":
            assert kwargs["max_seq_len_hint"] == 16
            assert (kwargs["k_scale"], kwargs["v_scale"]) == (1.25, 0.75)
            if codec != "auto":
                for name in ("k_scale_cache", "v_scale_cache"):
                    assert kwargs[name].dtype == torch.float32
                    assert kwargs[name].untyped_storage() == cache.untyped_storage()
        assert torch.all(output == {"xqa": 3, "scalar": 2, "triton": 4}[expected])


def test_xqa_import_does_not_require_scalar_or_prefill(monkeypatch):
    import sys

    from vllm.v1.attention.backends.flash_v100 import ops

    def xqa(*args, **kwargs):
        return None

    monkeypatch.setitem(
        sys.modules,
        "flash_attn_v100",
        SimpleNamespace(flash_attn_decode_paged_xqa=xqa),
    )
    for name in (
        "_flash_attn_func",
        "_flash_attn_bhmd_func",
        "_flash_attn_decode_paged",
        "_flash_attn_decode_paged_xqa",
        "_flash_attn_decode_paged_wmma",
        "_flash_attn_prefill_paged",
        "_flash_attn_prefill_paged_bhmd",
        "_flash_attn_prefill_paged_bfla",
        "_flash_attn_prefill_paged_splitkv",
    ):
        monkeypatch.setattr(ops, name, None)
    prepared = []
    monkeypatch.setattr(ops, "prepare_attention_runtime", lambda: prepared.append(True))
    monkeypatch.setattr(ops, "bind_attention_operation", lambda operation: operation)
    monkeypatch.setattr(
        ops._config, "options", lambda: SimpleNamespace(value=lambda name: False)
    )
    loaded = ops.get_flash_ops()
    assert loaded[3] is xqa
    assert all(operation is None for i, operation in enumerate(loaded) if i != 3)
    assert prepared == [True]
