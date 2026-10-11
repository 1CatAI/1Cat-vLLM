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
    "use_xqa,use_scalar,expected",
    [
        (True, True, "xqa"),
        (True, False, "xqa"),
        (False, True, "scalar"),
        (False, False, "triton"),
    ],
)
def test_decode_executor_runs_with_independent_operators(
    monkeypatch, codec, use_xqa, use_scalar, expected
):
    with cpu_cuda(monkeypatch, False), strict_shim() as legacy:
        install_ops(monkeypatch, Recorder(), legacy, {})
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
                paged=scalar,
                xqa=xqa,
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
