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
)
from vllm.v1.attention.backends.flash_v100.workspace import V100Workspace

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("use_xqa", [False, True])
def test_decode_executor_runs_with_independent_operators(monkeypatch, use_xqa):
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
            replace(backend.config, use_decode_xqa=use_xqa),
            backend.scale,
            backend.kv_cache_dtype,
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
                paged_keywords={"max_seq_len_hint"},
                scalar_tail=None,
                reject_xqa=lambda codec, metadata: False,
                reserve_bhmd_compare=unused,
                write_bhmd_compare=unused,
                compare_bhmd=unused,
                compare_triton=unused,
                triton_forward=unused,
                profile_trace=unused,
                draft_debug_enabled=unused,
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
        cache = torch.zeros((2, 1, 16, 1, 256), dtype=torch.float16)
        metadata = SimpleNamespace(
            num_actual_tokens=1,
            seq_lens=torch.tensor([16], dtype=torch.int32),
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            flash_v100_decode_max_seq_len_hint=16,
        )
        layer = SimpleNamespace(_k_scale_float=1.25, _v_scale_float=0.75)
        assert (
            executor._flash_v100_decode(
                layer, query, query, query, cache, metadata, output
            )
            is output
        )
        assert len(calls) == 1
        route, kwargs = calls[0]
        assert route == ("xqa" if use_xqa else "scalar")
        assert kwargs["max_seq_len_hint"] == 16
        assert (kwargs["k_scale"], kwargs["v_scale"]) == (1.25, 0.75)
        assert torch.all(output == (3 if use_xqa else 2))
