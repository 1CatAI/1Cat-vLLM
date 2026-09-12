# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops.sm70_e4m3_grouped import (
    grouped_e4m3_fp32_allowed,
    load_grouped_e4m3_fp32,
)


@pytest.mark.parametrize(
    "change",
    [
        None,
        "unavailable",
        "e5m2",
        "batch2",
        "table2",
        "q1",
        "q9",
        "page",
        "lengths",
        "dtype",
        "partition",
        "window",
        "causal",
        "capacity",
    ],
)
def test_single_request_shape_and_metadata_admission(monkeypatch, change):
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    instance = SimpleNamespace(
        kv_cache_dtype="fp8_e4m3",
        use_smallq_decode_xqa=True,
        flash_attn_grouped_e4m3_fp32_paged=object(),
        _flash_v100_window_size=lambda causal: (-1, -1),
    )
    rows = 1 if change == "q1" else 9 if change == "q9" else 5
    q = torch.empty((rows, 6, 256), dtype=torch.float16)
    k = torch.empty((1, 848, 1, 256), dtype=torch.uint8)
    table = torch.zeros((rows, 310), dtype=torch.int32)
    lengths = torch.zeros(rows, dtype=torch.int32)
    metadata = SimpleNamespace(
        seq_lens=torch.ones(1, dtype=torch.int32),
        block_table=table[:1],
        causal=True,
    )
    partition = None
    if change == "unavailable":
        instance.flash_attn_grouped_e4m3_fp32_paged = None
    elif change == "e5m2":
        instance.kv_cache_dtype = "fp8_e5m2"
    elif change == "batch2":
        metadata.seq_lens = torch.ones(2, dtype=torch.int32)
    elif change == "table2":
        metadata.block_table = table.expand(2, -1) if rows == 1 else table[:2]
    elif change == "page":
        k = k[:, :816]
    elif change == "lengths":
        lengths = lengths.long()
    elif change == "dtype":
        q = q.float()
    elif change == "partition":
        partition = 1024
    elif change == "window":
        instance._flash_v100_window_size = lambda causal: (4096, 0)
    elif change == "causal":
        metadata.causal = False
    elif change == "capacity":
        table = torch.zeros((rows, 999), dtype=torch.int32)
        metadata.block_table = table[:1]
    assert grouped_e4m3_fp32_allowed(
        instance,
        q,
        k,
        k,
        table,
        lengths,
        metadata,
        out=q,
        partition_size_hint=partition,
    ) is (change is None)


@pytest.mark.parametrize("available", [False, True])
def test_native_capability_fail_closed(monkeypatch, available):
    sentinel = object()
    module = SimpleNamespace(
        flash_attn_grouped_e4m3_fp32_available=lambda: available,
        flash_attn_grouped_e4m3_fp32_paged=sentinel,
    )
    monkeypatch.setitem(sys.modules, "flash_attn_v100", module)
    # With the long-context route off the loader must hand back the native
    # binding untouched; the wrapping itself is covered by the tail graph tests.
    monkeypatch.setenv("VLLM_SM70_E4M3_LONG_ATTENTION", "0")
    assert load_grouped_e4m3_fp32() is (sentinel if available else None)


def test_default_on_with_explicit_rollback(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_FLASH_V100_E4M3_GROUPED_FP32", raising=False)
    envs.disable_envs_cache()
    assert envs.VLLM_FLASH_V100_E4M3_GROUPED_FP32 is True
    monkeypatch.setenv("VLLM_FLASH_V100_E4M3_GROUPED_FP32", "0")
    assert envs.VLLM_FLASH_V100_E4M3_GROUPED_FP32 is False
