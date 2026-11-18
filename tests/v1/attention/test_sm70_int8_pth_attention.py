# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SM70 per-token-head INT8 KV cache routing and defaults."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.config.kernel import KernelConfig
from vllm.engine.arg_utils import _resolve_sm70_flash_v100_kv_cache_dtype_alias
from vllm.platforms import current_platform
from vllm.v1.attention.kv_codecs import INT8_PER_TOKEN_HEAD
from vllm.v1.attention.ops import sm70_grouped as grouped

pytestmark = pytest.mark.cpu_test


def test_sm70_default_kv_cache_dtype_resolves_to_int8_pth(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda *args, **kwargs: SimpleNamespace(major=7, minor=0),
    )
    model_cfg = SimpleNamespace(
        is_attention_free=False,
        use_mla=False,
        runner_type="generate",
    )
    kernel_cfg = KernelConfig()
    assert kernel_cfg.qsa_host_kv_dtype == "int8_per_token_head"

    dtype, from_ckpt = _resolve_sm70_flash_v100_kv_cache_dtype_alias(
        "auto",
        "auto",
        model_config=model_cfg,
        kernel_config=kernel_cfg,
        attention_backend=None,
    )
    assert dtype == "int8_per_token_head"
    assert not from_ckpt

    # Checkpoint FP8 directives also yield to default INT8 on SM70 when auto.
    dtype_ckpt, from_ckpt2 = _resolve_sm70_flash_v100_kv_cache_dtype_alias(
        "auto",
        "fp8",
        model_config=model_cfg,
        kernel_config=kernel_cfg,
        attention_backend="FLASH_ATTN_V100",
    )
    assert dtype_ckpt == "int8_per_token_head"
    assert not from_ckpt2

    # Explicit overrides remain untouched.
    assert _resolve_sm70_flash_v100_kv_cache_dtype_alias(
        "float16",
        "float16",
        model_config=model_cfg,
        kernel_config=kernel_cfg,
    ) == ("float16", False)
    assert _resolve_sm70_flash_v100_kv_cache_dtype_alias(
        "fp8_e4m3",
        "fp8_e4m3",
        model_config=model_cfg,
        kernel_config=kernel_cfg,
    ) == ("fp8_e4m3", False)
    assert _resolve_sm70_flash_v100_kv_cache_dtype_alias(
        "int8",
        "int8",
        model_config=model_cfg,
        kernel_config=kernel_cfg,
    ) == ("int8_per_token_head", False)

    # Host-KV QSA keeps device stub cache on auto while host KV uses INT8.
    host_kv_cfg = KernelConfig(qsa_host_kv=True)
    assert _resolve_sm70_flash_v100_kv_cache_dtype_alias(
        "auto",
        "auto",
        model_config=model_cfg,
        kernel_config=host_kv_cfg,
    ) == ("auto", False)


def test_grouped_contract_admits_int8_per_token_head():
    dev = torch.device("cuda:0")

    class Spec:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.device = dev
            self.ndim = len(shape)

        def is_contiguous(self):
            return True

        def data_ptr(self):
            return 64

        def stride(self, dim=None):
            s = []
            step = 1
            for d in reversed(self.shape):
                s.append(step)
                step *= d
            res = tuple(reversed(s))
            return res if dim is None else res[dim]

    op = SimpleNamespace(
        request_query_rows=(2, 3, 4, 5, 6, 7, 8),
        max_batch_size=16,
        multi_kv_heads=True,
        runtime_pages=True,
    )
    inst = SimpleNamespace(
        flash_attn_grouped_e4m3_fp32_paged=op,
        kv_cache_dtype="int8_per_token_head",
        use_smallq_decode_xqa=True,
        _flash_v100_window_size=lambda **kw: (-1, -1),
    )
    q = Spec((12, 12, 256), torch.float16)
    out = Spec((12, 12, 256), torch.float16)
    k_i8 = Spec((10, 544, 2, 256), torch.int8)
    v_i8 = Spec((10, 544, 2, 256), torch.int8)
    k_u8 = Spec((10, 544, 2, 256), torch.uint8)
    row_bt = Spec((12, 10), torch.int32)
    lengths = Spec((12,), torch.int32)
    meta = SimpleNamespace(
        block_table=Spec((4, 10), torch.int32),
        seq_lens=Spec((4,), torch.int32),
        causal=True,
        query_start_loc_cpu=torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32),
    )
    assert grouped.grouped_e4m3_fp32_allowed(
        inst, q, k_i8, v_i8, row_bt, lengths, meta, out=out, partition_size_hint=None
    )
    assert not grouped.grouped_e4m3_fp32_allowed(
        inst, q, k_u8, v_i8, row_bt, lengths, meta, out=out, partition_size_hint=None
    )
    assert grouped.GROUPED_CONTRACTS[INT8_PER_TOKEN_HEAD].codec is INT8_PER_TOKEN_HEAD
