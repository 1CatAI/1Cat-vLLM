# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa


@pytest.mark.parametrize("blocks", [21, 82, 656])
def test_exact_m1_cache_geometry(monkeypatch, blocks):
    monkeypatch.setattr(qsa.current_platform, "is_device_capability", lambda c: c == 70)
    q = NS(shape=(1, 6, 256), dtype=torch.float16)
    k = NS(shape=(blocks, 400, 1, 256), dtype=torch.float16)
    ids = NS(shape=(1, 2051), dtype=torch.int32)
    assert qsa._use_sm70_qsa_resolved_indices(q, k, ids, "float16")


@pytest.mark.parametrize(
    "bad", ["arch", "rows", "dtype", "page", "width", "overflow", "kv"]
)
def test_unsupported_routes_unchanged(monkeypatch, bad):
    monkeypatch.setattr(
        qsa.current_platform, "is_device_capability", lambda c: bad != "arch"
    )
    q = NS(shape=(1, 6, 256), dtype=torch.float16)
    k = NS(shape=(656, 400, 1, 256), dtype=torch.float16)
    ids = NS(shape=(1, 2051), dtype=torch.int32)
    if bad == "rows":
        q.shape = (2, 6, 256)
    if bad == "dtype":
        q.dtype = torch.bfloat16
    if bad == "page":
        k.shape = (400, 784, 1, 256)
    if bad == "width":
        ids.shape = (1, 512)
    if bad == "overflow":
        k.shape = (2**31, 400, 1, 256)
    dtype = "fp8_e4m3" if bad == "kv" else "float16"
    assert not qsa._use_sm70_qsa_resolved_indices(q, k, ids, dtype)
