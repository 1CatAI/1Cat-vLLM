# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dispatch tests; numerical evidence lives in test_sm70_dense_batch."""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv


@pytest.fixture(autouse=True)
def clear_env_cache(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SM70_QWEN38_BATCH_FASTPATH", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setattr(gemv.current_platform, "is_device_capability", lambda _: True)
    backend = torch.backends.cuda.matmul
    old_reduction = backend.allow_fp16_reduced_precision_reduction
    old_accumulation = backend.allow_fp16_accumulation
    backend.allow_fp16_reduced_precision_reduction = False
    backend.allow_fp16_accumulation = False
    yield
    backend.allow_fp16_reduced_precision_reduction = old_reduction
    backend.allow_fp16_accumulation = old_accumulation
    envs.disable_envs_cache()


def tensor_descriptor(m, k):
    return SimpleNamespace(
        ndim=2,
        shape=(m, k),
        dtype=torch.float16,
        is_cuda=True,
        device=torch.device("cuda:0"),
        stride=lambda: (k, 1),
        data_ptr=lambda: 16,
    )


@pytest.mark.parametrize("rows", (1, 2, 3, 4, 7, 8, 9, 16, 32))
@pytest.mark.parametrize(
    "role,n,k,limit",
    (
        ("mlp.gate", 512, 2560, 4),
        ("linear_attn.out_proj", 2560, 1536, 8),
        ("self_attn.o_proj", 2560, 1536, 8),
        ("self_attn.qkv_proj", 3584, 2560, 0),
    ),
)
def test_only_winning_batch_shapes_are_admitted(rows, role, n, k, limit):
    x, w = tensor_descriptor(rows, k), tensor_descriptor(n, k)
    assert gemv._can_use_dense_batch(x, w, "model.layers.0." + role) == (
        2 <= rows <= limit
    )


@pytest.mark.parametrize(
    "flag,value",
    (("VLLM_SM70_QWEN38_BATCH_FASTPATH", "0"), ("VLLM_BATCH_INVARIANT", "1")),
)
def test_batch_policy_can_disable_native_dispatch(monkeypatch, flag, value):
    monkeypatch.setenv(flag, value)
    assert not gemv._can_use_dense_batch(
        tensor_descriptor(2, 2560), tensor_descriptor(512, 2560), "layers.0.mlp.gate"
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("dtype", torch.float32),
        ("is_cuda", False),
        ("data_ptr", lambda: 18),
        ("stride", lambda: (5120, 1)),
    ),
)
def test_invalid_storage_stays_on_fallback(field, value):
    x, w = tensor_descriptor(2, 2560), tensor_descriptor(512, 2560)
    setattr(x, field, value)
    assert not gemv._can_use_dense_batch(x, w, "layers.0.mlp.gate")


def test_dense_permission_survives_fake_export():
    class Project(torch.nn.Module):
        def forward(self, x, weight):
            return torch.ops.vllm.qwen38_sm70_fp16_gemv(
                x, weight, "model.layers.0.linear_attn.out_proj", dense_batch=True
            )

    args = tuple(
        torch.empty(shape, dtype=torch.float16, device="meta")
        for shape in ((2, 1536), (2560, 1536))
    )
    graph = torch.export.export(Project(), args).graph
    calls = [
        node
        for node in graph.nodes
        if node.target == torch.ops.vllm.qwen38_sm70_fp16_gemv.default
    ]
    assert len(calls) == 1 and calls[0].args[-1] is True


@pytest.mark.parametrize("rows", (2, 4, 5, 8))
def test_reduced_precision_output_keeps_baseline_schedule(rows):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    assert gemv._can_use_dense_batch(
        tensor_descriptor(rows, 1536),
        tensor_descriptor(2560, 1536),
        "layers.0.linear_attn.out_proj",
    ) == (rows == 8)
    assert gemv._can_use_dense_batch(
        tensor_descriptor(4, 2560),
        tensor_descriptor(512, 2560),
        "layers.0.mlp.gate",
    )
