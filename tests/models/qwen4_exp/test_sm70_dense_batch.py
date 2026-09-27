# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native small-batch dense projection arithmetic; no runtime admission."""

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as _ops  # noqa: F401


@pytest.fixture(autouse=True)
def full_precision_sm70():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    backend = torch.backends.cuda.matmul
    names = (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    )
    previous = [getattr(backend, name) for name in names]
    for name in names:
        setattr(backend, name, False)
    yield
    for name, value in zip(names, previous):
        setattr(backend, name, value)


@pytest.mark.parametrize("n,k", ((512, 2560), (2560, 1536)))
@pytest.mark.parametrize("m", (2, 3, 4, 7, 8, 9, 15, 16))
def test_dense_batch_graph_bits_and_canaries(m, n, k):
    torch.manual_seed(20927)
    weight = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
    x = torch.randn(m, k, dtype=torch.float16, device="cuda")
    storage = torch.full((m * n + 16,), 17, dtype=torch.float16, device="cuda")
    actual = storage[8:-8].view(m, n)
    op = torch.ops._C.qwen38_dense_batch_sm70_out
    op(actual, x, weight)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(actual, x, weight)
    for scale in (0, 0.001, 0.03, 0.1, 1, 3):
        x.normal_(0, scale)
        actual.fill_(float("nan"))
        graph.replay()
        expected = F.linear(x, weight)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert torch.all(storage[:8] == 17) and torch.all(storage[-8:] == 17)


@pytest.mark.parametrize("m,n,k", ((1, 512, 2560), (17, 512, 2560), (2, 640, 2560)))
def test_dense_batch_unsupported_geometry(m, n, k):
    x = torch.empty(m, k, dtype=torch.float16, device="cuda")
    w = torch.empty(n, k, dtype=torch.float16, device="cuda")
    out = torch.empty(m, n, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops._C.qwen38_dense_batch_sm70_out(out, x, w)


def test_dense_batch_unaligned_input_rejected():
    x = torch.empty(2 * 2560 + 1, dtype=torch.float16, device="cuda")[1:].view(2, 2560)
    w = torch.empty(512, 2560, dtype=torch.float16, device="cuda")
    out = torch.empty(2, 512, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="aligned contiguous FP16"):
        torch.ops._C.qwen38_dense_batch_sm70_out(out, x, w)


@pytest.mark.parametrize("m", (1, 2, 4, 8, 16))
@pytest.mark.parametrize(
    "role,n,k,limit",
    (
        ("mlp.gate", 512, 2560, 4),
        ("linear_attn.out_proj", 2560, 1536, 8),
        ("self_attn.o_proj", 2560, 1536, 8),
    ),
)
def test_dense_runtime_route_and_fallback_bits(m, role, n, k, limit, monkeypatch):
    import vllm.envs as envs
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv

    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SM70_QWEN38_BATCH_FASTPATH", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    torch.manual_seed(20928)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    role = "model.layers.0." + role
    reference = gemv._qwen38_sm70_fp16_gemv(x, w, role, dense_batch=False)
    original = torch.ops._C.qwen38_dense_batch_sm70_out
    calls = []

    def tracked(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(torch.ops._C, "qwen38_dense_batch_sm70_out", tracked)
    actual = torch.ops.vllm.qwen38_sm70_fp16_gemv(x, w, role, dense_batch=True)
    assert len(calls) == int(2 <= m <= limit)
    assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
    envs.disable_envs_cache()
