# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.kernels.linear.fp16_gemv_silu import Sm70Fp16GemvSiluKernel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an SM70 device",
)


@pytest.mark.parametrize(
    "m,n,k,prefix,pad",
    [
        (1, 17, 73, 9, 24),
        (2, 41, 4096, 40, 48),
        (3, 81, 10240, 80, 88),
        (4, 7, 255, 0, 16),
        (4, 37, 8192, 32, 40),
        (8, 129, 256, 64, 136),
        (16, 33, 64, 32, 40),
        (1, 2560, 160, 0, 2560),
        (5, 321, 160, 159, 328),
        (17, 37, 255, 19, 43),
        (33, 19, 73, 0, 24),
    ],
)
def test_rows_padding_and_changed_graph_inputs(m, n, k, prefix, pad):
    torch.manual_seed(m + n)
    weight = torch.randint(-3, 4, (n + 7, k), device="cuda").half()
    x = torch.randint(-3, 4, (m, k), device="cuda").half()
    out = torch.empty(m, pad, device="cuda", dtype=torch.float16)
    args = (x, weight, out, n, prefix, 2, prefix + 5, 4.0)
    for _ in range(3):
        Sm70Fp16GemvSiluKernel.apply_out(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        Sm70Fp16GemvSiluKernel.apply_out(*args)
    rows = torch.cat(
        [
            torch.arange(2, prefix + 2, device="cuda"),
            torch.arange(prefix + 5, n + 5, device="cuda"),
        ]
    )
    for _ in range(8):
        x.random_(-3, 4)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        ref = (x.double() @ weight[rows].double().T).half().float()
        ref[:, :prefix] = torch.nn.functional.silu(ref[:, :prefix] / 4)
        assert torch.allclose(
            out[:, :n].float(), ref.half().float(), rtol=2e-6, atol=1e-6
        )
        assert torch.equal(out[:, n:], torch.zeros_like(out[:, n:]))


def test_fp16_subnormal_projection():
    x = torch.full((1, 64), 2**-24, device="cuda", dtype=torch.float16)
    w = torch.ones(17, 64, device="cuda", dtype=torch.float16)
    out = torch.empty(1, 17, device="cuda", dtype=torch.float16)
    Sm70Fp16GemvSiluKernel.apply_out(x, w, out, 17, 0)
    assert torch.equal(out, (x.double() @ w.double().T).half())


def test_capability_rejects_invalid_ranges_and_layouts():
    x = torch.ones(1, 64, device="cuda", dtype=torch.float16)
    w = torch.ones(17, 64, device="cuda", dtype=torch.float16)
    out = torch.empty(1, 24, device="cuda", dtype=torch.float16)
    valid = (x, w, out, 17, 10, 0, 10, 4.0)
    assert Sm70Fp16GemvSiluKernel.can_implement(*valid)
    for changes in (
        {3: 25},
        {4: 18},
        {5: -1},
        {6: 17},
        {7: 0},
        {7: float("inf")},
        {0: x.float()},
        {1: w[:, ::2]},
        {2: out[:, ::2]},
        {2: x},
    ):
        args = list(valid)
        for i, v in changes.items():
            args[i] = v
        assert not Sm70Fp16GemvSiluKernel.can_implement(*args)
