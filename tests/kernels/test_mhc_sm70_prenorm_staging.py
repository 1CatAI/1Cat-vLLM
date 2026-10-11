# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise regression oracle for the former two-launch prenorm stage."""

import pytest
import torch

from vllm.model_executor.kernels.mhc.triton import sm70_mhc_prenorm_staging
from vllm.triton_utils import tl, triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 numerical and launch contract",
)


# Retain the pre-fusion reductions independently: torch.sum/matmul use different
# reduction trees and cannot serve as a bitwise oracle for these FP32 partials.
@triton.jit
def _separate_sqrsum(X, S, XS, SS, ST, K: tl.constexpr, MULT: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    offsets = split * K + tl.arange(0, K)
    x = tl.load(X + row * XS + offsets).to(tl.float32)
    sqrsum = tl.sum(x * x, axis=0) * MULT
    tl.store(S + split * SS + row * ST, sqrsum)


@triton.jit
def _separate_dot(X, W, Y, XS, WS, YS, YT, K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    split = tl.program_id(2)
    offsets = split * K + tl.arange(0, K)
    x = tl.load(X + row * XS + offsets).to(tl.float32)
    w = tl.load(W + col * WS + offsets).to(tl.float32)
    dot = tl.sum(x * w, axis=0)
    tl.store(Y + split * YS + row * YT + col, dot)


def separate_staging(x, w, y, s, mult=1):
    splits = y.shape[0]
    k = x.shape[1] // splits
    _separate_sqrsum[(x.shape[0], splits)](
        x, s, x.stride(0), s.stride(0), s.stride(1), k, mult, num_warps=8
    )
    _separate_dot[(x.shape[0], w.shape[0], splits)](
        x,
        w,
        y,
        x.stride(0),
        w.stride(0),
        y.stride(0),
        y.stride(1),
        k,
        num_warps=8,
    )


def _assert_bits_equal(actual, expected):
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("tokens", [0, 1, 2, 4, 8, 16, 17, 128, 129, 256])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("pattern", ["random", "cancellation"])
def test_prenorm_staging_exact(tokens, broadcast, pattern):
    torch.manual_seed(617)
    k = 4096 if broadcast else 16384
    splits = (
        (2 if tokens <= 16 else 1)
        if broadcast
        else (8 if tokens <= 16 else 4 if tokens <= 128 else 2)
    )
    mult = 4 if broadcast else 1
    x = torch.randn(tokens, k, dtype=torch.float16, device="cuda")
    # Padded rows exercise the existing weight/output stride contract.
    w = torch.randn(24, k + 8, dtype=torch.float32, device="cuda")[:, :k]
    y = torch.full((splits, tokens, 32), float("nan"), device="cuda")[:, :, :24]
    s = torch.full((tokens, splits), float("nan"), device="cuda").t()
    ref_y, ref_s = torch.empty_like(y), torch.empty_like(s)
    if pattern == "cancellation":
        values = torch.tensor(
            [65504, -65504, 1, -1, 2**-24, -(2**-24), 0, -0.0],
            dtype=torch.float16,
            device="cuda",
        )
        x.copy_(values.repeat(k // 8).expand(tokens, k))
        w[:, 1::2].copy_(w[:, ::2])
    separate_staging(x, w, ref_y, ref_s, mult)
    sm70_mhc_prenorm_staging(x, w, y, s, sqrsum_mult=mult)
    _assert_bits_equal(y, ref_y)
    _assert_bits_equal(s, ref_s)


def test_prenorm_staging_no_projection_columns():
    x = torch.randn(2, 4096, dtype=torch.float16, device="cuda")
    w = torch.empty(0, 4096, dtype=torch.float32, device="cuda")
    y = torch.empty(2, 2, 0, dtype=torch.float32, device="cuda")
    s = torch.empty(2, 2, dtype=torch.float32, device="cuda")
    ref_s = torch.empty_like(s)
    separate_staging(x, w, y, ref_s, 4)
    sm70_mhc_prenorm_staging(x, w, y, s, sqrsum_mult=4)
    _assert_bits_equal(s, ref_s)


def test_prenorm_staging_replay_and_independent_buffers():
    graphs = []
    for seed in (32, 71):
        torch.manual_seed(seed)
        x = torch.randn(4, 16384, dtype=torch.float16, device="cuda")
        w = torch.randn(24, 16384, dtype=torch.float32, device="cuda")
        y = torch.empty(8, 4, 24, device="cuda")
        s = torch.empty(8, 4, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            sm70_mhc_prenorm_staging(x, w, y, s)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            sm70_mhc_prenorm_staging(x, w, y, s)
        graphs.append((graph, x, w, y, s))

    for round_idx in range(3):
        for graph, x, w, y, s in graphs[:: 1 if round_idx % 2 == 0 else -1]:
            x.add_(0.125)
            w.mul_(0.875)
            y.fill_(float("nan"))
            s.fill_(float("nan"))
            graph.replay()
            ref_y, ref_s = torch.empty_like(y), torch.empty_like(s)
            separate_staging(x, w, ref_y, ref_s)
            _assert_bits_equal(y, ref_y)
            _assert_bits_equal(s, ref_s)
    assert graphs[0][3].data_ptr() != graphs[1][3].data_ptr()
    assert graphs[0][4].data_ptr() != graphs[1][4].data_ptr()
