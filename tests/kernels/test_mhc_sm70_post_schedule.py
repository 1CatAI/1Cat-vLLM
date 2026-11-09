# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the small-batch post schedule against the former 512-element tiles."""

import pytest
import torch

from vllm.model_executor.kernels.mhc.triton import (
    _sm70_mhc_post_kernel,
    sm70_mhc_post,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 numerical and launch contract",
)


def _legacy_post(x, residual, post, comb, out):
    # Keep the old launch geometry: changing the tile can change compiler
    # contraction, even when the scalar expressions are unchanged.
    _sm70_mhc_post_kernel[(x.shape[0], 8)](
        x,
        residual,
        post,
        comb,
        out,
        x.stride(0),
        *residual.stride(),
        *post.stride(),
        *comb.stride(),
        *out.stride(),
        HIDDEN_SIZE=4096,
        HC_MULT=4,
        BLOCK_H=512,
        num_warps=4,
    )


def _inputs(tokens, layout, seed=617):
    torch.manual_seed(seed)

    def rand(shape, dtype, *, unit_stride=False):
        if layout == "contiguous":
            return torch.randn(shape, dtype=dtype, device="cuda")
        step = 2 if layout == "strided" and not unit_stride else 1
        storage = torch.randn(
            (*shape[:-1], shape[-1] * step + 8), dtype=dtype, device="cuda"
        )
        return storage[..., : shape[-1] * step : step]

    # The existing x contract requires unit hidden stride; residual, mixes
    # and output have explicit strides in the kernel interface.
    x = rand((tokens, 4096), torch.float16, unit_stride=True)
    residual = rand((tokens, 4, 4096), torch.float16)
    post = rand((tokens, 4), torch.float32)
    comb = rand((tokens, 4, 4), torch.float32)
    out = rand((tokens, 4, 4096), torch.float16)
    return x, residual, post, comb, out


@pytest.mark.parametrize("tokens", [0, 1, 2, 4, 8, 9, 16])
@pytest.mark.parametrize("layout", ["contiguous", "padded", "strided"])
@pytest.mark.parametrize("pattern", ["random", "range"])
def test_post_schedule_exact(tokens, layout, pattern):
    x, residual, post, comb, out = _inputs(tokens, layout)
    if pattern == "range":
        values = torch.tensor(
            [65504, -65504, 256, -256, 2**-24, -(2**-24), 0, -0.0],
            dtype=torch.float16,
            device="cuda",
        ).repeat(512)
        x.copy_(values.expand_as(x))
        residual.copy_(values.expand_as(residual))
    expected = torch.empty_strided(
        out.shape, out.stride(), dtype=out.dtype, device=out.device
    )
    _legacy_post(x, residual, post, comb, expected)
    actual = sm70_mhc_post(x, residual, post, comb, out)
    assert actual is out
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def test_post_replay_updates_independent_buffers():
    graphs = []
    for tokens, seed in ((1, 27), (8, 51)):
        inputs = _inputs(tokens, "padded", seed)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            sm70_mhc_post(*inputs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            sm70_mhc_post(*inputs)
        graphs.append((graph, inputs))

    for step in range(3):
        for graph, inputs in graphs[:: 1 if step % 2 == 0 else -1]:
            x, residual, post, comb, out = inputs
            x.add_(0.125)
            residual.mul_(0.75)
            post.add_(0.25)
            comb.mul_(0.875)
            out.fill_(float("nan"))
            graph.replay()
            expected = torch.empty_strided(
                out.shape, out.stride(), dtype=out.dtype, device=out.device
            )
            _legacy_post(x, residual, post, comb, expected)
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    assert graphs[0][1][-1].data_ptr() != graphs[1][1][-1].data_ptr()
