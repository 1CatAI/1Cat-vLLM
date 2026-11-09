# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The batched-matmul sparse MLA decode must return what the SM70 paged FP8
kernel returns for the same packed cache, indices and lengths."""

import pytest
import torch

from vllm.models.deepseek_v4.common.ops import (
    sparse_attn_decode_bmm,
    sparse_decode_bmm_workspace_specs,
)
from vllm.models.deepseek_v4.sm70.sparse_kernels import (
    sm70_sparse_attention_paged_fp8,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="the SM70 paged FP8 reference kernel runs on Volta",
)

HEAD_DIM = 512
NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
WINDOW = 128
BLOCK_PADDING_BYTES = 40


def _make_cache(num_rows: int, block_size: int) -> torch.Tensor:
    num_blocks = -(-num_rows // block_size)
    # The serving cache pads its blocks: stride(0) exceeds the bytes a block
    # holds.
    padded = torch.zeros(
        (num_blocks, block_size * 584 + BLOCK_PADDING_BYTES),
        dtype=torch.uint8,
        device="cuda",
    )
    cache = padded[:, : block_size * 584]
    data = cache[:, : block_size * 576].view(num_blocks, block_size, 576)
    nope = torch.randn((num_blocks, block_size, NOPE_HEAD_DIM), device="cuda")
    data[:, :, :NOPE_HEAD_DIM].copy_(nope.to(torch.float8_e4m3fn).view(torch.uint8))
    rope = 0.125 * torch.randn(
        (num_blocks, block_size, ROPE_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    data[:, :, NOPE_HEAD_DIM:].copy_(
        rope.view(torch.uint8).reshape_as(data[:, :, NOPE_HEAD_DIM:])
    )
    cache[:, block_size * 576 :].fill_(124)
    # Row 0 stands for a row nobody has written: its rope half is NaN. Slots
    # past the length and -1 slots must not read it into the result.
    nan = torch.full((ROPE_HEAD_DIM,), float("nan"), dtype=torch.bfloat16)
    data[0, 0, NOPE_HEAD_DIM:].copy_(nan.view(torch.uint8).to("cuda"))
    return cache.view(num_blocks, block_size, 584)


def _indices(num_tokens: int, width: int, length: int, num_rows: int):
    indices = torch.randint(
        1, num_rows, (num_tokens, width), device="cuda", dtype=torch.int32
    )
    indices[:, length:] = -1
    indices[:, length - 1 :].clamp_(max=0)  # the last used slot reads row 0
    lengths = torch.full((num_tokens,), length - 1, device="cuda", dtype=torch.int32)
    return indices, lengths


# (extra width, extra length): SWA-only, C4 top-k, C128 over the compressed keys.
@pytest.mark.parametrize("extra", [(0, 0), (512, 512), (192, 144)])
@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("num_tokens", [1, 6])
@torch.inference_mode()
def test_bmm_decode_matches_the_paged_fp8_kernel(extra, num_heads, num_tokens):
    torch.manual_seed(0)
    extra_width, extra_length = extra
    num_rows = 4096
    main_cache = _make_cache(num_rows, 256)
    main_indices, main_lengths = _indices(num_tokens, WINDOW, WINDOW, num_rows)
    extra_cache = extra_indices = extra_lengths = None
    if extra_width:
        extra_cache = _make_cache(num_rows, 64)
        extra_indices, extra_lengths = _indices(
            num_tokens, extra_width, extra_length, num_rows
        )
    q = (0.5 * torch.randn(num_tokens, num_heads, HEAD_DIM, device="cuda")).half()
    sink = torch.randn(num_heads, device="cuda")
    scale = HEAD_DIM**-0.5

    expected = torch.empty_like(q)
    sm70_sparse_attention_paged_fp8(
        q,
        main_cache,
        main_indices,
        main_lengths,
        scale,
        sink,
        expected,
        extra_cache,
        extra_indices,
        extra_lengths,
    )

    buffers = [
        torch.empty(shape, dtype=dtype, device="cuda")
        for shape, dtype in sparse_decode_bmm_workspace_specs(
            num_tokens, num_heads, HEAD_DIM, WINDOW, extra_width, torch.float16
        )
    ]
    output = torch.empty_like(q)

    def run():
        sparse_attn_decode_bmm(
            q,
            main_cache,
            main_indices,
            main_lengths,
            extra_cache,
            extra_indices,
            extra_lengths,
            scale,
            sink,
            output,
            *buffers,
        )

    # All shapes are static: the call captures into a CUDA graph.
    stream = torch.cuda.Stream()
    # vLLM's stream, so that leaving the contexts below restores it instead of
    # recording torch's default stream as vLLM's current stream, which breaks
    # later graph captures in the same process.
    stream.wait_stream(current_stream())
    with torch.cuda.stream(stream):
        run()
    current_stream().wait_stream(stream)
    output.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.accelerator.synchronize()

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-3)
