# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in tests for the acceptance-only CUDA kernel; no runtime registration."""

import os
from pathlib import Path

import pytest
import torch

from benchmarks.kernels.benchmark_h3_vsa_fp32 import fp32_reference, load_binary

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("H3_VSA_FP32_DIAGNOSTIC"),
    reason="requires a leased SM70 GPU and H3_VSA_FP32_DIAGNOSTIC binary",
)


@pytest.fixture(scope="module")
def ops():
    return load_binary(Path(os.environ["H3_VSA_FP32_DIAGNOSTIC"]))


def make_case(lengths, prefix, topk, amplitude=1):
    torch.manual_seed(821)
    blocks = len(lengths)
    sizes = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    q, k, v = [
        torch.randn(2, blocks * 64, 3, 128, device="cuda", dtype=torch.float16)
        * amplitude
        for _ in range(3)
    ]
    for i, length in enumerate(lengths):
        for tensor in (q, k, v):
            tensor[:, i * 64 + length : (i + 1) * 64] = float("nan")
    # Independent mask construction: prefix queries see every block; video
    # queries see every prefix block and their own top-k video blocks.
    scores = torch.randn(2, 3, blocks - prefix, blocks - prefix, device="cuda")
    selected = scores.topk(min(topk, blocks - prefix), dim=-1).indices
    mask = torch.zeros(2, 3, blocks, blocks, device="cuda", dtype=torch.bool)
    mask[:, :, :prefix] = True
    mask[:, :, :, :prefix] = True
    mask[:, :, prefix:, prefix:].scatter_(-1, selected, True)
    return [q, k, v, mask, sizes, 128**-0.5, prefix, topk]


@pytest.mark.parametrize(
    "lengths,prefix,topk,amplitude",
    [
        ((1,), 0, 1, 1),
        ((63, 1), 1, 1, 1),
        ((17, 64, 3, 64, 1), 2, 1, 1),
        ((17, 64, 3, 64, 1), 2, 99, 1),
        ((64, 3, 1), 1, 1, 16),
        ((64,) * 33 + (17, 64, 3, 64, 1), 2, 8, 1),
    ],
)
@torch.inference_mode()
def test_exact_fp32_sparse_math(ops, lengths, prefix, topk, amplitude):
    args = make_case(lengths, prefix, topk, amplitude)
    expected = fp32_reference(*args[:6])
    actual = ops.forward(*args)
    private = ops._forward_prevalidated(*args)
    valid = torch.cat(
        [torch.arange(i * 64, i * 64 + n, device="cuda") for i, n in enumerate(lengths)]
    )
    assert torch.equal(
        actual[:, valid].view(torch.int16), expected[:, valid].view(torch.int16)
    )
    assert torch.equal(actual.view(torch.int16), private.view(torch.int16))
    assert torch.isfinite(actual).all()
    for i, length in enumerate(lengths):
        assert not actual[:, i * 64 + length : (i + 1) * 64].count_nonzero()


@pytest.mark.parametrize(
    "invalid",
    [
        "empty_row",
        "missing_prefix_key",
        "zero_size",
        "large_size",
        "size_dtype",
        "map_dtype",
        "map_shape",
        "q_dtype",
        "q_stride",
        "q_grad",
        "prefix",
        "topk",
        "scale",
        "unaligned",
    ],
)
def test_public_entry_rejects_invalid_inputs(ops, invalid):
    args = make_case((64, 17, 64), 1, 1)
    q, _, _, mask, sizes, _, _, _ = args
    if invalid == "empty_row":
        mask[:, :, 1] = False
    elif invalid == "missing_prefix_key":
        mask[:, :, 1, 0] = False
        mask[:, :, 1, 1:] = True  # Correct count, wrong prefix selection.
    elif invalid == "zero_size":
        sizes[1] = 0
    elif invalid == "large_size":
        sizes[1] = 65
    elif invalid == "size_dtype":
        args[4] = sizes.long()
    elif invalid == "map_dtype":
        args[3] = mask.int()
    elif invalid == "map_shape":
        args[3] = mask[..., :-1].contiguous()
    elif invalid == "q_dtype":
        args[0] = q.float()
    elif invalid == "q_stride":
        args[0] = q.transpose(1, 2)
    elif invalid == "q_grad":
        q.requires_grad_()
    elif invalid == "prefix":
        args[6] = 3
    elif invalid == "topk":
        args[7] = 0
    elif invalid == "scale":
        args[5] = float("nan")
    elif invalid == "unaligned":
        args[0] = torch.empty(q.numel() + 1, device=q.device, dtype=q.dtype)[
            1:
        ].view_as(q)
    with pytest.raises(RuntimeError):
        ops.forward(*args)
    torch.accelerator.synchronize()
