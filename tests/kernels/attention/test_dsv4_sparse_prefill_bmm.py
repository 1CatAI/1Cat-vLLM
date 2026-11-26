# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The batched-matmul sparse MLA prefill attention against a float64 reference."""

import pytest
import torch

from tests.kernels.utils import requires_tensor_core_triton
from vllm.models.deepseek_v4.common.ops import (
    sparse_attn_prefill_bmm,
    sparse_attn_prefill_tensor_cores,
    sparse_prefill_bmm_workspace_specs,
)
from vllm.models.deepseek_v4.common.ops.sparse_prefill_bmm import tokens_per_pass
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="measured and enabled on CUDA only"
)

HEAD_DIM = 512
# The BLAS route and the tensor-core route that reads the selected rows itself.
ROUTES = [False, pytest.param(True, marks=requires_tensor_core_triton)]


def _prefill(tensor_cores: bool):
    return sparse_attn_prefill_tensor_cores if tensor_cores else sparse_attn_prefill_bmm


def _reference(q, kv, indices, lengths, scale, attn_sink) -> torch.Tensor:
    out = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    for token in range(q.shape[0]):
        row = indices[token, : int(lengths[token])]
        keys = kv[row[row >= 0].long()].double()
        logits = torch.einsum("hd,kd->hk", q[token].double(), keys) * scale
        logits = torch.cat([logits, attn_sink.double()[:, None]], dim=1)
        probs = torch.softmax(logits, dim=1)[:, :-1]
        out[token] = probs @ keys
    return out


# 300 tokens take three passes, the last one partial.
@pytest.mark.parametrize("num_tokens", [7, 300])
@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("width", [128, 640])
@pytest.mark.parametrize("stale_tail", [False, True])
@pytest.mark.parametrize("tensor_cores", ROUTES)
@torch.inference_mode()
def test_sparse_attn_prefill_bmm_matches_reference(
    num_tokens: int, num_heads: int, width: int, stale_tail: bool, tensor_cores: bool
):
    torch.manual_seed(0)
    num_kv = 900
    q = torch.randn((num_tokens, num_heads, HEAD_DIM), device="cuda").half()
    kv = (0.3 * torch.randn((num_kv, HEAD_DIM), device="cuda")).half()
    # Row 0 stands for workspace memory no sequence has written yet (a short
    # prompt has no compressed entries); unused slots must not read it.
    kv[0] = float("nan")
    indices = torch.randint(
        1, num_kv, (num_tokens, width), dtype=torch.int32, device="cuda"
    )
    lengths = torch.tensor(
        [width, 0, 1, width // 2, width - 1, 5, width], dtype=torch.int32
    ).cuda()
    lengths = lengths.repeat(-(-num_tokens // lengths.numel()))[:num_tokens]
    past_length = torch.arange(width, device="cuda")[None, :] >= lengths[:, None]
    if stale_tail:
        # combine_topk_swa_indices fills a reused workspace and writes only the
        # valid slots, so past the length lies whatever was there before.
        stale = torch.randint_like(indices, num_kv, 2**30)
        indices = torch.where(past_length, stale, indices)
    else:
        indices[past_length] = -1
    indices[3, 2] = -1  # an unused slot inside the valid length
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device="cuda")
    scale = HEAD_DIM**-0.5

    # Reserved for more tokens and a wider index tensor than this call uses.
    buffers = [
        torch.full(shape, float("nan"), dtype=dtype, device="cuda")
        for shape, dtype in sparse_prefill_bmm_workspace_specs(
            num_tokens + 9,
            num_heads,
            HEAD_DIM,
            width + 128,
            torch.float16,
            tensor_cores=tensor_cores,
        )
    ]
    output = torch.empty_like(q)
    _prefill(tensor_cores)(q, kv, indices, lengths, scale, attn_sink, output, *buffers)

    expected = _reference(q, kv, indices, lengths, scale, attn_sink)
    assert torch.isfinite(output).all()
    assert not output[1].any()  # no keys: the sink takes all the weight
    assert buffers[0].shape[0] == min(num_tokens + 9, tokens_per_pass(tensor_cores))
    # The tensor-core route reads the selected rows itself: no gathered keys.
    assert len(buffers) == (2 if tensor_cores else 3)
    torch.testing.assert_close(output.double(), expected, atol=1e-3, rtol=1e-3)


@requires_tensor_core_triton
@torch.inference_mode()
def test_tensor_core_route_spans_several_passes():
    # Two full passes of the tensor-core route and a partial one match the BLAS
    # route, which the reference test above covers.
    torch.manual_seed(1)
    device = torch.device("cuda")
    num_tokens = 2 * tokens_per_pass(True) + 3
    num_heads, width, num_kv = 64, 640, 900
    q = torch.randn((num_tokens, num_heads, HEAD_DIM), device=device).half()
    kv = (0.3 * torch.randn((num_kv, HEAD_DIM), device=device)).half()
    indices = torch.randint(
        0, num_kv, (num_tokens, width), dtype=torch.int32, device=device
    )
    lengths = torch.randint(
        1, width + 1, (num_tokens,), dtype=torch.int32, device=device
    )
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    outputs = []
    for tensor_cores in (False, True):
        buffers = [
            torch.empty(shape, dtype=dtype, device=device)
            for shape, dtype in sparse_prefill_bmm_workspace_specs(
                num_tokens,
                num_heads,
                HEAD_DIM,
                width,
                torch.float16,
                tensor_cores=tensor_cores,
            )
        ]
        output = torch.empty_like(q)
        _prefill(tensor_cores)(
            q, kv, indices, lengths, HEAD_DIM**-0.5, attn_sink, output, *buffers
        )
        outputs.append(output)
    torch.testing.assert_close(outputs[1], outputs[0], atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("tensor_cores", ROUTES)
@torch.inference_mode()
def test_prefill_retains_fp32_scores_before_softmax(tensor_cores: bool):
    # Finite FP16 inputs can produce scaled QK scores beyond FP16 range. The
    # existing attention softmax consumes FP32 scores and returns finite values.
    q = torch.full((1, 8, HEAD_DIM), 64.0, device="cuda", dtype=torch.float16)
    kv = torch.full((2, HEAD_DIM), 64.0, device="cuda", dtype=torch.float16)
    indices = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([2], device="cuda", dtype=torch.int32)
    sink = torch.zeros(8, device="cuda", dtype=torch.float32)
    buffers = [
        torch.empty(shape, device="cuda", dtype=dtype)
        for shape, dtype in sparse_prefill_bmm_workspace_specs(
            1, 8, HEAD_DIM, 2, q.dtype, tensor_cores=tensor_cores
        )
    ]
    output = torch.empty_like(q)
    _prefill(tensor_cores)(
        q, kv, indices, lengths, HEAD_DIM**-0.5, sink, output, *buffers
    )
    assert buffers[-2].dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.full_like(output, 64.0))
