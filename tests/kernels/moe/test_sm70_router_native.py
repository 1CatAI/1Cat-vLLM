# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native router semantics and changed-input graph replay."""

import pytest
import torch

native = pytest.importorskip("vllm._sm70_router_C")

from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (  # noqa: E402
    _sm70_qwen38_router_topk_kernel,
    vllm_topk_softmax,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 required",
)


def buffers(m):
    return [
        torch.empty((m, 10), device="cuda", dtype=t)
        for t in (torch.float32, torch.int32, torch.int32)
    ]


def control(logits, output):
    m = logits.size(0)
    if m > 16:
        vllm_topk_softmax(*output, logits, renormalize=True)
        return
    _sm70_qwen38_router_topk_kernel[(m,)](
        logits,
        *output,
        E=512,
        K=10,
        M=m,
        BLOCK_E=512,
        PACKED_HALF_KEY=m <= 16,
        SELECT_TOP16=m in (5, 10),
        num_warps=8,
    )


def oracle(logits, output):
    """Stable logit order and FP64 softmax, independent of the router kernel."""
    m = logits.size(0)
    invalid = (
        logits.isnan().any(1) | logits.isposinf().any(1) | logits.isneginf().all(1)
    )
    fallback = -torch.arange(512, device=logits.device, dtype=torch.float64)
    safe = torch.where(invalid[:, None], fallback, logits.double())
    ids = torch.argsort(safe, dim=1, descending=True, stable=True)[:, :10]
    weights = safe.gather(1, ids).softmax(1)
    weights[invalid] = 0
    source = torch.arange(10, device=logits.device)[None, :] * m
    source = source + torch.arange(m, device=logits.device)[:, None]
    for target, value in zip(output, (weights, ids, source)):
        target.copy_(value)


@pytest.mark.parametrize("m", [1, 5, 20])
@pytest.mark.parametrize("quantize", [False, True])
def test_native_router_changed_graph_inputs(m, quantize):
    logits = torch.zeros(m, 512, device="cuda", dtype=torch.float16)
    x = torch.zeros(m, 2560, device="cuda", dtype=torch.float16)
    out, ref, dispatched = buffers(m), buffers(m), buffers(m)
    q8 = torch.empty(m, 80, 36, device="cuda", dtype=torch.uint8)
    ref_q8 = torch.empty_like(q8)

    def candidate():
        if quantize:
            torch.ops.vllm_sm70_router.select_quantize(*out, q8, logits, x)
        else:
            torch.ops.vllm_sm70_router.top10(*out, logits)

    candidate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    for case in range(15):
        logits.normal_(0, (0.001, 0.1, 1.0, 30.0)[case % 4])
        x.normal_()
        if case == 0:
            logits.zero_()
            logits[:, ::2] = -0.0
            x.zero_()
        elif case == 1:
            logits.copy_((torch.arange(512, device="cuda") % 7).half())
        elif case == 2:
            logits[:, 0] = float("nan")
        elif case == 3:
            logits.fill_(-float("inf"))
        elif case == 4:
            logits[:, -1] = float("inf")
        elif case == 5:
            logits[:, ::2] = -float("inf")
        elif case == 6:
            logits.normal_(0, 5e-8)
            x.normal_(0, 5e-8)
        elif case == 7:
            logits.fill_(65504)
        elif case == 8:
            logits.fill_(-65504)
        oracle(logits, ref)
        # Generic M20 CUDA ranks rounded full-softmax scores. Subnormal
        # logits can collapse into ties there; retain exact raw-logit
        # ordering in the mathematical oracle instead of reproducing this.
        if not (m == 20 and case == 6):
            control(logits, dispatched)
            assert torch.equal(dispatched[1], ref[1]), (m, case)
            assert torch.equal(dispatched[2], ref[2]), (m, case)
            torch.testing.assert_close(dispatched[0], ref[0], atol=3e-7, rtol=0)
        if quantize:
            torch.ops._C.gguf_quantize_q8_1_sm70_out(ref_q8, x)
        for tensor in out:
            tensor.fill_(-777)
        graph.replay()
        assert torch.equal(out[1], ref[1]), (m, quantize, case)
        assert torch.equal(out[2], ref[2]), (m, quantize, case)
        torch.testing.assert_close(out[0], ref[0], atol=3e-7, rtol=0)
        if quantize:
            assert torch.equal(q8, ref_q8)


@pytest.mark.parametrize("failure", ["dtype", "shape", "rows", "stride", "output"])
def test_native_router_rejects_unsupported_storage(failure):
    logits = torch.zeros(5, 512, device="cuda", dtype=torch.float16)
    out = buffers(5)
    if failure == "dtype":
        logits = logits.float()
    elif failure == "shape":
        logits = logits[:, :256].contiguous()
    elif failure == "rows":
        logits = torch.zeros(21, 512, device="cuda", dtype=torch.float16)
    elif failure == "stride":
        logits = torch.zeros(5, 1024, device="cuda", dtype=torch.float16)[:, ::2]
    else:
        out[1] = out[1].long()
    with pytest.raises(RuntimeError):
        torch.ops.vllm_sm70_router.top10(*out, logits)
