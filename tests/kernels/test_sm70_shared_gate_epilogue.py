# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _sm70_ops as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an idle CUDA GPU"
)


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 8, 15, 16])
def test_batch_gate_replays_changed_inputs(rows):
    if not ops.has_qwen38_shared_gate_sigmoid_mul():
        pytest.skip("build the new native _C operator first")
    generator = torch.Generator(device="cuda").manual_seed(17)
    logits = torch.empty(rows, 1, dtype=torch.float16, device="cuda")
    # Aligned output view with guards on both sides, shared across replays.
    storage = torch.full((rows * 2560 + 16,), -37, dtype=torch.float16, device="cuda")
    out = storage[8:-8].view(rows, 2560)
    source = torch.empty_like(out)
    logits.zero_()
    ops.qwen38_shared_gate_sigmoid_mul_out(out, logits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(source)
        ops.qwen38_shared_gate_sigmoid_mul_out(out, logits)
    for scale in (0.01, 1, 8, 100):
        logits.normal_(generator=generator).mul_(scale)
        source.normal_(generator=generator).mul_(scale)
        out.fill_(float("nan"))
        graph.replay()
        expected = source * torch.sigmoid(logits)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
        assert (storage[:8] == -37).all()
        assert (storage[-8:] == -37).all()


@pytest.mark.parametrize("rows", [0, 1, 17])
def test_batch_gate_rejects_unsupported_rows(rows):
    if not ops.has_qwen38_shared_gate_sigmoid_mul():
        pytest.skip("build the new native _C operator first")
    out = torch.empty(rows, 2560, device="cuda", dtype=torch.float16)
    logits = torch.empty(rows, 1, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="expected M2-16"):
        ops.qwen38_shared_gate_sigmoid_mul_out(out, logits)


def test_batch_gate_rejects_unaligned_output():
    if not ops.has_qwen38_shared_gate_sigmoid_mul():
        pytest.skip("build the new native _C operator first")
    storage = torch.empty(5121, dtype=torch.float16, device="cuda")
    logits = torch.empty(2, 1, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="half2 aligned"):
        ops.qwen38_shared_gate_sigmoid_mul_out(storage[1:].view(2, 2560), logits)
