# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.model_executor.layers.quantization.gguf_dp4a_formats import (
    transcode_integer_dot,
)
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def oracle(x):
    groups = x.float().reshape(x.shape[0], -1, 32)
    maximum = groups.abs().amax(-1, keepdim=True)
    d = (maximum.double() / 127).float()
    ratio = torch.where(maximum == 0, 0, (groups.double() / d.double()).float())
    q = (ratio.sign() * (ratio.abs() + 0.5).floor()).to(torch.int8)
    return (q.float() * d.half().float()).reshape_as(x), groups.sum(-1).half().float()


@pytest.mark.parametrize(
    "kind,activated", [(12, False), (14, False), (23, False), (23, True)]
)
@pytest.mark.parametrize("m", [1, 5, 20])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("cooperative", [False, True])
def test_dense_matches_official_q8_formula_and_output_views(
    kind, activated, m, split, cooperative
):
    n, k = 64, 768
    _, size = quant_size(kind)
    rng = np.random.default_rng(kind)
    blocks = rng.integers(0, 256, (n, k // 256, size), dtype=np.uint8)
    # Keep the fused FP16 multiplication finite; arbitrary full-range IQ4
    # scale bytes otherwise overflow both the official oracle and the kernel.
    upper = 0.001 if activated else 0.01
    d = rng.uniform(upper / 10, upper, blocks.shape[:2]).astype("<f2")
    start = 208 if kind == 14 else 0
    blocks[:, :, start : start + 2] = d[..., None].view(np.uint8)
    if kind == 12:
        blocks[:, :, 2:4] = (d * np.float16(0.5))[..., None].view(np.uint8)
    raw = blocks.reshape(n, -1)
    codec = transcode_integer_dot(raw, kind)
    packed = [torch.from_numpy(t).cuda() for t in codec.packed()]
    reference = torch.from_numpy(dequantize(raw, kind)).cuda()
    torch.manual_seed(970 + m)
    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    q8 = torch.empty((m, k // 32, 36), device="cuda", dtype=torch.uint8)
    width = n // 2 if activated else n
    parent = torch.full((m, width + 16), -3.0, device="cuda", dtype=torch.float16)
    out = parent[:, 8 : width + 8]  # Direct mixed-projection output, no copyback.
    scratch = torch.empty((split, m, n), device="cuda", dtype=torch.float32)

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        torch.ops._C.gguf_dp4a_dense_sm70_out(
            out, scratch, q8, *packed, kind, split, cooperative, activated
        )

    def check():
        decoded, original_sum = oracle(x)
        expected = decoded @ reference.T
        if kind == 12:
            minimum = codec.dmin.astype(np.float32).repeat(8, axis=1)
            minimum *= codec.small_mins.astype(np.float32)
            delta = original_sum - decoded.reshape(m, -1, 32).sum(-1)
            # Standard Q8_1 stores the original (rounded) sum for affine offsets.
            expected -= delta @ torch.from_numpy(minimum).cuda().T
        if activated:
            rounded = expected.half()
            expected = (
                torch.nn.functional.silu(rounded[:, :width]) * rounded[:, width:]
            ).float()
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.003)
        assert torch.isfinite(out).all()
        assert torch.all(parent[:, :8] == -3) and torch.all(parent[:, -8:] == -3)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x))
    graph.replay()
    check()
