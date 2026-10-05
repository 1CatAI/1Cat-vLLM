# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.model_executor.layers.quantization.gguf_dp4a_formats import (
    pack_mixed_u4_pair,
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
    "kind,activated,raw_source,row_major",
    [
        (kind, activated, raw_source, False)
        for kind, activated, raw_source in [
            (12, False, False),
            (14, False, False),
            (23, False, False),
            (12, True, False),
            (14, True, False),
            (23, True, False),
            (14, False, True),
            (14, True, True),
        ]
    ]
    + [
        (kind, activated, False, True)
        for kind in (12, 14, 23)
        for activated in (False, True)
    ],
)
@pytest.mark.parametrize("m", [1, 5, 20])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("cooperative", [False, True])
def test_dense_matches_official_q8_formula_and_output_views(
    kind, activated, raw_source, row_major, m, split, cooperative
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
    if row_major:
        packed = [torch.from_numpy(t).cuda() for t in codec.row_storage()]
    if raw_source:
        packed = [
            torch.from_numpy(raw).cuda(),
            torch.empty(0, device="cuda", dtype=torch.float16),
            torch.empty(0, device="cuda", dtype=torch.int8),
            torch.empty(0, device="cuda", dtype=torch.float16),
            torch.empty(0, device="cuda", dtype=torch.int8),
        ]
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

    if (raw_source or row_major) and cooperative:
        with pytest.raises(
            RuntimeError, match="Raw integer dense cooperative reduction unavailable"
        ):
            run()
        return
    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x))
    graph.replay()
    check()


@pytest.mark.parametrize("kinds", [(12, 23), (23, 12)])
@pytest.mark.parametrize("m", [5, 20])
def test_mixed_u4_pair_keeps_official_affine_correction(kinds, m):
    n, k, split = 160, 2560, 16
    codecs, weights = [], []
    for kind in kinds:
        _, size = quant_size(kind)
        rng = np.random.default_rng(kind)
        raw = rng.integers(0, 256, (n, k // 256, size), dtype=np.uint8)
        d = rng.uniform(0.0001, 0.001, raw.shape[:2]).astype("<f2")
        raw[:, :, :2] = d[..., None].view(np.uint8)
        if kind == 12:
            raw[:, :, 2:4] = (d * np.float16(0.5))[..., None].view(np.uint8)
        raw = raw.reshape(n, -1)
        codecs.append(transcode_integer_dot(raw, kind))
        weights.append(torch.from_numpy(dequantize(raw, kind)).cuda())
    packed = [torch.from_numpy(a).cuda() for a in pack_mixed_u4_pair(*codecs)]
    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    q8 = torch.empty((m, k // 32, 36), device="cuda", dtype=torch.uint8)
    partial = torch.empty((split, m, 2 * n), device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        torch.ops._C.gguf_dp4a_dense_sm70_out(
            out, partial, q8, *packed, kinds[0], split, False, True, kinds[1]
        )

    def check():
        decoded, original_sum = oracle(x)
        values = []
        for kind, codec, weight in zip(kinds, codecs, weights):
            value = decoded @ weight.T
            if kind == 12:
                mins = codec.dmin.astype(np.float32).repeat(8, axis=1)
                mins *= codec.small_mins.astype(np.float32)
                delta = original_sum - decoded.reshape(m, -1, 32).sum(-1)
                value -= delta @ torch.from_numpy(mins).cuda().T
            values.append(value.half())
        expected = torch.nn.functional.silu(values[0]) * values[1]
        torch.testing.assert_close(out, expected, rtol=0.003, atol=0.003)
        assert torch.isfinite(out).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x))
    graph.replay()
    check()
