# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm import _custom_ops  # noqa: F401
from vllm.model_executor.kernels.linear import (
    Sm70GgufAffineConfig,
    TurboMindGgufAffineKernel,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.gguf_transcode import transcode_affine
from vllm.scalar_type import ScalarType
from vllm.transformers_utils.gguf_tensor_reader import quant_size

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def source(weight_type, n=160, k=2560, seed_offset=0):
    block, size = quant_size(weight_type)
    raw = np.random.default_rng(20261003 + weight_type + seed_offset).integers(
        0, 256, (n * k // block, size), dtype=np.uint8
    )
    d = (np.arange(raw.shape[0]) % 7 + 1).astype(np.float16) / np.float16(2048)
    d = d.reshape(-1, 1)
    offset = size - 2 if weight_type in (11, 14) else 0
    raw[:, offset : offset + 2] = d.view(np.uint8)
    if weight_type in (7, 13):
        raw[:, 2:4] = (d * np.float16(-0.5)).view(np.uint8)
    return raw.reshape(n, -1)


def prepare(canonical):
    weight, stats, meta = torch.ops._C.gguf_affine_sm70_prepare(
        torch.from_numpy(canonical.codes).cuda(),
        torch.from_numpy(canonical.scales).cuda(),
        torch.from_numpy(canonical.mins).cuda(),
        canonical.bits,
        canonical.group_size,
    )
    assert stats.dtype == torch.int64
    return weight, stats, *meta.tolist()


@pytest.mark.parametrize("weight_type", [11, 13, 14])
def test_bitplane_framework_storage_and_fullgraph_tracing(weight_type):
    canonical = transcode_affine(source(weight_type), weight_type)
    n, k = canonical.codes.shape
    config = Sm70GgufAffineConfig(
        (k, n),
        (k, n),
        ScalarType.uint(canonical.bits, None),
        torch.float16,
        canonical.group_size,
        True,
        False,
        source_type=weight_type,
    )
    selected = choose_mp_linear_kernel(config, compute_capability=70)
    assert selected is TurboMindGgufAffineKernel
    kernel = selected(config, "codes", "scales", "mins")
    layer = torch.nn.Module()
    for name in ("codes", "scales", "mins"):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.from_numpy(getattr(canonical, name)).cuda(), requires_grad=False
            ),
        )
    kernel.process_weights_after_loading(layer)
    assert layer.scales.dtype == torch.int64 and layer.mins is None
    torch.manual_seed(20261003 + weight_type)
    x = (torch.randn((8, k), device="cuda") * 0.125).half()
    expected = (
        x.float() @ torch.from_numpy(canonical.dequantize()).half().cuda().float().T
    )
    torch.testing.assert_close(
        kernel.apply_weights(layer, x).float(), expected, rtol=0.003, atol=0.0002
    )
    compiled = torch.compile(
        lambda x: kernel.apply_weights(layer, x), backend="eager", fullgraph=True
    )
    torch.testing.assert_close(
        compiled(x), kernel.apply_weights(layer, x), rtol=0, atol=0
    )


@pytest.mark.parametrize("weight_type", [6, 7, 11, 13, 14])
def test_bitplane_decode_batch_prefill_and_graph(weight_type):
    canonical = transcode_affine(source(weight_type), weight_type)
    weight, stats, kl, sl = prepare(canonical)
    decoded = torch.from_numpy(canonical.dequantize()).half().cuda()
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192):
        torch.manual_seed(20261003 + m)
        x = (torch.randn((m, decoded.shape[1]), device="cuda") * 0.125).half()
        out = torch.empty((m, decoded.shape[0]), dtype=torch.float16, device="cuda")

        def run(out=out, x=x):
            torch.ops._C.gguf_affine_gemm_sm70_out(
                out, x, weight, stats, canonical.bits, kl, sl, canonical.group_size
            )

        run()
        expected = x.float() @ decoded.float().T
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
        assert (out.float() - expected).norm() <= expected.norm() * 0.003 + 1e-6
        if m in (1, 8, 512):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
            torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)


@pytest.mark.parametrize("weight_type", [11, 13, 14])
def test_bitplane_grouped_with_empty_experts_and_graph(weight_type):
    e, n, k, m = 4, 160, 2560, 64
    canonical = [
        transcode_affine(source(weight_type, seed_offset=i), weight_type)
        for i in range(e)
    ]
    prepared = [prepare(p) for p in canonical]
    weights = torch.stack([p[0] for p in prepared])
    stats = torch.stack([p[1] for p in prepared])
    kl, sl = prepared[0][2:]
    wp, sp = torch.ops._C.awq_moe_build_strided_ptrs(weights, stats, kl, sl, e)
    boundaries = [0, 17, 17, 64, 64]
    offsets = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    torch.manual_seed(20261003 + weight_type)
    x = (torch.randn((m, k), device="cuda") * 0.125).half()
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    expected = torch.empty((m, n), device="cuda")
    for expert, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        decoded = torch.from_numpy(canonical[expert].dequantize()).half().cuda()
        expected[start:end] = x[start:end].float() @ decoded.float().T

    def run():
        torch.ops._C.gguf_affine_grouped_gemm_sm70_out(
            out, x, offsets, wp, sp, canonical[0].bits, e, canonical[0].group_size
        )

    run()
    torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
