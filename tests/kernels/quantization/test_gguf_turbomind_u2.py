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
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.gguf_tensor_reader import quant_size

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def source(weight_type, n=160, k=2560):
    block, size = quant_size(weight_type)
    rng = np.random.default_rng(20261003 + weight_type)
    data = rng.integers(0, 256, (n * k // block, size), dtype=np.uint8)
    d = (rng.integers(1, 8, (data.shape[0], 1)) * 0.0009765625).astype("<f2")
    offset = 80 if weight_type == 10 else (0 if weight_type in (41, 42) else size - 2)
    data[:, offset : offset + 2] = d.view(np.uint8)
    if weight_type == 10:
        data[:, 82:84] = (d * 0.5).astype("<f2").view(np.uint8)
    return data.reshape(n, -1)


def prepare(canonical):
    weight, stats, meta = torch.ops._C.gguf_affine_sm70_prepare(
        torch.from_numpy(canonical.codes).cuda(),
        torch.from_numpy(canonical.scales).cuda(),
        torch.from_numpy(canonical.mins).cuda(),
        2,
        canonical.group_size,
    )
    return weight, stats, *meta.tolist()


@pytest.mark.parametrize("weight_type", [10, 42])
def test_u2_framework_prepares_and_traces_canonical_groups(weight_type):
    canonical = transcode_affine(source(weight_type), weight_type)
    n, k = canonical.codes.shape
    config = Sm70GgufAffineConfig(
        (k, n),
        (k, n),
        scalar_types.uint2,
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
    assert layer.codes.dtype == torch.int32 and layer.mins is None
    torch.manual_seed(20261003)
    x = (torch.randn((8, k), device="cuda") * 0.125).half()
    decoded = torch.from_numpy(canonical.dequantize()).half().cuda()
    expected = x.float() @ decoded.float().T
    torch.testing.assert_close(
        kernel.apply_weights(layer, x).float(), expected, rtol=0.003, atol=0.0002
    )
    compiled = torch.compile(
        lambda x: kernel.apply_weights(layer, x), backend="eager", fullgraph=True
    )
    torch.testing.assert_close(
        compiled(x), kernel.apply_weights(layer, x), rtol=0, atol=0
    )
    config.group_size = 128
    admitted, reason = selected.can_implement(config)
    assert (
        not admitted and reason == "canonical_group_or_activation_order_not_supported"
    )


@pytest.mark.parametrize("weight_type", [10, 34, 35, 41, 42])
def test_u2_decode_batch_prefill_and_graph(weight_type):
    canonical = transcode_affine(source(weight_type), weight_type)
    w, s, kl, sl = prepare(canonical)
    decoded = torch.from_numpy(canonical.dequantize()).half().cuda()
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192):
        torch.manual_seed(20261003 + m)
        x = (torch.randn((m, decoded.shape[1]), device="cuda") * 0.125).half()
        out = torch.empty((m, decoded.shape[0]), device="cuda", dtype=torch.float16)

        def run(out=out, x=x):
            torch.ops._C.gguf_affine_gemm_sm70_out(
                out, x, w, s, 2, kl, sl, canonical.group_size
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


def test_q2_0_tp4_k160_reconstructs_full_projection():
    full = transcode_affine(source(42, n=2560, k=640), 42)
    torch.manual_seed(20261003)
    x = (torch.randn((8, 640), device="cuda") * 0.125).half()
    combined = torch.zeros((8, 2560), device="cuda")
    for rank in range(4):
        shard = full.tp_slice(rank, 4, axis=1)
        w, s, kl, sl = prepare(shard)
        part = torch.empty((8, 2560), device="cuda", dtype=torch.float16)
        torch.ops._C.gguf_affine_gemm_sm70_out(
            part, x[:, rank * 160 : (rank + 1) * 160].contiguous(), w, s, 2, kl, sl, 32
        )
        combined += part.float()
    decoded = torch.from_numpy(full.dequantize()).half().cuda()
    expected = x.float() @ decoded.float().T
    torch.testing.assert_close(combined, expected, rtol=0.003, atol=0.0002)


@pytest.mark.parametrize("group", [16, 32])
def test_u2_grouped_with_empty_experts_and_graph(group):
    e, n, k, m = 4, 160, 640, 64
    torch.manual_seed(20261003 + group)
    codes = torch.randint(0, 4, (e, n, k), device="cuda", dtype=torch.uint8)
    scale = torch.rand((e, n, k // group), device="cuda").half() * 0.0078125
    mins = torch.rand_like(scale) * -0.00390625
    prepared = [
        torch.ops._C.gguf_affine_sm70_prepare(codes[i], scale[i], mins[i], 2, group)
        for i in range(e)
    ]
    weights = torch.stack([p[0] for p in prepared])
    stats = torch.stack([p[1] for p in prepared])
    kl, sl = prepared[0][2].tolist()
    wp, sp = torch.ops._C.awq_moe_build_strided_ptrs(weights, stats, kl, sl, e)
    boundaries = [0, 17, 17, 64, 64]
    offsets = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    x = (torch.randn((m, k), device="cuda") * 0.125).half()
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    decoded = (
        (
            codes.float().reshape(e, n, -1, group) * scale.float()[..., None]
            + mins.float()[..., None]
        )
        .reshape(e, n, k)
        .half()
    )
    expected = torch.empty((m, n), device="cuda")
    for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        expected[start:end] = x[start:end].float() @ decoded[i].float().T

    def run():
        torch.ops._C.gguf_affine_grouped_gemm_sm70_out(
            out, x, offsets, wp, sp, 2, e, group
        )

    run()
    torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
