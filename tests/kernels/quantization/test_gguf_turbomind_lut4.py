# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from test_gguf_lut_transcode import source

from vllm import _custom_ops  # noqa: F401
from vllm.model_executor.kernels.linear import (
    Sm70GgufLut4Config,
    TurboMindGgufLut4Kernel,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.gguf_lut_transcode import transcode_lut4
from vllm.scalar_type import scalar_types

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def projection(weight_type, expert=0):
    return transcode_lut4(
        source(
            weight_type,
            n=160,
            k=2560,
            scale=0.00390625,
            seed=20261003 + expert,
            nv_max=0x40,
        ),
        weight_type,
    )


def prepare(p):
    w, s, meta = torch.ops._C.gguf_lut4_sm70_prepare(
        torch.from_numpy(p.codes).cuda(),
        torch.from_numpy(p.scales).cuda(),
        p.lut_id,
        p.group_size,
    )
    assert s.dtype == torch.int16
    return w, s, *meta.tolist()


@pytest.mark.parametrize("weight_type", [20, 23, 39, 40])
def test_lut4_decode_batch_prefill_and_graph(weight_type):
    p = projection(weight_type)
    w, s, kl, sl = prepare(p)
    dense = torch.from_numpy(p.dequantize()).half().cuda()
    n, k = p.codes.shape
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192):
        torch.manual_seed(20261003 + m)
        x = (torch.randn((m, k), device="cuda") * 0.125).half()
        out = torch.empty((m, n), dtype=torch.float16, device="cuda")

        def run(out=out, x=x):
            torch.ops._C.gguf_lut4_gemm_sm70_out(
                out, x, w, s, p.lut_id, kl, sl, p.group_size
            )

        run()
        expected = x.float() @ dense.float().T
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.003)
        assert (out.float() - expected).norm() <= expected.norm() * 0.003 + 1e-6
        if m in (1, 8, 512):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
            torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.003)


@pytest.mark.parametrize("weight_type", [20, 23, 39, 40])
def test_lut4_grouped_distinct_empty_experts_and_graph(weight_type):
    e, n, k = 4, 160, 2560
    canonical = [projection(weight_type, expert=i) for i in range(e)]
    prepared = [prepare(p) for p in canonical]
    w = torch.stack([p[0] for p in prepared])
    s = torch.stack([p[1] for p in prepared])
    kl, sl = prepared[0][2:]
    wp, sp = torch.ops._C.awq_moe_build_strided_ptrs(w, s, kl, sl, e)
    for m in (1, 8, 64, 512):
        boundaries = [0, m // 4, m // 4, m, m]
        offsets = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
        torch.manual_seed(20261003 + m)
        x = (torch.randn((m, k), device="cuda") * 0.125).half()
        out = torch.empty((m, n), dtype=torch.float16, device="cuda")
        expected = torch.empty((m, n), device="cuda")
        for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            dense = torch.from_numpy(canonical[i].dequantize()).half().cuda()
            expected[start:end] = x[start:end].float() @ dense.float().T

        def run(out=out, x=x, offsets=offsets):
            torch.ops._C.gguf_lut4_grouped_gemm_sm70_out(
                out,
                x,
                offsets,
                wp,
                sp,
                canonical[0].lut_id,
                e,
                canonical[0].group_size,
            )

        run()
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.003)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.003)


@pytest.mark.parametrize("weight_type", [20, 23, 39, 40])
def test_lut4_framework_and_fullgraph_tracing(weight_type):
    p = projection(weight_type)
    n, k = p.codes.shape
    config = Sm70GgufLut4Config(
        (k, n),
        (k, n),
        scalar_types.uint4,
        torch.float16,
        p.group_size,
        False,
        False,
        source_type=weight_type,
    )
    selected = choose_mp_linear_kernel(config, compute_capability=70)
    assert selected is TurboMindGgufLut4Kernel
    kernel = selected(config, "codes", "scales")
    layer = torch.nn.Module()
    for name in ("codes", "scales"):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.from_numpy(getattr(p, name)).cuda(), requires_grad=False
            ),
        )
    kernel.process_weights_after_loading(layer)
    assert layer.scales.dtype == torch.int16
    x = (torch.randn((8, k), device="cuda") * 0.125).half()
    expected = x.float() @ torch.from_numpy(p.dequantize()).half().cuda().float().T
    torch.testing.assert_close(
        kernel.apply_weights(layer, x).float(), expected, rtol=0.003, atol=0.003
    )
    compiled = torch.compile(
        lambda x: kernel.apply_weights(layer, x), backend="eager", fullgraph=True
    )
    torch.testing.assert_close(
        compiled(x), kernel.apply_weights(layer, x), rtol=0, atol=0
    )
