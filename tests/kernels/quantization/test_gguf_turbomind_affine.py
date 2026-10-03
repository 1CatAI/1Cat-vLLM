# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm import _custom_ops  # noqa: F401 -- registers packaged _C operators
from vllm.model_executor.kernels.linear import (
    Sm70GgufAffineConfig,
    TurboMindGgufAffineKernel,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.gguf_transcode import (
    transcode_affine_group32,
)
from vllm.scalar_type import scalar_types
from vllm.sm70_profiles.acceleration import loaded_linear_kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_framework_selects_canonical_storage_and_reports_capability():
    config = Sm70GgufAffineConfig(
        (256, 128),
        (256, 128),
        scalar_types.uint4,
        torch.float16,
        32,
        True,
        False,
        source_type=3,
    )
    selected = choose_mp_linear_kernel(config, compute_capability=70)
    assert selected is TurboMindGgufAffineKernel
    kernel = selected(config, "codes", "scales", "mins")
    layer = torch.nn.Module()
    for name, tensor in {
        "codes": torch.randint(0, 16, (128, 256), dtype=torch.uint8, device="cuda"),
        "scales": torch.zeros((128, 8), dtype=torch.float16, device="cuda"),
        "mins": torch.full((128, 8), 0.125, dtype=torch.float16, device="cuda"),
    }.items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    kernel.process_weights_after_loading(layer)
    assert layer.codes.dtype == torch.int32 and layer.mins is None
    x = torch.ones((8, 256), dtype=torch.float16, device="cuda")
    torch.testing.assert_close(
        kernel.apply_weights(layer, x),
        torch.full((8, 128), 32.0, device="cuda", dtype=torch.float16),
        rtol=0,
        atol=0,
    )
    compiled = torch.compile(
        lambda x: kernel.apply_weights(layer, x), backend="eager", fullgraph=True
    )
    torch.testing.assert_close(
        compiled(x), kernel.apply_weights(layer, x), rtol=0, atol=0
    )
    layer.quant_method = SimpleNamespace(kernel=kernel)
    admission = next(iter(loaded_linear_kernels(layer).values()))["operator_admission"]
    assert admission["source_type"] == "Q4_1"
    assert admission["min_m"] == 1 and admission["max_m"] is None
    config.partition_weight_shape = (256, 48)
    with pytest.raises(ValueError, match="cuts_canonical_group_or_output_pack"):
        choose_mp_linear_kernel(config, compute_capability=70)
    config.partition_weight_shape = (256, 128)
    config.enabled = False
    with pytest.raises(ValueError, match="disabled_by_kernel_config"):
        choose_mp_linear_kernel(config, compute_capability=70)


@pytest.mark.parametrize("weight_type", [2, 3, 8, 12])
@pytest.mark.parametrize("shape", [(128, 256), (160, 2560), (2560, 160)])
def test_turbomind_affine_decode_batch_prefill_and_graph(weight_type, shape):
    n, k = shape
    block, size = gguf.GGML_QUANT_SIZES[weight_type]
    if k % block:
        pytest.skip(
            "Source GGUF superblock does not fit; reblocking is covered separately"
        )
    random = np.random.default_rng(20261003 + weight_type)
    blocks = random.integers(0, 256, (n * k // block, size), dtype=np.uint8)
    blocks[:, :2] = np.frombuffer(np.float16(0.001337).tobytes(), dtype=np.uint8)
    if weight_type in (3, 12):
        blocks[:, 2:4] = np.frombuffer(np.float16(0.000739).tobytes(), dtype=np.uint8)
    source = blocks.reshape(n, -1)
    canonical = transcode_affine_group32(source, weight_type)
    codes = torch.from_numpy(canonical.codes).cuda()
    scale = torch.from_numpy(canonical.scales).cuda()
    mins = torch.from_numpy(canonical.mins).cuda()
    weight, stats, meta = torch.ops._C.gguf_affine_sm70_prepare(
        codes, scale, mins, canonical.bits
    )
    k_ld, q_ld = meta.tolist()
    decoded = torch.from_numpy(canonical.dequantize()).half().cuda()
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 512):
        torch.manual_seed(20261003 + m)
        x = (torch.randn((m, k), device="cuda") * 0.125).half()
        out = torch.empty((m, n), dtype=torch.float16, device="cuda")
        torch.ops._C.gguf_affine_gemm_sm70_out(
            out, x, weight, stats, canonical.bits, k_ld, q_ld
        )
        expected = x.float() @ decoded.float().T
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
        assert (out.float() - expected).norm() <= expected.norm() * 0.003 + 1e-6
        if m in (1, 8, 512):
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                torch.ops._C.gguf_affine_gemm_sm70_out(
                    out, x, weight, stats, canonical.bits, k_ld, q_ld
                )
            capture.replay()
            torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)


def test_zero_scale_nonzero_min_is_a_constant_block():
    codes = torch.randint(0, 16, (128, 256), dtype=torch.uint8, device="cuda")
    scale = torch.zeros((128, 8), dtype=torch.float16, device="cuda")
    mins = torch.full_like(scale, 0.125)
    weight, stats, meta = torch.ops._C.gguf_affine_sm70_prepare(codes, scale, mins, 4)
    k_ld, q_ld = meta.tolist()
    x = torch.ones((8, 256), dtype=torch.float16, device="cuda")
    out = torch.empty((8, 128), dtype=torch.float16, device="cuda")
    torch.ops._C.gguf_affine_gemm_sm70_out(out, x, weight, stats, 4, k_ld, q_ld)
    torch.testing.assert_close(out, torch.full_like(out, 32.0), rtol=0, atol=0)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("m", [1, 8, 64, 512])
def test_grouped_affine_with_empty_experts_and_graph(bits, m):
    experts, n, k = 4, 160, 640
    torch.manual_seed(20261003 + bits + m)
    codes = torch.randint(
        0, 1 << bits, (experts, n, k), dtype=torch.uint8, device="cuda"
    )
    scales = torch.full(
        (experts, n, k // 32), 0.00390625, dtype=torch.float16, device="cuda"
    )
    mins = torch.full_like(scales, -0.125)
    prepared = [
        torch.ops._C.gguf_affine_sm70_prepare(codes[e], scales[e], mins[e], bits)
        for e in range(experts)
    ]
    weights = torch.stack([p[0] for p in prepared])
    stats = torch.stack([p[1] for p in prepared])
    k_ld, q_ld = prepared[0][2].tolist()
    wp, sp = torch.ops._C.awq_moe_build_strided_ptrs(
        weights, stats, k_ld, q_ld, experts
    )
    boundaries = [0, m // 2, m // 2, m, m]
    offsets = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    x = (torch.randn((m, k), device="cuda") * 0.125).half()
    out = torch.empty((m, n), dtype=torch.float16, device="cuda")
    decoded = (
        codes.float().view(experts, n, -1, 32) * scales.float()[..., None]
        + mins.float()[..., None]
    )
    decoded = decoded.view(experts, n, k).half()
    expected = torch.empty((m, n), device="cuda")
    for e, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        expected[start:end] = x[start:end].float() @ decoded[e].float().T

    def run():
        torch.ops._C.gguf_affine_grouped_gemm_sm70_out(
            out, x, offsets, wp, sp, bits, experts
        )

    run()
    torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
    assert (out.float() - expected).norm() <= expected.norm() * 0.003 + 1e-6
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
