# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 Marlin must preserve GLM's SwiGLU clamp on pre-Blackwell GPUs."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.oracle import nvfp4


@pytest.mark.parametrize("requested", ["auto", "marlin"])
def test_clamped_nvfp4_can_select_marlin(monkeypatch, requested):
    class Supported:
        @staticmethod
        def is_supported_config(*args):
            return True, None

    monkeypatch.setattr(
        nvfp4,
        "backend_to_kernel_cls",
        lambda backend: [Supported] if backend == nvfp4.NvFp4MoeBackend.MARLIN else [],
    )
    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP4", raising=False)
    monkeypatch.delenv("VLLM_TEST_FORCE_FP8_MARLIN", raising=False)
    monkeypatch.delenv("VLLM_NVFP4_GEMM_BACKEND", raising=False)
    config = SimpleNamespace(
        swiglu_limit=10.0,
        moe_backend=requested,
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
    )
    with set_current_vllm_config(VllmConfig()):
        backend, _ = nvfp4.select_nvfp4_moe_backend(config, None, None)
    assert backend == nvfp4.NvFp4MoeBackend.MARLIN


@pytest.mark.parametrize("limit", [None, 10.0])
def test_marlin_quant_config_preserves_clamp(limit):
    scale = torch.ones(1)
    config = nvfp4.make_nvfp4_moe_quant_config(
        nvfp4.NvFp4MoeBackend.MARLIN,
        scale,
        scale,
        scale,
        scale,
        scale,
        scale,
        swiglu_limit=limit,
    )
    assert config.use_nvfp4_w4a16
    assert config.gemm1_clamp_limit == limit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 8, 33])
@pytest.mark.parametrize(
    "projection_ratios", [None, [(1, 10), (3, 0.25)], [(0, 0), (1, 2)]]
)
def test_nvfp4_marlin_clamp_matches_reference(rows, projection_ratios):
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        rand_marlin_weight_nvfp4_like,
    )
    from vllm.scalar_type import scalar_types

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("regression targets Ampere and Ada")
    torch.manual_seed(74)
    device = "cuda"
    experts, hidden, intermediate, topk = 2, 256, 256, 2

    def weights(n, k):
        tensors = [
            rand_marlin_weight_nvfp4_like(
                torch.randn(n, k, device=device, dtype=torch.float16) * 0.2,
                16,
            )
            for _ in range(experts)
        ]
        return [torch.stack(items).contiguous() for items in zip(*tensors)]

    ref1, w1, s1, g1 = weights(2 * intermediate, hidden)
    ref2, w2, s2, g2 = weights(hidden, intermediate)
    if projection_ratios is not None:
        ratios = torch.tensor(projection_ratios, device=device, dtype=torch.float32)
        g1 = g1.reshape(experts, 1) * ratios
        ref1[:, :, :intermediate] *= ratios[:, 0, None, None]
        ref1[:, :, intermediate:] *= ratios[:, 1, None, None]
    x = torch.randn(rows, hidden, device=device, dtype=torch.float16) * 4
    ids = torch.arange(topk, device=device, dtype=torch.int32).repeat(rows, 1)
    probs = torch.softmax(torch.randn(rows, topk, device=device), dim=-1)
    config = nvfp4.make_nvfp4_moe_quant_config(
        nvfp4.NvFp4MoeBackend.MARLIN,
        s1,
        s2,
        g1,
        g2,
        g1,
        g2,
        swiglu_limit=10.0,
    )
    with set_current_vllm_config(VllmConfig()):
        actual = fused_marlin_moe(
            x,
            w1,
            w2,
            None,
            None,
            s1,
            s2,
            probs,
            ids,
            scalar_types.float4_e2m1f.id,
            global_scale1=g1,
            global_scale2=g2,
            clamp_limit=config.gemm1_clamp_limit,
        )
    expected = torch.zeros_like(x, dtype=torch.float32)
    unclamped = torch.zeros_like(expected)
    for e in range(experts):
        gate, up = (x.float() @ ref1[e].float()).chunk(2, dim=-1)
        activated = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
        expected += (activated @ ref2[e].float()) * probs[:, e : e + 1]
        unclamped += (F.silu(gate) * up @ ref2[e].float()) * probs[:, e : e + 1]
    assert torch.isfinite(actual).all()
    relative_error = (actual.float() - expected).norm() / expected.norm()
    assert relative_error < 0.015, relative_error.item()
    # Ensure the test would catch a silently discarded clamp parameter.
    assert (unclamped - expected).norm() / expected.norm() > 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("experts,n,k", [(3, 256, 512), (4, 512, 256), (2, 4096, 4096)])
def test_nvfp4_expert_repack_reuses_storage_without_changing_bits(experts, n, k):
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        _repack_nvfp4_experts_in_place,
    )

    weight = torch.randint(
        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
    )
    perm = torch.empty(0, device="cuda", dtype=torch.int32)
    expected = torch.stack(
        [
            ops.gptq_marlin_repack(
                weight[e].view(torch.int32).T.contiguous(), perm, k, n, 4, False
            )
            for e in range(experts)
        ]
    )
    pointer = weight.data_ptr()
    actual = _repack_nvfp4_experts_in_place(weight)
    assert actual.data_ptr() == pointer
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_glm_nvfp4_repack_has_bounded_temporary_memory():
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        _repack_nvfp4_experts_in_place,
    )

    weight = torch.zeros((288, 4096, 2048), device="cuda", dtype=torch.uint8)
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    actual = _repack_nvfp4_experts_in_place(weight)
    torch.accelerator.synchronize()
    temporary_bytes = torch.accelerator.max_memory_allocated() - before
    assert actual.data_ptr() == weight.data_ptr()
    assert temporary_bytes < 64 * 1024**2, temporary_bytes
