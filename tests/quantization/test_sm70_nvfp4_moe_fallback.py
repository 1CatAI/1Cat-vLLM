# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 NVFP4 MoE fallback + W4A4 weight-only routing unit tests.

Drives real selection helpers (oracle reorder / prefer flag) and prepare
hooks — not reimplementations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.nn.parameter import Parameter


def test_sm70_compat_backend_order_puts_marlin_emulation_first():
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        sm70_compat_nvfp4_moe_backend_order,
    )

    original = [
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.VLLM_CUTLASS,
        NvFp4MoeBackend.MARLIN,
        NvFp4MoeBackend.EMULATION,
    ]
    ordered = sm70_compat_nvfp4_moe_backend_order(original)
    assert ordered[0] is NvFp4MoeBackend.MARLIN
    assert ordered[1] is NvFp4MoeBackend.EMULATION
    assert set(ordered) == set(original)
    # Relative order of non-compat backends preserved.
    assert ordered[2:] == [
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.VLLM_CUTLASS,
    ]


def test_prefer_sm70_nvfp4_moe_fallback_env_override(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle import nvfp4 as oracle

    monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_FALLBACK", "1")
    assert oracle.prefer_sm70_nvfp4_moe_fallback() is True

    monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_FALLBACK", "0")
    assert oracle.prefer_sm70_nvfp4_moe_fallback() is False


def test_prefer_sm70_detects_volta_from_platform(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle import nvfp4 as oracle

    monkeypatch.delenv("VLLM_SM70_NVFP4_MOE_FALLBACK", raising=False)

    class _SM70:
        def is_cuda(self):
            return True

        def has_device_capability(self, cap):
            return cap == 70

    class _SM80:
        def is_cuda(self):
            return True

        def has_device_capability(self, cap):
            return cap in (70, 75, 80)

    with patch(
        "vllm.platforms.current_platform",
        _SM70(),
    ):
        # prefer imports current_platform inside the function
        with patch.object(
            oracle,
            "prefer_sm70_nvfp4_moe_fallback",
            wraps=oracle.prefer_sm70_nvfp4_moe_fallback,
        ):
            pass

    # Call with patched platforms module used by the helper
    import vllm.platforms as platforms_mod

    monkeypatch.setattr(platforms_mod, "current_platform", _SM70())
    # Re-import path: helper does `from vllm.platforms import current_platform`
    # so patch the attribute on the platforms package.
    assert oracle.prefer_sm70_nvfp4_moe_fallback() is True

    monkeypatch.setattr(platforms_mod, "current_platform", _SM80())
    assert oracle.prefer_sm70_nvfp4_moe_fallback() is False


def _minimal_moe_config(moe_backend: str = "auto"):
    """Build a lightweight FusedMoEConfig-like object for oracle selection."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
    )

    # Prefer constructing a real config when possible; fall back to MagicMock
    # with the fields the oracle reads.
    try:
        # Parallel config with non-batched format.
        pc = MagicMock(spec=FusedMoEParallelConfig)
        pc.use_batched_activation_format = False
        pc.use_fi_nvl_two_sided_kernels = False
        pc.use_fi_nvl_one_sided_kernels = False
        cfg = MagicMock(spec=FusedMoEConfig)
        cfg.swiglu_limit = None
        cfg.moe_backend = moe_backend
        cfg.moe_parallel_config = pc
        cfg.is_act_and_mul = True
        cfg.activation = MoEActivation.SILU
        cfg.routing_method = None
        cfg.router_logits_dtype = None
        cfg.hidden_dim = 128
        cfg.is_lora_enabled = False
        return cfg
    except Exception:
        cfg = MagicMock()
        cfg.swiglu_limit = None
        cfg.moe_backend = moe_backend
        cfg.moe_parallel_config.use_batched_activation_format = False
        return cfg


def test_select_nvfp4_moe_backend_sm70_auto_picks_marlin(monkeypatch):
    """On SM70 fallback, auto-select must land on Marlin (weight-only)."""
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        select_nvfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )

    monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_FALLBACK", "1")
    # Avoid FlashInfer env forcing a different branch.
    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP4", raising=False)
    monkeypatch.delenv("VLLM_TEST_FORCE_FP8_MARLIN", raising=False)

    # Make only Marlin (and optionally Emulation) report support so we
    # exercise the real is_supported_config path for Marlin.
    def _supported(cls, moe_config, weight_key, activation_key, activation_format):
        name = getattr(cls, "__name__", str(cls))
        if name == "MarlinExperts":
            return True, None
        if name == "Nvfp4QuantizationEmulationTritonExperts":
            # Only for W4A4 scheme
            if (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic):
                return True, None
            return False, "scheme"
        return False, f"mocked-unsupported:{name}"

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEExperts."
        "is_supported_config",
        staticmethod(_supported),
    )
    # Also patch on MarlinExperts if it overrides (it doesn't — uses base).

    cfg = _minimal_moe_config("auto")
    backend, experts_cls = select_nvfp4_moe_backend(
        config=cfg,
        weight_key=kNvfp4Static,
        activation_key=kNvfp4Dynamic,
    )
    assert backend is NvFp4MoeBackend.MARLIN
    assert experts_cls is MarlinExperts


def test_select_nvfp4_moe_backend_sm70_w4a16_still_marlin(monkeypatch):
    """W4A16 (activation_key=None): Emulation rejects scheme; Marlin wins."""
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        select_nvfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static

    monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_FALLBACK", "1")
    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP4", raising=False)
    monkeypatch.delenv("VLLM_TEST_FORCE_FP8_MARLIN", raising=False)

    def _supported(cls, moe_config, weight_key, activation_key, activation_format):
        name = getattr(cls, "__name__", str(cls))
        if name == "MarlinExperts":
            return True, None
        return False, f"mocked-unsupported:{name}"

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEExperts."
        "is_supported_config",
        staticmethod(_supported),
    )

    cfg = _minimal_moe_config("auto")
    backend, experts_cls = select_nvfp4_moe_backend(
        config=cfg,
        weight_key=kNvfp4Static,
        activation_key=None,
    )
    assert backend is NvFp4MoeBackend.MARLIN
    assert experts_cls is MarlinExperts


def test_select_falls_through_to_emulation_when_marlin_unsupported(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        select_nvfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )

    monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_FALLBACK", "1")
    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP4", raising=False)
    monkeypatch.delenv("VLLM_TEST_FORCE_FP8_MARLIN", raising=False)

    def _supported(cls, moe_config, weight_key, activation_key, activation_format):
        name = getattr(cls, "__name__", str(cls))
        if name == "Nvfp4QuantizationEmulationTritonExperts":
            return True, None
        return False, f"mocked-unsupported:{name}"

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEExperts."
        "is_supported_config",
        staticmethod(_supported),
    )

    cfg = _minimal_moe_config("auto")
    backend, experts_cls = select_nvfp4_moe_backend(
        config=cfg,
        weight_key=kNvfp4Static,
        activation_key=kNvfp4Dynamic,
    )
    assert backend is NvFp4MoeBackend.EMULATION
    assert experts_cls is Nvfp4QuantizationEmulationTritonExperts


def test_ct_w4a4_prepare_routes_through_sm70_try_prepare(monkeypatch):
    """CT W4A4 process_weights must call try_prepare_sm70_nvfp4_linear."""
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (  # noqa: E501
        CompressedTensorsW4A4Fp4,
    )
    from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm

    layer = torch.nn.Module()
    layer.weight_packed = Parameter(
        torch.zeros((4, 2), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_global_scale = Parameter(
        torch.ones(1, dtype=torch.float32), requires_grad=False
    )
    layer.weight_scale = Parameter(
        torch.ones((4, 1), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.input_global_scale = Parameter(
        torch.ones(1, dtype=torch.float32), requires_grad=False
    )

    called = {"prepare": False}

    def fake_prepare(prepared: torch.nn.Module) -> bool:
        called["prepare"] = True
        return True

    monkeypatch.setattr(sm70_tm, "try_prepare_sm70_nvfp4_linear", fake_prepare)
    # Avoid init_nvfp4_linear_kernel in scheme __init__ when TM is on.
    monkeypatch.setattr(
        sm70_tm, "use_turbomind", lambda default_enabled: True
    )

    scheme = CompressedTensorsW4A4Fp4()
    scheme.process_weights_after_loading(layer)
    assert called["prepare"] is True


def test_modelopt_w4a4_prepare_routes_through_sm70_try_prepare(monkeypatch):
    """ModelOpt W4A4 linear must use the shared SM70 prepare path."""
    from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4LinearMethod,
    )

    config = ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=16,
    )

    # Avoid heavy kernel init; pin a stub.
    class _StubKernel:
        def process_weights_after_loading(self, layer):
            raise AssertionError("should not fall through to kernel on SM70 hit")

        def apply_weights(self, layer, x, bias=None):
            raise AssertionError("should not fall through")

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel",
        lambda: _StubKernel(),
    )

    method = ModelOptNvFp4LinearMethod(config)
    method.kernel = _StubKernel()

    layer = torch.nn.Module()
    layer.weight = Parameter(torch.zeros((4, 2), dtype=torch.uint8), requires_grad=False)
    layer.weight_scale = Parameter(
        torch.ones((4, 1), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale_2 = Parameter(
        torch.ones(1, dtype=torch.float32), requires_grad=False
    )
    layer.input_scale = Parameter(
        torch.ones(1, dtype=torch.float32), requires_grad=False
    )

    called = {"prepare": False}

    def fake_prepare(prepared: torch.nn.Module) -> bool:
        called["prepare"] = True
        return True

    monkeypatch.setattr(sm70_tm, "try_prepare_sm70_nvfp4_linear", fake_prepare)

    method.process_weights_after_loading(layer)
    assert called["prepare"] is True
    assert hasattr(layer, "weight_global_scale")

    # apply path
    applied = {"hit": False}

    def fake_apply(layer, x, bias=None):
        applied["hit"] = True
        return torch.zeros((*x.shape[:-1], 4), dtype=x.dtype)

    monkeypatch.setattr(sm70_tm, "try_apply_sm70_nvfp4_linear", fake_apply)
    out = method.apply(layer, torch.ones((1, 4), dtype=torch.float16), None)
    assert applied["hit"] is True
    assert out.shape == (1, 4)
