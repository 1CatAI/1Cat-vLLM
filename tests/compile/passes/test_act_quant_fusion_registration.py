# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.compilation.passes.fusion.act_quant_fusion import (
    ActivationQuantFusionPass,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
)


def test_activation_quant_fusion_skips_missing_backend_ops(monkeypatch):
    import vllm.compilation.passes.fusion.act_quant_fusion as act_quant_fusion

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(fuse_act_quant=True),
        ),
    )

    monkeypatch.setattr(act_quant_fusion, "QUANT_OPS", {})
    monkeypatch.setattr(act_quant_fusion, "FUSED_OPS", {})

    ActivationQuantFusionPass(config)


@pytest.mark.parametrize("capability", [70, 75, 80, 90])
@pytest.mark.parametrize("available", [False, True])
def test_static_fp8_fusion_requires_supported_cuda_device(
    monkeypatch, capability, available
):
    import vllm.compilation.passes.fusion.act_quant_fusion as fusion

    operators = SimpleNamespace()
    if available:
        operators.silu_and_mul_quant = object()
    monkeypatch.setattr(
        fusion, "torch", SimpleNamespace(ops=SimpleNamespace(_C=operators))
    )
    monkeypatch.setattr(
        fusion,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            has_device_capability=lambda minimum: capability >= minimum,
        ),
    )
    assert fusion._supports_static_fp8_silu_mul_quant() == (
        available and capability >= 80
    )


def test_static_fp8_fusion_preserves_non_cuda_backends(monkeypatch):
    import vllm.compilation.passes.fusion.act_quant_fusion as fusion

    monkeypatch.setattr(
        fusion,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(_C=SimpleNamespace(silu_and_mul_quant=object()))
        ),
    )
    monkeypatch.setattr(
        fusion, "current_platform", SimpleNamespace(is_cuda=lambda: False)
    )
    assert fusion._supports_static_fp8_silu_mul_quant()
