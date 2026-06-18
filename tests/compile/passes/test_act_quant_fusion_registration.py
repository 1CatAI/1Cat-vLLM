# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
