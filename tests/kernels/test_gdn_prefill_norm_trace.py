# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.fla.ops.gdn_profiling import bind_gdn_profiler
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.platforms import current_platform

PREFILL_TOKENS = 128


def _projection_layer(vllm_config) -> QwenGatedDeltaNetAttention:
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.prefix = "model.layers.0.linear_attn"
    layer._gdn_profiler = bind_gdn_profiler(vllm_config)
    layer.enable_sm70_gdn_rmsnorm_onepass = False
    layer.enable_sm70_dflash2_fused_gdn_norm = False
    layer.norm = RMSNormGated(
        128,
        eps=1e-6,
        norm_before_gate=True,
        activation="silu",
        dtype=torch.float16,
        device="cuda",
    )
    layer.output_projection_override = lambda hidden: hidden
    return layer


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_traced_prefill_projection_keeps_the_custom_op_norm(
    default_vllm_config, monkeypatch
):
    layer = _projection_layer(default_vllm_config)
    direct_calls = []
    forward_cuda = RMSNormGated.forward_cuda

    def counting_forward_cuda(self, x, z=None):
        direct_calls.append(x.shape[0])
        return forward_cuda(self, x, z)

    monkeypatch.setattr(RMSNormGated, "forward_cuda", counting_forward_cuda)
    torch.manual_seed(20261015)
    core = torch.randn(PREFILL_TOKENS, 4, 128, device="cuda", dtype=torch.float16)
    z = torch.randn_like(core)

    # Eager prefill (the opaque-op case) takes the direct Triton norm.
    eager = layer._compute_output_projection(core, z, PREFILL_TOKENS)
    assert direct_calls == [PREFILL_TOKENS * 4]

    # A traced prefill projection must not reach the Triton norm's device guard,
    # which Dynamo refuses to inline.
    direct_calls.clear()

    def project(core_attn_out, gate):
        return layer._compute_output_projection(core_attn_out, gate, PREFILL_TOKENS)

    with set_current_vllm_config(default_vllm_config):
        traced = torch.compile(project, backend="eager", fullgraph=True)(core, z)
    assert direct_calls == []
    torch.testing.assert_close(traced, eager, atol=2e-3, rtol=2e-3)
