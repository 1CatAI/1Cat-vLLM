# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which kernel serves FusedRMSNormGated: the CustomOp switch and the call form.

The fused norm kernel is the body of forward_cuda for the calls it covers; the
CustomOp decides, as for every other op, whether forward_cuda runs at all. These
tests run on the CPU: they replace the kernels by recorders.
"""

import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fla.ops import kda as kda_ops
from vllm.model_executor.layers.fla.ops.kda import FusedRMSNormGated

HEAD_DIM = 128


def _norm(
    custom_ops: list[str], hidden_size: int = HEAD_DIM, activation: str = "sigmoid"
) -> FusedRMSNormGated:
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=custom_ops))
    with set_current_vllm_config(config):
        return FusedRMSNormGated(
            hidden_size, activation=activation, device="cpu", dtype=torch.float16
        )


@pytest.mark.parametrize(
    ("custom_ops", "expected"),
    [
        (["none"], "native"),
        (["none", "-fused_rms_norm_gated"], "native"),
        (["none", "+fused_rms_norm_gated"], "cuda"),
    ],
)
def test_custom_op_switch_selects_the_forward(monkeypatch, custom_ops, expected):
    calls = []
    monkeypatch.setattr(
        FusedRMSNormGated, "forward_native", lambda *a, **k: calls.append("native")
    )
    monkeypatch.setattr(
        FusedRMSNormGated, "forward_cuda", lambda *a, **k: calls.append("cuda")
    )
    _norm(custom_ops)(None, None, out_dtype=torch.float16)
    assert calls == [expected]


@pytest.fixture
def recorders(monkeypatch):
    calls = []

    def fused(x, gate, weight, eps, out_dtype):
        calls.append(("fused", out_dtype))
        return x

    def general(x, g, weight, bias, activation, **kwargs):
        calls.append(("general", kwargs["out_dtype"]))
        return x

    monkeypatch.setattr(kda_ops, "kda_gated_rmsnorm", fused)
    monkeypatch.setattr(kda_ops, "rms_norm_gated", general)
    return calls


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_covered_call_uses_the_fused_kernel(recorders, x_dtype):
    norm = _norm(["none", "+fused_rms_norm_gated"])
    x = torch.zeros(1, 3, 4, HEAD_DIM, dtype=x_dtype)
    gate = torch.zeros(3, 4, HEAD_DIM, dtype=torch.float16)
    norm.forward_cuda(x, gate, out_dtype=torch.float16)
    norm.forward_cuda(x, gate)
    assert recorders == [("fused", torch.float16), ("fused", x_dtype)]


def test_other_calls_use_the_general_kernel(recorders):
    norm = _norm(["none", "+fused_rms_norm_gated"])
    x = torch.zeros(3, HEAD_DIM, dtype=torch.float16)
    gate = torch.zeros(3, HEAD_DIM, dtype=torch.float16)
    # residual, prenorm
    norm.forward_cuda(x, gate, residual=x.clone())
    norm.forward_cuda(x, gate, prenorm=True)
    # another row width, another activation, an input the kernel does not take
    wide = _norm(["none", "+fused_rms_norm_gated"], hidden_size=2 * HEAD_DIM)
    wide.forward_cuda(torch.zeros(3, 2 * HEAD_DIM), torch.zeros(3, 2 * HEAD_DIM))
    swish = _norm(["none", "+fused_rms_norm_gated"], activation="swish")
    swish.forward_cuda(x, gate)
    norm.forward_cuda(x.double(), gate)
    assert [kind for kind, _ in recorders] == ["general"] * 5
