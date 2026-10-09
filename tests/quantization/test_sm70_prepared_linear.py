# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.qpn import nvfp4_dequant
from vllm.model_executor.kernels.linear.sm70_provider import apply_prepared
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    SM70TurboMindLinearState,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "kind,flags,name,contiguous",
    [
        ("uint4", {}, "awq_gemm_sm70_out", False),
        ("fp8", {}, "fp8_gemm_sm70_out", False),
        ("mxfp4", {}, "mxfp4_gemm_sm70_out", False),
        ("nvfp4", {}, "nvfp4_gemm_sm70_out", False),
        ("nvfp4", {"prescaled_scales": True}, "nvfp4_gemm_sm70_prescaled_out", False),
        (
            "nvfp4",
            {"use_scale_code": True},
            "nvfp4_qpn2_compact_tm_gemm_sm70_out",
            False,
        ),
        ("nvfp4_qpn4", {}, "sm70_nvfp4_qpn4_dispatch", True),
        ("nvfp4_qpn2_dense", {}, "nvfp4_qpn2_dispatch_linear", True),
    ],
)
@pytest.mark.parametrize("rows", [0, 3])
def test_prepared_provider_preserves_dispatch_and_shared_output_contract(
    monkeypatch, kind, flags, name, contiguous, rows
):
    calls = []
    state = SM70TurboMindLinearState(
        torch.empty(4, 8),
        torch.empty(1),
        16,
        4,
        8,
        6,
        kind,
        padded_output_size=8,
        **flags,
    )
    x = torch.arange(2 * rows * 8, dtype=torch.float16).reshape(2, rows, 8)[..., ::2]
    bias = torch.arange(6, dtype=x.dtype)

    def write(out, inp):
        calls.append((name, inp.stride(-1), tuple(inp.shape)))
        out.copy_(torch.arange(out.numel(), dtype=out.dtype).reshape_as(out))
        return out

    if kind == "nvfp4_qpn4":
        monkeypatch.setattr(
            torch.ops.vllm, name, lambda out, prefix, inp, *args: write(out, inp)
        )
    elif kind == "nvfp4_qpn2_dense":

        def qpn2(inp, codes, scales, global_scale, n, k, split_k, nacc, policy):
            return write(torch.empty((inp.shape[0], n), dtype=inp.dtype), inp)

        monkeypatch.setattr(nvfp4_dequant, name, qpn2)
    state.native_ops = SimpleNamespace(
        arguments=(), **{name: lambda out, inp, *args: write(out, inp)}
    )
    result = apply_prepared(state, x, bias, "layer")
    expected = (
        torch.arange(2 * rows * 8, dtype=x.dtype).reshape(2 * rows, 8)[:, :6] + bias
    )
    assert torch.equal(result, expected.reshape(2, rows, 6))
    assert len(calls) == 1
    if rows:
        assert calls[0][1] == (1 if contiguous else 2)


@pytest.mark.parametrize("kind", ["uint4", "nvfp4"])
def test_interleaved_preparation_restores_logical_gate_up_before_bias(kind):
    state = SM70TurboMindLinearState(
        torch.empty(4, 8), torch.empty(1), 16, 4, 8, 8, kind, gated_silu=True
    )

    def gemm(out, *args):
        out.copy_(torch.tensor([[0, 4, 1, 5, 2, 6, 3, 7]], dtype=out.dtype))

    state.native_ops = SimpleNamespace(**{state.provider.name: gemm})
    output = apply_prepared(state, torch.ones(1, 4), torch.arange(8), "layer")
    assert output.tolist() == [[0, 2, 4, 6, 8, 10, 12, 14]]


def test_scale_code_conversion_rebinds_provider_without_runtime_strategy_lookup():
    state = SM70TurboMindLinearState(
        torch.empty(4, 8), torch.empty(1), 16, 4, 8, 8, "nvfp4"
    )
    assert state.provider.name == "nvfp4_gemm_sm70_out"
    state.use_scale_code = True
    state.refresh_provider()
    assert state.provider.name == "nvfp4_compact"
    calls = []
    state.native_ops = SimpleNamespace(
        nvfp4_qpn2_compact_tm_gemm_sm70_out=lambda *args: calls.append(args)
    )
    apply_prepared(state, torch.ones(1, 4), None, "layer")
    assert len(calls) == 1


def test_fused_empty_input_does_not_launch_and_dtype_error_is_retained():
    state = SM70TurboMindLinearState(
        torch.empty(4, 8), torch.empty(1), 16, 4, 8, 8, "nvfp4", gated_silu=True
    )
    state.native_ops = SimpleNamespace(
        nvfp4_gemm_sm70_out=lambda *args: pytest.fail("empty fused launch")
    )
    output = apply_prepared(
        state, torch.empty(0, 4, dtype=torch.float16), None, "layer", gated=True
    )
    assert output.shape == (0, 4)
    with pytest.raises(RuntimeError, match="requires float16"):
        apply_prepared(
            state, torch.ones(1, 4, dtype=torch.float32), None, "layer", gated=True
        )
