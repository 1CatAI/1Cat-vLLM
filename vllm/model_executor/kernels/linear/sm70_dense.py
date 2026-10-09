# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared FP16 provider; model eligibility arrives as a loading contract."""

import torch

from vllm import _sm70_ops as sm70_ops
from vllm._sm70.policy import NativeBindings
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear_io import (
    flatten_linear_input,
    restore_linear_output,
)

logger = init_logger(__name__)


class DenseLinearState(torch.nn.Module):
    def __init__(
        self,
        weight,
        *,
        prefix,
        policy,
        trace,
        forbidden=False,
        max_m=None,
        glm_cublaslt=False,
    ):
        super().__init__()
        object.__setattr__(self, "weight", weight)
        self.prefix = prefix
        self.policy = policy
        self.trace = trace
        self.native_ops = sm70_ops
        self._sm70_f16_forbidden = forbidden
        self._sm70_f16_max_m = max_m
        self._sm70_glm53_tp8_cublaslt = glm_cublaslt

    def apply(self, x, bias):
        return _maybe_sm70_dense_forward(self, x, bias)

    def apply_glm(self, x, bias):
        return _maybe_sm70_glm53_tp8_cublaslt(self, x, bias)


def _interleave_output_rows_for_gated_silu(weight: torch.Tensor) -> torch.Tensor:
    half = weight.shape[0] // 2
    return torch.stack((weight[:half], weight[half:]), dim=1).reshape(weight.shape)


def _maybe_sm70_glm53_tp8_cublaslt(
    state: DenseLinearState,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    if not getattr(state, "_sm70_glm53_tp8_cublaslt", False):
        return None
    if bias is not None or x.dtype != torch.float16 or x.shape[-1] not in (1024, 4096):
        return None
    x_2d = flatten_linear_input(x)
    if x_2d.shape[0] != 8 or not x_2d.is_contiguous():
        return None
    weight = state.weight
    if weight.dtype != torch.float16 or not weight.is_contiguous():
        return None
    shape = (tuple(weight.shape), tuple(x_2d.shape))
    if shape not in (
        ((3336, 4096), (8, 4096)),
        ((4096, 1024), (8, 1024)),
    ):
        return None
    if not hasattr(torch.ops._C, "sm70_glm53_tp8_cublaslt_out"):
        raise RuntimeError(
            "The SM70 GLM-5.3 TP8 cuBLASLt projection requires its native "
            "op. Rebuild vLLM from source with CUDA 12.8 and arch 7.0."
        )
    out = torch.empty((x_2d.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    state.native_ops.sm70_glm53_tp8_cublaslt_out(out, x_2d, weight)
    logger.info_once("SM70 GLM-5.3 TP8 cuBLASLt M8 projection path enabled.")
    return restore_linear_output(out, x)


def _maybe_sm70_dense_forward(
    state: DenseLinearState,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    glm53_output = _maybe_sm70_glm53_tp8_cublaslt(state, x, bias)
    if glm53_output is not None:
        return glm53_output
    if getattr(state, "_sm70_f16_forbidden", False):
        return None
    if not getattr(state, "_sm70_f16_prepared", False):
        return None
    if not hasattr(torch.ops._C, "sm70_f16_gemm"):
        return None
    max_m = getattr(state, "_sm70_f16_max_m", None)
    if max_m is not None and x.numel() // x.shape[-1] > max_m:
        return None

    if state.trace.dense_debug or state.trace.qwen_next_trace:
        rows = x.numel() // x.shape[-1]
        logger.info(
            "SM70 dense apply active for %s m=%d n=%d k=%d",
            getattr(state, "prefix", "<unknown>"),
            rows,
            state.weight.shape[0],
            x.shape[-1],
        )

    x_2d = flatten_linear_input(x)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    tm_weight = getattr(state, "_sm70_f16_tm_weight", None)
    k_ld = getattr(state, "_sm70_f16_k_ld", None)
    if not torch.compiler.is_compiling() and tm_weight is not None and k_ld is not None:
        out = torch.empty(
            (x_2d.size(0), tm_weight.shape[0]),
            dtype=x_2d.dtype,
            device=x_2d.device,
        )
        state.native_ops.sm70_f16_gemm_out(out, x_2d, tm_weight, k_ld, False)
    else:
        out = state.native_ops.sm70_f16_gemm(x_2d, state.weight)

    if bias is not None:
        out = out + bias
    return restore_linear_output(out, x)


def prepare_dense(state, *, force_enable, input_parallel, suffix_allowed):
    if state._sm70_glm53_tp8_cublaslt:
        state.native_ops = NativeBindings(state.policy.native.values)
    if not state.policy.dense_f16 and not force_enable:
        return
    if getattr(state, "_sm70_f16_forbidden", False):
        return
    if not force_enable and not suffix_allowed:
        return
    if not input_parallel:
        return
    if not hasattr(torch.ops._C, "sm70_f16_prepare"):
        return
    if state.weight.dtype != torch.float16 or not state.weight.is_cuda:
        return
    if torch.cuda.get_device_capability(state.weight.device) != (7, 0):
        return
    if (
        state.weight.ndim != 2
        or (state.weight.shape[1] % 16) != 0
        or (state.weight.shape[0] % 32) != 0
    ):
        return

    state.native_ops = NativeBindings(state.policy.native.values)
    prepared = state.native_ops.sm70_f16_prepare(state.weight)
    state.register_buffer("_sm70_f16_tm_weight", prepared[0], persistent=False)
    state._sm70_f16_k_ld = int(prepared[1][0].item())
    prefix = getattr(state, "prefix", "")
    if prefix.rsplit(".", 1)[-1] == "gate_up_proj" and state.weight.shape[0] % 2 == 0:
        gated_weight = _interleave_output_rows_for_gated_silu(state.weight).contiguous()
        gated_prepared = state.native_ops.sm70_f16_prepare(gated_weight)
        state.register_buffer("_sm70_f16_gated_weight", gated_weight, persistent=False)
        state.register_buffer(
            "_sm70_f16_gated_tm_weight", gated_prepared[0], persistent=False
        )
        state._sm70_f16_gated_k_ld = int(gated_prepared[1][0].item())
    state._sm70_f16_prepared = True
    logger.info_once("SM70 dense fp16 fast path enabled for small decode projections.")
    if state.trace.dense_debug or state.trace.qwen_next_trace:
        logger.info(
            "SM70 dense prepared for %s weight_shape=%s",
            getattr(state, "prefix", "<unknown>"),
            tuple(state.weight.shape),
        )
