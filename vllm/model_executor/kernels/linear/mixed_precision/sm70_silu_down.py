# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured SiLU/down fusion in the mixed-precision kernel lifecycle."""

import math
import statistics
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.model_executor.kernels.linear.fp16_silu_down import apply
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op


@dataclass
class Sm70SiluDownConfig(MPLinearLayerConfig):
    """Unpacked FP16 weight consuming two contiguous activation segments."""


def _reference(x, weight):
    activated = x.new_empty((x.shape[0], weight.shape[1]))
    torch.ops._C.silu_and_mul(activated, x)
    return torch.nn.functional.linear(activated, weight)


def _dispatch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return apply(x.reshape(-1, x.shape[-1]), weight).reshape(
        *x.shape[:-1], weight.shape[0]
    )


def _fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="sm70_fp16_silu_down", op_func=_dispatch, fake_impl=_fake
)


def _measure(weight, width):
    """Pair both graph arms with a 16-MiB read/write L2 flush."""
    k = weight.shape[1]
    row = ((torch.arange(2 * k, device=weight.device) % 23) - 11).half() / 16
    x = row[None, :].expand(width, 2 * k).contiguous()
    candidate, reference = apply(x, weight), _reference(x, weight)
    if not torch.allclose(candidate, reference, atol=0.002, rtol=0.002):
        return {"accepted": False, "reason": "startup_numeric_check"}
    flush = torch.zeros(4 * 1024 * 1024, device=weight.device, dtype=torch.int32)
    graphs = []
    for fused in (False, True):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flush.add_(1)
            apply(x, weight) if fused else _reference(x, weight)
        graphs.append(graph)
    timings: list[list[float]] = [[], []]
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for repeat in range(9):
        for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
            start.record()
            graphs[arm].replay()
            end.record()
            end.synchronize()
            timings[arm].append(start.elapsed_time(end) * 1000)
    quartiles = [statistics.quantiles(t, n=4, method="inclusive") for t in timings]
    accepted = quartiles[1][2] < quartiles[0][0]
    return {
        "accepted": accepted,
        "reason": None if accepted else "cold_probe_not_faster",
        "cold_probe_including_flush_us": [statistics.median(t) for t in timings],
        "cold_probe_interquartile_us": [[q[0], q[2]] for q in quartiles],
    }


class Sm70SiluDownKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls):
        return 70

    @classmethod
    def can_implement(cls, c):
        if not isinstance(c, Sm70SiluDownConfig):
            return False, "requires a SiLU/down FP16 configuration"
        if not current_platform.is_device_capability(70):
            return False, "requires SM70"
        if (
            c.weight_type != scalar_types.float16
            or c.act_type != torch.float16
            or c.out_type not in (None, torch.float16)
        ):
            return False, "requires FP16 input, weight and output"
        if c.group_size != -1 or c.zero_points or c.has_g_idx:
            return False, "does not consume quantization metadata"
        k, n = c.partition_weight_shape
        if not (0 < k <= 4096 and n > 0):
            return False, "invalid shape or excessive static K unrolling"
        reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        if (
            reduced[0] if isinstance(reduced, tuple) else reduced
        ) or torch.backends.cuda.matmul.allow_fp16_accumulation:
            return False, "requires FP32 accumulation policy"
        return True, None

    def process_weights_after_loading(self, layer):
        from vllm.config import get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
        widths = (cfg.compilation_config.cudagraph_capture_sizes if cfg else None) or [
            1
        ]
        weight = getattr(layer, self.w_q_name)
        self.measurements = {
            width: _measure(weight, width) for width in sorted(set(widths)) if width > 0
        }
        self.widths = [
            width for width, result in self.measurements.items() if result["accepted"]
        ]
        if cfg:
            cfg.kernel_config.fused_fp16_silu_down_applicable = True
            cfg.kernel_config.linear_kernel_selections[
                f"silu_down:{getattr(layer, 'prefix', '')}:{tuple(weight.shape)}"
            ] = {
                "selected": type(self).__name__ if self.widths else "vendor",
                "input_dtype": "float16",
                "weight_dtype": "float16",
                "accumulation": "float32",
                "measured_widths": self.widths,
                "measurements": self.measurements,
            }

    def apply_weights(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, getattr(layer, self.w_q_name), bias)

    def apply_silu_down(self, layer, x):
        weight = getattr(layer, self.w_q_name)
        if (
            math.prod(x.shape[:-1]) not in self.widths
            or x.dtype != torch.float16
            or x.shape[-1] != 2 * weight.shape[1]
            or not x.is_contiguous()
        ):
            return None
        return torch.ops.vllm.sm70_fp16_silu_down(x, weight)


def maybe_prepare_silu_down(layer):
    """Prepare only operators explicitly consuming SiLU/multiply output."""
    from vllm.config import get_current_vllm_config_or_none
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel

    cfg = get_current_vllm_config_or_none()
    weight = getattr(layer, "weight", None)
    if (
        cfg is None
        or envs.VLLM_BATCH_INVARIANT
        or not cfg.kernel_config.fused_fp16_silu_down
        or not getattr(layer, "_consumes_silu_and_mul", False)
        or layer.bias is not None
        or not layer.input_is_parallel
        or weight is None
        or weight.ndim != 2
        or not weight.is_cuda
        or weight.dtype != torch.float16
        or not weight.is_contiguous()
    ):
        return
    k, n = weight.shape[1], weight.shape[0]
    config = Sm70SiluDownConfig(
        full_weight_shape=(layer.input_size, layer.output_size),
        partition_weight_shape=(k, n),
        weight_type=scalar_types.float16,
        act_type=torch.float16,
        group_size=-1,
        zero_points=False,
        has_g_idx=False,
        out_type=torch.float16,
    )
    try:
        kernel_cls = choose_mp_linear_kernel(config)
    except ValueError:
        return
    kernel = kernel_cls(config, "weight", "")
    kernel.process_weights_after_loading(layer)
    layer._sm70_silu_down_kernel = kernel
