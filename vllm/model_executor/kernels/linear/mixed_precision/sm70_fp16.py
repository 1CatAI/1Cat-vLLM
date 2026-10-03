# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured FP16 linears in the existing mixed-precision kernel lifecycle."""

import statistics
from dataclasses import dataclass

import torch

from vllm.model_executor.kernels.linear.fp16_gemv_silu import (
    Sm70Fp16GateUpKernel,
    Sm70Fp16GemvSiluKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op


@dataclass
class Sm70Fp16LinearLayerConfig(MPLinearLayerConfig):
    """Unpacked FP16 weights; no scales, zero points or reordering."""


@dataclass
class Sm70Fp16Admission:
    input_dtype: str
    weight_dtype: str
    accumulation: str
    plain_widths: list[int]
    gated_widths: list[int]
    measurements: dict


def _reference(x, weight, gated):
    projected = torch.nn.functional.linear(x, weight)
    if not gated:
        return projected
    out = x.new_empty((x.shape[0], weight.shape[0] // 2))
    torch.ops._C.silu_and_mul(out, projected)
    return out


def _candidate(x, weight, out, gated):
    if gated:
        Sm70Fp16GateUpKernel.apply_out(x, weight, out)
    else:
        Sm70Fp16GemvSiluKernel.apply_out(x, weight, out, out.shape[1], 0)


def _dispatch(
    x: torch.Tensor, weight: torch.Tensor, gated: bool, measured_widths: list[int]
) -> torch.Tensor:
    shape = (*x.shape[:-1], weight.shape[0] // (2 if gated else 1))
    flattened = x.reshape(-1, x.shape[-1])
    if flattened.shape[0] not in measured_widths or not flattened.is_contiguous():
        return _reference(flattened, weight, gated).reshape(shape)
    out = x.new_empty((flattened.shape[0], shape[-1]))
    _candidate(flattened, weight, out, gated)
    return out.reshape(shape)


def _fake(
    x: torch.Tensor, weight: torch.Tensor, gated: bool, measured_widths: list[int]
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0] // (2 if gated else 1)))


direct_register_custom_op(
    op_name="sm70_fp16_measured_linear",
    op_func=_dispatch,
    fake_impl=_fake,
)


def _measure(weight, width, gated):
    """Compare captured operations with an L2 flush before every operation.

    Flush cost is included equally in both arms. This is a startup admission
    measurement, not a reported projection latency or an endpoint benchmark.
    The deterministic input does not consume model/sampler RNG state.
    """
    with torch.accelerator.device_index(weight.device.index):
        k, n = weight.shape[1], weight.shape[0] // (2 if gated else 1)
        row = ((torch.arange(k, device=weight.device) % 23) - 11).half() / 16
        x = row[None, :].expand(width, k).contiguous()
        out = x.new_empty((width, n))
        _candidate(x, weight, out, gated)
        ref = _reference(x, weight, gated)
        if not torch.allclose(out, ref, atol=0.002, rtol=0.002):
            return {"accepted": False, "reason": "startup_numeric_check"}
        # V100 L2 is 6 MiB. Touch 16 MiB through a compute read/write kernel.
        flush = torch.zeros(4 * 1024 * 1024, device=weight.device, dtype=torch.int32)
        graphs = []
        for candidate in (False, True):
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream(device=weight.device)
            with torch.cuda.graph(graph, stream=stream):
                flush.add_(1)
                if candidate:
                    _candidate(x, weight, out, gated)
                else:
                    _reference(x, weight, gated)
            graphs.append(graph)
        timings: list[list[float]] = [[], []]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for repeat in range(7):
            for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                start.record()
                graphs[arm].replay()
                end.record()
                end.synchronize()
                timings[arm].append(start.elapsed_time(end) * 1000)
        medians = [statistics.median(t) for t in timings]
        accepted = medians[1] < medians[0]
        return {
            "accepted": accepted,
            "reason": None if accepted else "cold_probe_not_faster",
            "cold_probe_including_flush_us": medians,
        }


class Sm70Fp16LinearKernel(MPLinearKernel):
    """FP16/FP16 is the unpacked 16-bit case of the existing MP interface."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not isinstance(c, Sm70Fp16LinearLayerConfig):
            return False, "requires unpacked FP16 configuration"
        if not current_platform.is_device_capability(70):
            return False, "requires SM70"
        if c.weight_type != scalar_types.float16 or c.act_type != torch.float16:
            return False, "requires FP16 weights and input"
        if c.zero_points or c.has_g_idx or c.group_size != -1:
            return False, "does not consume quantization metadata"
        k, n = c.partition_weight_shape
        if k <= 0 or n <= 0 or k > 16384:
            return False, "invalid shape or excessive static K unrolling"
        reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        if (reduced[0] if isinstance(reduced, tuple) else reduced) or getattr(
            torch.backends.cuda.matmul, "allow_fp16_accumulation", False
        ):
            return False, "requires the existing FP32 accumulation policy"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.config import get_current_vllm_config_or_none

        weight = getattr(layer, self.w_q_name)
        self.plain_widths: list[int] = []
        self.gated_widths: list[int] = []
        self.measurements = {}
        cfg = get_current_vllm_config_or_none()
        widths = (cfg.compilation_config.cudagraph_capture_sizes if cfg else None) or [
            1
        ]
        sizes: list[int] = list(getattr(layer, "output_sizes", ()))
        gated = len(sizes) == 2 and sizes[0] == sizes[1]
        for width in sorted(set(widths)):
            for fusion in (False, True) if gated else (False,):
                result = _measure(weight, width, fusion)
                self.measurements[f"M{width}/{'silu_mul' if fusion else 'linear'}"] = (
                    result
                )
                if result["accepted"]:
                    (self.gated_widths if fusion else self.plain_widths).append(width)
        self.capability = Sm70Fp16Admission(
            input_dtype="float16",
            weight_dtype="float16",
            accumulation="float32",
            plain_widths=self.plain_widths,
            gated_widths=self.gated_widths,
            measurements=self.measurements,
        )

    def apply_weights(self, layer, x, bias=None):
        weight = getattr(layer, self.w_q_name)
        if bias is not None or x.dtype != torch.float16:
            return torch.nn.functional.linear(x, weight, bias)
        return torch.ops.vllm.sm70_fp16_measured_linear(
            x, weight, False, self.plain_widths
        )

    def apply_fused_silu_and_mul(self, layer, x):
        sizes: list[int] = list(getattr(layer, "output_sizes", ()))
        if len(sizes) != 2 or sizes[0] != sizes[1] or x.dtype != torch.float16:
            return None
        return torch.ops.vllm.sm70_fp16_measured_linear(
            x, getattr(layer, self.w_q_name), True, self.gated_widths
        )
