# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm import _sm70_ops as sm70_ops
from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)


def _is_sm70_tp4_nvfp4_gate_up(layer: torch.nn.Module) -> bool:
    return bool(
        getattr(layer, "tp_size", 1) == 4
        and getattr(layer, "prefix", "").rsplit(".", 1)[-1] == "gate_up_proj"
        and getattr(layer, "input_size_per_partition", 0) == 5120
        and getattr(layer, "output_size_per_partition", 0) == 8704
        and getattr(layer, "logical_widths", None) == [4352, 4352]
    )


def _is_sm70_tp4_nvfp4_down(layer: torch.nn.Module) -> bool:
    return bool(
        getattr(layer, "tp_size", 1) == 4
        and getattr(layer, "prefix", "").rsplit(".", 1)[-1] == "down_proj"
        and getattr(layer, "input_size_per_partition", 0) == 4352
        and getattr(layer, "output_size_per_partition", 0) == 5120
    )


def _is_sm70_nvfp4_qpn4_runtime_contract() -> bool:
    """Admit only the measured single-sequence, no-MTP decode contract."""
    vllm_config = get_current_vllm_config()
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_seqs = int(getattr(scheduler_config, "max_num_seqs", 1))
    speculative_config = getattr(vllm_config, "speculative_config", None)
    return max_num_seqs == 1 and speculative_config is None


_SM70_NVFP4_QPN4_REQUIRED_OPS = (
    "nvfp4_qpn4_prepare_sm70",
    "nvfp4_qpn4_prepare_scale_code_sm70",
    "nvfp4_qpn4_dequantize_sm70_out",
    "nvfp4_qpn4_prefill_sm70_out",
    "nvfp4_qpn4_dispatch_sm70_out",
)


def _missing_sm70_nvfp4_qpn4_ops() -> list[str]:
    return [
        name
        for name in _SM70_NVFP4_QPN4_REQUIRED_OPS
        if not hasattr(torch.ops._C, name)
    ]


__all__ = ["CompressedTensorsW4A4Fp4"]

_SM70_NVFP4_QPN2_CONFIGS = {
    # (K, N, fused gated-SiLU): (split-K, independent accumulator chains)
    (5120, 8704, False): (8, 2),
    (5120, 8704, True): (8, 2),
    (4352, 5120, False): (16, 2),
}
_SM70_NVFP4_QPN2_SHAPES = {
    # Checkpoint-native packed tensors are [N, K/2].
    "gate_up_proj": (8704, 2560),
    "down_proj": (5120, 2176),
}
_SM70_NVFP4_QPN2_REQUIRED_OPS = (
    "nvfp4_qpn2_prepare_sm70",
    "nvfp4_qpn2_gemm_sm70_out",
    "nvfp4_qpn2_gated_sm70_out",
    "nvfp4_qpn2_dispatch_sm70_out",
)
_SM70_NVFP4_QPN2_PREFILL_REQUIRED_OPS = ("nvfp4_qpn4_prefill_sm70_out",)


def _is_qpn2_layer(layer: torch.nn.Module) -> bool:
    if getattr(layer, "tp_size", 1) != 4:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    expected = _SM70_NVFP4_QPN2_SHAPES.get(suffix)
    if expected is None or tuple(layer.weight.shape) != expected:
        return False
    expected_n, expected_packed_k = expected
    return bool(
        getattr(layer, "input_size_per_partition", 0) == expected_packed_k * 2
        and getattr(layer, "output_size_per_partition", 0) == expected_n
    )


def _missing_qpn2_ops() -> list[str]:
    return [
        name
        for name in _SM70_NVFP4_QPN2_REQUIRED_OPS
        if not hasattr(torch.ops._C, name)
    ]


def _missing_qpn2_prefill_ops() -> list[str]:
    return [
        name
        for name in _SM70_NVFP4_QPN2_PREFILL_REQUIRED_OPS
        if not hasattr(torch.ops._C, name)
    ]


def _explicit_nvfp4_emulation_requested() -> bool:
    if envs.VLLM_USE_NVFP4_CT_EMULATIONS or envs.VLLM_NVFP4_GEMM_BACKEND == "emulation":
        return True

    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    return (
        vllm_config is not None
        and vllm_config.kernel_config.linear_backend == "emulation"
    )


class CompressedTensorsW4A4Fp4(CompressedTensorsScheme):
    def __init__(self):
        self.kernel = None
        if not sm70_tm.use_turbomind(envs.VLLM_SM70_NVFP4_TURBOMIND):
            self.kernel = init_nvfp4_linear_kernel()
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        if (
            sm70_tm.use_turbomind(envs.VLLM_SM70_NVFP4_TURBOMIND)
            or sm70_tm.forces_marlin()
        ):
            return 70
        if _explicit_nvfp4_emulation_requested():
            return 70
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Rename CT checkpoint names to standardized names
        layer.weight = layer.weight_packed
        del layer.weight_packed

        if (
            torch.unique(layer.input_global_scale).numel() != 1
            or torch.unique(layer.weight_global_scale).numel() != 1
        ):
            logger.warning_once(
                "In NVFP4 linear, the global scale for input or weight are different"
                " for parallel layers (e.g. q_proj, k_proj, v_proj). This "
                " will likely result in reduced accuracy. Please verify the model"
                " accuracy. Consider using a checkpoint with a shared global NVFP4"
                " scale for fused layers."
            )

        # Process global scales (CT stores as divisors, i.e. 1/scale)
        input_global_scale_inv = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(
            (1.0 / input_global_scale_inv).to(torch.float32), requires_grad=False
        )
        weight_global_scale = layer.weight_global_scale.max().to(torch.float32)
        layer.weight_global_scale = Parameter(
            1.0 / weight_global_scale, requires_grad=False
        )

        # Pre-compute alpha and inverse for runtime quantization
        layer.input_global_scale_inv = Parameter(
            input_global_scale_inv, requires_grad=False
        )
        layer.alpha = Parameter(
            layer.input_global_scale * layer.weight_global_scale, requires_grad=False
        )

        if sm70_tm.should_prepare_turbomind(
            layer.weight, envs.VLLM_SM70_NVFP4_TURBOMIND
        ):
            logger.info_once(
                "SM70 compressed-tensors NVFP4 TurboMind W4A16 dense path enabled."
            )
            is_qpn4_gate = _is_sm70_tp4_nvfp4_gate_up(layer)
            is_qpn4_down = _is_sm70_tp4_nvfp4_down(layer)
            qpn4_model_layer = envs.VLLM_SM70_NVFP4_QPN4 and (
                is_qpn4_down or (is_qpn4_gate and envs.VLLM_SM70_NVFP4_DENSE_GATED_SILU)
            )
            qpn4_runtime = (
                _is_sm70_nvfp4_qpn4_runtime_contract() if qpn4_model_layer else False
            )
            if qpn4_model_layer and not qpn4_runtime:
                logger.info_once(
                    "The SM70 NVFP4 QPN4 route retains TurboMind unless the "
                    "runtime contract is max_num_seqs=1 with no MTP."
                )
            if qpn4_model_layer and qpn4_runtime:
                missing_ops = _missing_sm70_nvfp4_qpn4_ops()
                if missing_ops:
                    logger.warning_once(
                        "The automatic SM70 NVFP4 QPN4 route is unavailable "
                        "in the loaded vllm._C; retaining TurboMind. Missing "
                        f"ops: {missing_ops}."
                    )
                workspace = (
                    None
                    if missing_ops
                    else sm70_tm.get_nvfp4_qpn4_dense_workspace(layer.weight)
                )
                if not missing_ops and workspace is not None:
                    sm70_tm.prepare_nvfp4_qpn4_linear(
                        layer,
                        workspace,
                        gated_silu=is_qpn4_gate,
                    )
                    layer.weight = Parameter(
                        torch.empty(0, dtype=torch.uint8, device=layer.weight.device),
                        requires_grad=False,
                    )
                    layer.weight_scale = Parameter(
                        torch.empty(
                            0,
                            dtype=torch.float8_e4m3fn,
                            device=layer.weight_scale.device,
                        ),
                        requires_grad=False,
                    )
                    logger.info_once(
                        "Memory-neutral SM70 NVFP4 QPN4 M=1 decode "
                        "path enabled with bounded FP16 prefill workspace."
                    )
                    return
                if not missing_ops:
                    logger.warning_once(
                        "Insufficient memory for the bounded SM70 NVFP4 QPN4 "
                        "prefill workspace; retaining TurboMind."
                    )
            use_qpn2 = bool(envs.VLLM_SM70_NVFP4_QPN2 and _is_qpn2_layer(layer))
            if use_qpn2:
                missing_ops = _missing_qpn2_ops()
                if missing_ops:
                    logger.warning_once(
                        "The requested SM70 NVFP4 QPN2 route is unavailable; "
                        f"retaining TurboMind. Missing ops: {missing_ops}."
                    )
                    use_qpn2 = False
            if use_qpn2:
                qpn2_codes, qpn2_scales = sm70_ops.nvfp4_qpn2_prepare_sm70(
                    layer.weight.data, layer.weight_scale.data
                )
                qpn2_global_scale = float(layer.weight_global_scale.item())
                qpn2_prefill_workspace = None
                if envs.VLLM_SM70_NVFP4_QPN2_PREFILL:
                    missing_prefill_ops = _missing_qpn2_prefill_ops()
                    if missing_prefill_ops:
                        logger.warning_once(
                            "The requested SM70 NVFP4 QPN2-packed prefill "
                            "route is unavailable; retaining TurboMind for "
                            f"large M. Missing ops: {missing_prefill_ops}."
                        )
                    else:
                        qpn2_prefill_workspace = sm70_tm.get_nvfp4_qpn4_dense_workspace(
                            layer.weight
                        )
                        if qpn2_prefill_workspace is None:
                            logger.warning_once(
                                "Insufficient memory for the bounded SM70 "
                                "NVFP4 QPN2-packed prefill workspace; "
                                "retaining TurboMind for large M."
                            )

            use_gated_silu = bool(
                envs.VLLM_SM70_NVFP4_DENSE_GATED_SILU and is_qpn4_gate and not use_qpn2
            )
            sm70_tm.prepare_nvfp4_linear(
                layer,
                interleave_gated_silu=use_gated_silu,
            )
            if use_qpn2:
                suffix = layer.prefix.rsplit(".", 1)[-1]
                k = layer.input_size_per_partition
                n = layer.output_size_per_partition
                split_k, nacc = _SM70_NVFP4_QPN2_CONFIGS[(k, n, False)]
                layer.register_buffer(
                    "sm70_nvfp4_qpn2_codes", qpn2_codes, persistent=False
                )
                layer.register_buffer(
                    "sm70_nvfp4_qpn2_scales", qpn2_scales, persistent=False
                )
                layer.sm70_nvfp4_qpn2 = True
                layer.sm70_nvfp4_qpn2_global_scale = qpn2_global_scale
                layer.sm70_nvfp4_qpn2_split_k = split_k
                layer.sm70_nvfp4_qpn2_nacc = nacc
                layer.sm70_nvfp4_qpn2_gated_silu = suffix == "gate_up_proj"
                layer.sm70_nvfp4_qpn2_prefill_dense_weight_ptr = (
                    0
                    if qpn2_prefill_workspace is None
                    else qpn2_prefill_workspace.data_ptr()
                )
                if qpn2_prefill_workspace is not None:
                    # QPN2 and QPN4 use the same physical tile order, but
                    # expose checkpoint-native [N, K/2] and GEMM-native
                    # [K, N/2] shapes respectively.  Keep zero-copy views so
                    # prefill does not retain a third multi-GB weight layout.
                    layer.sm70_nvfp4_qpn2_prefill_codes = qpn2_codes.view(k, n // 2)
                    layer.sm70_nvfp4_qpn2_prefill_scales = qpn2_scales.view(k // 16, n)
                logger.info_once(
                    "SM70 NVFP4 QPN2 M<=8 route enabled for a compatible "
                    "TP4 projection contract."
                )
                if qpn2_prefill_workspace is not None:
                    logger.info_once(
                        "SM70 NVFP4 QPN2-packed bounded FP16 prefill route "
                        "enabled for M>=%d.",
                        envs.VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M,
                    )
            elif use_gated_silu:
                logger.info_once(
                    "SM70 NVFP4 TurboMind gated-SiLU single-layout path enabled."
                )
            layer.weight = Parameter(
                torch.empty(0, dtype=torch.uint8, device=layer.weight.device),
                requires_grad=False,
            )
            layer.weight_scale = Parameter(
                torch.empty(
                    0, dtype=torch.float8_e4m3fn, device=layer.weight_scale.device
                ),
                requires_grad=False,
            )
            return

        # Convert layer to NVFP4 linear kernel format
        self._fallback_kernel().process_weights_after_loading(layer)

    def _fallback_kernel(self):
        if self.kernel is None:
            self.kernel = init_nvfp4_linear_kernel()
        return self.kernel

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "sm70_nvfp4_qpn2", False):
            return self._apply_qpn2(layer, x, bias, gated_silu=False)
        if sm70_tm.has_prepared_linear(layer):
            return sm70_tm.apply_prepared_linear(layer, x, bias)
        return self._fallback_kernel().apply_weights(layer=layer, x=x, bias=bias)

    def apply_fused_silu_and_mul(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        if getattr(layer, "sm70_nvfp4_qpn2_gated_silu", False):
            return self._apply_qpn2(layer, x, None, gated_silu=True)
        if not sm70_tm.has_prepared_linear(layer):
            return None
        return sm70_tm.apply_prepared_fused_silu_and_mul(layer, x)

    @staticmethod
    def _apply_qpn2(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        gated_silu: bool,
    ) -> torch.Tensor:
        if x.dtype != torch.float16:
            raise RuntimeError(
                f"SM70 NVFP4 QPN2 requires float16 activations, got {x.dtype}."
            )
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()
        output_size = layer.output_size_per_partition
        if gated_silu:
            output_size //= 2
        out_2d = torch.empty(
            (x_2d.shape[0], output_size), dtype=x.dtype, device=x.device
        )
        if x_2d.shape[0] == 0:
            return out_2d.reshape(*x.shape[:-1], output_size)
        prefill_dense_weight_ptr = int(
            getattr(layer, "sm70_nvfp4_qpn2_prefill_dense_weight_ptr", 0)
        )
        if (
            x_2d.shape[0] >= envs.VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M
            and prefill_dense_weight_ptr
        ):
            sm70_ops.nvfp4_qpn4_prefill_sm70_out(
                out_2d,
                prefill_dense_weight_ptr,
                x_2d,
                layer.sm70_nvfp4_qpn2_prefill_codes,
                layer.sm70_nvfp4_qpn2_prefill_scales,
                float(layer.sm70_nvfp4_qpn2_global_scale),
                True,
                gated_silu,
            )
            if bias is not None:
                out_2d.add_(bias)
            return out_2d.reshape(*x.shape[:-1], output_size)
        state = getattr(layer, sm70_tm.STATE_ATTR)
        split_k = int(layer.sm70_nvfp4_qpn2_split_k)
        nacc = int(layer.sm70_nvfp4_qpn2_nacc)
        if gated_silu:
            split_k, nacc = _SM70_NVFP4_QPN2_CONFIGS[
                (x_2d.shape[1], output_size * 2, True)
            ]
        sm70_ops.nvfp4_qpn2_dispatch_sm70_out(
            out_2d,
            x_2d,
            layer.sm70_nvfp4_qpn2_codes,
            layer.sm70_nvfp4_qpn2_scales,
            float(layer.sm70_nvfp4_qpn2_global_scale),
            split_k,
            nacc,
            state.weight,
            state.scales,
            state.group_size,
            state.k_ld,
            state.q_ld,
            gated_silu,
        )
        if bias is not None:
            out_2d.add_(bias)
        return out_2d.reshape(*x.shape[:-1], output_size)
