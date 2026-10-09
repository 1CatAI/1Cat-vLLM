# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboMind AWQ in the common mixed-precision linear kernel lifecycle."""

import weakref
from dataclasses import dataclass, field

import torch

from vllm import _sm70_ops as sm70_ops
from vllm.config.kernel import Sm70AwqConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import sm70_turbomind as tm
from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
    workspace_pool,
)
from vllm.model_executor.models.config import sm70_awq_prefill_projection_qualified
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)


@dataclass
class Sm70AwqLinearLayerConfig(MPLinearLayerConfig):
    # Legacy AWQ GEMM checkpoint packing differs from the GPTQ layout used by
    # other MP kernels. They cannot consume these tensors without conversion.
    policy: Sm70AwqConfig = field(default_factory=Sm70AwqConfig)
    gated_silu: bool = False


_SM70_AWQ_PREFILL_DENSE_M = 4096
_SM70_AWQ_PREFILL_DENSE_SHAPES = {
    "gate_up_proj": (5120, 8704),
    "down_proj": (4352, 5120),
    "in_proj_qkvz": (5120, 4096),
    "out_proj": (1536, 5120),
    "o_proj": (1536, 5120),
}
_SM70_AWQ_PREFILL_DENSE_WORKSPACE_ELEMENTS = max(
    k * n for k, n in _SM70_AWQ_PREFILL_DENSE_SHAPES.values()
)
_SM70_AWQ_PREFILL_DENSE_WORKSPACE_BYTES = (
    _SM70_AWQ_PREFILL_DENSE_WORKSPACE_ELEMENTS * torch.float16.itemsize
)
_sm70_awq_prefill_dense_workspaces: weakref.WeakValueDictionary[
    tuple[int, torch.dtype, int], torch.Tensor
] = weakref.WeakValueDictionary()


def _is_sm70_awq_prefill_exact_dense_layer(layer: torch.nn.Module) -> bool:
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    if (
        not sm70_awq_prefill_projection_qualified(suffix)
        or len(layer.qweight.shape) != 2
    ):
        return False
    k, packed_n = layer.qweight.shape
    return k > 0 and k % 128 == 0 and packed_n > 0 and packed_n % 16 == 0


def _get_sm70_awq_prefill_exact_dense_workspace(
    weight: torch.Tensor,
) -> torch.Tensor | None:
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    elements = max(_SM70_AWQ_PREFILL_DENSE_WORKSPACE_ELEMENTS, weight.numel() * 8)
    cache_key = (device_index, torch.float16, elements)
    workspace = workspace_pool("awq_prefill", _sm70_awq_prefill_dense_workspaces).get(
        cache_key
    )
    if workspace is not None:
        return workspace
    try:
        workspace = torch.empty(
            (elements,),
            dtype=torch.float16,
            device=weight.device,
        )
    except torch.OutOfMemoryError:
        logger.warning_once(
            "Insufficient memory for the bounded SM70 AWQ prefill workspace; "
            "falling back to TurboMind AWQ."
        )
        return None
    workspace_pool("awq_prefill", _sm70_awq_prefill_dense_workspaces)[cache_key] = (
        workspace
    )
    return workspace


class TurboMindAwqLinearKernel(MPLinearKernel):
    config: Sm70AwqLinearLayerConfig

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls, config: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not isinstance(config, Sm70AwqLinearLayerConfig):
            return False, "requires AWQ GEMM checkpoint packing"
        if config.weight_type != scalar_types.uint4 or not config.zero_points:
            return False, "requires asymmetric uint4 AWQ weights"
        if config.act_type != torch.float16:
            return False, "requires float16 activations"
        if config.group_size not in (32, 64, 128):
            return False, "requires group_size 32/64/128"
        k, n = config.partition_weight_shape
        if k <= 0 or n <= 0 or k % config.group_size or n % 8:
            return False, "requires positive K/N, complete groups and uint4 packing"
        if config.has_g_idx:
            return False, "does not support activation-order group indices"
        if not hasattr(torch.ops._C, "awq_sm70_prepare"):
            return False, "native awq_sm70_prepare is unavailable"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_awq_sm70_prepared", False):
            return

        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        group_size = self.config.group_size

        is_gated_silu_layer = self.config.gated_silu
        use_gated_silu = is_gated_silu_layer and bool(self.config.policy.fused_silu)

        use_prefill_exact_dense = (
            self.config.policy.prefill_exact_dense
            and group_size == 128
            and _is_sm70_awq_prefill_exact_dense_layer(layer)
            and not use_gated_silu
            and hasattr(torch.ops._C, "awq_sm70_dequantize_out")
        )

        tm_weight, tm_scales, meta = sm70_ops.awq_sm70_prepare(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            group_size,
            use_gated_silu,
        )
        layer._awq_sm70_weight = tm_weight
        layer._awq_sm70_scales = tm_scales
        layer._awq_sm70_k_ld = int(meta[0])
        layer._awq_sm70_q_ld = int(meta[1])
        layer._awq_sm70_group_size = group_size
        state = tm.SM70TurboMindLinearState(
            tm_weight,
            tm_scales,
            group_size,
            int(meta[0]),
            int(meta[1]),
            tm_weight.shape[-1] * 8,
            "uint4",
            gated_silu=use_gated_silu,
        )
        setattr(layer, tm.STATE_ATTR, state)
        self.native_ops = state.native_ops
        layer._awq_sm70_prepared = True
        if use_gated_silu:
            layer._awq_sm70_gated_silu = True
            layer._awq_sm70_gated_silu_primary = True
            logger.info_once(
                "SM70 AWQ dense MLP gated-SiLU single-layout path enabled."
            )

        # The runtime path consumes only the TurboMind-packed tensors above.
        # Releasing the original AWQ tensors matches the 0.0.3 SM70 path and
        # avoids carrying duplicate quantized weights in long-context runs.
        layer.qweight = torch.nn.Parameter(
            torch.empty(0, dtype=torch.int32, device=tm_weight.device),
            requires_grad=False,
        )
        layer.qzeros = torch.nn.Parameter(
            torch.empty(0, dtype=torch.int32, device=tm_weight.device),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(
            torch.empty(0, dtype=tm_scales.dtype, device=tm_weight.device),
            requires_grad=False,
        )
        if use_prefill_exact_dense:
            workspace = _get_sm70_awq_prefill_exact_dense_workspace(tm_weight)
            if workspace is not None:
                layer._awq_sm70_prefill_exact_dense_workspace = workspace
                logger.info_once(
                    "SM70 AWQ exact-dense prefill path enabled with a bounded "
                    "layout-sized workspace."
                )
        logger.info_once("SM70 AWQ TurboMind dense path enabled.")

    def apply_fused_silu_and_mul(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.config.policy.fused_silu:
            return None
        if not getattr(layer, "_awq_sm70_gated_silu", False):
            return None
        if not getattr(layer, "_awq_sm70_prepared", False):
            return None
        if getattr(layer, "tp_size", 1) != 2:
            return None

        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.shape[0] != 1:
            return None
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()

        out_features = layer.output_size_per_partition // 2
        out_2d = torch.empty(
            (x_2d.shape[0], out_features),
            dtype=x.dtype,
            device=x.device,
        )
        self.native_ops.awq_gemm_sm70_out(
            out_2d,
            x_2d,
            layer._awq_sm70_weight,
            layer._awq_sm70_scales,
            layer._awq_sm70_group_size,
            layer._awq_sm70_k_ld,
            layer._awq_sm70_q_ld,
            True,
        )
        return out_2d.reshape(*x.shape[:-1], out_features)

    def apply_weights(self, layer, x, bias=None):
        reshaped_x = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (layer._awq_sm70_weight.shape[-1] * 8,)
        prefill_workspace = getattr(
            layer, "_awq_sm70_prefill_exact_dense_workspace", None
        )
        if (
            prefill_workspace is not None
            and reshaped_x.dtype == torch.float16
            and reshaped_x.shape[0] == _SM70_AWQ_PREFILL_DENSE_M
        ):
            logger.info_once(
                "SM70 AWQ bounded-workspace exact-dense 4096-token "
                "prefill runtime path active."
            )
            k = reshaped_x.shape[1]
            n = out_shape[-1]
            prefill_weight = prefill_workspace[: k * n].view(k, n)
            self.native_ops.awq_sm70_dequantize_out(
                prefill_weight,
                layer._awq_sm70_weight,
                layer._awq_sm70_scales,
                layer._awq_sm70_group_size,
            )
            out = torch.mm(reshaped_x, prefill_weight)
            if bias is not None:
                out.add_(bias)
            return out.reshape(out_shape)
        return tm.apply_prepared_linear(layer, x, bias)
