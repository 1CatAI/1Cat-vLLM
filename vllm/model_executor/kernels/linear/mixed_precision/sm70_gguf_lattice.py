# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical GGUF LATTICE weights in the shared kernel lifecycle."""

from dataclasses import dataclass

import torch

from vllm.model_executor.kernels.gguf import GGUFDecoderFamily, GGUFOperatorCapability
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import LATTICE_TYPES
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.gguf_tensor_reader import quant_type_name

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig
from .sm70_gguf import _get_affine_blas_workspace

# Matched Flash-Next TP4 sweeps; unmeasured descriptors retain fused MMA.
# The narrow expert crossover is nonmonotonic, so it has two intervals.
_LATTICE_BLAS_BANDS = {
    (17, 2560, 160): ((8, 1024), (4096, None)),
    (18, 2560, 160): ((8, 1024), (4096, None)),
    (21, 2560, 1536): ((2048, None),),
}


@dataclass
class Sm70GgufLatticeConfig(MPLinearLayerConfig):
    source_type: int = 18
    enabled: bool = True


class TurboMindGgufLatticeKernel(MPLinearKernel):
    config: Sm70GgufLatticeConfig

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls, config: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not isinstance(config, Sm70GgufLatticeConfig):
            return False, "requires_canonical_gguf_lattice_storage"
        if not config.enabled:
            return False, "disabled_by_kernel_config"
        if config.source_type not in LATTICE_TYPES:
            return False, "source_format_codec_unavailable"
        if config.act_type != torch.float16:
            return False, "requires_fp16_activations"
        expected_carrier = (
            scalar_types.uint4 if config.source_type in (18, 21) else scalar_types.uint2
        )
        if config.weight_type != expected_carrier:
            return False, "requires_canonical_lattice_carriers"
        expected_group = 16 if config.source_type in (17, 22, 29) else 32
        if config.group_size != expected_group or config.has_g_idx:
            return False, "canonical_group_or_activation_order_not_supported"
        k, n = config.partition_weight_shape
        if k <= 0 or n <= 0 or k % config.group_size or n % 32:
            return False, "local_shape_cuts_canonical_group_or_output_pack"
        for name in ("gguf_lattice_sm70_prepare", "gguf_lattice_gemm_sm70_out"):
            if not hasattr(torch.ops._C, name):
                return False, f"operator_missing:{name}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_gguf_tm_lattice_prepared", False):
            return
        codes, scales, _, _ = self._get_weight_params(layer)
        self.source_type = self.config.source_type
        weight, stats, meta = torch.ops._C.gguf_lattice_sm70_prepare(
            codes, scales, self.source_type, self.config.group_size
        )
        replace_parameter(layer, self.w_q_name, weight)
        replace_parameter(layer, self.w_s_name, stats)
        layer.gguf_tm_k_ld, layer.gguf_tm_q_ld = meta.tolist()
        self.capability = GGUFOperatorCapability(
            GGUFDecoderFamily.LATTICE,
            quant_type_name(self.config.source_type),
            "gguf_lattice_gemm_sm70_out",
            True,
        )
        k, n = self.config.partition_weight_shape
        bands = _LATTICE_BLAS_BANDS.get((self.source_type, k, n))
        reason = None
        if not hasattr(torch.ops._C, "gguf_lattice_blas_sm70_out"):
            reason = "operator_missing:gguf_lattice_blas_sm70_out"
        elif bands is None:
            reason = "local_shape_has_no_prefill_calibration"
        else:
            workspace = _get_affine_blas_workspace(weight)
            if workspace is None:
                reason = "prefill_workspace_allocation_failed"
            else:
                layer.gguf_tm_blas_workspace = workspace[: k * n].view(k, n)
        self.prefill_capabilities = tuple(
            GGUFOperatorCapability(
                GGUFDecoderFamily.LATTICE,
                quant_type_name(self.source_type),
                "gguf_lattice_blas_sm70_out",
                True,
                min_m=minimum,
                max_m=maximum,
                reason=reason,
            )
            for minimum, maximum in (bands or ((128, None),))
        )
        self.operator_capabilities = (self.capability, *self.prefill_capabilities)
        layer._gguf_tm_lattice_prepared = True

    def apply_weights(self, layer, x, bias=None):
        if not self.capability.supports_m(x.numel() // x.shape[-1]):
            raise ValueError("M outside the GGUF LATTICE operator capability")
        n = self.config.partition_weight_shape[1]
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        output = torch.empty((rows.shape[0], n), dtype=x.dtype, device=x.device)
        if any(
            capability.reason is None and capability.supports_m(rows.shape[0])
            for capability in self.prefill_capabilities
        ):
            torch.ops._C.gguf_lattice_blas_sm70_out(
                output,
                rows,
                getattr(layer, self.w_q_name),
                getattr(layer, self.w_s_name),
                self.source_type,
                layer.gguf_tm_blas_workspace,
                self.config.group_size,
            )
        else:
            torch.ops._C.gguf_lattice_gemm_sm70_out(
                output,
                rows,
                getattr(layer, self.w_q_name),
                getattr(layer, self.w_s_name),
                self.source_type,
                layer.gguf_tm_k_ld,
                layer.gguf_tm_q_ld,
                self.config.group_size,
            )
        if bias is not None:
            output.add_(bias)
        return output.reshape(*x.shape[:-1], n)
