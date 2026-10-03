# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical GGUF affine weights in the shared mixed-precision lifecycle."""

from dataclasses import dataclass

import torch

from vllm.model_executor.kernels.gguf import GGUFDecoderFamily, GGUFOperatorCapability
from vllm.model_executor.layers.quantization.gguf_transcode import AFFINE_GROUP32_TYPES
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.gguf_tensor_reader import quant_type_name

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


@dataclass
class Sm70GgufAffineConfig(MPLinearLayerConfig):
    source_type: int = 0
    enabled: bool = True


class TurboMindGgufAffineKernel(MPLinearKernel):
    """Only native FP16 activation/MMA compute; no activation quantization."""

    config: Sm70GgufAffineConfig

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls, config: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not isinstance(config, Sm70GgufAffineConfig):
            return False, "requires_canonical_gguf_affine_storage"
        if not config.enabled:
            return False, "disabled_by_kernel_config"
        if config.source_type not in AFFINE_GROUP32_TYPES:
            return False, "source_format_codec_unavailable"
        if config.act_type != torch.float16:
            return False, "requires_fp16_activations"
        if config.weight_type not in (scalar_types.uint4, scalar_types.uint8):
            return False, "canonical_integer_width_unavailable"
        if config.group_size != 32 or config.has_g_idx:
            return False, "requires_group32_without_activation_order"
        k, n = config.partition_weight_shape
        if k <= 0 or n <= 0 or k % 32 or n % 32:
            return False, "local_shape_cuts_canonical_group_or_output_pack"
        for name in ("gguf_affine_sm70_prepare", "gguf_affine_gemm_sm70_out"):
            if not hasattr(torch.ops._C, name):
                return False, f"operator_missing:{name}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_gguf_tm_affine_prepared", False):
            return
        codes, scales, mins, _ = self._get_weight_params(layer)
        assert mins is not None
        bits = 4 if self.config.weight_type == scalar_types.uint4 else 8
        weight, stats, meta = torch.ops._C.gguf_affine_sm70_prepare(
            codes, scales, mins, bits
        )
        replace_parameter(layer, self.w_q_name, weight)
        replace_parameter(layer, self.w_s_name, stats)
        if self.w_zp_name is not None:
            layer.register_parameter(self.w_zp_name, None)
        layer.gguf_tm_k_ld, layer.gguf_tm_q_ld = meta.tolist()
        self.bits = bits
        self.capability = GGUFOperatorCapability(
            GGUFDecoderFamily.AFFINE,
            quant_type_name(self.config.source_type),
            "gguf_affine_gemm_sm70_out",
            True,
        )
        layer._gguf_tm_affine_prepared = True

    def apply_weights(self, layer, x, bias=None):
        if not self.capability.supports_m(x.numel() // x.shape[-1]):
            raise ValueError("M outside the GGUF affine operator capability")
        n = self.config.partition_weight_shape[1]
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        output = torch.empty((rows.shape[0], n), dtype=x.dtype, device=x.device)
        torch.ops._C.gguf_affine_gemm_sm70_out(
            output,
            rows,
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            self.bits,
            layer.gguf_tm_k_ld,
            layer.gguf_tm_q_ld,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(*x.shape[:-1], n)
