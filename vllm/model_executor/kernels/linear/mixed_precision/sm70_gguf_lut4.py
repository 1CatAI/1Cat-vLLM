# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical GGUF LUT4 weights in the shared kernel lifecycle."""

from dataclasses import dataclass

import torch

from vllm._sm70.policy import NativeBindings
from vllm.config.sm70_native import capture_linear_native_config
from vllm.model_executor.kernels.gguf import GGUFDecoderFamily, GGUFOperatorCapability
from vllm.model_executor.layers.quantization.gguf_lut_transcode import LUT4_TYPES
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.gguf_tensor_reader import quant_type_name

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


@dataclass
class Sm70GgufLut4Config(MPLinearLayerConfig):
    source_type: int = 20
    enabled: bool = True


class TurboMindGgufLut4Kernel(MPLinearKernel):
    config: Sm70GgufLut4Config

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls, config: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not isinstance(config, Sm70GgufLut4Config):
            return False, "requires_canonical_gguf_lut4_storage"
        if not config.enabled:
            return False, "disabled_by_kernel_config"
        if config.source_type not in LUT4_TYPES:
            return False, "source_format_codec_unavailable"
        if config.act_type != torch.float16:
            return False, "requires_fp16_activations"
        if config.weight_type != scalar_types.uint4:
            return False, "requires_preserved_nibble_indices"
        expected_group = 16 if config.source_type == 40 else 32
        if config.group_size != expected_group or config.has_g_idx:
            return False, "canonical_group_or_activation_order_not_supported"
        k, n = config.partition_weight_shape
        if k <= 0 or n <= 0 or k % config.group_size or n % 32:
            return False, "local_shape_cuts_canonical_group_or_output_pack"
        for name in ("gguf_lut4_sm70_prepare", "gguf_lut4_gemm_sm70_out"):
            if not hasattr(torch.ops._C, name):
                return False, f"operator_missing:{name}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.native_ops = NativeBindings(capture_linear_native_config("gguf").values)
        if getattr(layer, "_gguf_tm_lut4_prepared", False):
            return
        codes, scales, _, _ = self._get_weight_params(layer)
        self.lut_id = 0 if self.config.source_type in (20, 23) else 1
        weight, stats, meta = torch.ops._C.gguf_lut4_sm70_prepare(
            codes, scales, self.lut_id, self.config.group_size
        )
        replace_parameter(layer, self.w_q_name, weight)
        replace_parameter(layer, self.w_s_name, stats)
        layer.gguf_tm_k_ld, layer.gguf_tm_q_ld = meta.tolist()
        self.capability = GGUFOperatorCapability(
            GGUFDecoderFamily.LUT4,
            quant_type_name(self.config.source_type),
            "gguf_lut4_gemm_sm70_out",
            True,
        )
        layer._gguf_tm_lut4_prepared = True

    def apply_weights(self, layer, x, bias=None):
        if not self.capability.supports_m(x.numel() // x.shape[-1]):
            raise ValueError("M outside the GGUF LUT4 operator capability")
        n = self.config.partition_weight_shape[1]
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        output = torch.empty((rows.shape[0], n), dtype=x.dtype, device=x.device)
        self.native_ops.gguf_lut4_gemm_sm70_out(
            output,
            rows,
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            self.lut_id,
            layer.gguf_tm_k_ld,
            layer.gguf_tm_q_ld,
            self.config.group_size,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(*x.shape[:-1], n)
