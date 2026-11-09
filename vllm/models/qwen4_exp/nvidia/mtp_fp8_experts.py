# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint and online weight-only FP8 for standalone SM70 MTP experts."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8_sm70_moe import Fp8SM70MoEMethod
from vllm.model_executor.utils import set_weight_attrs

from .mtp_fp8_checkpoint import padded_mtp_fp8_width


def checkpoint_fp8_prefixes(fallback, expert_prefixes: set[str]) -> set[str]:
    """Resolve supported checkpoint formats independently of target quantization."""
    if fallback.get_name() == "fp8" and fallback.is_checkpoint_fp8_serialized:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            is_layer_skipped,
        )

        selected = {
            prefix
            for prefix in expert_prefixes
            if not is_layer_skipped(
                prefix=prefix,
                ignored_layers=fallback.ignored_layers,
                fused_mapping=fallback.packed_modules_mapping,
                match_mode=fallback.ignored_layers_match_mode,
            )
        }
        if not selected:
            return set()
        if fallback.weight_block_size != [128, 128]:
            raise ValueError("MTP checkpoint FP8 requires 128x128 block scales")
        return selected
    result = set()
    if fallback.get_name() == "modelopt_mixed":
        from vllm.model_executor.layers.quantization.modelopt import (
            _BLOCK_FP8_MOE_ALGOS,
        )

        for prefix in expert_prefixes:
            if fallback.is_layer_excluded(prefix):
                continue
            if fallback._resolve_quant_algo(prefix) in _BLOCK_FP8_MOE_ALGOS:
                if fallback.fp8_block_config.weight_block_size != [128, 128]:
                    raise ValueError("MTP checkpoint FP8 requires 128x128 block scales")
                result.add(prefix)
    return result


class MTPCheckpointFp8SM70MoEMethod(Fp8SM70MoEMethod):
    """Load padded checkpoint FP8 blocks without an FP16 expert allocation."""

    def __init__(self, layer: RoutedExperts):
        super().__init__(
            SimpleNamespace(weight_block_size=[128, 128], activation_scheme="dynamic"),
            layer,
        )

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        if act_dtype != torch.float16 or hidden_size % 128:
            raise ValueError(
                "SM70 MTP FP8 requires FP16 and hidden size divisible by 128"
            )
        return hidden_size, padded_mtp_fp8_width(
            intermediate_size_per_partition, moe_parallel_config.tp_size
        )


def quantize_expert_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one already-sharded matrix; zero rows retain a finite scale.

    Channel scales are rounded to FP16 before quantizing because the SM70
    weight-only kernel consumes FP16 scales. No target-model tensor is modified.
    """
    if weight.ndim != 2 or weight.dtype != torch.float16:
        raise ValueError("MTP online FP8 requires a two-dimensional FP16 matrix")
    if not torch.isfinite(weight).all():
        raise ValueError("Cannot quantize non-finite MTP expert weights")
    limit = torch.finfo(torch.float8_e4m3fn).max
    scale = (
        (weight.float().abs().amax(dim=1, keepdim=True) / limit)
        .clamp_min(torch.finfo(torch.float16).tiny)
        .half()
        .float()
    )
    quantized = (weight.float() / scale).clamp(-limit, limit)
    return quantized.to(torch.float8_e4m3fn), scale


class MTPFp8SM70MoEMethod(Fp8SM70MoEMethod):
    """Load ordinary TP shards, then retain only packed FP8 expert weights."""

    def __init__(self, layer: RoutedExperts):
        super().__init__(
            SimpleNamespace(weight_block_size=[128, 128], activation_scheme="dynamic"),
            layer,
        )

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.float16 or not self.moe.is_act_and_mul:
            raise ValueError("SM70 MTP FP8 requires FP16 gated experts")
        if hidden_size % 128:
            raise ValueError("SM70 MTP FP8 requires a hidden size divisible by 128")
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        # The checkpoint is unquantized. The normal loader must slice its
        # weights in element units, including a TP4 intermediate width of 160.
        layer.weight_block_size = None
        for name, shape in (
            (
                "w13_weight",
                (num_experts, 2 * intermediate_size_per_partition, hidden_size),
            ),
            ("w2_weight", (num_experts, hidden_size, intermediate_size_per_partition)),
        ):
            param = nn.Parameter(
                torch.empty(shape, dtype=params_dtype), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
        layer.w13_input_scale = layer.w2_input_scale = None

    def process_weights_after_loading(self, layer):
        # Work one expert at a time, rather than creating a full FP32 MoE copy.
        # Channelwise scales avoid crossing gate/up or TP shard boundaries.
        for name in ("w13", "w2"):
            source = getattr(layer, name + "_weight")
            padded_shape = pad_expert_matrix(source[0], name == "w13").shape
            quantized = torch.empty(
                (source.shape[0], *padded_shape),
                dtype=torch.float8_e4m3fn,
                device=source.device,
            )
            scales = torch.empty(
                (source.shape[0], padded_shape[0], 1),
                dtype=torch.float32,
                device=source.device,
            )
            for expert in range(source.shape[0]):
                padded = pad_expert_matrix(source[expert], name == "w13")
                quantized[expert], scales[expert] = quantize_expert_rows(padded)
            setattr(
                layer, name + "_weight", nn.Parameter(quantized, requires_grad=False)
            )
            layer.register_parameter(
                name + "_weight_scale_inv", nn.Parameter(scales, requires_grad=False)
            )
        # The existing packer accepts channel scales and the existing method
        # discards the unpacked weights after preparing the runtime layout.
        super().process_weights_after_loading(layer)


def pad_expert_matrix(weight: torch.Tensor, gate_up: bool) -> torch.Tensor:
    """Keep gate/up halves aligned with the zero-padded down-projection input."""
    intermediate = weight.shape[0] // 2 if gate_up else weight.shape[1]
    padded = (intermediate + 127) // 128 * 128
    if padded == intermediate:
        return weight
    if gate_up:
        result = weight.new_zeros((2 * padded, weight.shape[1]))
        result[:intermediate].copy_(weight[:intermediate])
        result[padded : padded + intermediate].copy_(weight[intermediate:])
    else:
        result = weight.new_zeros((weight.shape[0], padded))
        result[:, :intermediate].copy_(weight)
    return result


class MTPExpertFp8Config(QuantizationConfig):
    """Override only excluded, unquantized MTP experts in a draft config."""

    def __init__(
        self,
        fallback: QuantizationConfig,
        checkpoint_prefixes: set[str] | None = None,
        quantize_unquantized: bool = True,
    ):
        super().__init__()
        self.fallback = fallback
        self.packed_modules_mapping = fallback.packed_modules_mapping
        self.checkpoint_prefixes = checkpoint_prefixes or set()
        self.quantize_unquantized = quantize_unquantized

    def get_name(self):
        return self.fallback.get_name()

    def get_supported_act_dtypes(self):
        return [torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 70

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError("Construct from the resolved MTP draft config")

    def get_cache_scale(self, name):
        return self.fallback.get_cache_scale(name)

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts) and prefix in self.checkpoint_prefixes:
            return MTPCheckpointFp8SM70MoEMethod(layer)
        method = self.fallback.get_quant_method(layer, prefix)
        if (
            self.quantize_unquantized
            and isinstance(layer, RoutedExperts)
            and prefix.startswith("mtp.layers.")
        ):
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptMixedPrecisionConfig,
            )

            excluded = getattr(self.fallback, "is_layer_excluded", lambda _: False)
            # Mixed checkpoints also leave layers unquantized by omitting them
            # from quantized_layers; an explicit exclusion is not required.
            absent_from_mixed = (
                isinstance(self.fallback, ModelOptMixedPrecisionConfig)
                and self.fallback._resolve_quant_algo(prefix) is None
            )
            if not isinstance(method, UnquantizedFusedMoEMethod) and not (
                method is None and (excluded(prefix) or absent_from_mixed)
            ):
                raise ValueError(
                    "Online MTP FP8 requires unquantized checkpoint experts"
                )
            return MTPFp8SM70MoEMethod(layer)
        return method
