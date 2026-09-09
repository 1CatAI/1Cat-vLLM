# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ComfyUI tensor-wise INT8 ConvRot checkpoint support.

The on-disk format stores an already rotated, row-wise quantized weight and a
per-output-row scale.  At runtime the activation must receive the matching
Hadamard rotation before FP16 matrix multiplication; treating these weights as
ordinary INT8 silently produces incorrect output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import Module

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)

from .ops import convrot_reference, dequantize_int8_reference


def create_weight_parameter(
    output_size_per_partition, input_size_per_partition, weight_loader, params_dtype
):
    return ModelWeightParameter(
        data=torch.empty(
            output_size_per_partition, input_size_per_partition, dtype=params_dtype
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )


logger = init_logger(__name__)

_FORMAT = "int8_tensorwise"


def fp16_gemm_input(x):
    """Scale wide-range activations by exact powers of two before FP16 GEMM.

    Leave headroom for the 256-channel rotation's worst-case amplification.
    Row scaling is restored in FP32 after the projection.
    """
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    if flat.dtype == torch.float16:
        return flat, None
    if flat.dtype != torch.float32:
        raise ValueError("H3 GEMM activations must be FP16 or FP32")
    if flat.is_cuda:
        from .cuda_ops import w8a16_extension

        return w8a16_extension().prepare_fp16(flat)
    maximum = flat.abs().amax(-1, keepdim=True)
    _, exponent = torch.frexp(maximum)
    scale = torch.ldexp(torch.ones_like(maximum), (exponent - 11).clamp_min(0))
    return (flat / scale).half(), scale


class FP32OutputLinearMethod(UnquantizedLinearMethod):
    """Keep wide-range projection outputs in FP32 with FP16 Tensor Core inputs."""

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.contiguous()

    def apply(self, layer, x, bias=None):
        if x.is_cuda:
            from .cuda_ops import w8a16_extension

            values, scale = fp16_gemm_input(x)
            output = w8a16_extension().gemm(values, layer.weight, True)
            if scale is not None:
                output = output * scale
            output = output.reshape(*x.shape[:-1], layer.weight.shape[0])
        else:
            output = torch.nn.functional.linear(x.float(), layer.weight.float())
        return output if bias is None else output + bias.float()


def preserve_fp32_output(layer):
    layer.h3_output_fp32 = True
    if isinstance(layer.quant_method, UnquantizedLinearMethod):
        layer.quant_method = FP32OutputLinearMethod()


@dataclass(frozen=True)
class Int8ConvRotLayerConfig:
    """Validated per-linear metadata decoded from ``.comfy_quant``."""

    convrot: bool
    convrot_groupsize: int = 256

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Int8ConvRotLayerConfig:
        fmt = value.get("format")
        if fmt != _FORMAT:
            raise ValueError(
                f"Unsupported Comfy quant format {fmt!r}; expected {_FORMAT!r}."
            )
        convrot = value.get("convrot", False)
        if not isinstance(convrot, bool):
            raise ValueError(f"Comfy ConvRot flag must be boolean, got {convrot!r}.")
        group_size = value.get("convrot_groupsize", 256)
        if not isinstance(group_size, int) or group_size < 4:
            raise ValueError(
                f"Comfy ConvRot group size must be an integer >= 4, got {group_size!r}."
            )
        # comfy-kitchen's regular Hadamard is defined for powers of four.
        value_left = group_size
        while value_left > 1 and value_left % 4 == 0:
            value_left //= 4
        if value_left != 1:
            raise ValueError(
                f"Comfy ConvRot group size must be a power of four, got {group_size}."
            )
        return cls(convrot=convrot, convrot_groupsize=group_size)


class DiffusionInt8ConvRotConfig(QuantizationConfig):
    """Offline ComfyUI W8A16 ConvRot configuration.

    Quantized layers are explicit because a Comfy checkpoint can mix INT8
    linears with BF16/FP16 linears.  The MiniMax-H3 pipeline fills this mapping
    from each layer's ``.comfy_quant`` tensor before constructing the model.
    """

    def __init__(
        self,
        layer_configs: Mapping[str, Mapping[str, Any] | Int8ConvRotLayerConfig]
        | None = None,
        quantized_layers: list[str] | None = None,
        convrot_groupsize: int = 256,
        ignored_layers: list[str] | None = None,
        weight_layout: str = "column",
    ) -> None:
        super().__init__()
        if weight_layout not in ("row", "column"):
            raise ValueError("H3 INT8 weight layout must be row or column")
        self.weight_layout = weight_layout
        self.ignored_layers = ignored_layers or []
        self.layer_configs: dict[str, Int8ConvRotLayerConfig] = {}
        self.is_checkpoint_quantized = True
        self.is_checkpoint_int8_convrot_serialized = True

        if layer_configs:
            self.configure_layers(layer_configs)
        if quantized_layers:
            default = Int8ConvRotLayerConfig.from_mapping(
                {
                    "format": _FORMAT,
                    "convrot": True,
                    "convrot_groupsize": convrot_groupsize,
                }
            )
            for prefix in quantized_layers:
                self.layer_configs.setdefault(prefix, default)
        self._validate_checkpoint_layers_are_quantized(self.layer_configs)

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "int8_convrot"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionInt8ConvRotConfig:
        return cls(
            layer_configs=config.get("layer_configs"),
            quantized_layers=config.get("quantized_layers"),
            convrot_groupsize=config.get("convrot_groupsize", 256),
            ignored_layers=config.get("ignored_layers"),
            weight_layout=config.get("weight_layout", "column"),
        )

    def configure_layers(
        self,
        layer_configs: Mapping[str, Mapping[str, Any] | Int8ConvRotLayerConfig],
    ) -> None:
        """Install checkpoint-derived metadata before model construction."""
        parsed: dict[str, Int8ConvRotLayerConfig] = {}
        for prefix, value in layer_configs.items():
            if not prefix or prefix.endswith(
                (".weight", ".weight_scale", ".comfy_quant")
            ):
                raise ValueError(
                    f"ConvRot layer metadata must use a module prefix, got {prefix!r}."
                )
            parsed[prefix] = (
                value
                if isinstance(value, Int8ConvRotLayerConfig)
                else Int8ConvRotLayerConfig.from_mapping(value)
            )
        self._validate_checkpoint_layers_are_quantized(parsed)
        if self.layer_configs and self.layer_configs != parsed:
            raise ValueError(
                "ConvRot layer metadata was already configured with different "
                "checkpoint values."
            )
        self.layer_configs = parsed

    def _validate_checkpoint_layers_are_quantized(
        self,
        layer_configs: Mapping[str, Int8ConvRotLayerConfig],
    ) -> None:
        conflicts = sorted(
            prefix
            for prefix in layer_configs
            if is_layer_skipped(prefix, self.ignored_layers)
        )
        if conflicts:
            raise ValueError(
                "Checkpoint-marked INT8 ConvRot layers cannot also be ignored: "
                f"{conflicts[:5]}. Remove them from ignored_layers."
            )

    def validate_model_bindings(self, model: Module) -> None:
        """Require every checkpoint marker to bind an executable ConvRot layer."""
        expected = set(self.layer_configs)
        bound = {
            method.prefix
            for module in model.modules()
            if isinstance(
                method := getattr(module, "quant_method", None),
                Int8ConvRotLinearMethod,
            )
            and method.quant_config is self
        }
        if bound != expected:
            missing = sorted(expected - bound)
            unexpected = sorted(bound - expected)
            raise ValueError(
                "MiniMax-H3 ConvRot checkpoint metadata does not match executable "
                "quantized layers: "
                f"unbound markers={missing[:5]}, unexpected bindings={unexpected[:5]}."
            )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(prefix, self.ignored_layers):
            return UnquantizedLinearMethod()
        layer_config = self.layer_configs.get(prefix)
        if layer_config is None:
            return UnquantizedLinearMethod()
        return Int8ConvRotLinearMethod(self, layer_config, prefix=prefix)


class Int8ConvRotLinearMethod(LinearMethodBase):
    """Execute signed INT8 weights after FP16 dequantization."""

    def __init__(
        self,
        quant_config: DiffusionInt8ConvRotConfig,
        layer_config: Int8ConvRotLayerConfig,
        *,
        prefix: str,
    ) -> None:
        self.quant_config = quant_config
        self.layer_config = layer_config
        self.prefix = prefix
        self._cuda_impl: Callable[..., torch.Tensor] | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.layer_config.convrot_groupsize
        if self.layer_config.convrot and input_size_per_partition % group_size:
            raise ValueError(
                f"{self.prefix} has TP-local input width {input_size_per_partition}, "
                f"which is not aligned to its ConvRot group size {group_size}. "
                "Choose a tensor-parallel degree whose row-parallel shards preserve "
                "group boundaries."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = create_weight_parameter(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            weight_loader=weight_loader,
            params_dtype=torch.int8,
        )
        layer.register_parameter("weight", weight)
        scale = ChannelQuantScaleParameter(
            data=torch.empty((output_size_per_partition, 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        if layer.weight.dtype != torch.int8 or layer.weight.dim() != 2:
            raise ValueError(
                f"{self.prefix} expected a 2-D INT8 weight, got "
                f"{tuple(layer.weight.shape)} {layer.weight.dtype}."
            )
        scale = layer.weight_scale
        if scale.dtype != torch.float32 or scale.numel() != layer.weight.shape[0]:
            raise ValueError(
                f"{self.prefix} expected one FP32 scale per output row; got "
                f"{tuple(scale.shape)} {scale.dtype} for weight "
                f"{tuple(layer.weight.shape)}."
            )
        if not torch.isfinite(scale).all() or not torch.all(scale > 0):
            raise ValueError(
                f"{self.prefix} contains non-finite or non-positive INT8 weight scales."
            )
        # Preserve logical [N,K] coordinates and every signed INT8 byte. The
        # CPU stager keeps strides, so this costs no additional GPU residency.
        # Limit the new layout to DiT projections validated with TP4.
        if self.quant_config.weight_layout == "column" and self.prefix.startswith(
            "blocks."
        ):
            layer.weight.data = layer.weight.data.t().contiguous().t()
        else:
            layer.weight.data = layer.weight.data.contiguous()
        layer.weight_scale.data = scale.data.reshape(-1).contiguous()

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.dtype not in (torch.float16, torch.float32):
            raise ValueError("H3 W8A16 requires FP16 or FP32 activations")
        if x.is_cuda:
            shape = x.shape
            x, scale = fp16_gemm_input(x)
            output = self.apply_prepared(layer, x, scale)
            output = output.reshape(*shape[:-1], layer.weight.shape[0])
            return output if bias is None else output + bias
        original_shape = x.shape
        x, scale = fp16_gemm_input(x)
        if self.layer_config.convrot:
            x = convrot_reference(x, self.layer_config.convrot_groupsize)
        weight = dequantize_int8_reference(layer.weight, layer.weight_scale)
        if getattr(layer, "h3_output_fp32", False) or scale is not None:
            output = torch.nn.functional.linear(x.float(), weight.float())
            if scale is not None:
                output = output * scale
            if bias is not None:
                output = output + bias.float()
        else:
            output = torch.nn.functional.linear(x, weight, bias)
        return output.reshape(*original_shape[:-1], layer.weight.shape[0])

    def apply_prepared(self, layer, values, scale, *, input_is_rotated=False):
        """Project FP16 rows with an explicit scale restored before TP reduction."""
        from .cuda_ops import fp16_gemm, w8a16_extension

        ops = w8a16_extension()
        x = values.reshape(-1, values.shape[-1])
        if self.layer_config.convrot and not input_is_rotated:
            if self.layer_config.convrot_groupsize != 256:
                raise ValueError("H3 SM70 ConvRot implements 256-channel groups")
            x = ops.rotate(x)
        weight = getattr(layer, "h3_fp16_weight", None)
        if weight is None:
            weight = ops.dequantize(layer.weight, layer.weight_scale)
        output = fp16_gemm(
            x, weight, getattr(layer, "h3_output_fp32", False) or scale is not None
        )
        if scale is not None:
            output = output * scale
        return output.reshape(*values.shape[:-1], layer.weight.shape[0])


def rotate_local_fp16(layer, values):
    """Rotate local residual rows before their bit-preserving TP all-gather."""
    method = layer.quant_method
    if (
        values.is_cuda
        and values.dtype == torch.float16
        and isinstance(method, Int8ConvRotLinearMethod)
        and method.layer_config.convrot
        and method.layer_config.convrot_groupsize == 256
        and layer.bias is None
        and not layer.gather_output
    ):
        from .cuda_ops import w8a16_extension

        return w8a16_extension().rotate(values), True
    return values, False


class _H3RotatedColumnInput(ColumnParallelLinear):
    def forward(self, input_, *, input_is_rotated=False):
        if not input_is_rotated:
            return super().forward(input_)
        method = self.quant_method
        if (
            not input_.is_cuda
            or input_.dtype != torch.float16
            or self.bias is not None
            or self.gather_output
            or not isinstance(method, Int8ConvRotLinearMethod)
            or not method.layer_config.convrot
            or method.layer_config.convrot_groupsize != 256
        ):
            raise ValueError(
                "Pre-rotated H3 columns require bias-free INT8 FP16 inputs"
            )
        # The module still receives every gathered row: ordinary forward hooks
        # and useful-FLOP accounting retain the original full projection shape.
        output = method.apply_prepared(self, input_, None, input_is_rotated=True)
        return (output, None) if self.return_bias else output


class H3QKVParallelLinear(_H3RotatedColumnInput, QKVParallelLinear):
    """QKV projection accepting explicitly rotated, gathered FP16 rows."""


class H3MergedColumnParallelLinear(_H3RotatedColumnInput, MergedColumnParallelLinear):
    """Gate/up projection accepting explicitly rotated, gathered FP16 rows."""


class H3RowParallelLinear(RowParallelLinear):
    """H3-only optional prepared input; retain normal module hooks and TP sum."""

    def forward(self, input_, input_scale=None):
        if input_scale is None:
            return super().forward(input_)
        if (
            not self.input_is_parallel
            or self.bias is not None
            or not isinstance(self.quant_method, Int8ConvRotLinearMethod)
        ):
            raise ValueError(
                "Prepared H3 rows require a bias-free INT8 local projection"
            )
        output = self.quant_method.apply_prepared(self, input_, input_scale)
        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return (output, None) if self.return_bias else output
