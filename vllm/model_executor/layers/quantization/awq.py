# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Union

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from transformers import PretrainedConfig

from vllm import _custom_ops as ops
from vllm import _sm70_ops as sm70_ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_safetensors_params_metadata

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (
            f"AWQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    def get_name(self) -> "QuantizationMethods":
        return "awq"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        if envs.VLLM_SM70_AWQ_TURBOMIND:
            return 70
        # The default AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        elif isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if (
                current_platform.is_cuda()
                and current_platform.has_device_capability(70)
                and not current_platform.has_device_capability(75)
                and envs.VLLM_SM70_AWQ_TURBOMIND
            ):
                if envs.VLLM_SM70_AWQ_MOE_DISABLE:
                    logger.warning_once(
                        "Layer '%s' SM70 AWQ TurboMind MoE path disabled by "
                        "VLLM_SM70_AWQ_MOE_DISABLE=1. Falling back to MoeWNA16.",
                        prefix,
                    )
                    from .moe_wna16 import MoeWNA16Config

                    config = {
                        "quant_method": "awq",
                        "bits": self.weight_bits,
                        "group_size": self.group_size,
                        "zero_point": self.zero_point,
                        "lm_head": False,
                        "modules_to_not_convert": self.modules_to_not_convert,
                    }
                    return MoeWNA16Config.from_config(config).get_quant_method(
                        layer, prefix
                    )

                from vllm.model_executor.layers.quantization.awq_sm70_moe import (
                    AWQSM70MoEMethod,
                )

                return AWQSM70MoEMethod(
                    self.weight_bits,
                    self.group_size,
                    self.zero_point,
                    layer,
                )

            # Lazy import to avoid circular import.
            from .awq_marlin import AWQMarlinConfig
            from .moe_wna16 import MoeWNA16Config
            from .utils.marlin_utils import check_moe_marlin_supports_layer

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AWQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                config = {
                    "quant_method": "awq",
                    "bits": self.weight_bits,
                    "group_size": self.group_size,
                    "zero_point": self.zero_point,
                    "lm_head": False,
                    "modules_to_not_convert": self.modules_to_not_convert,
                }
                return MoeWNA16Config.from_config(config).get_quant_method(
                    layer, prefix
                )
            marlin_compatible_config_dict = {
                "quant_method": "awq",
                "bits": self.weight_bits,
                "group_size": self.group_size,
                "zero_point": self.zero_point,
                "lm_head": False,
                "modules_to_not_convert": self.modules_to_not_convert,
            }
            awq_marlin_config = AWQMarlinConfig.from_config(
                marlin_compatible_config_dict
            )
            return awq_marlin_config.get_quant_method(layer, prefix)
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.modules_to_not_convert:
            self.modules_to_not_convert = hf_to_vllm_mapper.apply_list(
                self.modules_to_not_convert
            )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ):
        if self.modules_to_not_convert:
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        layers = {param_name.rsplit(".", 1)[0] for param_name in metadata}
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
            and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_to_not_convert = list(layers - quant_layers)


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_awq_sm70_prepared", False):
            return

        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        if not envs.VLLM_SM70_AWQ_TURBOMIND or not layer.qweight.is_cuda:
            return

        cap = torch.cuda.get_device_capability(layer.qweight.device)
        if cap != (7, 0):
            return

        group_size = self.quant_config.group_size
        if group_size == -1:
            group_size = layer.qweight.shape[0]
        if group_size not in (32, 64, 128):
            raise RuntimeError(
                "SM70 TurboMind AWQ supports group_size 32/64/128, "
                f"but got {group_size}."
            )
        if not hasattr(torch.ops._C, "awq_sm70_prepare"):
            raise RuntimeError(
                "VLLM_SM70_AWQ_TURBOMIND=1 requires a build with CUDA arch 7.0 "
                "and the SM70 TurboMind extension."
            )

        tm_weight, tm_scales, meta = sm70_ops.awq_sm70_prepare(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            group_size,
        )
        layer._awq_sm70_weight = tm_weight
        layer._awq_sm70_scales = tm_scales
        layer._awq_sm70_k_ld = int(meta[0])
        layer._awq_sm70_q_ld = int(meta[1])
        layer._awq_sm70_group_size = group_size
        layer._awq_sm70_prepared = True

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
        logger.info_once("SM70 AWQ TurboMind dense path enabled.")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        reshaped_x = x.reshape(-1, x.shape[-1])

        # num_tokens >= threshold
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256
        if getattr(layer, "_awq_sm70_prepared", False):
            out_shape = x.shape[:-1] + (
                layer._awq_sm70_weight.shape[-1] * pack_factor,
            )
            out = torch.empty(
                (reshaped_x.shape[0], out_shape[-1]),
                dtype=x.dtype,
                device=x.device,
            )
            sm70_ops.awq_gemm_sm70_out(
                out,
                reshaped_x,
                layer._awq_sm70_weight,
                layer._awq_sm70_scales,
                layer._awq_sm70_group_size,
                layer._awq_sm70_k_ld,
                layer._awq_sm70_q_ld,
            )
        elif FP16_MATMUL_HEURISTIC_CONDITION or envs.VLLM_BATCH_INVARIANT:
            # Batch invariant mode requires torch.matmul path for Triton override.
            out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
            out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
            out = torch.matmul(reshaped_x, out)
        else:
            out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
            out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
