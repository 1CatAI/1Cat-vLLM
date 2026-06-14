# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, get_args

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)

QuantizationMethods = Literal[
    "awq",
    "fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "gguf",
    "auto_gptq",
    "gptq",
    "gptq_marlin",
    "awq_marlin",
    "humming",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "gpt_oss_mxfp4",
    "deepseek_v4_fp8",
    "online",
    # Below are online quant shorthand names (see vllm.config.quantization).
    # Listed here as strings to avoid a circular import; kept in sync with
    # _ONLINE_SHORTHANDS by the assertion in get_quantization_config().
    "fp8_per_tensor",
    "fp8_per_block",
    "int8_per_channel_weight_only",
    "mxfp8",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

DEPRECATED_QUANTIZATION_METHODS = [
    "fbgemm_fp8",
    "fp_quant",
]

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.debug(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # lazy import to avoid triggering `torch.compile` too early
    if quantization == "humming":
        try:
            from .humming import HummingConfig
        except ModuleNotFoundError as exc:
            if exc.name != "humming":
                raise

            class HummingConfig(QuantizationConfig):
                """Placeholder used when the optional humming package is absent."""

                def get_name(self) -> QuantizationMethods:
                    return "humming"

                def get_supported_act_dtypes(self) -> list:
                    return []

                @classmethod
                def get_min_capability(cls) -> int:
                    return 0

                @staticmethod
                def get_config_filenames() -> list[str]:
                    return []

                @classmethod
                def from_config(cls, config: dict) -> "HummingConfig":
                    del config
                    raise ModuleNotFoundError(
                        "The optional 'humming' package is required for "
                        "quantization='humming'."
                    )

                @classmethod
                def override_quantization_method(
                    cls, hf_quant_cfg: dict, user_quant: str | None, hf_config=None
                ) -> QuantizationMethods | None:
                    del hf_quant_cfg, hf_config
                    if user_quant == "humming":
                        raise ModuleNotFoundError(
                            "The optional 'humming' package is required for "
                            "quantization='humming'."
                        )
                    return None

                def get_quant_method(self, layer, prefix):
                    del layer, prefix
                    raise ModuleNotFoundError(
                        "The optional 'humming' package is required for "
                        "quantization='humming'."
                    )

        return HummingConfig

    from vllm.config.quantization import _ONLINE_SHORTHANDS
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.models.deepseek_v4 import DeepseekV4FP8Config

    from .auto_gptq import AutoGPTQConfig
    from .awq import AWQConfig
    from .awq_marlin import AWQMarlinConfig
    from .bitsandbytes import BitsAndBytesConfig
    from .compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from .experts_int8 import ExpertsInt8Config
    from .fbgemm_fp8 import FBGEMMFp8Config
    from .fp8 import Fp8Config
    from .fp_quant import FPQuantConfig
    from .gguf import GGUFConfig
    from .inc import INCConfig
    from .modelopt import (
        ModelOptFp8Config,
        ModelOptMixedPrecisionConfig,
        ModelOptMxFp8Config,
        ModelOptNvFp4Config,
    )
    from .moe_wna16 import MoeWNA16Config
    from .mxfp4 import GptOssMxfp4Config, Mxfp4Config
    from .online.base import OnlineQuantizationConfig
    from .torchao import TorchAOConfig

    method_to_config: dict[str, type[QuantizationConfig]] = {
        "awq": AWQConfig,
        "fp8": Fp8Config,
        "fbgemm_fp8": FBGEMMFp8Config,
        "fp_quant": FPQuantConfig,
        "modelopt": ModelOptFp8Config,
        "modelopt_fp4": ModelOptNvFp4Config,
        "modelopt_mxfp8": ModelOptMxFp8Config,
        "modelopt_mixed": ModelOptMixedPrecisionConfig,
        "gguf": GGUFConfig,
        "auto_gptq": AutoGPTQConfig,
        "gptq": AutoGPTQConfig,
        "gptq_marlin": AutoGPTQConfig,
        "awq_marlin": AWQMarlinConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "bitsandbytes": BitsAndBytesConfig,
        "experts_int8": ExpertsInt8Config,
        "quark": QuarkConfig,
        "moe_wna16": MoeWNA16Config,
        "torchao": TorchAOConfig,
        "auto-round": INCConfig,
        "inc": INCConfig,
        "mxfp4": Mxfp4Config,
        "gpt_oss_mxfp4": GptOssMxfp4Config,
        "deepseek_v4_fp8": DeepseekV4FP8Config,
        "online": OnlineQuantizationConfig,
    }

    # Register online shorthands as quantization methods so the user can
    # specify "LLM(..., quantization='fp8_per_tensor')" as shorthand for
    # creating a more complicated online quant config object.
    for shorthand in _ONLINE_SHORTHANDS:
        assert shorthand not in method_to_config, (
            f"Online quant shorthand {shorthand!r} conflicts with an "
            f"existing quantization method"
        )
        method_to_config[shorthand] = OnlineQuantizationConfig

    # Update the `method_to_config` with customized quantization methods.
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]
