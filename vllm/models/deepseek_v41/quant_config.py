# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization config for DeepSeek-V4.1 on SM70 (PORT_DESIGN §0 A2, §1 "Dense FP8 weights", §2.2).

The official checkpoint stores every dense linear as FP8 E4M3 with UE8M0 scales per 32x32 block and the routed
experts as MXFP4. On Volta:

* every dense FP8 linear is dequantized to FP16 once at load (1Cat's SM70 dequant fallback,
  ``Fp8LinearMethod._dequantize_block_weight``) and run as an FP16 GEMM -- forced here, independent of the
  ``VLLM_SM70_FP8_*`` environment (TurboMind group-32 kernels are a P5 item). Dense scale exponents are
  -13..-6, so every E4M3 x 2^s value is an FP16 value; the load path verifies that bit-exactly and raises
  otherwise (no silent flush to zero);
* grouped (``is_bmm``) linears -- ``wo_a`` -- use the dequantized einsum path;
* FusedMoE layers get L-MOE's ``vllm.models.deepseek_v41.sm70.moe.make_v41_moe_method`` (imported lazily);
* Engram ``wkv`` must not come through here: its scales reach 2^-18 and it is loaded with a 2^10 bias by
  ``DeepseekV41Engram`` itself (quant_config=None). Building it through this config raises.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import replace_parameter

QUANT_METHOD_NAME = "deepseek_v41_fp8"
V41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
V41_WEIGHT_BLOCK = [32, 32]


def dequantize_fp8_block32_exact(weight: torch.Tensor, scale: torch.Tensor,
                                 block: tuple[int, int] = (32, 32)) -> torch.Tensor:
    """E4M3 [N, K] x UE8M0 [ceil(N/bn), ceil(K/bk)] -> FP16 [N, K], verified bit-exact.

    The product is formed in FP32 (exact: 4 significant bits x a power of two) and rounded once to FP16; the
    round trip FP16 -> FP32 must reproduce it for every element, otherwise some value under/overflowed FP16 and
    the load fails loudly (A2: dense scale exponents -13..-6 keep every value exact).
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected float8_e4m3fn weight, got {weight.dtype}")
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"expected 2-D weight/scale, got {tuple(weight.shape)} / {tuple(scale.shape)}")
    bn, bk = block
    n, k = weight.shape
    if scale.shape[0] != -(-n // bn) or scale.shape[1] != -(-k // bk):
        raise ValueError(f"scale shape {tuple(scale.shape)} does not tile weight {tuple(weight.shape)} by {block}")
    scale32 = scale.to(torch.float32)
    expanded = scale32.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)[:n, :k]
    exact = weight.to(torch.float32) * expanded
    out = exact.to(torch.float16)
    mismatch = out.to(torch.float32) != exact
    if bool(mismatch.any()):
        bad = int(mismatch.sum())
        raise ValueError(
            f"FP8 32x32 block weight is not exactly representable in FP16 ({bad} of {exact.numel()} values "
            f"change; scale range [{float(scale32.min())}, {float(scale32.max())}]). PORT_DESIGN A2 assumes "
            "dense scale exponents -13..-6; a layer outside that range needs a scale bias (as Engram wkv).")
    return out.contiguous()


class DeepseekV41SM70Fp8LinearMethod(Fp8LinearMethod):
    """Fp8LinearMethod pinned to the dequant-to-FP16 fallback (A2), with the grouped wo_a einsum."""

    def __init__(self, quant_config: DeepseekV41FP8Config):
        super().__init__(quant_config)
        if self.weight_block_size != V41_WEIGHT_BLOCK:
            raise ValueError(f"DeepSeek-V4.1 dense FP8 expects weight_block_size {V41_WEIGHT_BLOCK}, "
                             f"got {self.weight_block_size}")
        self.use_sm70_dequant_fallback = True
        self.use_sm70_fp8_turbomind = False
        self.use_marlin = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, weight_scale_inv = process_fp8_weight_block_strategy(layer.weight, layer.weight_scale_inv)
        if weight.shape != layer.weight.shape:
            raise ValueError(f"FP8 weight padding changed the shape {tuple(layer.weight.shape)} -> "
                             f"{tuple(weight.shape)}; the V4.1 fallback does not support padded weights")
        dequant = dequantize_fp8_block32_exact(weight, weight_scale_inv, tuple(self.weight_block_size))
        replace_parameter(layer, "weight", dequant)
        layer.input_scale = None
        if getattr(layer, "is_bmm", False):
            layer.dequantized_bmm = True

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if getattr(layer, "dequantized_bmm", False):
            # x ends in [groups, K]; group g multiplies rows g*R:(g+1)*R (as Fp8LinearMethod.apply).
            group_count = int(layer.bmm_batch_size)
            weight = layer.weight.view(group_count, -1, x.shape[-1])
            out = torch.einsum("...gk,grk->...gr", x, weight)
            if bias is not None:
                out.add_(bias.view(group_count, -1))
            return out
        return torch.nn.functional.linear(x, layer.weight, bias)


@register_quantization_config(QUANT_METHOD_NAME)
class DeepseekV41FP8Config(Fp8Config):
    """Claims DeepSeek-V4.1 FP8 checkpoints (model_type deepseek_v41 / deepseek_v41_text)."""

    is_scale_e8m0 = True

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return QUANT_METHOD_NAME  # type: ignore[return-value]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg: Any, user_quant: str | None,
                                     hf_config: Any = None) -> QuantizationMethods | None:
        if not (isinstance(hf_quant_cfg, dict)
                and hf_quant_cfg.get("quant_method") in ("fp8", QUANT_METHOD_NAME)):
            return None
        model_type = getattr(hf_config, "model_type", None)
        if model_type in V41_MODEL_TYPES or user_quant == QUANT_METHOD_NAME:
            return QUANT_METHOD_NAME  # type: ignore[return-value]
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DeepseekV41FP8Config:
        block = config.get("weight_block_size")
        if block != V41_WEIGHT_BLOCK:
            raise ValueError(f"DeepSeek-V4.1 FP8 checkpoint must use weight_block_size {V41_WEIGHT_BLOCK}, "
                             f"got {block}")
        if config.get("scale_fmt", "ue8m0") != "ue8m0":
            raise ValueError(f"DeepSeek-V4.1 FP8 checkpoint must use scale_fmt ue8m0, got {config['scale_fmt']}")
        expert_dtype = config.get("expert_dtype", "fp4")
        if expert_dtype != "fp4":
            raise ValueError(f"the SM70 DeepSeek-V4.1 port supports expert_dtype fp4 only, got {expert_dtype!r}")
        fp8 = super().from_config(config)
        assert isinstance(fp8, DeepseekV41FP8Config)
        return fp8

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if ".engram." in f".{prefix}.":
                raise ValueError(
                    f"{prefix}: Engram linears must be built with quant_config=None and load their FP8 weight "
                    "with the 2^10 scale bias (PORT_DESIGN §4.1); the plain FP16 dequant would flush 23% of "
                    "the Engram wkv blocks")
            if is_layer_skipped(prefix=prefix, ignored_layers=self.ignored_layers,
                                fused_mapping=self.packed_modules_mapping,
                                match_mode=self.ignored_layers_match_mode):
                return UnquantizedLinearMethod()
            return DeepseekV41SM70Fp8LinearMethod(self)
        if isinstance(layer, FusedMoE):
            from vllm.models.deepseek_v41.sm70.moe import make_v41_moe_method

            return make_v41_moe_method(layer.moe_config)
        return super().get_quant_method(layer, prefix)
