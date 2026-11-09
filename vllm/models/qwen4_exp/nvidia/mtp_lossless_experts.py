# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact FP16 expert values with omitted zero mantissa bits, not quantization."""

import numpy as np
import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm._sm70.policy import register_policy_op
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    MoEActivation,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


def pack_lossless_fp16(weight: torch.Tensor) -> torch.Tensor:
    """Pack CPU FP16 values into 52 bytes per 32; reject any lossy input.

    FP16 subnormals keep their full mantissa. Normal BF16-origin values have
    three zero low bits. The magnitude interval is explicit and checked for
    every loaded expert; no scaling, rounding or clipping is performed here.
    """
    if (
        weight.dtype != torch.float16
        or weight.device.type != "cpu"
        or weight.shape[-1] % 32
    ):
        raise ValueError("Lossless MTP packing needs CPU FP16 and whole groups of 32")
    bits = weight.contiguous().view(torch.uint16).numpy().reshape(-1, 32)
    mag = bits & 0x7FFF
    if np.any(mag >= 0x6000) or np.any((mag >= 1024) & ((mag & 7) != 0)):
        raise ValueError("MTP weights are outside the exact 13-bit FP16 representation")
    code = np.where(mag < 1024, mag, 1024 + ((mag - 1024) >> 3)).astype(np.uint16)
    packed = np.empty((bits.shape[0], 52), dtype=np.uint8)
    packed[:, :32] = code & 255
    packed[:, 32:48] = (code[:, ::2] >> 8) | ((code[:, 1::2] >> 8) << 4)
    packed[:, 48:] = np.packbits(bits >> 15, axis=1, bitorder="little")
    return torch.from_numpy(
        packed.reshape(*weight.shape[:-1], weight.shape[-1] * 13 // 8)
    )


def unpack_lossless_fp16(packed: torch.Tensor) -> torch.Tensor:
    """Independent CPU decoding oracle; production reads the planes directly."""
    if (
        packed.dtype != torch.uint8
        or packed.device.type != "cpu"
        or packed.shape[-1] % 52
    ):
        raise ValueError("Lossless MTP oracle requires CPU 52-byte groups")
    rows = packed.contiguous().numpy().reshape(-1, 52)
    hi = np.empty((rows.shape[0], 32), dtype=np.uint16)
    hi[:, ::2] = rows[:, 32:48] & 15
    hi[:, 1::2] = rows[:, 32:48] >> 4
    code = rows[:, :32].astype(np.uint16) | (hi << 8)
    mag = np.where(code < 1024, code, ((code - 1024) << 3) + 1024).astype(np.uint16)
    sign = np.unpackbits(rows[:, 48:], axis=1, bitorder="little").astype(np.uint16)
    bits = mag | (sign << 15)
    return torch.from_numpy(
        bits.view(np.float16).reshape(*packed.shape[:-1], packed.shape[-1] * 8 // 13)
    )


def _lossless_moe(x, w13, w2, ids, probabilities):
    m = x.shape[0]
    ids = ids.to(torch.int32).flatten().contiguous()
    probabilities = probabilities.contiguous()
    padded = torch.full((1,), m * 20, dtype=torch.int32, device=x.device)
    gate_up = x.new_empty((m, 10, 320))
    torch.ops._C.sm70_mtp_moe_u13_out(
        gate_up, x.contiguous(), w13, ids, probabilities, padded, False
    )
    hidden = x.new_empty((m * 10, 160))
    torch.ops._C.silu_and_mul(hidden, gate_up.view(m * 10, 320))
    down = x.new_empty((m, 10, 2560))
    torch.ops._C.sm70_mtp_moe_u13_out(
        down, hidden, w2, ids, probabilities, padded, True
    )
    output = torch.empty_like(x)
    ops.moe_sum(down, output)
    return output


def _lossless_moe_fake(x, w13, w2, ids, probabilities):
    return torch.empty_like(x)


register_policy_op(
    "sm70_mtp_lossless_moe",
    "(Tensor x, Tensor w13, Tensor w2, Tensor ids, Tensor probabilities) -> Tensor",
    _lossless_moe,
    _lossless_moe_fake,
)


class MTPLosslessMoEMethod(FusedMoEMethodBase):
    @property
    def topk_indices_dtype(self):
        return torch.int32

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if (
            (num_experts, hidden_size, intermediate_size_per_partition)
            != (512, 2560, 160)
            or params_dtype != torch.float16
            or not self.moe.is_act_and_mul
            or self.moe.has_bias
            or self.moe.tp_size != 4
            or self.moe.ep_size != 1
            or self.moe.dp_size != 1
        ):
            raise ValueError(
                "Lossless MTP experts require TP4 FP16 512x2560x160 gated weights"
            )
        self.loaded: dict[str, set[int]] = {
            shard: set() for shard in ("w1", "w3", "w2")
        }
        for name, n, k in (("w13_weight", 320, 2560), ("w2_weight", 2560, 160)):
            param = nn.Parameter(
                torch.empty((512, n, k * 13 // 8), dtype=torch.uint8),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
            set_weight_attrs(param, {"packed_expert_loader": self.load_expert})

    def load_expert(self, layer, param, weight, shard, expert):
        if shard not in self.loaded or expert in self.loaded[shard]:
            raise ValueError("Duplicate or invalid lossless MTP expert projection")
        expected = (2560, 640) if shard == "w2" else (640, 2560)
        if tuple(weight.shape) != expected or weight.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise ValueError(
                "Lossless MTP requires the admitted BF16/FP16 checkpoint geometry"
            )
        start = layer.tp_rank * 160
        local = (
            weight[:, start : start + 160]
            if shard == "w2"
            else weight[start : start + 160]
        )
        packed = pack_lossless_fp16(local.to(device="cpu", dtype=torch.float16))
        if shard == "w2":
            target = param[expert]
        else:
            first = 0 if shard == "w1" else 160
            target = param[expert, first : first + 160]
        target.copy_(packed)
        self.loaded[shard].add(expert)

    def process_weights_after_loading(self, layer):
        if any(experts != set(range(512)) for experts in self.loaded.values()):
            raise ValueError("Incomplete lossless MTP expert checkpoint")
        logger.info(
            "Lossless FP16 MTP expert storage: 975 MiB, saving 225 MiB per TP rank"
        )

    def maybe_make_prepare_finalize(self, routing_tables=None):
        return None

    def get_fused_moe_quant_config(self, layer):
        return None

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if (
            layer.expert_map is not None
            or layer.activation != MoEActivation.SILU
            or layer.apply_router_weight_on_input
        ):
            raise ValueError(
                "Lossless MTP requires TP-only output-weighted SiLU experts"
            )
        return torch.ops.vllm.sm70_mtp_lossless_moe(
            x, layer.w13_weight, layer.w2_weight, topk_ids, topk_weights
        )


class MTPLosslessConfig(QuantizationConfig):
    def get_name(self):
        return "lossless_fp16_mtp"

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
        raise NotImplementedError("Construct from the resolved unquantized MTP config")

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts) and prefix.startswith("mtp.layers."):
            return MTPLosslessMoEMethod(layer.moe_config)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None
