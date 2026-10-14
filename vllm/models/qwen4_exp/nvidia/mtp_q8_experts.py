# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF Q8_0 storage for Qwen4Exp MTP draft experts.

Draft experts only propose tokens; the target verifies every one, so their
precision changes acceptance, never the output distribution. Q8_0 halves the
FP16 draft expert storage (1.17 -> 0.62 GiB per TP4 rank) and runs through the
existing GGUF MoE operator (dp4a MMVQ for small batches, MMQ above 64 rows).
"""

import torch
from torch import nn

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

# Tokens per draft MoE call during prefill (bounds its temporaries).
_PREFILL_CHUNK = 2048

GGML_Q8_0 = 8
_BLOCK = 32
_BLOCK_BYTES = 34  # FP16 scale + 32 int8 values


def quantize_q8_0(weight: torch.Tensor) -> torch.Tensor:
    """ggml ``quantize_row_q8_0_ref``: per 32 values d = amax / 127 (stored as
    FP16), q = round(x / d) computed with the FP32 scale."""
    if weight.shape[-1] % _BLOCK:
        raise ValueError("Q8_0 needs whole blocks of 32 values")
    x = weight.to(torch.float32).reshape(-1, _BLOCK)
    d = x.abs().amax(dim=1, keepdim=True) / 127
    inverse = torch.where(d > 0, 1 / d, torch.zeros_like(d))
    q = torch.round(x * inverse).clamp_(-128, 127).to(torch.int8)
    packed = torch.empty((x.shape[0], _BLOCK_BYTES), dtype=torch.uint8, device=x.device)
    packed[:, :2] = d.to(torch.float16).view(torch.uint8)
    packed[:, 2:] = q.view(torch.uint8)
    return packed.reshape(*weight.shape[:-1], weight.shape[-1] // _BLOCK * _BLOCK_BYTES)


def dequantize_q8_0(packed: torch.Tensor) -> torch.Tensor:
    """Reference decoder for tests."""
    blocks = packed.reshape(-1, _BLOCK_BYTES)
    d = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
    q = blocks[:, 2:].contiguous().view(torch.int8).to(torch.float32)
    values = (q * d).to(torch.float16)
    return values.reshape(*packed.shape[:-1], packed.shape[-1] // _BLOCK_BYTES * _BLOCK)


class MTPQ80MoEMethod(FusedMoEMethodBase):
    """TP-sharded Q8_0 copies of the FP16/BF16 MTP experts."""

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
            hidden_size % _BLOCK
            or intermediate_size_per_partition % _BLOCK
            or params_dtype != torch.float16
            or not self.moe.is_act_and_mul
            or self.moe.has_bias
            or self.moe.ep_size != 1
            or self.moe.dp_size != 1
        ):
            raise ValueError(
                "Q8_0 MTP experts need FP16 gated TP-only experts in 32-value blocks"
            )
        self.num_experts = num_experts
        self.hidden = hidden_size
        self.local = intermediate_size_per_partition
        self.loaded: dict[str, set[int]] = {s: set() for s in ("w1", "w3", "w2")}
        for name, n, k in (
            ("w13_weight", 2 * self.local, hidden_size),
            ("w2_weight", hidden_size, self.local),
        ):
            param = nn.Parameter(
                torch.empty(
                    (num_experts, n, k // _BLOCK * _BLOCK_BYTES), dtype=torch.uint8
                ),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
            set_weight_attrs(param, {"packed_expert_loader": self.load_expert})

    def load_expert(self, layer, param, weight, shard, expert):
        if shard not in self.loaded or expert in self.loaded[shard]:
            raise ValueError("Duplicate or invalid Q8_0 MTP expert projection")
        full = self.local * layer.tp_size
        expected = (self.hidden, full) if shard == "w2" else (full, self.hidden)
        if tuple(weight.shape) != expected or weight.dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            raise ValueError("Q8_0 MTP experts need the floating checkpoint geometry")
        start = layer.tp_rank * self.local
        local = (
            weight[:, start : start + self.local]
            if shard == "w2"
            else weight[start : start + self.local]
        )
        device = param.device if param.device.type == "cuda" else local.device
        packed = quantize_q8_0(local.to(device=device, dtype=torch.float32))
        if shard == "w2":
            target = param[expert]
        else:
            first = 0 if shard == "w1" else self.local
            target = param[expert, first : first + self.local]
        target.copy_(packed)
        self.loaded[shard].add(expert)

    def process_weights_after_loading(self, layer):
        if any(e != set(range(self.num_experts)) for e in self.loaded.values()):
            raise ValueError("Incomplete Q8_0 MTP expert checkpoint")
        stored = layer.w13_weight.numel() + layer.w2_weight.numel()
        fp16 = 2 * self.num_experts * 3 * self.local * self.hidden
        logger.info(
            "Q8_0 MTP expert storage: %.0f MiB per rank (FP16: %.0f MiB).",
            stored / 2**20,
            fp16 / 2**20,
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
            raise ValueError("Q8_0 MTP experts need TP-only output-weighted SiLU")
        from vllm.model_executor.layers.quantization.gguf import fused_moe_gguf

        def run(rows):
            return fused_moe_gguf(
                x[rows],
                layer.w13_weight,
                layer.w2_weight,
                topk_weights[rows],
                topk_ids[rows],
                GGML_Q8_0,
                GGML_Q8_0,
                "silu",
            )

        tokens = x.shape[0]
        if tokens <= _PREFILL_CHUNK:
            return run(slice(None))
        # Prefill: the GGUF MoE path keeps [tokens * top_k, hidden]
        # temporaries; token chunks bound them (tokens are independent).
        out = torch.empty_like(x)
        for start in range(0, tokens, _PREFILL_CHUNK):
            rows = slice(start, min(tokens, start + _PREFILL_CHUNK))
            out[rows] = run(rows)
        return out


class MTPQ80Config(QuantizationConfig):
    def get_name(self):
        return "q8_0_mtp_experts"

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
            return MTPQ80MoEMethod(layer.moe_config)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None
