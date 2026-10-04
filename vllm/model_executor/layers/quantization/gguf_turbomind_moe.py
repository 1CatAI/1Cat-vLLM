# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical expert banks and graph-compatible GGUF routing."""

from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from vllm.model_executor.kernels.gguf import (
    GGUFDecoderFamily,
    GGUFOperatorCapability,
    admit_moe_fallback,
    decoder_family,
    lattice_grouped_capabilities,
    select_lattice_grouped_capability,
)
from vllm.model_executor.layers.fused_moe.sm70_small_routing import (
    SM70_SMALL_ROUTING,
)
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    LATTICE_TYPES,
    LatticeGGUFProjection,
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_lut_transcode import (
    LUT4_TYPES,
    Lut4GGUFProjection,
    transcode_lut4,
)
from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
from vllm.model_executor.layers.quantization.gguf_native import pad_weight_tail
from vllm.model_executor.layers.quantization.gguf_transcode import (
    AffineGGUFProjection,
    transcode_affine,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_type_name


class GGUFExpertBank(torch.nn.Module):
    def __init__(self, source_type, experts, device, dtype):
        super().__init__()
        self.source_type = source_type
        self.experts = experts
        self.device = device
        self.dtype = dtype
        self.family = decoder_family(source_type)
        self.pending: dict[int, Any] = {}
        self.capabilities: tuple[GGUFOperatorCapability, ...] = ()

    def add(self, index, weight, rank, size, axis):
        if weight.ndim != 2 or not 0 <= index < self.experts:
            raise ValueError("GGUF expert requires a valid index and 2D projection")
        if index in self.pending:
            raise ValueError("Duplicate canonical GGUF expert")
        if self.family == GGUFDecoderFamily.FLOAT:
            if weight.shape[axis] % size or not 0 <= rank < size:
                raise ValueError("Floating GGUF expert has invalid TP boundary")
            local = weight.chunk(size, dim=axis)[rank]
            self.n, self.k = local.shape
            self.pending[index] = local.to(self.device).contiguous()
            return
        source = weight.detach().cpu().numpy()
        canonical: LatticeGGUFProjection | Lut4GGUFProjection | AffineGGUFProjection
        if self.source_type in LATTICE_TYPES:
            canonical = transcode_lattice(source, self.source_type)
        elif self.source_type in LUT4_TYPES:
            canonical = transcode_lut4(source, self.source_type)
        else:
            canonical = transcode_affine(source, self.source_type)
        canonical = canonical.tp_slice(rank, size, axis=axis)
        self.group = canonical.group_size
        self.n, self.k = (
            canonical.shape
            if isinstance(canonical, LatticeGGUFProjection)
            else canonical.codes.shape
        )
        if self.n % 32:
            raise ValueError("Canonical GGUF expert output cuts a 32-row pack")
        if isinstance(canonical, LatticeGGUFProjection):
            codes, stats = canonical.mma884_storage()
            stats = stats.view({2: np.int16, 4: np.int32, 8: np.int64}[stats.itemsize])
            prepared = torch.ops._C.gguf_lattice_sm70_prepare(
                torch.from_numpy(codes).to(self.device),
                torch.from_numpy(stats).to(self.device),
                self.source_type,
                self.group,
            )
        else:
            codes = torch.from_numpy(canonical.codes).to(self.device)
            scales = torch.from_numpy(canonical.scales).to(self.device)
            if isinstance(canonical, Lut4GGUFProjection):
                self.decoder = canonical.lut_id
                prepared = torch.ops._C.gguf_lut4_sm70_prepare(
                    codes, scales, self.decoder, self.group
                )
            else:
                assert isinstance(canonical, AffineGGUFProjection)
                self.decoder = canonical.bits
                prepared = torch.ops._C.gguf_affine_sm70_prepare(
                    codes,
                    scales,
                    torch.from_numpy(canonical.mins).to(self.device),
                    self.decoder,
                    self.group,
                )
        self.pending[index] = prepared

    def finalize(self):
        if set(self.pending) != set(range(self.experts)):
            raise ValueError("Incomplete canonical GGUF expert bank")
        prepared = [self.pending[i] for i in range(self.experts)]
        if self.family == GGUFDecoderFamily.FLOAT:
            weights = pad_weight_tail(torch.stack(prepared), self.source_type)
            self.register_buffer("weights", weights, persistent=False)
            self.capabilities = (
                admit_moe_fallback(weights, self.source_type, self.dtype),
            )
        else:
            metadata = prepared[0][2].tolist()
            if any(p[2].tolist() != metadata for p in prepared[1:]):
                raise ValueError("Canonical GGUF expert layouts disagree")
            weights = torch.stack([p[0] for p in prepared])
            stats = torch.stack([p[1] for p in prepared])
            self.register_buffer("weights", weights, persistent=False)
            self.register_buffer("stats", stats, persistent=False)
            wp, sp = torch.ops._C.awq_moe_build_strided_ptrs(
                weights, stats, *metadata, self.experts
            )
            self.register_buffer("weight_ptrs", wp, persistent=False)
            self.register_buffer("stat_ptrs", sp, persistent=False)
            if self.family == GGUFDecoderFamily.LATTICE:
                self.capabilities = lattice_grouped_capabilities(
                    self.source_type, self.k, self.n, self.experts, self.dtype
                )
            else:
                name = (
                    "gguf_lut4_grouped_gemm_sm70_out"
                    if self.family == GGUFDecoderFamily.LUT4
                    else "gguf_affine_grouped_gemm_sm70_out"
                )
                self.capabilities = (
                    GGUFOperatorCapability(
                        self.family,
                        quant_type_name(self.source_type),
                        name,
                        True,
                        reason=None
                        if hasattr(torch.ops._C, name)
                        else f"operator_missing:{name}",
                    ),
                )
        self.pending.clear()
        if not any(c.reason is None for c in self.capabilities):
            raise ValueError("No admitted canonical GGUF expert operator")

    def forward(self, x, offsets, ids):
        output = torch.empty((x.shape[0], self.n), dtype=x.dtype, device=x.device)
        if self.family == GGUFDecoderFamily.FLOAT:
            op = getattr(torch.ops._C_gguf, self.capabilities[0].operator)
            return op(
                x,
                self.weights,
                ids[:, None].int().contiguous(),
                self.source_type,
                self.n,
                1,
                x.shape[0],
            )
        capability = (
            select_lattice_grouped_capability(self.capabilities, x.shape[0])
            if self.family == GGUFDecoderFamily.LATTICE
            else self.capabilities[0]
        )
        if capability.reason or not capability.supports_m(x.shape[0]):
            raise ValueError("Canonical GGUF expert operator rejects routed rows")
        decoder = (
            self.source_type
            if self.family == GGUFDecoderFamily.LATTICE
            else self.decoder
        )
        getattr(torch.ops._C, capability.operator)(
            output,
            x,
            offsets,
            self.weight_ptrs,
            self.stat_ptrs,
            decoder,
            self.experts,
            self.group,
        )
        return output


class GGUFTurboMindMoEMethod(GGUFNativeMoEMethod):
    def __init__(self, quant_config, moe):
        super().__init__(quant_config, moe)
        self.builders: dict[str, GGUFExpertBank] = {}

    def load_expert(self, layer, param, weight, shard_id, expert_id):
        if param.is_gguf_weight_type:
            value = int(weight.item())
            if self.weight_types.setdefault(shard_id, value) != value:
                raise ValueError("GGUF projection has inconsistent expert formats")
            return
        if shard_id not in self.weight_types:
            raise ValueError("Missing GGUF expert type before payload")
        if shard_id not in self.builders:
            self.builders[shard_id] = GGUFExpertBank(
                self.weight_types[shard_id],
                self.num_experts,
                param.device,
                self.params_dtype,
            )
        self.builders[shard_id].add(
            expert_id,
            weight,
            layer.tp_rank,
            layer.tp_size,
            1 if shard_id == "w2" else 0,
        )
        bank = self.builders[shard_id]
        expected = (
            (self.hidden_size, self.intermediate_size)
            if shard_id == "w2"
            else (self.intermediate_size, self.hidden_size)
        )
        actual = (bank.n, bank.k)
        if actual != expected:
            raise ValueError(f"GGUF expert shape {actual} != {expected}")
        self.loaded_experts[shard_id].add(expert_id)

    def process_weights_after_loading(self, layer):
        banks = torch.nn.ModuleDict()
        for shard in ("w1", "w3", "w2"):
            bank = self.builders[shard]
            bank.finalize()
            banks[shard] = bank
        layer.gguf_expert_banks = banks
        self.small_routing = bool(
            self.native_enabled
            and self.params_dtype == torch.float16
            and self.num_experts == 512
            and self.hidden_size == 2560
            and layer.ep_size == 1
        )
        self.native_admission = {
            "enabled": True,
            "tp_size": layer.tp_size,
            "ep_size": layer.ep_size,
            "routing": {
                **asdict(SM70_SMALL_ROUTING),
                "enabled": self.small_routing,
                "reason": None
                if self.small_routing
                else "requires_fp16_512_experts_hidden2560_without_ep",
                "outside_m_band": "legacy_alignment_and_fp32_weighted_sum",
            },
            "projections": {
                name: {
                    "source_type": quant_type_name(bank.source_type),
                    "shape": [bank.experts, bank.n, bank.k],
                    "operators": [asdict(c) for c in bank.capabilities],
                }
                for name, bank in banks.items()
            },
        }
        self.builders.clear()

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        from vllm.model_executor.layers.fused_moe import MoEActivation

        if layer.apply_router_weight_on_input or layer.activation != MoEActivation.SILU:
            raise ValueError(
                "Canonical GGUF experts require output-weighted SiLU routing"
            )
        if x.shape[0] == 0:
            return torch.empty_like(x)
        ids = topk_ids
        mask = None
        if layer.expert_map is not None:
            ids = layer.expert_map[ids]
            mask = ids >= 0
            ids = ids.clamp_min(0)
        tokens, top_k = ids.shape
        small_routing = self.small_routing and layer.expert_map is None
        if small_routing:
            routed, offsets, sorted_ids, inverse = (
                torch.ops.vllm.sm70_small_expert_route(x, topk_ids, self.num_experts)
            )
        else:
            ids = ids.long()
            sorted_ids, order = ids.reshape(-1).sort()
            boundaries = torch.arange(
                self.num_experts + 1, device=x.device, dtype=torch.int64
            )
            offsets = torch.searchsorted(sorted_ids, boundaries).to(torch.int32)
            routed = x[torch.div(order, top_k, rounding_mode="floor")].contiguous()
        bank = layer.gguf_expert_banks
        gate = bank["w1"](routed, offsets, sorted_ids)
        up = bank["w3"](routed, offsets, sorted_ids)
        hidden = torch.nn.functional.silu(gate) * up
        down = bank["w2"](hidden.contiguous(), offsets, sorted_ids)
        if small_routing:
            return torch.ops.vllm.sm70_small_expert_unroute(down, inverse, topk_weights)
        restored = down[order.argsort()].view(tokens, top_k, self.hidden_size)
        if mask is not None:
            restored = torch.where(mask[..., None], restored, 0)
        return (restored.float() * topk_weights[..., None].float()).sum(1).to(x.dtype)
