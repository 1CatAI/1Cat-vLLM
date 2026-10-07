# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF experts with independent gate/up/down storage and aligned TP slices."""

from dataclasses import asdict
from typing import TYPE_CHECKING

import torch
from torch.nn import Parameter

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.kernels.gguf import (
    GGUFDecoderFamily,
    admit_moe_fallback,
    decoder_family,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    MoEActivation,
)
from vllm.model_executor.layers.quantization.gguf_native import (
    NATIVE_TYPES,
    empty_guarded_weight,
    native_available,
    pad_weight_tail,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.gguf_tensor_reader import quant_size, quant_type_name

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig


class GGUFNativeMoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: "GGUFConfig", moe):
        super().__init__(moe)
        self.quant_config = quant_config
        config = get_current_vllm_config_or_none()
        self.native_enabled = (
            config.kernel_config.sm70_gguf.enabled if config is not None else True
        )
        self.weight_types: dict[str, int] = {}
        self.guarded_shards: set[str] = set()
        self.loaded_experts: dict[str, set[int]] = {
            shard: set() for shard in ("w1", "w3", "w2")
        }

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
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition
        self.params_dtype = params_dtype
        if self.moe.dp_size != 1 or self.moe.pcp_size != 1 or self.moe.ep_size != 1:
            raise ValueError(
                "Native GGUF experts currently require DP=PCP=EP=1; use TP"
            )
        # Keep vLLM's expert name mapping, while the callback stores the three
        # projections independently. No expert weights are merged by byte type.
        for name in (
            "w13_qweight",
            "w2_qweight",
            "w13_qweight_type",
            "w2_qweight_type",
        ):
            param = Parameter(torch.empty(0, dtype=params_dtype), requires_grad=False)
            set_weight_attrs(param, extra_weight_attrs)
            set_weight_attrs(
                param,
                {
                    "is_gguf_weight_type": name.endswith("_type"),
                    "is_gguf_weight": not name.endswith("_type"),
                    "gguf_expert_loader": self.load_expert,
                    "ignore_warning": name.endswith("_type"),
                },
            )
            layer.register_parameter(name, param)

    def load_expert(self, layer, param, weight, shard_id, expert_id):
        if param.is_gguf_weight_type:
            value = int(weight.item())
            previous = self.weight_types.setdefault(shard_id, value)
            if previous != value:
                raise ValueError(f"GGUF {shard_id} has inconsistent expert types")
            return
        if shard_id not in self.weight_types:
            raise ValueError(f"Missing GGUF {shard_id} type before expert payload")
        value = self.weight_types[shard_id]
        if (
            value not in NATIVE_TYPES
            and decoder_family(value) != GGUFDecoderFamily.FLOAT
        ):
            raise ValueError(f"Unsupported native GGUF expert {quant_type_name(value)}")
        rows, k = (
            (self.hidden_size, self.intermediate_size)
            if shard_id == "w2"
            else (self.intermediate_size, self.hidden_size)
        )
        block, size = quant_size(value)
        if k % block:
            raise ValueError(
                f"GGUF {shard_id} local K={k} is not aligned to block {block}; "
                "convert the checkpoint projection before TP slicing"
            )
        packed_k = k // block * size if weight.dtype == torch.uint8 else k
        tp_size, tp_rank = layer.tp_size, layer.tp_rank
        expected = (
            (rows, packed_k * tp_size)
            if shard_id == "w2"
            else (rows * tp_size, packed_k)
        )
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"GGUF {shard_id} shape {tuple(weight.shape)} != {expected}"
            )
        local = (
            weight[:, tp_rank * packed_k : (tp_rank + 1) * packed_k]
            if shard_id == "w2"
            else weight[tp_rank * rows : (tp_rank + 1) * rows]
        )
        name = "gguf_" + shard_id
        if not hasattr(layer, name):
            shape = (self.num_experts, rows, packed_k)
            if value in NATIVE_TYPES and weight.dtype == torch.uint8:
                storage = empty_guarded_weight(shape, value, param.device)
                self.guarded_shards.add(shard_id)
            else:
                storage = torch.empty(shape, dtype=weight.dtype, device=param.device)
            layer.register_buffer(name, storage, persistent=False)
        if expert_id in self.loaded_experts[shard_id]:
            raise ValueError(f"Duplicate GGUF {shard_id} expert {expert_id}")
        getattr(layer, name)[expert_id].copy_(local)
        self.loaded_experts[shard_id].add(expert_id)

    def process_weights_after_loading(self, layer):
        if not self.native_enabled or not native_available():
            raise ValueError(
                "Native GGUF experts require enabled packaged _C_gguf operators"
            )
        self.native_admission = {
            "enabled": True,
            "tp_size": layer.tp_size,
            "ep_size": layer.ep_size,
            "projections": {},
        }
        self.projection_capabilities = {}
        for shard in ("w1", "w3", "w2"):
            if self.loaded_experts[shard] != set(range(self.num_experts)):
                raise ValueError(f"Incomplete GGUF {shard} expert payloads")
            weight = getattr(layer, "gguf_" + shard)
            if not weight.is_cuda:
                raise ValueError("Native GGUF experts require CUDA storage")
            value = self.weight_types[shard]
            prepared = pad_weight_tail(
                weight, value, storage_has_zero_tail=shard in self.guarded_shards
            )
            setattr(layer, "gguf_" + shard, prepared)
            capability = admit_moe_fallback(prepared, value, self.params_dtype)
            self.projection_capabilities[shard] = capability
            self.native_admission["projections"][shard] = {
                **asdict(capability),
                "shape": list(weight.shape),
            }

    def get_fused_moe_quant_config(self, layer):
        return None

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # DP=PCP=1 has replicated attention-TP inputs. The existing MoERunner
        # reduces the final TP partials.
        return None

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if layer.apply_router_weight_on_input or layer.activation != MoEActivation.SILU:
            raise ValueError("Native GGUF experts require output-weighted SiLU routing")
        ids = topk_ids.to(torch.int32).contiguous()
        mask = None
        if layer.expert_map is not None:
            ids = layer.expert_map[ids.long()].to(torch.int32)
            mask = ids >= 0
            ids = ids.clamp_min(0).contiguous()
        tokens, top_k = ids.shape
        native = torch.ops._C_gguf

        def projection(shard):
            capability = self.projection_capabilities[shard]
            return getattr(native, capability.operator)

        gate = projection("w1")(
            x.contiguous(),
            layer.gguf_w1,
            ids,
            self.weight_types["w1"],
            self.intermediate_size,
            top_k,
            tokens,
        )
        up = projection("w3")(
            x.contiguous(),
            layer.gguf_w3,
            ids,
            self.weight_types["w3"],
            self.intermediate_size,
            top_k,
            tokens,
        )
        hidden = torch.nn.functional.silu(gate) * up
        routed_ids = ids.reshape(-1, 1)
        down = projection("w2")(
            hidden.contiguous(),
            layer.gguf_w2,
            routed_ids,
            self.weight_types["w2"],
            self.hidden_size,
            1,
            tokens * top_k,
        )
        down = down.view(tokens, top_k, self.hidden_size)
        if mask is not None:
            down = torch.where(mask[..., None], down, 0)
        return (down.float() * topk_weights[..., None].float()).sum(1).to(x.dtype)
