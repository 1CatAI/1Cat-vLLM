# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF experts with independent gate/up/down storage and aligned TP slices."""

from dataclasses import asdict
from typing import TYPE_CHECKING

import torch
from torch.nn import Parameter

from vllm._sm70.policy import register_policy_op
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.kernels.gguf import (
    GGUFDecoderFamily,
    admit_moe_fallback,
    decoder_family,
    original_moe_prefill_capability,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    MoEActivation,
)
from vllm.model_executor.layers.quantization.gguf_native import (
    NATIVE_TYPES,
    empty_guarded_weight,
    native_available,
    packed_tp_span,
    pad_weight_tail,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.gguf_tensor_reader import quant_size, quant_type_name

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig


def _original_expert_dp4a(
    x,
    ids,
    probabilities,
    gate,
    up,
    down,
    source_type,
    down_type,
    left,
    quantized_hidden,
    bank_aware,
):
    m, top_k = ids.shape
    n = gate.shape[1]
    q8 = torch.empty((m, x.shape[1] // 32, 36), dtype=torch.uint8, device=x.device)
    hidden = (
        torch.empty((m, top_k, n // 32, 36), dtype=torch.uint8, device=x.device)
        if quantized_hidden
        else x.new_empty((m, top_k, n))
    )
    output = torch.empty_like(x)
    torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
    torch.ops._C.gguf_dp4a_gate_up_sm70_out(
        hidden,
        q8,
        ids,
        gate,
        up,
        source_type,
        True,
        4 if quantized_hidden and source_type in (20, 23) else 16,
        bank_aware,
    )
    torch.ops._C.gguf_dp4a_raw_down_unroute_sm70_out(
        output,
        hidden,
        ids,
        probabilities,
        down,
        down_type,
        left,
    )
    return output


def _original_expert_dp4a_fake(
    x,
    ids,
    probabilities,
    gate,
    up,
    down,
    source_type,
    down_type,
    left,
    quantized_hidden,
    bank_aware,
):
    return torch.empty_like(x)


register_policy_op(
    "gguf_original_expert_dp4a",
    "(Tensor x, Tensor ids, Tensor probabilities, Tensor gate, Tensor up, "
    "Tensor down, int source_type, int down_type, int left, "
    "bool quantized_hidden, bool bank_aware) -> Tensor",
    _original_expert_dp4a,
    _original_expert_dp4a_fake,
)


class GGUFNativeMoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: "GGUFConfig", moe):
        super().__init__(moe)
        self.quant_config = quant_config
        config = get_current_vllm_config_or_none()
        self.native_enabled = (
            config.kernel_config.sm70_gguf.enabled if config is not None else True
        )
        self.small_m_dp4a = bool(config and config.kernel_config.sm70_gguf.small_m_dp4a)
        self.lut4_expert_dp4a = bool(
            config and config.kernel_config.sm70_gguf.lut4_expert_dp4a
        )
        self.q8_intermediate = bool(
            config and config.kernel_config.sm70_gguf.q8_expert_intermediate
        )
        self.weight_types: dict[str, int] = {}
        self.guarded_shards: set[str] = set()
        self.input_padding: dict[str, tuple[int, int]] = {}
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
        if k % block and not (
            shard_id == "w2" and value in NATIVE_TYPES and weight.dtype == torch.uint8
        ):
            raise ValueError(
                f"GGUF {shard_id} local K={k} is not aligned to block {block}; "
                "convert the checkpoint projection before TP slicing"
            )
        tp_size, tp_rank = layer.tp_size, layer.tp_rank
        if shard_id == "w2" and weight.dtype == torch.uint8:
            first, last, left, physical_k = packed_tp_span(
                k * tp_size, value, tp_size, tp_rank
            )
            packed_k = last - first
            expected = (rows, k * tp_size // block * size)
            local = weight[:, first:last]
            self.input_padding[shard_id] = (left, physical_k)
        else:
            packed_k = k // block * size if weight.dtype == torch.uint8 else k
            expected = (
                (rows, packed_k * tp_size)
                if shard_id == "w2"
                else (rows * tp_size, packed_k)
            )
            local = (
                weight[:, tp_rank * packed_k : (tp_rank + 1) * packed_k]
                if shard_id == "w2"
                else weight[tp_rank * rows : (tp_rank + 1) * rows]
            )
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"GGUF {shard_id} shape {tuple(weight.shape)} != {expected}"
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
        self.prefill_capabilities = {}
        self.native_admission["prefill_projections"] = {}
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
            block, size = quant_size(value)
            k = (
                prepared.shape[-1] // size * block
                if prepared.dtype == torch.uint8
                else prepared.shape[-1]
            )
            prefill = original_moe_prefill_capability(
                value,
                k,
                prepared.shape[1],
                self.num_experts,
                self.params_dtype,
                is_sm70=current_platform.is_device_capability(70),
                enabled=self.native_enabled,
            )
            self.prefill_capabilities[shard] = prefill
            self.native_admission["prefill_projections"][shard] = asdict(prefill)
            self.native_admission["projections"][shard] = {
                **asdict(capability),
                "shape": list(weight.shape),
                "activation_padding": list(self.input_padding.get(shard, (0, 0))),
            }
        self.dp4a_admitted = bool(
            self.small_m_dp4a
            and self.params_dtype == torch.float16
            and (self.num_experts, self.hidden_size, self.intermediate_size)
            == (512, 2560, 160)
            and self.weight_types["w1"] == self.weight_types["w3"]
            and self.weight_types["w1"] in (18, 20, 21, 22, 23)
            and (self.weight_types["w1"] not in (20, 23) or self.lut4_expert_dp4a)
            and self.weight_types["w2"] in (20, 42)
            and hasattr(torch.ops._C, "gguf_dp4a_raw_down_unroute_sm70_out")
        )
        self.native_admission["small_m_dp4a"] = {
            "enabled": self.dp4a_admitted,
            "reason": None
            if self.dp4a_admitted
            else "original_dp4a_shape_or_operator_unavailable",
            "min_m": 1,
            "max_m": 20,
            "storage": "original",
            "accumulation": "FP32",
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
        if (
            self.dp4a_admitted
            and 1 <= tokens <= 20
            and layer.expert_map is None
            and topk_weights.dtype == torch.float32
        ):
            return torch.ops.vllm.gguf_original_expert_dp4a(
                x.contiguous(),
                ids,
                topk_weights.contiguous(),
                layer.gguf_w1,
                layer.gguf_w3,
                layer.gguf_w2,
                self.weight_types["w1"],
                self.weight_types["w2"],
                self.input_padding.get("w2", (0, 0))[0],
                self.q8_intermediate,
                self.weight_types["w1"] == 21,
            )
        native = torch.ops._C_gguf

        def projection(shard):
            capability = self.projection_capabilities[shard]
            prefill = getattr(self, "prefill_capabilities", {}).get(shard)
            m = tokens * top_k if shard == "w2" else tokens
            if prefill is not None and prefill.reason is None and prefill.supports_m(m):
                capability = prefill
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
        left, physical_k = self.input_padding.get("w2", (0, self.intermediate_size))
        if left or physical_k != self.intermediate_size:
            hidden = torch.nn.functional.pad(
                hidden, (left, physical_k - left - self.intermediate_size)
            )
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
