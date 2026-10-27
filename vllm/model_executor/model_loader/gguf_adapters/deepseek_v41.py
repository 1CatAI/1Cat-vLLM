# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict V4.1 Flash backbone mapping, with independent Engram table loading.

Only stacked-expert transport is shared with the other adapters. DeepSeek
norms have no Qwen offset, and FP32 coefficients/hash integers must not pass
through the engine's FP16 parameter dtype.
"""

from types import SimpleNamespace
from typing import Any, cast

import gguf
import numpy as np
import torch

from .qwen35 import _dequantize_embedding, _validate_embedding_range
from .qwen35_moe import Qwen35MoeAdapter

_GLOBAL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
    "engram_token_map.weight": "model.engram_token_map",
    "engram_primes.weight": "model.engram_primes",
    "engram_offsets.weight": "model.engram_offsets",
    "engram_multipliers.weight": "model.engram_multipliers",
}
_LAYER = {
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_kv.weight": "attn.wkv.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_output_a.weight": "attn.wo_a.weight",
    "attn_output_b.weight": "attn.wo_b.weight",
    "attn_sinks.weight": "attn.attn_sink",
    "attn_compressor_kv.weight": "attn.compressor.wkv.weight",
    "attn_compressor_gate.weight": "attn.compressor.wgate.weight",
    "attn_compressor_norm.weight": "attn.compressor.norm.weight",
    "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
    "indexer.proj.weight": "attn.indexer.weights_proj.weight",
    "indexer.attn_k.weight": "attn.indexer.wk.weight",
    "indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "ffn_gate_exps.weight": "ffn.experts.gate_proj.weight",
    "ffn_up_exps.weight": "ffn.experts.up_proj.weight",
    "ffn_down_exps.weight": "ffn.experts.down_proj.weight",
    "ffn_gate_shexp.weight": "ffn.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.down_proj.weight",
    "engram_wkv.weight": "engram.wkv.weight",
    "engram_q.weight": "engram.q_weight",
    "engram_k.weight": "engram.k_weight",
    **{
        f"hc_{branch}_{part}.weight": f"hc_{branch}_{part}"
        for branch in ("attn", "ffn")
        for part in ("fn", "base", "scale")
    },
}
_PROJECTIONS = (
    ".wq_a.weight",
    ".wq_b.weight",
    ".wkv.weight",
    ".wo_a.weight",
    ".wo_b.weight",
    ".wgate.weight",
    ".weights_proj.weight",
    ".wk.weight",
    ".shared_experts.gate_proj.weight",
    ".shared_experts.up_proj.weight",
    ".shared_experts.down_proj.weight",
)
_FLOATING = (
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.BF16,
)


class DeepseekV41Adapter:
    architecture_label = "DeepSeek V4.1 Flash"
    language_model_prefix = ""
    packed_embeddings = True
    native_expert_storage = True
    _dense = staticmethod(Qwen35MoeAdapter._dense)
    _expert_weights = cast(Any, Qwen35MoeAdapter._expert_weights)

    def __init__(self, config, tp_size=1):
        fields = config.to_dict()
        fields["num_experts"] = fields.get("n_routed_experts")
        self.config = SimpleNamespace(**fields)
        self.tp_size, self.fallback_reasons = tp_size, {}
        c = self.config
        if (
            c.hidden_size,
            c.moe_intermediate_size,
            c.num_hidden_layers,
            c.n_routed_experts,
            c.num_experts_per_tok,
            c.hc_mult,
        ) != (5120, 2304, 40, 384, 6, 4):
            raise ValueError("GGUF adapter requires the V4.1 Flash geometry")
        self.expected = self.expected_shapes()

    def expected_shapes(self):
        c = self.config
        h, inter, experts = c.hidden_size, c.moe_intermediate_size, c.n_routed_experts
        common = {
            "attn_norm.weight": (h,),
            "ffn_norm.weight": (h,),
            "attn_q_a.weight": (h, c.q_lora_rank),
            "attn_q_a_norm.weight": (c.q_lora_rank,),
            "attn_q_b.weight": (c.q_lora_rank, c.num_attention_heads * c.head_dim),
            "attn_kv.weight": (h, c.head_dim),
            "attn_kv_a_norm.weight": (c.head_dim,),
            "attn_output_a.weight": (
                c.num_attention_heads * c.head_dim // c.o_groups,
                c.o_lora_rank,
                c.o_groups,
            ),
            "attn_output_b.weight": (c.o_groups * c.o_lora_rank, h),
            "attn_sinks.weight": (c.num_attention_heads,),
            "ffn_gate_inp.weight": (h, experts),
            "exp_probs_b.bias": (experts,),
            "ffn_gate_exps.weight": (h, inter, experts),
            "ffn_up_exps.weight": (h, inter, experts),
            "ffn_down_exps.weight": (inter, h, experts),
            "ffn_gate_shexp.weight": (h, inter),
            "ffn_up_shexp.weight": (h, inter),
            "ffn_down_shexp.weight": (inter, h),
            **{
                f"hc_{branch}_{part}.weight": shape
                for branch in ("attn", "ffn")
                for part, shape in (
                    ("fn", (c.hc_mult * h, (2 + c.hc_mult) * c.hc_mult)),
                    ("base", ((2 + c.hc_mult) * c.hc_mult,)),
                    ("scale", (3,)),
                )
            },
        }
        result = {
            "token_embd.weight": (h, c.vocab_size),
            "output.weight": (h, c.vocab_size),
            "output_norm.weight": (h,),
            "engram_token_map.weight": (c.vocab_size,),
            "engram_primes.weight": (24, 2),
            "engram_offsets.weight": (24, 2),
            "engram_multipliers.weight": (4, 2),
        }
        for layer in range(c.num_hidden_layers):
            values = dict(common)
            if layer in c.kv_source_layer_ids:
                values.update(
                    {
                        "attn_compressor_kv.weight": (h, c.head_dim),
                        "attn_compressor_norm.weight": (c.head_dim,),
                        "indexer.attn_k.weight": (c.head_dim, c.index_head_dim),
                        "indexer.k_norm.weight": (c.index_head_dim,),
                    }
                )
                if c.compress_ratios[layer] == 2:
                    values["attn_compressor_gate.weight"] = (h, c.head_dim)
            if layer in c.index_source_layer_ids:
                values.update(
                    {
                        "indexer.attn_q_b.weight": (
                            c.q_lora_rank,
                            c.index_n_heads * c.index_head_dim,
                        ),
                        "indexer.proj.weight": (h, c.index_n_heads),
                    }
                )
            if layer in c.engram_layer_ids:
                values.update(
                    {
                        "engram_q.weight": (h, c.hc_mult),
                        "engram_k.weight": (h, c.hc_mult),
                        "engram_wkv.weight": (6144, (c.hc_mult + 1) * h),
                    }
                )
            result.update(
                {f"blk.{layer}.{name}": shape for name, shape in values.items()}
            )
        return result

    def build_name_map(self, tensors):
        missing, extra = (
            self.expected.keys() - tensors.keys(),
            tensors.keys() - self.expected.keys(),
        )
        if missing or extra:
            raise ValueError(
                f"V4.1 GGUF coverage mismatch: missing {len(missing)} "
                f"{sorted(missing)[:8]}, extra {len(extra)} {sorted(extra)[:8]}"
            )
        result = {}
        for raw, tensor in tensors.items():
            if tuple(map(int, tensor.shape)) != self.expected[raw]:
                raise ValueError(f"V4.1 GGUF shape mismatch: {raw}")
            if "_exps.weight" in raw and int(tensor.tensor_type) not in (16, 17, 29):
                raise ValueError(f"Unsupported V4.1 original expert format: {raw}")
            if raw in _GLOBAL:
                result[raw] = _GLOBAL[raw]
            else:
                _, layer, suffix = raw.split(".", 2)
                result[raw] = f"model.layers.{layer}.{_LAYER[suffix]}"
        if len(set(result.values())) != len(result):
            raise ValueError("V4.1 mapping aliases distinct source tensors")
        for layer in range(self.config.num_hidden_layers):
            if (
                tensors[f"blk.{layer}.ffn_gate_exps.weight"].tensor_type
                != tensors[f"blk.{layer}.ffn_up_exps.weight"].tensor_type
            ):
                raise ValueError(f"V4.1 gate/up formats differ at layer {layer}")
        return result

    @staticmethod
    def is_linear(name):
        return name == "lm_head.weight" or name.endswith(_PROJECTIONS)

    def needs_dense_fallback(self, name, tensor):
        if name.endswith(".shared_experts.down_proj.weight"):
            block, _ = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            if int(tensor.shape[0]) // self.tp_size % block:
                self.fallback_reasons[name.removesuffix(".weight")] = (
                    "shared_down_tp_boundary:bounded_FP16_restore"
                )
                return True
        return False

    @staticmethod
    def _restore_shared_down(tensor, dtype):
        # One unconditional shared expert, not the routed expert library.
        # The complete FP16 source is <24 MiB; 128-row FP32 decode chunks
        # keep the restore's temporary footprint below 32 MiB. The ordinary
        # row-parallel loader then owns only 576 FP16 columns on each rank.
        rows, columns = tensor.data.shape[0], int(tensor.shape[0])
        result = torch.empty(rows, columns, dtype=dtype)
        for first in range(0, rows, 128):
            values = gguf.quants.dequantize(
                tensor.data[first : first + 128], tensor.tensor_type
            )
            if (
                not np.isfinite(values).all()
                or np.abs(values).max() > torch.finfo(dtype).max
            ):
                raise ValueError("Shared-expert GGUF restore exceeds compute dtype")
            result[first : first + len(values)].copy_(torch.from_numpy(values))
        return result

    def linear_layouts(self, name_map):
        return {}

    def _projection_dtype(self, name, dtype):
        # The ratio-2 compressor pools unrounded FP32 linear values/scores.
        # Its checkpoint weights are BF16; ratio 1 keeps the FP16 boundary.
        if ".attn.compressor." in name and name.endswith(
            (".wkv.weight", ".wgate.weight")
        ):
            layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
            if self.config.compress_ratios[layer] == 2:
                return torch.float32
        return dtype

    def weights(self, tensors, name_map, dtype):
        experts = {
            raw: name for raw, name in name_map.items() if ".ffn.experts." in name
        }
        regular = {raw: name for raw, name in name_map.items() if raw not in experts}
        for raw, name in regular.items():
            if self.is_linear(name) or (
                self.packed_embeddings and raw == "token_embd.weight"
            ):
                source_type = tensors[raw].tensor_type
                if source_type in _FLOATING or self.needs_dense_fallback(
                    name, tensors[raw]
                ):
                    source_type = {
                        torch.float16: gguf.GGMLQuantizationType.F16,
                        torch.bfloat16: gguf.GGMLQuantizationType.BF16,
                        torch.float32: gguf.GGMLQuantizationType.F32,
                    }[self._projection_dtype(name, dtype)]
                yield (
                    name.removesuffix(".weight") + ".qweight_type",
                    torch.tensor(int(source_type)),
                )
        for raw, name in regular.items():
            tensor = tensors[raw]
            if raw.startswith("engram_"):
                expected = (
                    gguf.GGMLQuantizationType.I32
                    if raw == "engram_token_map.weight"
                    else gguf.GGMLQuantizationType.I64
                )
                if tensor.tensor_type != expected:
                    raise ValueError(
                        f"V4.1 hash tensor has an invalid integer type: {raw}"
                    )
                yield name, torch.from_numpy(tensor.data.copy())
            elif raw == "token_embd.weight":
                if self.packed_embeddings:
                    if tensor.tensor_type in _FLOATING:
                        weight = self._dense(tensor, dtype)
                    else:
                        _validate_embedding_range(tensor, dtype, raw)
                        weight = torch.from_numpy(tensor.data.copy())
                    yield "model.embed_tokens.qweight", weight
                else:
                    yield name, _dequantize_embedding(tensor, dtype, raw)
            elif self.is_linear(name):
                weight = (
                    self._restore_shared_down(tensor, dtype)
                    if self.needs_dense_fallback(name, tensor)
                    else self._dense(tensor, self._projection_dtype(name, dtype))
                    if tensor.tensor_type in _FLOATING
                    else torch.from_numpy(np.ascontiguousarray(tensor.data).copy())
                )
                # The eight independent O groups are contiguous output row
                # ranges. Keep their packed bytes, flattening only the group axis.
                if raw.endswith("attn_output_a.weight"):
                    weight = weight.reshape(-1, weight.shape[-1])
                yield name.removesuffix(".weight") + ".qweight", weight
            else:
                yield name, self._dense(tensor, torch.float32)
        yield from self._expert_weights(tensors, experts, dtype)
