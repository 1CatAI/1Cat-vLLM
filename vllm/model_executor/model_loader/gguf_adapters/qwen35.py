# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore llama.cpp Qwen3.5 dense weights to vLLM checkpoint layout.

Name and GDN layout rules are adapted from vllm-gguf-plugin at
e2b8ad532b8b5ea175100202c30430c1d2b5e6a8 (Apache-2.0).
"""

import gguf
import numpy as np
import regex as re
import torch

from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size

_GLOBALS = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
_LAYERS = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_a.weight": "linear_attn.A_log",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def _embedding_chunks(tensor, dtype, name, rows_per_chunk=1024):
    rows, columns = map(int, tensor.shape[::-1])
    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        data = gguf.quants.dequantize(tensor.data[start:stop], tensor.tensor_type)
        decoded = torch.from_numpy(data)
        converted = decoded.to(dtype)
        if torch.any(torch.isfinite(decoded) & ~torch.isfinite(converted)):
            raise ValueError(
                f"GGUF {name}: values overflow {dtype}; use --dtype float32"
            )
        yield start, stop, converted


def _dequantize_embedding(tensor, dtype, name, rows_per_chunk=1024):
    """Bound temporary decode/range-check memory for dense vocabulary tables."""
    rows, columns = map(int, tensor.shape[::-1])
    weight = torch.empty((rows, columns), dtype=dtype, device="cpu")
    for start, stop, converted in _embedding_chunks(
        tensor, dtype, name, rows_per_chunk
    ):
        weight[start:stop].copy_(converted)
    return weight


class Qwen35Adapter:
    global_names = _GLOBALS
    layer_names = _LAYERS
    architecture_label = "Qwen3.5"
    packed_embeddings = True

    def __init__(self, config, tp_size=1):
        self.config = config
        self.tp_size = tp_size
        self.fallback_reasons = {}
        repeat, remainder = divmod(
            config.linear_num_value_heads, config.linear_num_key_heads
        )
        if remainder or repeat < 1:
            raise ValueError("GGUF GDN value heads must be a multiple of key heads")
        self.layout = (
            GGUFHeadTilingLayout(repeat, config.linear_value_head_dim)
            if repeat > 1
            else None
        )

    def build_name_map(self, tensors):
        result = {}
        for name in tensors:
            if name in self.global_names:
                result[name] = self.global_names[name]
                continue
            match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
            if match:
                block, suffix = int(match[1]), match[2]
                if block >= self.config.num_hidden_layers:
                    # The target does not execute nextn. Its draft adapter has a
                    # separate mapping and must consume these tensors explicitly.
                    if block < self.config.num_hidden_layers + getattr(
                        self.config, "num_nextn_predict_layers", 0
                    ):
                        continue
                elif suffix in self.layer_names:
                    result[name] = f"model.layers.{block}.{self.layer_names[suffix]}"
                    continue
            raise ValueError(f"Unmapped {self.architecture_label} GGUF tensor: {name}")
        return result

    @staticmethod
    def is_linear(name):
        return name.endswith(".weight") and not name.endswith(
            ("norm.weight", "conv1d.weight", "embed_tokens.weight")
        )

    def needs_dense_fallback(self, name, tensor):
        if not name.endswith(
            (".down_proj.weight", ".out_proj.weight", ".o_proj.weight")
        ):
            return False
        block_size, _ = quant_size(tensor.tensor_type)
        local_k, remainder = divmod(int(tensor.shape[0]), self.tp_size)
        tiled_span = local_k
        if self.layout is not None and name.endswith("linear_attn.out_proj.weight"):
            tiled_span //= self.layout.heads_per_group
        if remainder or local_k % block_size or tiled_span % block_size:
            self.fallback_reasons[name.removesuffix(".weight")] = (
                f"dequantize_to_fp16_before_tp:local_K={local_k}, "
                f"stored_tile_K={tiled_span}, GGML_block={block_size}"
            )
            return True
        return False

    def linear_layouts(self, name_map):
        if self.layout is None:
            return {}
        return {
            name.removesuffix(".weight"): self.layout
            for name in name_map.values()
            if name.endswith("linear_attn.out_proj.weight")
        }

    def restore(self, name, weight):
        if name.endswith(".A_log"):
            if not torch.all(weight < 0):
                raise ValueError(f"GGUF {name} must contain negative exp(A_log)")
            weight = torch.log(-weight.float())
        if self.layout is not None:
            layout = self.layout
            if name.endswith(
                (".A_log", ".dt_bias", "in_proj_a.weight", "in_proj_b.weight")
            ):
                weight = layout.weight_to_vllm(weight, dim=0, head_dim=1)
            elif name.endswith("in_proj_z.weight"):
                weight = layout.weight_to_vllm(weight, dim=0)
            elif name.endswith(("in_proj_qkv.weight", "conv1d.weight")):
                qk = (
                    self.config.linear_key_head_dim
                    * self.config.linear_num_key_heads
                    * 2
                )
                weight = torch.cat(
                    [weight[:qk], layout.weight_to_vllm(weight[qk:], dim=0)]
                )
        if name.endswith("norm.weight") and not name.endswith(
            "linear_attn.norm.weight"
        ):
            weight = weight - 1
        if name.endswith("conv1d.weight"):
            weight = weight.unsqueeze(1)
        return weight

    def weights(self, tensors, name_map, dtype):
        # All linear type descriptors must arrive before any weight payload,
        # including F16/F32 shards mixed with packed GGML shards.
        for raw, name in name_map.items():
            packed_embedding = (
                self.packed_embeddings
                and name.endswith("embed_tokens.weight")
                and tensors[raw].tensor_type
                not in (
                    gguf.GGMLQuantizationType.F32,
                    gguf.GGMLQuantizationType.F16,
                    gguf.GGMLQuantizationType.BF16,
                )
            )
            if self.is_linear(name) or packed_embedding:
                yield (
                    name.removesuffix(".weight") + ".qweight_type",
                    torch.tensor(
                        int(gguf.GGMLQuantizationType.F16)
                        if self.needs_dense_fallback(name, tensors[raw])
                        else int(tensors[raw].tensor_type)
                    ),
                )
        for raw, name in name_map.items():
            tensor = tensors[raw]
            quantized = tensor.tensor_type not in (
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.BF16,
            )
            if quantized and name.endswith("embed_tokens.weight"):
                if self.packed_embeddings:
                    # Validate the same FP16 range as the dense loader, using
                    # only a bounded row chunk. Serving decodes selected rows
                    # from the original GGML bytes through GGUFEmbeddingMethod.
                    for _ in _embedding_chunks(tensor, dtype, raw):
                        pass
                    yield (
                        name.removesuffix(".weight") + ".qweight",
                        torch.from_numpy(tensor.data.copy()),
                    )
                    continue
                # Embedding row order is unchanged by restoration. Keep the
                # existing global-table contract for the TP weight loader,
                # without full FP32 decode and boolean temporary tables.
                yield name, _dequantize_embedding(tensor, dtype, raw)
                continue
            dense_fallback = self.needs_dense_fallback(name, tensor)
            if quantized and (not self.is_linear(name) or dense_fallback):
                # Embeddings and convolution use the model's dense parameters.
                data = dequantize(tensor.data, tensor.tensor_type)
                # Dequantization owns this dense array; avoid a second copy.
                weight = torch.from_numpy(data)
            elif tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
                data = tensor.data.view(np.uint16).copy()
                weight = torch.from_numpy(data).view(torch.bfloat16)
            else:
                weight = torch.from_numpy(tensor.data.copy())
            weight = self.restore(name, weight)
            if not quantized or dense_fallback or not self.is_linear(name):
                # The model keeps A_log in FP32 even under --dtype half.
                # Preserve inverse-log precision rather than rounding through
                # FP16 before the destination parameter copies it back to FP32.
                target_dtype = torch.float32 if name.endswith(".A_log") else dtype
                converted = weight.to(target_dtype)
                if torch.any(torch.isfinite(weight) & ~torch.isfinite(converted)):
                    raise ValueError(
                        f"GGUF {raw}: values overflow {dtype}; use --dtype float32"
                    )
                weight = converted
            if self.is_linear(name):
                name = name.removesuffix(".weight") + ".qweight"
            yield name, weight
