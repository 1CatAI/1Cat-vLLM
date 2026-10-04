# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp tensor names and inverse converter transformations.

Format oracle: ggml-org/llama.cpp conversion/qwen4exp.py and tensor_mapping.py
at bed0a856606ee4a24a164066f73d2379447033f5 (MIT). GDN rules are shared with
the Apache-2.0 Qwen3.5/plugin adapter; no llama.cpp implementation is copied.
"""

import gguf
import numpy as np
import torch

from vllm.model_executor.layers.quantization.gguf_repack import q2_0_to_q4_1
from vllm.transformers_utils.gguf_tensor_reader import dequantize

from .qwen35 import _LAYERS, Qwen35Adapter

_HC = {
    f"hc_{branch}_{raw}.weight": f"{module}.{target}.weight"
    for branch, module in (
        ("attn", "attn_hyper_connection"),
        ("ffn", "mlp_hyper_connection"),
    )
    for raw, target in (
        ("norm", "hc_norm"),
        ("down", "input_mix_weight_down"),
        ("up", "input_mix_weight_up"),
        ("inject", "block_inject_weight"),
    )
}
_EXPERTS = {
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ffn_gate_exps.weight": "mlp.experts.gate_proj.weight",
    "ffn_up_exps.weight": "mlp.experts.up_proj.weight",
    "ffn_down_exps.weight": "mlp.experts.down_proj.weight",
}
_EXTRA = {
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
    "indexer.q_proj.weight": "self_attn.indexer.q_proj.weight",
    "indexer.k_proj.weight": "self_attn.indexer.k_proj.weight",
    "ple_key.weight": "ple.key_proj.weight",
    "ple_value.weight": "ple.value_proj.weight",
    "ple_norm_key.weight": "ple.norm_key.weight",
    "ple_norm_query.weight": "ple.norm_query.weight",
    "ple_norm_conv.weight": "ple.norm_conv.weight",
    "ple_conv1d.weight": "ple.conv1d.weight",
}


class Qwen4ExpAdapter(Qwen35Adapter):
    native_expert_storage = True
    architecture_label = "Qwen4Exp"
    layer_names = {**_LAYERS, **_HC, **_EXPERTS, **_EXTRA}
    global_names = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output.weight": "lm_head.weight",
        **{
            f"output_hc_{raw}.weight": f"model.hyper_connection_mixer.{target}.weight"
            for raw, target in (
                ("norm", "hc_norm"),
                ("down", "input_mix_weight_down"),
                ("up", "input_mix_weight_up"),
            )
        },
    }

    def build_name_map(self, tensors):
        table = "per_layer_token_embd.weight"
        regular = {key: value for key, value in tensors.items() if key != table}
        result = super().build_name_map(regular)
        if table in tensors:
            layers = self.config.ple_layer_ids
            if len(layers) != 1:
                raise ValueError(
                    "Qwen4Exp GGUF table requires one configured PLE layer"
                )
            result[table] = (
                f"model.layers.{layers[0] - 1}.ple.ple_embedding.ngram_embedding.weight"
            )
        return result

    @staticmethod
    def is_linear(name):
        # HC uses explicitly unquantized replicated/merged linears.
        return (
            Qwen35Adapter.is_linear(name)
            and "hyper_connection" not in name
            and not name.endswith("ngram_embedding.weight")
            and name != "lm_head.weight"
            and not name.endswith(
                (
                    ".ple.norm_key.weight",
                    ".ple.norm_query.weight",
                    ".ple.norm_conv.weight",
                )
            )
            and not name.endswith(
                (".mlp.gate.weight", ".mlp.shared_expert_gate.weight")
            )
        )

    def restore(self, name, weight):
        if name.endswith(".mlp.shared_expert_gate.weight") and weight.ndim == 1:
            return weight.unsqueeze(0)
        if ".ple." in name:
            if name.endswith(".conv1d.weight"):
                return weight.unsqueeze(1)
            if name.endswith(
                (".norm_key.weight", ".norm_query.weight", ".norm_conv.weight")
            ):
                return weight - 1
        return super().restore(name, weight)

    def needs_dense_fallback(self, name, tensor):
        if ".mlp.experts." in name:
            # Expert TP/EP admission is separate from the attention/dense TP
            # layout. Never silently dequantize a stacked expert checkpoint.
            return False
        return super().needs_dense_fallback(name, tensor)

    @staticmethod
    def _dense(tensor, dtype):
        if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            raw = torch.from_numpy(tensor.data.view(np.uint16).copy())
            value = raw.view(torch.bfloat16).float()
        elif tensor.tensor_type in (
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.F32,
        ):
            value = torch.from_numpy(tensor.data.copy())
        else:
            value = torch.from_numpy(dequantize(tensor.data, tensor.tensor_type))
        converted = value.to(dtype)
        if torch.any(torch.isfinite(value) & ~torch.isfinite(converted)):
            raise ValueError("Qwen4Exp GGUF indexer projection overflows target dtype")
        return converted

    def weights(self, tensors, name_map, dtype):
        regular = {}
        indexer_pairs: dict[str, dict[str, gguf.ReaderTensor]] = {}
        expert_tensors = {}
        table_entry = None
        for raw, name in name_map.items():
            if name.endswith("ngram_embedding.weight"):
                table_entry = (raw, name)
            elif ".mlp.experts." in name:
                expert_tensors[raw] = name
            elif name.endswith(("indexer.q_proj.weight", "indexer.k_proj.weight")):
                prefix, projection, _ = name.rsplit(".", 2)
                indexer_pairs.setdefault(prefix, {})[projection] = tensors[raw]
            else:
                regular[raw] = name
        yield from super().weights(tensors, regular, dtype)
        for prefix, pair in indexer_pairs.items():
            if set(pair) != {"q_proj", "k_proj"}:
                raise ValueError(f"Incomplete GGUF indexer Q/K pair: {prefix}")
            # The converter splits this replicated projection along output rows.
            # Its two parts can have distinct quantization. Restore a dense
            # projection before concatenating, rather than merging packed bytes.
            weight = torch.cat(
                [self._dense(pair[key], dtype) for key in ("q_proj", "k_proj")]
            )
            expected = (self.config.indexer_n_heads + 1) * self.config.indexer_head_dim
            if weight.shape != (expected, self.config.hidden_size):
                raise ValueError(f"Invalid GGUF indexer projection shape at {prefix}")
            yield (
                prefix + ".index_qk_proj.qweight_type",
                torch.tensor(int(gguf.GGMLQuantizationType.F16)),
            )
            yield prefix + ".index_qk_proj.qweight", weight
        for raw, name in expert_tensors.items():
            tensor = tensors[raw]
            if len(tensor.shape) != 3 or tensor.shape[2] != self.config.num_experts:
                raise ValueError(f"Invalid stacked GGUF expert shape: {raw}")
            prefix, projection, _ = name.rsplit(".", 2)
            floating = tensor.tensor_type in (
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.BF16,
            )
            repack = (
                int(tensor.tensor_type) == 42
                and projection == "down_proj"
                and int(tensor.shape[0]) % (64 * self.tp_size) != 0
            )
            storage_type = (
                int(
                    {
                        torch.float16: gguf.GGMLQuantizationType.F16,
                        torch.float32: gguf.GGMLQuantizationType.F32,
                        torch.bfloat16: gguf.GGMLQuantizationType.BF16,
                    }[dtype]
                )
                if floating
                else int(gguf.GGMLQuantizationType.Q4_1)
                if repack
                else int(tensor.tensor_type)
            )
            if repack:
                self.fallback_reasons[prefix + ".down_proj"] = (
                    "lossless_Q2_0_to_Q4_1_before_tp:"
                    f"local_K={int(tensor.shape[0]) // self.tp_size},block=32"
                )
            # Keep one type per logical gate/up/down projection. The expert
            # method handles local expert admission and TP storage boundaries.
            for expert in range(self.config.num_experts):
                module = f"{prefix}.{expert}.{projection}"
                yield module + ".qweight_type", torch.tensor(storage_type)
            packed = (
                self._dense(tensor, dtype)
                if floating
                else torch.from_numpy(tensor.data)
            )
            for expert in range(self.config.num_experts):
                weight = (
                    torch.from_numpy(q2_0_to_q4_1(tensor.data[expert]))
                    if repack
                    else packed[expert]
                )
                yield f"{prefix}.{expert}.{projection}.qweight", weight
        if table_entry is not None:
            raw, name = table_entry
            tensor = tensors[raw]
            constants = self.config.gguf_ple_constants
            count = (
                constants["ngram_heads_offsets"][-1]
                + constants["ngram_heads_vocab_sizes"][-1]
            )
            padded = (count + 127) // 128 * 128
            row_dim = self.config.ple_embed_dim // (
                (self.config.ngram_size - 1) * self.config.heads_per_ngram
            )
            if list(tensor.shape) != [row_dim, padded]:
                raise ValueError("GGUF PLE table shape disagrees with hash metadata")
            # Preserve packed mmap storage; never copy/dequantize the full table.
            yield (
                name.removesuffix(".weight") + ".qweight_type",
                torch.tensor(int(tensor.tensor_type)),
            )
            yield (
                name.removesuffix(".weight") + ".qweight",
                torch.from_numpy(tensor.data),
            )
            prefix = name.removesuffix(".ngram_embedding.weight")
            for constant, values in constants.items():
                yield f"{prefix}.{constant}", torch.tensor(values, dtype=torch.int64)
