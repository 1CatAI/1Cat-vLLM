# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp tensor names and inverse converter transformations.

Format oracle: ggml-org/llama.cpp conversion/qwen4exp.py and tensor_mapping.py
at bed0a856606ee4a24a164066f73d2379447033f5 (MIT). GDN rules are shared with
the Apache-2.0 Qwen3.5/plugin adapter; no llama.cpp implementation is copied.
"""

import gguf
import torch

from .qwen35 import _LAYERS, Qwen35Adapter
from .qwen35_moe import _EXPERTS, Qwen35MoeAdapter

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


class Qwen4ExpAdapter(Qwen35MoeAdapter):
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
        yield from self._expert_weights(tensors, expert_tensors, dtype)
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
