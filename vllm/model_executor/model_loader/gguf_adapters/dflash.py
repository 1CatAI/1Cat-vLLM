# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore DFlash2 GGUF tensors to the existing draft weight loader.

Tensor names follow llama.cpp conversion/qwen.py and gguf tensor_mapping.py
at bed0a856606ee4a24a164066f73d2379447033f5 (MIT). Qwen3 norms and NeoX
RoPE projections are already in runtime order, unlike Qwen3.5 GDN weights.
"""

import regex as re

from .qwen35 import Qwen35Adapter

_GLOBALS = {
    "enc.output_norm.weight": "hidden_norm.weight",
    "fc.weight": "fc.weight",
    "output_norm.weight": "norm.weight",
    "selector_hidden.weight": "candidate_selector.hidden_projection.weight",
    "selector_predecessor.weight": "candidate_selector.predecessor_codebook",
    "selector_successor.weight": "candidate_selector.successor_codebook",
}
_LAYERS = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_conv_base": "attention_conv.base_kernel",
    "attn_conv_proj.weight": "attention_conv.kernel_projection.weight",
    "ffn_conv_base": "mlp_conv.base_kernel",
    "ffn_conv_proj.weight": "mlp_conv.kernel_projection.weight",
}


class DFlashAdapter(Qwen35Adapter):
    packed_embeddings = False

    def __init__(self, config, tp_size=1):
        self.config = config
        self.tp_size = tp_size
        self.layout = None
        self.fallback_reasons = {}

    def build_name_map(self, tensors):
        result = {}
        for name in tensors:
            if name in _GLOBALS:
                result[name] = _GLOBALS[name]
                continue
            match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
            if (
                match
                and 0 <= int(match[1]) < self.config.num_hidden_layers
                and match[2] in _LAYERS
            ):
                # DFlashForCausalLM adds the model. prefix before delegation.
                result[name] = f"layers.{match[1]}.{_LAYERS[match[2]]}"
                continue
            raise ValueError(f"Unmapped DFlash GGUF tensor: {name}")
        return result

    @staticmethod
    def is_linear(name):
        # The convolution and selector modules own dense parameters. Decode
        # these during loading; keep backbone projections in canonical form.
        return name == "fc.weight" or any(
            name.endswith(f".{projection}.weight")
            for projection in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        )

    def restore(self, name, weight):
        return weight
