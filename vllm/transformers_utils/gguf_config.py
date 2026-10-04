# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read model configuration from GGUF, without Transformers' GGUF parser.

Metadata names follow ggml-org/llama.cpp's GGUF specification and converters.
This module constructs configuration objects only; tensor conversion belongs
to the model loader's architecture adapters.
"""

from pathlib import Path
from typing import Any

import gguf
from transformers import AutoConfig, PretrainedConfig

# GGUF architecture, HF configuration type, vLLM model implementation.
_ARCHITECTURES = {
    "llama": ("llama", "LlamaForCausalLM"),
    "qwen2": ("qwen2", "Qwen2ForCausalLM"),
    "qwen3": ("qwen3", "Qwen3ForCausalLM"),
    "qwen35": ("qwen3_5_text", "Qwen3_5ForCausalLM"),
    "qwen35moe": ("qwen3_5_moe_text", "Qwen3_5MoeForCausalLM"),
    "dflash": ("qwen3", "DFlash2DraftModel"),
}


class _MetadataReader(gguf.GGUFReader):
    def _build_tensors(self, _offset, _fields):
        # The config/tokenizer stage must work even when a checkpoint contains
        # a newer GGML type unknown to the installed gguf package. Tensor
        # payload validation belongs to the architecture/kernel loading stage.
        pass


def read_gguf_metadata(path: str | Path) -> dict[str, Any]:
    """Read the header and tensor directory; never materialize weight data."""
    reader = _MetadataReader(path)
    return {
        name: field.contents()
        for name, field in reader.fields.items()
        if not name.startswith("GGUF.")
    }


def gguf_config_dict(metadata: dict[str, Any]) -> dict[str, Any]:
    arch = metadata.get("general.architecture")
    if arch not in _ARCHITECTURES:
        raise ValueError(
            f"No native GGUF config adapter for architecture {arch!r}. "
            "Provide --hf-config-path with an explicit model config."
        )
    model_type, implementation = _ARCHITECTURES[arch]

    def required(key: str):
        full_key = f"{arch}.{key}"
        if full_key not in metadata:
            raise ValueError(f"Missing required GGUF metadata: {full_key}")
        return metadata[full_key]

    def positive(key: str) -> int:
        value = required(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"GGUF {arch}.{key} must be a positive integer")
        return value

    def optional(key: str, default=None):
        return metadata.get(f"{arch}.{key}", default)

    hidden = positive("embedding_length")
    heads = positive("attention.head_count")
    head_dim = optional("attention.key_length")
    if head_dim is None:
        head_dim, remainder = divmod(hidden, heads)
        if remainder:
            raise ValueError("GGUF embedding_length is not divisible by head_count")
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError("GGUF attention.key_length must be a positive integer")
    nextn = optional("nextn_predict_layers", 0)
    layers = positive("block_count") - nextn
    if layers <= 0:
        raise ValueError("GGUF contains no backbone layers (MTP-only checkpoint)")
    tokens = metadata.get("tokenizer.ggml.tokens")
    vocab_size = optional("vocab_size", len(tokens) if tokens is not None else None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("GGUF requires vocab_size or tokenizer.ggml.tokens")
    config = {
        "model_type": model_type,
        "architectures": [implementation],
        "hidden_size": hidden,
        "intermediate_size": (
            positive("expert_feed_forward_length")
            if arch == "qwen35moe"
            else positive("feed_forward_length")
        ),
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": optional("attention.head_count_kv", heads),
        "head_dim": head_dim,
        "max_position_embeddings": positive("context_length"),
        "vocab_size": vocab_size,
        "rms_norm_eps": required("attention.layer_norm_rms_epsilon"),
        "hidden_act": "silu",
        "torch_dtype": "float16",
        "tie_word_embeddings": False,
        "gguf_architecture": arch,
        "num_nextn_predict_layers": nextn,
    }
    for key in ("bos", "eos", "padding"):
        token_id = metadata.get(f"tokenizer.ggml.{key}_token_id")
        if token_id is not None:
            config[f"{'pad' if key == 'padding' else key}_token_id"] = token_id

    rope = {"rope_type": "default", "rope_theta": optional("rope.freq_base", 10000.0)}
    scaling = optional("rope.scaling.type", "none")
    if scaling not in ("none", "linear", "yarn"):
        # E.g. llama3 needs frequency factors not represented by these keys.
        raise ValueError(
            f"GGUF RoPE scaling {scaling!r} needs an explicit --hf-config-path"
        )
    if scaling != "none":
        rope.update(rope_type=scaling, factor=required("rope.scaling.factor"))
        if (original := optional("rope.scaling.original_context_length")) is not None:
            rope["original_max_position_embeddings"] = original
    rotary_dim = optional("rope.dimension_count", head_dim)
    rope["partial_rotary_factor"] = rotary_dim / head_dim
    config["partial_rotary_factor"] = rotary_dim / head_dim
    config["rope_parameters"] = rope
    if arch == "dflash":
        target_layers = required("target_layers")
        if (
            not isinstance(target_layers, list)
            or not target_layers
            or any(
                isinstance(i, bool) or not isinstance(i, int) or i <= 0
                for i in target_layers
            )
            or target_layers != sorted(set(target_layers))
        ):
            raise ValueError("GGUF dflash.target_layers must be ordered 1-based IDs")
        mask_id = metadata.get("tokenizer.ggml.mask_token_id")
        if (
            isinstance(mask_id, bool)
            or not isinstance(mask_id, int)
            or not 0 <= mask_id < vocab_size
        ):
            raise ValueError("GGUF DFlash requires a mask token within the vocabulary")
        draft: dict[str, Any] = {
            key: positive(key)
            for key in (
                "block_size",
                "conv_kernel_size",
                "conv_group_size",
                "selector_rank",
                "selector_top_k",
            )
        }
        if draft["block_size"] < 2 or hidden % draft["conv_group_size"]:
            raise ValueError("GGUF DFlash block/group dimensions are incompatible")
        draft.update(
            mask_token_id=mask_id,
            # llama.cpp writes extraction positions as layer index + 1.
            target_layer_ids=[i - 1 for i in target_layers],
        )
        for source, destination in {
            "logit_scale": "output_multiplier",
            "final_logit_softcapping": "final_logit_softcapping",
            "embedding_scale": "input_embedding_scale",
            "attention.value_scale": "attention_value_scale",
        }.items():
            if (value := optional(source)) is not None:
                draft[destination] = value
        window = optional("attention.sliding_window")
        pattern = optional("attention.sliding_window_pattern", [False] * layers)
        if len(pattern) != layers or any(not isinstance(i, bool) for i in pattern):
            raise ValueError("GGUF DFlash sliding-window pattern must match layers")
        if any(pattern) and (
            isinstance(window, bool) or not isinstance(window, int) or window <= 0
        ):
            raise ValueError("GGUF DFlash sliding layers require a positive window")
        causal = optional("attention.causal", False)
        if not isinstance(causal, bool):
            raise ValueError("GGUF DFlash attention.causal must be boolean")
        config.update(
            torch_dtype="bfloat16",
            is_causal=causal,
            dflash_config=draft,
            sliding_window=window,
            use_sliding_window=any(pattern),
            max_window_layers=layers,
            layer_types=[
                "sliding_attention" if sliding else "full_attention"
                for sliding in pattern
            ],
        )
    if arch.startswith("qwen35"):
        for key, target in {
            "ssm.conv_kernel": "linear_conv_kernel_dim",
            "ssm.state_size": "linear_key_head_dim",
            "ssm.group_count": "linear_num_key_heads",
            "ssm.time_step_rank": "linear_num_value_heads",
        }.items():
            config[target] = positive(key)
        inner = positive("ssm.inner_size")
        value_dim, remainder = divmod(inner, config["linear_num_value_heads"])
        if remainder:
            raise ValueError("GGUF ssm.inner_size is not divisible by time_step_rank")
        config["linear_value_head_dim"] = value_dim
        recurrent = optional("attention.recurrent_layers")
        if recurrent is None:
            interval = optional("full_attention_interval", 4)
            if not isinstance(interval, int) or interval <= 0:
                raise ValueError("GGUF full_attention_interval must be positive")
            recurrent = [(i + 1) % interval != 0 for i in range(layers)]
        if len(recurrent) not in (layers, layers + nextn):
            raise ValueError("GGUF recurrent_layers length does not match block_count")
        config["layer_types"] = [
            "linear_attention" if r else "full_attention" for r in recurrent[:layers]
        ]
        sections = required("rope.dimension_sections")
        if len(sections) != 4 or sections[-1] != 0:
            raise ValueError("Qwen3.5 GGUF requires three MRoPE sections and zero tail")
        rope.update(mrope_section=sections[:3], mrope_interleaved=True)
    if arch == "qwen35moe":
        config.update(
            num_experts=positive("expert_count"),
            num_experts_per_tok=positive("expert_used_count"),
            moe_intermediate_size=positive("expert_feed_forward_length"),
            shared_expert_intermediate_size=positive(
                "expert_shared_feed_forward_length"
            ),
            norm_topk_prob=optional("expert_weights_norm", True),
            decoder_sparse_step=1,
        )
    return config


def gguf_config_from_metadata(metadata: dict[str, Any]) -> PretrainedConfig:
    config_dict = gguf_config_dict(metadata)
    model_type = config_dict.pop("model_type")
    if model_type == "qwen3_5_text":
        from .configs.qwen3_5 import Qwen3_5TextConfig

        return Qwen3_5TextConfig(**config_dict)
    if model_type == "qwen3_5_moe_text":
        from .configs.qwen3_5_moe import Qwen3_5MoeTextConfig

        return Qwen3_5MoeTextConfig(**config_dict)
    return AutoConfig.for_model(model_type, **config_dict)


def load_gguf_config(path: str | Path) -> PretrainedConfig:
    return gguf_config_from_metadata(read_gguf_metadata(path))
