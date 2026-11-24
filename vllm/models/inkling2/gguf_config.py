# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 config from llama.cpp / Unsloth GGUF metadata.

The GGUF route serves the same `Inkling2LoopForConditionalGeneration` as the
official checkpoint (text only: the GGUFs carry no vision tower), so the GGUF
``inkling2.*`` keys are mapped onto the official ``text_config`` fields.

GGUF counts layer *visits* (``block_count`` = 48 for 24 physical layers with a
6..18 core run 3 times) and lists the sliding-window pattern per visit; the
official config counts physical layers and lists sliding-window layer ids.
"""

from typing import Any

from vllm.models.inkling2.configs import (
    INKLING2_LOOP_ARCHITECTURE,
    Inkling2LoopMMConfig,
)

_ARCH = "inkling2"


def _visit_layers(n_visits: int, start: int, end: int, passes: int) -> list[int]:
    """Physical layer of each visit: prefix, core x passes, suffix."""
    n_phys = n_visits - (passes - 1) * (end - start)
    order = list(range(start)) + list(range(start, end)) * passes
    order += list(range(end, n_phys))
    if len(order) != n_visits:
        raise ValueError("Inkling2 GGUF loop metadata does not match block_count")
    return order


def inkling2_config_from_gguf(metadata: dict[str, Any]) -> Inkling2LoopMMConfig:
    def get(key: str, default: Any = None) -> Any:
        full = f"{_ARCH}.{key}"
        if full in metadata:
            return metadata[full]
        if default is None:
            raise ValueError(f"Missing required Inkling2 GGUF metadata: {full}")
        return default

    def uniform(key: str) -> int:
        value = get(key)
        values = set(value) if isinstance(value, list) else {value}
        if len(values) != 1:
            raise ValueError(f"Inkling2 GGUF {key} must be the same for every layer")
        return int(values.pop())

    if metadata.get("general.architecture") != _ARCH:
        raise ValueError("not an inkling2 GGUF")
    passes = int(get("num_loops", 1))
    if passes < 2:
        raise ValueError("only looped Inkling2 GGUFs (num_loops > 1) are supported")
    start, end = int(get("loop.core_start")), int(get("loop.core_end"))
    visits = int(get("block_count"))
    order = _visit_layers(visits, start, end, passes)
    n_phys = max(order) + 1

    pattern = [bool(x) for x in get("attention.sliding_window_pattern")]
    if len(pattern) != visits:
        raise ValueError("sliding_window_pattern must list every layer visit")
    local = {}
    for visit, layer in enumerate(order):
        if local.setdefault(layer, pattern[visit]) != pattern[visit]:
            raise ValueError(f"layer {layer} changes window type between visits")

    head_dim = int(get("attention.key_length"))
    swa_head_dim = int(get("attention.key_length_swa", head_dim))
    text_config = {
        "model_type": "inkling2_loop_text",
        "vocab_size": int(get("vocab_size")),
        "unpadded_vocab_size": int(get("vocab_size_unpadded")),
        "hidden_size": int(get("embedding_length")),
        "num_hidden_layers": n_phys,
        "num_attention_heads": uniform("attention.head_count"),
        "num_key_value_heads": uniform("attention.head_count_kv"),
        "head_dim": head_dim,
        "swa_num_attention_heads": uniform("attention.head_count"),
        "swa_num_key_value_heads": uniform("attention.head_count_kv"),
        "swa_head_dim": swa_head_dim,
        "d_rel": int(get("attention.relative_length")),
        "rel_extent": int(get("attention.relative_extent")),
        "local_layer_ids": [i for i in range(n_phys) if local[i]],
        "sliding_window_size": int(get("attention.sliding_window")),
        "sconv_kernel_size": int(get("shortconv.l_cache")),
        "rms_norm_eps": float(get("attention.layer_norm_rms_epsilon")),
        "model_max_length": int(get("context_length")),
        "log_scaling_n_floor": int(get("attention.log_scale_floor")),
        "log_scaling_alpha": float(get("attention.log_scale_alpha")),
        "qk_l2_clip_threshold": float(get("attention.qk_clip")),
        "rel_l2_clip_threshold": float(get("attention.relative_clip")),
        "n_layers_per_attn_res_block": int(get("attn_res.block_size")),
        "attn_res_normalize_values": True,
        "use_value_norm_no_gamma": True,
        "final_logit_softcapping": float(get("final_logit_softcapping", 0.0)) or None,
        "n_routed_experts": int(get("expert_count")),
        "num_experts_per_tok": int(get("expert_used_count")),
        "n_shared_experts": 0,
        "intermediate_size": int(get("expert_feed_forward_length")),
        "mlp_gate_softcap": float(get("activation.situ_beta")),
        "mlp_up_softcap": float(get("activation.situ_linear_beta")),
        "route_scale": float(get("expert_weights_scale", 1.0)),
        "use_gate_bias": True,
        "loop_config": {
            "core_start": start,
            "core_end": end,
            "num_passes": passes,
            "gate_rank": int(get("loop.gate_rank")),
            "is_varloop": False,
        },
    }
    config = Inkling2LoopMMConfig(
        architectures=[INKLING2_LOOP_ARCHITECTURE],
        text_config=text_config,
        vision_config=None,
        eos_token_id=metadata.get("tokenizer.ggml.eos_token_id"),
        pad_token_id=metadata.get("tokenizer.ggml.padding_token_id"),
        torch_dtype="float16",
    )
    config.gguf_architecture = _ARCH
    return config
