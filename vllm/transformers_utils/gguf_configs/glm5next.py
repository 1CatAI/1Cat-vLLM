# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore the GLM5-Next configuration serialized by llama.cpp.

The exporter writes zero KV heads for KDA layers, one for sparse MLA, and
includes NextN blocks in block_count. MLA key/value_length describe its latent
cache; key/value_length_mla describe the attention heads instead.
"""

import math
from typing import Any


def glm5next_config_dict(metadata: dict[str, Any]) -> dict[str, Any]:
    arch = "glm5-next"
    if metadata.get("general.architecture") != arch:
        raise ValueError("GLM5-Next builder received a different architecture")

    def required(key):
        full = f"{arch}.{key}"
        if full not in metadata:
            raise ValueError(f"Missing required GGUF metadata: {full}")
        return metadata[full]

    def integer(key, *, minimum=1, default=None):
        value = metadata.get(f"{arch}.{key}", default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"GGUF {arch}.{key} must be an integer >= {minimum}")
        return value

    def number(key, *, default=None, positive=False):
        value = metadata.get(f"{arch}.{key}", default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (positive and value <= 0)
        ):
            raise ValueError(f"Invalid GGUF numeric metadata: {arch}.{key}")
        return value

    def boolean(key, default):
        value = metadata.get(f"{arch}.{key}", default)
        if not isinstance(value, bool):
            raise ValueError(f"GGUF {arch}.{key} must be boolean")
        return value

    blocks = integer("block_count")
    nextn = integer("nextn_predict_layers", minimum=0, default=0)
    layers = blocks - nextn
    if layers <= 0:
        raise ValueError("GGUF GLM5-Next contains no backbone layers")
    kv_heads = required("attention.head_count_kv")
    if (
        not isinstance(kv_heads, list)
        or len(kv_heads) != blocks
        or any(type(value) is not int or value not in (0, 1) for value in kv_heads)
        or any(value != 1 for value in kv_heads[layers:])
    ):
        raise ValueError("GLM5-Next KV-head layout must cover KDA/MLA and MLA NextN")
    indexer_types = metadata.get(f"{arch}.attention.indexer.types")
    if indexer_types is not None and (
        not isinstance(indexer_types, list)
        or len(indexer_types) != layers
        or any(value is not True for value in indexer_types)
    ):
        raise ValueError(
            "GLM5-Next GGUF currently requires a full indexer per MLA layer"
        )
    if integer("rope.dimension_count", minimum=0) != 0:
        raise ValueError("GLM5-Next GGUF requires NoPE MLA")
    kv_rank = integer("attention.kv_lora_rank")
    if (
        integer("attention.key_length") != kv_rank
        or integer("attention.value_length") != kv_rank
    ):
        raise ValueError("GLM5-Next latent cache dimensions disagree with kv_lora_rank")
    heads = integer("attention.head_count")
    experts = integer("expert_count")
    topk = integer("expert_used_count")
    groups = integer("expert_group_count")
    top_groups = integer("expert_group_used_count")
    if topk > experts or experts % groups or top_groups > groups:
        raise ValueError("Invalid GLM5-Next expert routing dimensions")
    if integer("expert_gating_func") != 2:
        raise ValueError("GLM5-Next GGUF requires sigmoid expert routing")
    dense_layers = integer("leading_dense_block_count", minimum=0)
    if dense_layers > layers:
        raise ValueError("GLM5-Next dense layer count exceeds backbone")
    shared = integer("expert_shared_count")
    moe_intermediate = integer("expert_feed_forward_length")
    if integer("expert_shared_feed_forward_length") != shared * moe_intermediate:
        raise ValueError("GLM5-Next shared expert dimensions are inconsistent")
    kpool = integer("attention.indexer.kpool")
    topk_index = integer("attention.indexer.top_k")
    if kpool <= 1 or topk_index % kpool:
        raise ValueError("GLM5-Next index_topk must be divisible by kpool > 1")
    clamps = []
    for field in ("swiglu_clamp_exp", "swiglu_clamp_shexp"):
        value = metadata.get(f"{arch}.{field}")
        if value is None:
            continue
        if (
            not isinstance(value, list)
            or len(value) != blocks
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                or item <= 0
                for item in value
            )
        ):
            raise ValueError(f"Invalid GLM5-Next {field} per-layer values")
        clamps.extend(value)
    if clamps and len(set(clamps)) != 1:
        raise ValueError("GLM5-Next runtime requires one shared/expert SwiGLU limit")
    config = {
        "model_type": "glm5_next_text",
        "architectures": ["Glm5NextForCausalLM"],
        "gguf_architecture": arch,
        "torch_dtype": "float16",
        "tie_word_embeddings": False,
        "hidden_size": integer("embedding_length"),
        "intermediate_size": integer("feed_forward_length"),
        "num_hidden_layers": layers,
        "num_nextn_predict_layers": nextn,
        "num_attention_heads": heads,
        # The GGUF KV-head array encodes layer kinds, not HF KV-head count.
        "num_key_value_heads": heads,
        "max_position_embeddings": integer("context_length"),
        "vocab_size": integer("vocab_size"),
        "rms_norm_eps": number("attention.layer_norm_rms_epsilon", positive=True),
        "hidden_act": "silu",
        "layer_types": [
            "linear_attention" if value == 0 else "deepseek_sparse_attention"
            for value in kv_heads[:layers]
        ],
        "mlp_layer_types": ["dense"] * dense_layers
        + ["sparse"] * (layers - dense_layers),
        "first_k_dense_replace": dense_layers,
        "mla": True,
        "mla_nope": True,
        "qk_rope_head_dim": 0,
        "qk_nope_head_dim": integer("attention.key_length_mla"),
        "v_head_dim": integer("attention.value_length_mla"),
        "q_lora_rank": integer("attention.q_lora_rank"),
        "kv_lora_rank": kv_rank,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "linear_head_dim": integer("kda.head_dim"),
        "linear_num_heads": heads,
        "linear_conv_kernel_dim": integer("ssm.conv_kernel"),
        "linear_lower_bound": number("kda.gate_lower_bound", default=-5.0),
        "index_head_dim": integer("attention.indexer.key_length"),
        "index_topk": topk_index,
        "index_n_heads": integer("attention.indexer.head_count"),
        "index_kpool": kpool,
        "index_kpool_always_select_tail": boolean(
            "attention.indexer.kpool_select_tail", True
        ),
        "index_dsa_use_layernorm": True,
        "index_kpool_compress": True,
        "mhc": True,
        "mhc_num_residual_streams": integer("hyper_connection.count"),
        "mhc_sinkhorn_iterations": integer("hyper_connection.sinkhorn_iterations"),
        "hc_eps": number("hyper_connection.epsilon", positive=True),
        "n_routed_experts": experts,
        "num_experts": experts,  # Shared GGUF stacked-expert storage contract.
        "num_experts_per_token": topk,
        "n_shared_experts": shared,
        "moe_intermediate_size": moe_intermediate,
        "routed_scaling_factor": number("expert_weights_scale", positive=True),
        "norm_topk_prob": boolean("expert_weights_norm", True),
        "moe_renormalize": boolean("expert_weights_norm", True),
        "scoring_func": "sigmoid",
        "moe_router_dtype": "float32",
        "topk_method": "noaux_tc",
        "use_grouped_topk": True,
        "n_group": groups,
        "topk_group": top_groups,
        "swiglu_limit": clamps[0] if clamps else None,
    }
    # The implemented indexer uses LayerNorm(eps=1e-6), and its Hadamard
    # transform is 128-wide. Do not silently admit a different checkpoint.
    if config["index_head_dim"] != 128 or not math.isclose(
        number("attention.layer_norm_epsilon", positive=True), 1e-6, rel_tol=1e-6
    ):
        raise ValueError("GLM5-Next indexer requires head_dim=128 and norm eps=1e-6")
    for key, target in (
        ("bos", "bos_token_id"),
        ("eos", "eos_token_id"),
        ("padding", "pad_token_id"),
    ):
        value = metadata.get(f"tokenizer.ggml.{key}_token_id")
        if value is not None:
            if type(value) is not int or not 0 <= value < config["vocab_size"]:
                raise ValueError(f"Invalid GLM5-Next tokenizer {key} token ID")
            config[target] = value
    stop_ids = []
    for key in ("eos", "eot", "eom"):
        value = metadata.get(f"tokenizer.ggml.{key}_token_id")
        if value is not None:
            if type(value) is not int or not 0 <= value < config["vocab_size"]:
                raise ValueError(f"Invalid GLM5-Next tokenizer {key} token ID")
            if value not in stop_ids:
                stop_ids.append(value)
    if stop_ids:
        config["eos_token_id"] = stop_ids if len(stop_ids) > 1 else stop_ids[0]
    return config
