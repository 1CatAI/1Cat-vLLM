# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject configurations whose semantics differ from the qualified kernels."""

_COMMON = {
    "moe_intermediate_size": 2304,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "sliding_window": 128,
    "rope_theta": 10000,
    "rms_norm_eps": 1e-20,
    "hc_mult": 4,
    "hc_eps": 1e-6,
    "hc_sinkhorn_iters": 20,
    "n_shared_experts": 1,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "attention_bias": False,
}


def validate_runtime_geometry(config, *, draft=False):
    for name, expected in _COMMON.items():
        if getattr(config, name, None) != expected:
            raise ValueError(f"V4.1 qualified runtime requires {name}={expected!r}")
    ratios = tuple(config.compress_ratios)
    if draft:
        if ratios != (0, 0, 0):
            raise ValueError("V4.1 draft requires exactly three uncompressed layers")
        return
    if (
        ratios[:40] != (0, 0) + (2,) * 18 + (1,) * 20
        or any(ratios[40:])
        or getattr(config, "compress_rope_theta", None) != 160000
    ):
        raise ValueError("V4.1 target compression or rotary schedule differs")
    rope = config.rope_scaling or {}
    expected_rope = dict(
        rope_type="yarn",
        factor=16,
        beta_fast=32,
        beta_slow=1,
        original_max_position_embeddings=65536,
    )
    if any(rope.get(k) != value for k, value in expected_rope.items()):
        raise ValueError("V4.1 target requires the checkpoint's compressed YaRN")
