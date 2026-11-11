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


def kv_source_of(config, layer):
    """The KV source layer whose caches ``layer`` reads (-1: window only)."""
    return max((i for i in config.kv_source_layer_ids if i <= layer), default=-1)


def pp_split_problem(config, split, *, staged_state=False):
    """Why a pipeline boundary before target layer ``split`` is not admissible.

    Returns None when admissible. Only the four-stream HC residual may cross a
    stage boundary (the KV20 rule):

    - every layer reads the KV and indexer caches of the nearest source at or
      below it, so the second stage must start at a KV and an index source;
    - Engram, its hash state and the ratio-2 compressor states are committed
      once per step from the accepted counts, so they stay on the first stage.

    For the V4.1 target (KV sources 2/8/14/20, Engram 1/14, ratio 2 on layers
    2-19) the only admissible split is 20.

    ``staged_state`` (the native-weights route) commits the Engram and ratio-2
    state on whichever stage holds it, and lets a stage start inside the
    ratio-1 source 20's consumers: the previous stage then ships the source's
    new compressed and index rows with the residual (a KV mirror). The split
    must start at an index source below layer 20; from 20 on any layer (the
    index group's top-k is shipped too). Cuts 8, 14, 20, 21, ..., 39.
    """
    layers = config.num_hidden_layers
    if not 0 < split < layers:
        return f"split must be inside 1..{layers - 1}"
    if staged_state:
        source = kv_source_of(config, split)
        ratio1 = source >= 0 and tuple(config.compress_ratios)[source] == 1
        if split not in tuple(config.index_source_layer_ids) and not ratio1:
            # Inside the ratio-1 layers the index group's top-k (512 int32 per
            # token) is shipped with the residual; elsewhere it is not.
            return "not an index source layer (index keys would cross the boundary)"
        if source != split and not ratio1:
            return "a stage may start inside the consumers of a ratio-1 source only"
        return None
    if split not in tuple(config.kv_source_layer_ids):
        return "not a KV source layer (KV would cross the boundary)"
    if split not in tuple(config.index_source_layer_ids):
        return "not an index source layer (index keys would cross the boundary)"
    if any(layer >= split for layer in config.engram_layer_ids):
        return "Engram layers and their hash state must be on the first stage"
    if any(ratio == 2 for ratio in tuple(config.compress_ratios)[split:layers]):
        return "ratio-2 compressor state must be on the first stage"
    return None
