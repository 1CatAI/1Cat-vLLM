# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape-compatible configuration changes must not silently change semantics."""

from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v41.contract import validate_runtime_geometry


def _config():
    return SimpleNamespace(
        moe_intermediate_size=2304,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        o_groups=8,
        index_n_heads=32,
        index_head_dim=128,
        sliding_window=128,
        rope_theta=10000,
        rms_norm_eps=1e-20,
        hc_mult=4,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
        n_shared_experts=1,
        hidden_act="silu",
        swiglu_limit=10.0,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        attention_bias=False,
        compress_rope_theta=160000,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        rope_scaling=dict(
            rope_type="yarn",
            factor=16,
            beta_fast=32,
            beta_slow=1,
            original_max_position_embeddings=65536,
        ),
    )


def test_checkpoint_geometry_and_its_distinct_uncompressed_draft_are_admitted():
    config = _config()
    validate_runtime_geometry(config)
    config.compress_ratios, config.rope_scaling = [0, 0, 0], None
    validate_runtime_geometry(config, draft=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sliding_window", 256),
        ("num_key_value_heads", 2),
        ("qk_rope_head_dim", 32),
        ("rms_norm_eps", 1e-6),
        ("hc_sinkhorn_iters", 10),
        ("hc_eps", 1e-5),
        ("n_shared_experts", 2),
        ("norm_topk_prob", False),
        ("scoring_func", "softmax"),
        ("compress_rope_theta", 10000),
    ],
)
def test_semantic_drift_is_rejected_before_weight_allocation(field, value):
    config = _config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match="V4.1"):
        validate_runtime_geometry(config)


def test_compression_and_yarn_cannot_silently_use_other_runtime_constants():
    config = _config()
    config.compress_ratios[31] = 2
    with pytest.raises(ValueError, match="compression"):
        validate_runtime_geometry(config)
    config = _config()
    config.rope_scaling["factor"] = 8
    with pytest.raises(ValueError, match="YaRN"):
        validate_runtime_geometry(config)
    config.compress_ratios = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="three"):
        validate_runtime_geometry(config, draft=True)
