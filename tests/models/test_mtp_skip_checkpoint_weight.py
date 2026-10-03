# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP drafters ship inside their target's checkpoint: the loader must skip
the target's tensors before reading them, and keep every tensor the drafter's
load_weights would use."""

import pytest

from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP


@pytest.mark.parametrize(
    "name,skipped",
    [
        ("mtp.layers.0.mlp.gate_proj.weight", False),
        ("mtp.fc.weight", False),
        ("model.language_model.embed_tokens.weight", False),
        ("lm_head.weight", False),
        ("model.language_model.layers.3.mlp.gate_proj.weight", True),
        ("model.visual.blocks.0.attn.qkv.weight", True),
    ],
)
def test_qwen3_5_mtp_skips_target_weights(name, skipped):
    drafter = Qwen3_5MTP.__new__(Qwen3_5MTP)
    assert drafter.skip_checkpoint_weight(name) is skipped


@pytest.mark.parametrize(
    "name,skipped",
    [
        ("mtp.layers.0.mlp.experts.0.down_proj.weight", False),
        ("model.language_model.mtp.fc.weight", False),
        ("model.language_model.embed_tokens.weight", False),
        ("mtp.shared_head.head.weight", False),
        ("lm_head.weight", False),
        ("model.language_model.layers.3.mlp.gate.weight", True),
        (
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
            ".shard_7.weight",
            True,
        ),
    ],
)
def test_qwen4_exp_mtp_skips_target_weights(name, skipped):
    drafter = Qwen4ExpMTP.__new__(Qwen4ExpMTP)
    assert drafter.skip_checkpoint_weight(name) is skipped
