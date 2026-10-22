# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.config.speculative import match_target_rope_scaling


def _config(rope, positions):
    text = SimpleNamespace(rope_parameters=rope, max_position_embeddings=positions)
    return SimpleNamespace(hf_text_config=text)


YARN = {
    "rope_type": "yarn",
    "rope_theta": 1e7,
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
}


def test_draft_adopts_extended_target_rope():
    target = _config(YARN, 1048576)
    draft = _config({"rope_type": "default", "rope_theta": 1e7}, 262144)
    assert match_target_rope_scaling(target, draft)
    assert draft.hf_text_config.rope_parameters == YARN
    assert draft.hf_text_config.max_position_embeddings == 1048576
    # Idempotent: an already scaled draft is left alone.
    assert not match_target_rope_scaling(target, draft)


def test_draft_keeps_rope_for_native_or_mismatched_targets():
    native = _config({"rope_type": "default", "rope_theta": 1e7}, 262144)
    draft = _config({"rope_type": "default", "rope_theta": 1e7}, 262144)
    assert not match_target_rope_scaling(native, draft)
    other_theta = _config({"rope_type": "default", "rope_theta": 1e6}, 262144)
    assert not match_target_rope_scaling(_config(YARN, 1048576), other_theta)
    assert other_theta.hf_text_config.max_position_embeddings == 262144
