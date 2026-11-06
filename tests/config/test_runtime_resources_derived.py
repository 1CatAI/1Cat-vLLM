# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configs derived for a drafter borrow the target's worker-local resources."""

from vllm.config import ModelConfig, VllmConfig, replace
from vllm.runtime_resources import runtime_resources_for, share_runtime_resources


def _config() -> VllmConfig:
    return VllmConfig(model_config=ModelConfig(model="facebook/opt-125m"))


def test_replace_alone_drops_the_runtime_resources():
    # The reason share_runtime_resources exists: a drafter built from this copy
    # would register its native owners in a map the target's forward never
    # activates (seen with a block-FP8 GLM-5.3-Flash DFlash2 drafter).
    parent = _config()
    copy = replace(parent, is_speculative_draft=True)
    assert runtime_resources_for(copy) is not runtime_resources_for(parent)


def test_derived_config_borrows_the_parent_resources():
    parent = _config()
    resources = runtime_resources_for(parent)
    derived = share_runtime_resources(
        replace(parent, is_speculative_draft=True), parent
    )
    assert derived.is_speculative_draft
    assert runtime_resources_for(derived) is resources
    # An owner bound while building the drafter is the one the target's
    # forward context activates.
    owners = runtime_resources_for(derived).setdefault("execution_context_owners", [])
    owners.append(object())
    assert runtime_resources_for(parent)["execution_context_owners"] is owners
