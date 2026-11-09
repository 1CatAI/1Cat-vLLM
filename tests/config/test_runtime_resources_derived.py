# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configs derived for a drafter borrow the target's worker-local resources."""

from types import SimpleNamespace

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


def test_v1_proposers_load_their_drafter_with_the_target_resources(monkeypatch):
    # V1 subclasses (DFlash, draft model, Gemma4, Step3.5 MTP) derive the
    # base draft config once more with replace(), so the sharing has to sit
    # where the drafter is built.
    from vllm.v1.spec_decode import draft_model, llm_base_proposer

    parent = _config()
    built = []

    def get_model(*, vllm_config, **kwargs):
        built.append(vllm_config)

    monkeypatch.setattr(llm_base_proposer, "get_model", get_model)
    monkeypatch.setattr(draft_model, "get_model", get_model)
    proposer = SimpleNamespace(
        vllm_config=parent,
        speculative_config=SimpleNamespace(
            draft_model_config=None, draft_load_config=None
        ),
        _create_draft_vllm_config=lambda: replace(parent, is_speculative_draft=True),
    )
    llm_base_proposer.SpecDecodeBaseProposer._get_model(proposer)
    draft_model.DraftModelProposer._get_model(proposer)
    assert len(built) == 2
    for config in built:
        assert config.is_speculative_draft
        assert runtime_resources_for(config) is runtime_resources_for(parent)
