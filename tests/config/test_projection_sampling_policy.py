# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm import envs
from vllm.config.gdn import GdnConfig
from vllm.config.gdn_projection import GdnProjectionConfig
from vllm.config.sm70_dflash2 import DFlashLookupPolicy
from vllm.config.speculative_sampling import SpeculativeSamplingPolicy


def _transfer(value):
    receiver, sender = multiprocessing.Pipe(duplex=False)
    try:
        sender.send(value)
        return receiver.recv()
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize("order", [(False, True), (True, False)])
def test_gdn_projection_is_per_engine_serialized_and_hashed(monkeypatch, order):
    alias = "VLLM_SM70_GDN_MIXED_QKV_CONTIGUOUS"
    policies = []
    for value in order:
        monkeypatch.setenv(alias, str(int(value)))
        before = dict(os.environ)
        policy = GdnConfig()
        policy.resolve()
        assert dict(os.environ) == before
        policies.append(_transfer(policy))
    monkeypatch.setenv(alias, "invalid-after-initialization")
    for policy, value in zip(policies, order):
        policy.resolve()
        assert policy.projection.mixed_qkv_contiguous is value
        assert policy.projection.sources["mixed_qkv_contiguous"] == alias
    assert policies[0].compute_hash() != policies[1].compute_hash()
    # An engine that does not bind GDN does not hash an unused projection knob.
    assert (
        GdnConfig().compute_hash()
        == GdnConfig(
            projection=GdnProjectionConfig(mixed_qkv_contiguous=True)
        ).compute_hash()
    )


@pytest.mark.parametrize("legacy", ["0", "1", "2", "false", ""])
def test_sampling_retains_distinct_legacy_boolean_parsers(monkeypatch, legacy):
    monkeypatch.setenv("VLLM_MTP_STOCHASTIC_TOKEN_MATCHING", legacy)
    monkeypatch.setenv("VLLM_SM70_REJECTION_COMBINE_BONUS", legacy)
    monkeypatch.setenv("VLLM_SM70_MTP_PROB_DRAFT_APPLY_TOP_P", legacy)
    policy = SpeculativeSamplingPolicy()
    policy.resolve(draft=True, vocab=False)
    assert policy.token_matching == (legacy == "1")
    assert policy.combine_bonus == (legacy != "0")
    assert policy.draft_apply_top_p == (legacy == "1")
    if legacy in ("false", ""):
        with pytest.raises(ValueError, match="invalid literal"):
            policy.use_fused_top_p()
    else:
        assert policy.use_fused_top_p() == bool(int(legacy))


def test_typed_proposal_overrides_both_legacy_views(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_MTP_PROB_DRAFT_APPLY_TOP_P", "invalid")
    policy = SpeculativeSamplingPolicy(draft_apply_top_p=True)
    policy.resolve(draft=True, vocab=False)
    assert policy.draft_apply_top_p and policy.use_fused_top_p()
    assert policy.sources["draft_apply_top_p"] == "typed"


def test_initialized_proposals_ignore_later_environment_and_match_legacy(monkeypatch):
    from vllm.v1.spec_decode import llm_base_proposer as proposal

    monkeypatch.setattr(proposal, "_sync_draft_token_ids_across_tp", lambda ids: ids)
    metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.ones(1),
        top_k=None,
        top_p=None,
        generators={},
    )
    logits = torch.tensor([[0.2, 0.9, -0.4]], dtype=torch.float32)
    policies, expected = [], []
    for scale in (1.0, 0.75):
        monkeypatch.setenv("VLLM_SM70_MTP_PROB_DRAFT_TEMPERATURE_SCALE", str(scale))
        policy = SpeculativeSamplingPolicy()
        policy.resolve(draft=True, vocab=False)
        policies.append(_transfer(policy))
        torch.manual_seed(91)
        expected.append(
            proposal.compute_probs_and_sample_next_token(logits.clone(), metadata)
        )
    for name in SpeculativeSamplingPolicy.aliases.values():
        monkeypatch.setenv(name, "invalid-after-initialization")
        monkeypatch.setitem(
            envs.environment_variables, name, Mock(side_effect=AssertionError(name))
        )
    for policy, oracle in zip(policies * 2, expected * 2):
        torch.manual_seed(91)
        actual = proposal.compute_probs_and_sample_next_token(
            logits.clone(), metadata, policy=policy
        )
        assert all(torch.equal(a, b) for a, b in zip(actual, oracle))


def test_lookup_limits_and_sources_survive_worker_transfer(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_NSTRONG", "-1")
    first = DFlashLookupPolicy()
    first.resolve()
    second = DFlashLookupPolicy(nstrong=8)
    second.resolve()
    first, second = _transfer((first, second))
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_NSTRONG", "99")
    first.resolve()
    second.resolve()
    assert first.nstrong == 1 and second.nstrong == 8
    assert first.compute_hash() != second.compute_hash()


def test_unused_proposal_and_vocab_options_do_not_change_other_paths():
    first = SpeculativeSamplingPolicy()
    second = SpeculativeSamplingPolicy(draft_temperature_scale=0.75, shortlist_size=64)
    first.resolve()
    second.resolve()
    assert first.compute_hash(draft=False, vocab=False) == second.compute_hash(
        draft=False, vocab=False
    )
    assert first.compute_hash() != second.compute_hash()


def test_disabling_input_core_keeps_legacy_short_circuit(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_INPUT_CORE_OP", "invalid-unused")
    policy = GdnProjectionConfig(disable_input_core=True)
    policy.resolve()
    assert not policy.input_core
    with pytest.raises(ValueError, match="invalid literal"):
        GdnProjectionConfig(disable_input_core=False).resolve()


@pytest.mark.parametrize(
    "paired,missing", [(False, []), (True, ["missing"]), (True, [])]
)
def test_configured_projection_preserves_pair_and_missing_operator_errors(
    monkeypatch, paired, missing
):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    policy = GdnProjectionConfig(qpn8_ba_split=True, rmsnorm_onepass=paired)
    policy.resolve()
    monkeypatch.setattr(gdn, "_missing_sm70_gdn_qpn8_ba_ops", lambda: missing)
    monkeypatch.setenv("VLLM_SM70_GDN_RMSNORM_ONEPASS", "invalid-after-init")
    if not paired:
        with pytest.raises(RuntimeError, match="requires the accepted"):
            gdn._sm70_gdn_qpn8_ba_split_enabled(policy)
    elif missing:
        with pytest.raises(RuntimeError, match="requires the source-built"):
            gdn._sm70_gdn_qpn8_ba_split_enabled(policy)
    else:
        assert gdn._sm70_gdn_qpn8_ba_split_enabled(policy)


def test_migrated_aliases_are_not_evaluated_again_for_graph_hash(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.config.sm70_dflash2 import Sm70DFlash2Config

    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    cfg.kernel_config.gdn.resolve()
    sampling = SpeculativeSamplingPolicy()
    sampling.resolve()
    cfg.speculative_config = SimpleNamespace(
        sampling_policy=sampling, sm70_dflash2=Sm70DFlash2Config()
    )
    first = envs.compile_factors(vllm_config=cfg)
    for name in (
        *sampling.aliases.values(),
        *cfg.kernel_config.gdn.projection.aliases.values(),
    ):
        monkeypatch.setenv(name, "invalid-after-init")
        monkeypatch.setitem(
            envs.environment_variables, name, Mock(side_effect=AssertionError(name))
        )
    assert first == envs.compile_factors(vllm_config=cfg)
