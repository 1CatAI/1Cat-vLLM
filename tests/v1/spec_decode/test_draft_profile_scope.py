# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


def test_draft_scope_is_set_before_backend_and_kv_revalidation():
    @dataclass
    class Cache:
        cache_dtype: str = "fp8_e4m3"

    @dataclass
    class Attention:
        backend: str = "target"

    checked_scopes = []

    @dataclass
    class Config:
        cache_config: Cache
        attention_config: Attention
        is_speculative_draft: bool = False

        def __post_init__(self):
            checked_scopes.append(self.is_speculative_draft)
            if not self.is_speculative_draft:
                assert self.cache_config.cache_dtype == "fp8_e4m3"
                assert self.attention_config.backend == "target"

    target = Config(Cache(), Attention())
    checked_scopes.clear()
    proposer = SimpleNamespace(
        vllm_config=target,
        speculative_config=SimpleNamespace(
            moe_backend=None, attention_backend="draft", kv_cache_dtype="auto"
        ),
    )
    draft = SpecDecodeBaseProposer._create_draft_vllm_config(proposer)
    assert checked_scopes and all(checked_scopes)
    assert draft.is_speculative_draft
    assert draft.cache_config.cache_dtype == "auto"
    assert draft.attention_config.backend == "draft"
    assert not target.is_speculative_draft
    assert target.cache_config.cache_dtype == "fp8_e4m3"
    assert target.attention_config.backend == "target"


def test_qwen4exp_mtp_model_config_is_marked_before_revalidation(monkeypatch):
    from vllm.models.qwen4_exp.nvidia import mtp

    target_model = SimpleNamespace(architectures=["Qwen4ExpForCausalLM"])
    draft_model = SimpleNamespace(architectures=["Qwen4ExpMTP"])
    checked_scopes = []

    @dataclass
    class Config:
        model_config: object
        speculative_config: object
        is_speculative_draft: bool = False

        def __post_init__(self):
            checked_scopes.append(self.is_speculative_draft)
            if self.model_config is draft_model:
                assert self.is_speculative_draft

    target = Config(target_model, SimpleNamespace(draft_model_config=draft_model))
    checked_scopes.clear()
    monkeypatch.setattr(mtp, "get_draft_quant_config", lambda _: None)
    draft = mtp._make_draft_vllm_config(target, 0)
    assert checked_scopes == [True]
    assert draft.model_config is draft_model
    assert draft.is_speculative_draft
    assert not target.is_speculative_draft
