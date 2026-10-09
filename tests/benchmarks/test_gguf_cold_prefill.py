# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.benchmark_gguf_prefill import (
    cold_prefill_evidence,
    reset_cold_prefix_cache,
)


def test_cold_reset_requires_idle_cache_reset():
    llm = SimpleNamespace(reset_prefix_cache=Mock(return_value=True))
    assert reset_cold_prefix_cache(llm, True)
    llm.reset_prefix_cache.assert_called_once_with()
    llm.reset_prefix_cache.return_value = False
    with pytest.raises(RuntimeError, match="reset failed"):
        reset_cold_prefix_cache(llm, True)


@pytest.mark.parametrize("cached", [1, 32767, None])
def test_cold_prefill_rejects_hits_or_missing_evidence(cached):
    result = SimpleNamespace(num_cached_tokens=cached, prompt_token_ids=[1] * 32768)
    with pytest.raises(RuntimeError, match="zero cached tokens"):
        cold_prefill_evidence(result, True)


def test_cold_prefill_counts_all_prompt_tokens():
    result = SimpleNamespace(num_cached_tokens=0, prompt_token_ids=[1] * 32768)
    assert cold_prefill_evidence(result, True) == {
        "num_cached_tokens": 0,
        "computed_prompt_tokens": 32768,
    }


def test_cold_concurrent_cohort_keeps_tokens_and_isolates_prefixes(monkeypatch):
    from benchmarks import benchmark_flashnext_acceptance as acceptance

    prompts = []

    def generate(llm, inputs, params, atomic):
        assert atomic
        prompts.extend(inputs)
        return []

    monkeypatch.setattr(acceptance, "generate_cohort", generate)
    client = SimpleNamespace(get_output=Mock())
    original = client.get_output
    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=client))
    ids = [1, 2, 3]
    salts = [f"cold-{i}" for i in range(4)]
    acceptance.observed_cohort(llm, ids, object(), 4, cache_salts=salts)
    assert all(row["prompt_token_ids"] == ids for row in prompts)
    assert [row["cache_salt"] for row in prompts] == salts
    assert client.get_output is original
    with pytest.raises(ValueError, match="distinct cache salt"):
        acceptance.observed_cohort(llm, ids, object(), 4, cache_salts=["same"] * 4)
