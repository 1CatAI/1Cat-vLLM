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
