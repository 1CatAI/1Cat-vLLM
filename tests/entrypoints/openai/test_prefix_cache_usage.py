# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.engine.protocol import (
    UsageInfo,
    populate_prompt_cache_usage,
)


def test_populate_prompt_cache_usage_hit_rate() -> None:
    u_none = UsageInfo(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    populate_prompt_cache_usage(u_none, 100, None)
    assert "prefix_cache_hit_rate" not in u_none.model_dump(exclude_none=True)

    u_zero = UsageInfo(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    populate_prompt_cache_usage(u_zero, 100, 0)
    d_zero = u_zero.model_dump(exclude_none=True)
    assert d_zero["prompt_tokens_details"] == {"cached_tokens": 0}
    assert d_zero["prompt_cache_hit_tokens"] == 0
    assert d_zero["prompt_cache_miss_tokens"] == 100
    assert d_zero["prefix_cache_hit_rate"] == 0.0

    u_hit = UsageInfo(prompt_tokens=4000, completion_tokens=20, total_tokens=4020)
    populate_prompt_cache_usage(u_hit, 4000, 3840)
    dumped = u_hit.model_dump(exclude_none=True)
    assert dumped["prompt_tokens_details"] == {"cached_tokens": 3840}
    assert dumped["prompt_cache_hit_tokens"] == 3840
    assert dumped["prompt_cache_miss_tokens"] == 160
    assert dumped["prefix_cache_hit_rate"] == 0.96
