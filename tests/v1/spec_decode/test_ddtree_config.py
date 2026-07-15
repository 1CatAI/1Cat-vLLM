# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.speculative import SpeculativeConfig


def _config(
    *,
    method: str = "dflash_ddtree",
    num_speculative_tokens: int = 16,
    ddtree_budget: int | None = 24,
    ddtree_disable_tree_verify: bool = False,
) -> SpeculativeConfig:
    config = object.__new__(SpeculativeConfig)
    config.method = method
    config.num_speculative_tokens = num_speculative_tokens
    config.ddtree_budget = ddtree_budget
    config.ddtree_disable_tree_verify = ddtree_disable_tree_verify
    return config


def test_dflash_ddtree_state_tokens_use_budget_when_tree_verify_enabled() -> None:
    config = _config(ddtree_budget=24, ddtree_disable_tree_verify=False)

    assert config.num_speculative_state_tokens() == 24


def test_dflash_ddtree_state_tokens_remain_flat_when_tree_verify_disabled() -> None:
    config = _config(ddtree_budget=24, ddtree_disable_tree_verify=True)

    assert config.num_speculative_state_tokens() == 16


def test_dflash_ddtree_state_tokens_never_shrink_below_flat_depth() -> None:
    config = _config(ddtree_budget=8, ddtree_disable_tree_verify=False)

    assert config.num_speculative_state_tokens() == 16


def test_non_ddtree_state_tokens_use_num_speculative_tokens() -> None:
    config = _config(method="dflash", ddtree_budget=24)

    assert config.num_speculative_state_tokens() == 16


def test_dflash_ddtree_defaults_match_reference_best_first() -> None:
    assert SpeculativeConfig.ddtree_top_k is None
    assert SpeculativeConfig.ddtree_chain_seed is False
    assert SpeculativeConfig.ddtree_disable_tree_verify is False
