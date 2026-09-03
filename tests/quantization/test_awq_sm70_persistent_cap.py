# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.model_executor.layers.quantization.awq_sm70_moe as awq_sm70_moe

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("max_num_seqs", "override", "expected"),
    [
        (1, 0, 1),
        (8, 0, 8),
        (16, 0, 16),
        (32, 0, 32),
        (48, 0, 32),
        (64, 0, 32),
        (64, 8, 8),
        (64, 16, 16),
        (64, 32, 32),
        (8, 16, 16),
        (0, 0, 1),
        (-1, 0, 1),
    ],
)
def test_resolve_persistent_max_tokens(
    max_num_seqs: int,
    override: int,
    expected: int,
) -> None:
    assert (
        awq_sm70_moe._resolve_persistent_max_tokens(max_num_seqs, override) == expected
    )


@pytest.mark.parametrize(
    ("config_max_num_seqs", "override", "expected"),
    [
        (8, 0, 8),
        (8, 16, 16),
        (16, 8, 8),
        (64, 0, 32),
        (64, 16, 16),
        (None, 0, 32),
    ],
)
def test_persistent_max_tokens_for_runtime(
    monkeypatch: pytest.MonkeyPatch,
    config_max_num_seqs: int | None,
    override: int,
    expected: int,
) -> None:
    if config_max_num_seqs is None:
        config = None
    else:
        scheduler_config = type(
            "SchedulerConfig",
            (),
            {"max_num_seqs": config_max_num_seqs},
        )()
        config = type(
            "VllmConfig",
            (),
            {"scheduler_config": scheduler_config},
        )()
    monkeypatch.setattr(
        awq_sm70_moe,
        "get_current_vllm_config_or_none",
        lambda: config,
    )
    monkeypatch.setattr(
        awq_sm70_moe.envs,
        "VLLM_SM70_AWQ_MOE_PERSISTENT_MAX_TOKENS",
        override,
    )

    assert awq_sm70_moe._persistent_max_tokens_for_runtime() == expected
