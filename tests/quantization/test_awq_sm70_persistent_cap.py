# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.model_executor.layers.quantization.awq_sm70_moe as awq_sm70_moe

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("max_num_seqs", "verifier_width", "override", "expected"),
    [
        (1, 1, 0, 1),
        (8, 1, 0, 8),
        (8, 2, 0, 16),
        (8, 3, 0, 24),
        (8, 4, 0, 32),
        (16, 2, 0, 32),
        (32, 1, 0, 32),
        (48, 1, 0, 32),
        (64, 1, 0, 32),
        (64, 4, 8, 8),
        (64, 4, 16, 16),
        (64, 4, 32, 32),
        (8, 1, 16, 16),
        (0, 0, 0, 1),
        (-1, -1, 0, 1),
    ],
)
def test_resolve_persistent_max_tokens(
    max_num_seqs: int,
    verifier_width: int,
    override: int,
    expected: int,
) -> None:
    assert (
        awq_sm70_moe._resolve_persistent_max_tokens(
            max_num_seqs, verifier_width, override
        )
        == expected
    )


@pytest.mark.parametrize(
    ("config_max_num_seqs", "spec_method", "spec_state_tokens", "override", "expected"),
    [
        (8, None, 0, 0, 8),
        (8, "mtp", 1, 0, 16),
        (8, "mtp", 2, 0, 24),
        (8, "mtp", 3, 0, 32),
        (16, "mtp", 1, 0, 32),
        (8, "eagle", 3, 0, 8),
        (8, "mtp", 2, 16, 16),
        (16, None, 0, 8, 8),
        (64, None, 0, 0, 32),
        (64, None, 0, 16, 16),
        (None, None, 0, 0, 32),
    ],
)
def test_persistent_max_tokens_for_runtime(
    monkeypatch: pytest.MonkeyPatch,
    config_max_num_seqs: int | None,
    spec_method: str | None,
    spec_state_tokens: int,
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
        speculative_config = None
        if spec_method is not None:
            speculative_config = type(
                "SpeculativeConfig",
                (),
                {
                    "method": spec_method,
                    "num_speculative_state_tokens": lambda self: spec_state_tokens,
                },
            )()
        config = type(
            "VllmConfig",
            (),
            {
                "scheduler_config": scheduler_config,
                "speculative_config": speculative_config,
            },
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
