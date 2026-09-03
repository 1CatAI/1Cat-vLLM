# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 warmup must reserve the same speculative KV tail as the scheduler."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.warmup import (
    _kernel_prefill_warmup_token_counts,
    _reserved_block_count,
    warmup_kernels,
)

BLOCK_SIZE = 16
MAX_MODEL_LEN = 1024
NUM_SPEC_TOKENS = 7


def test_kernel_prefill_warmup_profiles_are_capability_advertised() -> None:
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        max_model_len=64,
        compilation_config=SimpleNamespace(
            static_forward_context={
                "plain": object(),
                "qsa_0": SimpleNamespace(
                    kernel_warmup_prefill_token_counts=(33, 257, True, -1)
                ),
                "qsa_1": SimpleNamespace(kernel_warmup_prefill_token_counts=(33,)),
            }
        ),
    )

    assert _kernel_prefill_warmup_token_counts(runner, 6) == (6, 33)


def test_kernel_prefill_warmup_runs_default_batch_and_extra_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector_states: list[bool] = []
    attention_spec = _full_attention_spec()
    runner = SimpleNamespace(
        num_speculative_steps=4,
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=128,
        ),
        max_model_len=64,
        compilation_config=SimpleNamespace(
            static_forward_context={
                "profile": SimpleNamespace(kernel_warmup_prefill_token_counts=(33,))
            }
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=attention_spec)],
            num_blocks=64,
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=4),
        model_state=SimpleNamespace(max_encoder_len=0),
        is_pooling_model=False,
        is_last_pp_rank=True,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        kv_connector=SimpleNamespace(
            set_disabled=lambda disabled: connector_states.append(disabled)
        ),
    )
    executions: list[Any] = []
    samples: list[Any] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    warmup_kernels(runner, executions.append, samples.append)

    assert [output.total_num_scheduled_tokens for output in executions] == [
        24,
        20,
        0,
        33,
        5,
        0,
    ]
    assert connector_states == [True, False]
    assert len(samples) == 4


def _speculative_config(method: str) -> SpeculativeConfig:
    config = object.__new__(SpeculativeConfig)
    object.__setattr__(config, "method", method)
    object.__setattr__(config, "num_speculative_tokens", NUM_SPEC_TOKENS)
    object.__setattr__(config, "ddtree_disable_tree_verify", False)
    object.__setattr__(config, "ddtree_budget", 24)
    return config


class _Config:
    speculative_config: SpeculativeConfig | None = None
    diffusion_config = None
    num_speculative_tokens = VllmConfig.num_speculative_tokens
    num_lookahead_tokens = VllmConfig.num_lookahead_tokens


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("dflash", NUM_SPEC_TOKENS + 1),
        ("dflash_ddtree", 25),
        ("eagle3", NUM_SPEC_TOKENS),
        ("mtp", NUM_SPEC_TOKENS),
        ("dspark", NUM_SPEC_TOKENS),
        ("draft_model", NUM_SPEC_TOKENS),
        ("ngram", 0),
    ],
)
def test_num_lookahead_tokens_per_method(method: str, expected: int) -> None:
    config = _Config()
    config.speculative_config = _speculative_config(method)

    assert config.num_lookahead_tokens == expected


def test_num_lookahead_tokens_without_speculation() -> None:
    assert _Config().num_lookahead_tokens == 0


def _full_attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
    )


def _mamba_spec(mode: str) -> MambaSpec:
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mode,
        num_speculative_blocks=NUM_SPEC_TOKENS,
    )


def _circular_spec() -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (_full_attention_spec(), 2),
        (_circular_spec(), 1),
        (
            UniformTypeKVCacheSpecs(
                block_size=4, kv_cache_specs={"compressor_state": _circular_spec()}
            ),
            1,
        ),
        (_mamba_spec("align"), 8),
        (_mamba_spec("none"), 9),
        (_mamba_spec("all"), 9),
    ],
)
def test_reserved_block_count_matches_speculative_tail(spec, expected) -> None:
    # 14 scheduled tokens plus DFlash's eight lookahead positions crosses the
    # second attention block. Align-mode Mamba excludes lookahead from its
    # token range but always retains seven speculative state blocks.
    assert (
        _reserved_block_count(
            14,
            spec,
            num_lookahead_tokens=NUM_SPEC_TOKENS + 1,
            max_model_len=MAX_MODEL_LEN,
            max_encoder_len=0,
        )
        == expected
    )
