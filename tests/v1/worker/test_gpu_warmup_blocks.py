# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 warmup must reserve the same speculative KV tail as the scheduler."""

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.worker.gpu.warmup import _reserved_block_count

BLOCK_SIZE = 16
MAX_MODEL_LEN = 1024
NUM_SPEC_TOKENS = 7


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


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (_full_attention_spec(), 2),
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
