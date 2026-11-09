# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turing runs the fp16-only FA2 build: capability floor 7.5, dtype gate."""

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((7, 0), False), ((7, 5), True), ((8, 0), True), ((9, 0), True)],
)
def test_capability_floor_is_turing(capability, expected):
    assert (
        FlashAttentionBackend.supports_compute_capability(DeviceCapability(*capability))
        is expected
    )


def _combination(dtype: torch.dtype, capability: tuple[int, int]) -> str | None:
    return FlashAttentionBackend.supports_combination(
        head_size=128,
        dtype=dtype,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        device_capability=DeviceCapability(*capability),
    )


def test_turing_takes_fp16_only():
    assert _combination(torch.float16, (7, 5)) is None
    reason = _combination(torch.bfloat16, (7, 5))
    assert reason is not None and "fp16" in reason


def test_ampere_keeps_bf16():
    assert _combination(torch.bfloat16, (8, 0)) is None
