# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity checks for local FUSE47 fallbacks and upstream SM70 kernels."""

import pytest
import torch

from vllm import _sm70_ops
from vllm.model_executor.layers import sm70_fuse47
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def _require_sm70():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")


def test_fuse47_h1_matches_upstream_batch_gate():
    _require_sm70()
    if not _sm70_ops.has_qwen38_shared_gate_sigmoid_mul():
        pytest.skip("Requires upstream qwen38 shared-gate native op")

    generator = torch.Generator(device="cuda").manual_seed(271828)
    for rows in (2, 5, 16):
        source = torch.randn(
            (rows, 2560), device="cuda", dtype=torch.float16, generator=generator
        )
        logits = torch.randn(
            (rows, 1), device="cuda", dtype=torch.float16, generator=generator
        )
        expected = source.clone()
        _sm70_ops.qwen38_shared_gate_sigmoid_mul_out(expected, logits)
        actual = sm70_fuse47.shared_gate_sigmoid_mul(logits, source)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    ("rows", "real_tokens", "accepted", "state_idx"),
    [(5, 1, 1, 1), (5, 3, 5, 2), (10, 5, 8, 1), (10, 3, 4, 0)],
)
def test_fuse47_p2_matches_upstream_ple_spec(rows, real_tokens, accepted, state_idx):
    _require_sm70()
    if not hasattr(torch.ops._C, "qwen38_ple_spec_sm70_out"):
        pytest.skip("Requires upstream qwen38 PLE native op")

    generator = torch.Generator(device="cuda").manual_seed(314159)
    x = torch.randn(
        (rows, 10240), device="cuda", dtype=torch.float16, generator=generator
    )
    weights = (
        torch.randn((10240, 4), device="cuda", dtype=torch.float16, generator=generator)
        * 0.1
    )
    initial_state = torch.randn(
        (3, 10240, 13), device="cuda", dtype=torch.float16, generator=generator
    )
    expected_state = initial_state.clone()
    actual_state = initial_state.clone()
    state_indices = torch.tensor([state_idx], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, real_tokens], device="cuda", dtype=torch.int32)
    accepted_tokens = torch.tensor([accepted], device="cuda", dtype=torch.int32)
    expected = torch.empty_like(x)
    actual = torch.empty_like(x)

    torch.ops._C.qwen38_ple_spec_sm70_out(
        expected,
        expected_state,
        x,
        weights,
        state_indices,
        starts,
        accepted_tokens,
    )
    sm70_fuse47.ple_short_conv_spec(
        x,
        actual_state,
        weights,
        state_indices,
        starts,
        accepted_tokens,
        5,
        actual,
        NULL_BLOCK_ID,
    )

    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(actual_state.view(torch.int16), expected_state.view(torch.int16))
