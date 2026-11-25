# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN speculative replay around a Mamba block boundary (SM70 native verifier).

With replay, a request's running state slot holds the state committed at the
start of the round; the next round replays the accepted tokens onto it. A round
of 5 rows (MTP4) that starts at m * B - 5 ends exactly on the block boundary,
and its running slot is the block that ends there. An in-round checkpoint at
that boundary used to be stored into the running slot itself, so the next
round replayed the accepted tokens on top of a state that already contained
all 5 rows, including the rejected drafts.

The test runs the real replay row preparation and the native verifier for two
rounds - one ending on the boundary, one after it - and compares the committed
state and the boundary checkpoint with a float32 reference.
"""

import numpy as np
import pytest
import torch

KD = 128
BLOCK = 1632  # Mamba block size of the Qwen3.8 Flash-Next deployments
H, HV, ROWS = 4, 12, 5  # local k / v heads at TP4, MTP4 verifier rows


def _verify_op():
    return torch.ops._C.sm70_gdn_verify_out


def _available() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (7, 0):
        return False
    try:
        import vllm._C  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops._C, "sm70_gdn_verify_out")


def _round_inputs(gen, device):
    width = 2 * H * KD + HV * KD
    qkv = (torch.randn(ROWS, width, generator=gen) * 0.5).half().to(device)
    a = (torch.randn(ROWS, HV, generator=gen) * 0.5).half().to(device)
    b = (torch.randn(ROWS, HV, generator=gen) * 0.5).half().to(device)
    return qkv, a, b


def _reference_steps(qkv, a, b, A_log, dt_bias):
    """Per-token (k, v, decay, beta) as the verifier computes them, float32."""
    qkv = qkv.float()
    k = qkv[:, H * KD : 2 * H * KD].view(ROWS, H, KD)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    k = k.repeat_interleave(HV // H, dim=1)
    v = qkv[:, 2 * H * KD :].view(ROWS, HV, KD)
    x = a.float() + dt_bias.float()
    softplus = torch.where(x <= 20, torch.log1p(torch.exp(x)), x)
    decay = torch.exp(-torch.exp(A_log.float()) * softplus)
    beta = torch.sigmoid(b.float())
    return [(k[t], v[t], decay[t], beta[t]) for t in range(ROWS)]


def _apply(state, steps):
    for k, v, decay, beta in steps:
        state = state * decay[:, None, None]
        delta = (v - torch.einsum("hvk,hk->hv", state, k)) * beta[:, None]
        state = state + delta[:, :, None] * k[:, None, :]
    return state


@pytest.mark.skipif(not _available(), reason="needs the SM70 native GDN verifier")
@pytest.mark.parametrize("accepted", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("blocks_before", [1, 2])
def test_round_ending_on_block_boundary_keeps_committed_state(
    accepted: int, blocks_before: int
):
    from vllm.model_executor.layers.mamba.gdn.spec_replay import SpecReplayState

    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(1000 * blocks_before + accepted)
    A_log = (torch.randn(HV, generator=gen) * 0.3).to(device)
    dt_bias = (torch.randn(HV, generator=gen) * 0.3).to(device)
    s0 = (torch.randn(HV, KD, KD, generator=gen) * 0.05).to(device)

    max_reqs, slots, cols = 4, 16, 8
    state = torch.zeros(slots, HV, KD, KD, dtype=torch.float32, device=device)
    # Request 0 owns state slots 10, 11, ... in block-table columns 0, 1, ...
    block_table = torch.full((max_reqs, cols), -1, dtype=torch.int32, device=device)
    block_table[0] = torch.arange(10, 10 + cols, dtype=torch.int32)
    table_ptrs = torch.tensor(
        [block_table.data_ptr()], dtype=torch.int64, device=device
    )
    replay = SpecReplayState(max_reqs, 8, 1, device)
    rbuf = replay.layer_buffer("layer", H, HV, device)
    computed = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=device)
    starts = torch.tensor([0, ROWS], dtype=torch.int32, device=device)

    def column_slot(col: int) -> int:
        return int(block_table[0, col])

    def run_round(num_computed: int, qkv, a, b):
        computed[0] = num_computed
        replay.prepare(
            idx_mapping,
            starts,
            computed,
            np.array([ROWS - 1]),
            np.array([False]),
            1,
            table_ptrs,
            block_table.stride(0),
            BLOCK,
        )
        running = (num_computed + ROWS - 1) // BLOCK
        out = torch.empty(ROWS, HV, KD, dtype=torch.float16, device=device)
        index = torch.tensor([[column_slot(running)]], dtype=torch.int32, device=device)
        _verify_op()(
            qkv,
            a,
            b,
            A_log,
            dt_bias,
            state,
            out,
            starts,
            index,
            None,
            H,
            HV,
            KD**-0.5,
            1,
            rbuf,
            replay.spec_replay,
            None,
            ROWS,
            replay.spec_map,
            replay.spec_ckpt_token,
            replay.spec_ckpt_slot[0],
        )
        torch.cuda.synchronize()
        return running

    boundary = blocks_before * BLOCK
    first_start = boundary - ROWS
    # Prefill left the committed state of position first_start in its column.
    state[column_slot((first_start - 1) // BLOCK)] = s0
    round1 = _round_inputs(gen, device)
    round2 = _round_inputs(gen, device)
    steps1 = _reference_steps(round1[0], round1[1], round1[2], A_log, dt_bias)
    steps2 = _reference_steps(round2[0], round2[1], round2[2], A_log, dt_bias)

    # Round 1 ends exactly on the boundary.
    running1 = run_round(first_start, *round1)
    assert running1 == blocks_before - 1
    torch.testing.assert_close(state[column_slot(running1)], s0, rtol=0, atol=0)

    # Round 1 accepted `accepted` tokens. Round 2 crosses into the next column;
    # the align pre-copy moves the running state there first.
    replay.accepted[0] = accepted
    second_start = first_start + accepted
    running2 = (second_start + ROWS - 1) // BLOCK
    assert running2 == running1 + 1
    state[column_slot(running2)] = state[column_slot(running1)]
    run_round(second_start, *round2)

    committed = _apply(s0, steps1[:accepted])
    torch.testing.assert_close(
        state[column_slot(running2)], committed, rtol=2e-3, atol=2e-3
    )
    # The block that ends on the boundary holds the boundary state: stored from
    # round 2's replay when round 1 was fully accepted, in-round otherwise.
    at_boundary = _apply(committed, steps2[: ROWS - accepted])
    torch.testing.assert_close(
        state[column_slot(running1)], at_boundary, rtol=2e-3, atol=2e-3
    )
