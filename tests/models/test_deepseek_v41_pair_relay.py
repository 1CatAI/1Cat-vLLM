# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 TP4 prefill pair relay (VLLM_V41_PREFILL_PAIR_RELAY), CPU.

With the relay on, each rank of the row-split prefill publishes its FP16
activation slice to three of the four parity banks and skips the rank across
the pair boundary; after the existing bank fence it copies that slice from its
pair partner. Every bank must end bit-equal to the four-store result, which is
also checked against a torch reference of the activation. The verify mode's
compare kernel must count every live word and find a single flipped sign bit
(-0.0 vs +0.0), which float equality would miss. Kernels run in the Triton
interpreter (TRITON_INTERPRET=1): this proves indexing, skipping and copying,
not GPU memory ordering (the GPU check is the verify mode on a real server).

    python tests/models/test_deepseek_v41_pair_relay.py   # or pytest
"""

import os
import sys

import pytest

if "triton" in sys.modules and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip(
        "needs TRITON_INTERPRET=1 in the environment: triton was imported "
        "before this file could turn the interpreter on",
        allow_module_level=True,
    )
os.environ["TRITON_INTERPRET"] = "1"  # before triton is imported

from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402

from vllm.model_executor.layers.quantization.gguf_shared_prefill import (  # noqa: E402
    SharedOriginalPrefill,
    _activation_pair_plan,
    _v41_prefill_activation_rows_p,
)

WIDTH = 576
FULL = 4 * WIDTH
PROGRAMS = 3
ROWS = 11
NO_INDEX = 2**31 - 1


def test_pair_plan_admission():
    # Ranks 0-1 and 2-3 are the NVLink pairs: skip the rank diagonally
    # across the pair boundary, pull it from the partner.
    expected = ((3, 1), (2, 0), (1, 3), (0, 2))
    for rank in range(4):
        assert _activation_pair_plan("0", True, 4, PROGRAMS, rank) is None
        plan = _activation_pair_plan("1", True, 4, PROGRAMS, rank)
        assert (plan.skipped, plan.partner) == expected[rank]
    for setting, rows_mode, world, programs in (
        ("2", True, 4, PROGRAMS),
        ("1", False, 4, PROGRAMS),
        ("1", True, 2, PROGRAMS),
        ("1", True, 8, PROGRAMS),
        ("1", True, 4, 0),
    ):
        with pytest.raises(ValueError):
            _activation_pair_plan(setting, rows_mode, world, programs, 0)


def _workspaces(banks):
    pointers = [[t.data_ptr() for t in bank] for bank in banks]
    out = []
    for rank in range(4):
        w = SharedOriginalPrefill.__new__(SharedOriginalPrefill)
        w.cache = SimpleNamespace(rank=rank)
        w.activation_pair = _activation_pair_plan("1", True, 4, PROGRAMS, rank)
        w.rows_full = FULL
        w.rows_pointers = pointers
        w.rows_local = [bank[rank] for bank in banks]
        w.pair_verify = True
        w.pair_verify_sequence = 0
        w.pair_verify_stats = torch.tensor([[0, 0, NO_INDEX]], dtype=torch.int32)
        out.append(w)
    return out, pointers


def test_relay_matches_four_stores_bitwise():
    width_blocks = triton.cdiv(WIDTH, 256)
    order = torch.arange(ROWS, dtype=torch.int64).flip(0)
    probabilities = (torch.arange(ROWS, dtype=torch.float32) + 1) / 16
    banks = [
        [torch.full((ROWS, FULL), -99, dtype=torch.float16) for _ in range(4)]
        for _ in range(2)
    ]
    workspaces, pointers = _workspaces(banks)
    # Active routes, an empty bank (1, 1), tails, and reuse of both parities.
    spans = ((1, 10), (3, 8), (1, 1), (0, 7), (2, 9), (4, 11))
    for epoch, (lo, hi) in enumerate(spans):
        parity = epoch % 2
        bounds = torch.tensor([lo] + [hi] * 32, dtype=torch.int32)
        original = [t.clone() for t in banks[parity]]
        expected = [t.clone() for t in banks[parity]]
        for rank, w in enumerate(workspaces):
            w.pair_shadow_pointers = [None, None]
            w.pair_shadow_pointers[parity] = [t.data_ptr() for t in original]
            w.pair_shadow_local = [None, None]
            w.pair_shadow_local[parity] = original[rank]
            w.pair_verify_stats[0] = torch.tensor([0, 0, NO_INDEX])
        for rank in range(4):
            projection = torch.empty((ROWS, 2 * WIDTH), dtype=torch.float16)
            gate = float((epoch % 2) * 12 + rank + 1)  # crosses the +10 clamp
            up = torch.where(torch.arange(WIDTH) % 2 == 0, -8.0 - rank, 8.0 + rank)
            up[9] = 0
            projection[:, 0::2] = gate
            projection[:, 1::2] = up
            w = workspaces[rank]
            # The original four stores, into the shadow banks.
            w._publish_shadow_activation(
                parity,
                projection,
                probabilities,
                order,
                bounds,
                32,
                WIDTH,
                PROGRAMS,
                PROGRAMS,
            )
            # The relay's three stores, into the live banks.
            _v41_prefill_activation_rows_p[(PROGRAMS, width_blocks)](
                projection,
                probabilities,
                order,
                *pointers[parity],
                bounds,
                EXPERTS=32,
                WIDTH=WIDTH,
                FULL=FULL,
                COLUMN=rank * WIDTH,
                BLOCK=256,
                PROGRAMS=PROGRAMS,
                SKIP_DEST=w.activation_pair.skipped,
                enable_fp_fusion=False,
            )
            g = torch.tensor(min(gate, 10.0))
            reference = (
                (g * torch.sigmoid(g))
                * up.clamp(-10.0, 10.0)[None, :]
                * probabilities[order[lo:hi], None]
            ).to(torch.float16)
            for output in expected:
                output[lo:hi, rank * WIDTH : (rank + 1) * WIDTH] = reference
        # All four writers finish before any rank reads its partner.
        for w in workspaces:
            w._pull_activation_pair(parity, bounds, 32, WIDTH, PROGRAMS, PROGRAMS, 0)
        for actual, stock, ref in zip(banks[parity], original, expected, strict=True):
            assert torch.equal(actual.view(torch.int16), stock.view(torch.int16))
            assert torch.equal(actual.view(torch.int16), ref.view(torch.int16))
        for w in workspaces:
            assert w.pair_verify_stats[0].tolist() == [(hi - lo) * FULL, 0, NO_INDEX]
        if epoch == 1:
            # Flip the sign of a zero in a column the relay did not pull.
            first = lo * FULL + 9
            flat = banks[parity][1].view(torch.int16).flatten()
            assert flat[first] == 0
            flat[first] = -32768
            workspaces[1].pair_verify_stats[0] = torch.tensor([0, 0, NO_INDEX])
            workspaces[1]._pull_activation_pair(
                parity, bounds, 32, WIDTH, PROGRAMS, PROGRAMS, 0
            )
            assert workspaces[1].pair_verify_stats[0].tolist() == [
                (hi - lo) * FULL,
                1,
                first,
            ]
            workspaces[1].report_pair_verify(hi - lo, 1)
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
