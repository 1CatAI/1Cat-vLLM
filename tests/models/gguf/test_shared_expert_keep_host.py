# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_KEEP_HOST: every expert keeps a permanent host home, so a
free-slot exchange never writes the victim back to host memory."""

import pytest
import torch

from vllm.utils.shared_host import SharedHostArena

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA build needed"
)

BYTES, EXPERTS, RESIDENT = 16, 8, 4


def _planner(native, arena):
    return native.SharedExpertPlanner(
        arena.address,
        arena.size,
        [native.SharedExpertLayer(BYTES, EXPERTS, RESIDENT)],
        4,
        4,
        64,
    )


def test_keep_host_exchange_has_no_writeback(monkeypatch):
    from vllm.model_executor.layers.quantization.gguf_expert_cache import _native

    monkeypatch.setenv("VLLM_V41_SWAP_FREE", "1")
    monkeypatch.setenv("VLLM_V41_FREE_SLOTS", "2")
    monkeypatch.setenv("VLLM_V41_ASYNC_EVICT", "0")
    monkeypatch.setenv("VLLM_V41_KEEP_HOST", "1")
    native = _native()
    with SharedHostArena.create(4096) as arena:
        planner = _planner(native, arena)
        # Cold experts live at their permanent homes: expert * bytes.
        for expert in range(RESIDENT, EXPERTS):
            home = expert * BYTES
            assert planner.cold_offset(0, expert) == home
            arena.buffer[home : home + BYTES] = bytes([expert]) * BYTES
        copies = planner.plan(0, 1, [4, 4, 1, 1], 100, 1000, 0, 6, 0)
        assert len(copies) == 1
        op = copies[0]
        assert (op.incoming, op.victim, op.slot) == (4, 0, 0)
        assert op.load_offset == 4 * BYTES
        assert op.save_offset == 2**64 - 1  # no host allocation for the victim
        for rank in range(4):
            planner.acknowledge(rank, 1, 101 + rank)
        planner.finish(1, 106)
        assert planner.slot(0, 4) == 0 and planner.slot(0, 0) == -1
        # The victim is cold again at its own home; the incoming home is intact.
        assert planner.cold_offset(0, 0) == 0
        assert arena.buffer[4 * BYTES : 5 * BYTES] == b"\x04" * BYTES
        planner.stop()
        del planner


def test_keep_host_requires_free_slot_mode(monkeypatch):
    from vllm.model_executor.layers.quantization.gguf_expert_cache import _native

    monkeypatch.setenv("VLLM_V41_SWAP_FREE", "0")
    monkeypatch.setenv("VLLM_V41_KEEP_HOST", "1")
    native = _native()
    with SharedHostArena.create(4096) as arena, pytest.raises(ValueError):
        _planner(native, arena)
