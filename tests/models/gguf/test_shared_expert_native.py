# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native planner bindings operate directly on the single shared mapping."""

import pytest
import torch

from vllm.utils.shared_host import SharedHostArena

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA build needed"
)


def test_native_planner_compacts_only_after_all_ranks_and_leases_complete():
    from vllm.model_executor.layers.quantization.gguf_expert_cache import _native

    native = _native()
    with SharedHostArena.create(4096) as arena:
        planner = native.SharedExpertPlanner(
            arena.address,
            arena.size,
            [native.SharedExpertLayer(16, 8, 4)],
            4,
            4,
            64,
        )
        for expert in range(4, 8):
            offset = planner.cold_offset(0, expert)
            arena.buffer[offset : offset + 16] = bytes([expert]) * 16
        lease, original_offset = planner.acquire(0, 4)
        copies = planner.plan(0, 1, [4, 4, 1, 1], 100, 1000)
        assert len(copies) == 1
        op = copies[0]
        assert op.load_offset == original_offset and op.save_offset != original_offset
        assert (op.incoming, op.victim, op.slot) == (4, 0, 0)
        arena.buffer[op.save_offset : op.save_offset + 16] = b"\x00" * 16
        for rank in (3, 1, 0):
            planner.acknowledge(rank, 1, 101)
        assert not planner.ready(1, 102)
        planner.acknowledge(2, 1, 103)
        assert not planner.ready(1, 104)
        with pytest.raises(RuntimeError, match="not completed"):
            planner.finish(1, 105)
        assert arena.buffer[original_offset : original_offset + 16] == b"\x04" * 16
        planner.release(lease)
        planner.finish(1, 106)
        assert planner.slot(0, 4) == 0 and planner.slot(0, 0) == -1
        assert planner.cold_offset(0, 0) == original_offset
        assert arena.buffer[original_offset : original_offset + 16] == b"\x00" * 16
        planner.stop()
        with pytest.raises(RuntimeError, match="stopped"):
            planner.plan(0, 2, [4], 107, 1000)
        del planner
