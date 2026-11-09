# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.utils.cpu_triton_utils import _FuncWrapper
from vllm.v1.worker import block_table


@pytest.mark.parametrize("num_reqs", [1, 2])
@pytest.mark.parametrize("num_tokens", [1, 3, 5])
def test_repeated_cpu_slot_mapping_keeps_registered_fallback(
    monkeypatch, num_reqs, num_tokens
):
    table = block_table.BlockTable(
        block_size=16,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=16,
        cp_kv_cache_interleave_size=1,
    )
    fallback = Mock()
    monkeypatch.setattr(
        block_table, "_compute_slot_mapping_kernel", _FuncWrapper(fallback)
    )

    class GPUOnly:
        def __getitem__(self, grid):
            raise AssertionError("CPU slot mapping entered a Triton fast path")

    monkeypatch.setattr(block_table, "_compute_slot_mapping_no_pad_kernel", GPUOnly())
    monkeypatch.setattr(
        block_table, "_compute_slot_mapping_single_req_no_pad_kernel", GPUOnly()
    )
    positions = torch.arange(num_tokens)
    starts = torch.tensor(
        [0, num_tokens] if num_reqs == 1 else [0, 1, num_tokens], dtype=torch.int32
    )
    # A warmup or previous batch initializes padding; equal/growing batches
    # used to bypass the sole CPU-registered kernel on the next request.
    table._slot_mapping_pad_initialized = True
    table._slot_mapping_last_num_tokens = 3
    table.compute_slot_mapping(num_reqs, starts, positions)
    table.compute_slot_mapping(num_reqs, starts, positions)
    assert fallback.call_count == 2
    args = fallback.call_args.args
    assert args[0] == num_tokens
    assert args[1] == 8
    assert args[2] is starts
    assert args[3] is positions
    assert args[4] is table.block_table.gpu
    assert args[7] is table.slot_mapping.gpu


@pytest.mark.parametrize("num_reqs", [1, 2])
def test_accelerator_slot_mapping_keeps_no_pad_dispatch(monkeypatch, num_reqs):
    # Host tensors suffice to observe dispatch without launching GPU kernels.
    table = block_table.BlockTable(16, 2, 4, 8, False, torch.device("cpu"), 16, 1)
    table.device = torch.device("cuda")
    table._slot_mapping_pad_initialized = True
    table._slot_mapping_last_num_tokens = 2
    single, general, padded = Mock(), Mock(), Mock()
    monkeypatch.setattr(
        block_table,
        "_compute_slot_mapping_single_req_no_pad_kernel",
        _FuncWrapper(single),
    )
    monkeypatch.setattr(
        block_table, "_compute_slot_mapping_no_pad_kernel", _FuncWrapper(general)
    )
    monkeypatch.setattr(
        block_table, "_compute_slot_mapping_kernel", _FuncWrapper(padded)
    )
    starts = torch.tensor([0, 3] if num_reqs == 1 else [0, 1, 3])
    table.compute_slot_mapping(num_reqs, starts, torch.arange(3))
    (single if num_reqs == 1 else general).assert_called_once()
    (general if num_reqs == 1 else single).assert_not_called()
    padded.assert_not_called()
