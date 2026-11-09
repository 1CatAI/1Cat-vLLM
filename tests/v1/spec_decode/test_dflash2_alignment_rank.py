# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import _diagnostic_rank


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_alignment_dump_uses_initialized_process_rank(monkeypatch, rank):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    assert _diagnostic_rank() == rank


@pytest.mark.parametrize(
    "rank,local,expected", [(None, None, 0), (None, "2", 2), ("3", "1", 3)]
)
def test_alignment_dump_rank_fallback_before_initialization(
    monkeypatch, rank, local, expected
):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    if rank is not None:
        monkeypatch.setenv("RANK", rank)
    if local is not None:
        monkeypatch.setenv("LOCAL_RANK", local)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert _diagnostic_rank() == expected
