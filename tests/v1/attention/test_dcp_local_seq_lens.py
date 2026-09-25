# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP local sequence lengths: per-rank results and no host synchronization."""

import pytest
import torch

from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens


def _reference(seq_lens: torch.Tensor, dcp: int, rank: int, interleave: int):
    """Count the positions this rank owns under the interleaved layout."""
    out = []
    for length in seq_lens.tolist():
        positions = torch.arange(length)
        owned = (positions // interleave) % dcp == rank
        out.append(int(owned.sum()))
    return torch.tensor(out, dtype=torch.int32)


@pytest.mark.parametrize("dcp", [1, 2, 4])
@pytest.mark.parametrize("interleave", [1, 4, 16])
def test_rank_scalar_matches_reference_and_all_ranks(dcp: int, interleave: int):
    generator = torch.Generator().manual_seed(57431)
    seq_lens = torch.randint(0, 5000, (9,), generator=generator, dtype=torch.int32)
    all_ranks = get_dcp_local_seq_lens(seq_lens, dcp, None, interleave)
    for rank in range(dcp):
        local = get_dcp_local_seq_lens(seq_lens, dcp, rank, interleave)
        assert local.shape == seq_lens.shape
        assert local.dtype == torch.int32
        assert torch.equal(local, _reference(seq_lens, dcp, rank, interleave))
        column = all_ranks if all_ranks.dim() == 1 else all_ranks[:, rank]
        assert torch.equal(local, column)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rank_scalar_path_does_not_synchronize():
    """Metadata builds call this every step; a host sync here stalls decode.

    Building the rank offset as ``torch.tensor([[rank]], device="cuda")`` is a
    pageable H2D copy that synchronizes the stream. Sync debug mode turns any
    synchronizing call into an error.
    """
    seq_lens = torch.tensor([37, 4097, 30030], dtype=torch.int32, device="cuda")
    torch.accelerator.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        local = get_dcp_local_seq_lens(seq_lens, 2, 1, 1)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert torch.equal(
        local.cpu(), _reference(seq_lens.cpu(), dcp=2, rank=1, interleave=1)
    )
