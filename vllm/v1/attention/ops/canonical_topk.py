# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical ordering of selected sparse indices, without changing selection."""

import torch


def canonicalize_selected_indices(selected: torch.Tensor) -> torch.Tensor:
    """Sort each row's nonnegative IDs while preserving every negative slot.

    Repeated IDs retain their multiplicity. This does not select or deduplicate
    IDs, and does not make membership deterministic when TopK scores are tied.
    Values may be logical token IDs or pool IDs; no model or device policy is
    implied. Only shape/dtype control Python branches, so tensor data stays on
    the device and the operation has no host synchronization or dynamic output.
    """
    if selected.ndim != 2 or selected.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected a two-dimensional integer index tensor")
    if selected.numel() == 0:
        return selected.clone()
    valid = selected >= 0
    ordered = (
        torch.where(valid, selected, torch.iinfo(selected.dtype).max)
        .sort(dim=-1)
        .values
    )
    ordinal = valid.to(torch.int64).cumsum(dim=-1) - 1
    values = ordered.gather(-1, ordinal.clamp_min(0))
    return torch.where(valid, values, selected)
