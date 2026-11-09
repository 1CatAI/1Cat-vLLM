# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explain physical-slot and unused conv-storage differences in raw captures.

This never edits captures or replaces the raw byte comparison. The caller must
provide the convolution width from the frozen model. Unknown layouts fail
closed. All values consumed by convolution, and all verifier output storage,
remain subject to the exact comparison.
"""

from __future__ import annotations

import torch


def _field(states: dict, name: str) -> torch.Tensor:
    matches = [v for k, v in states.items() if k.startswith(name + ":")]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous state metadata: {name}")
    return matches[0]


def check_slot_mapping(
    left: dict, right: dict, mapping: dict[int, int], reverse: dict[int, int]
) -> None:
    """Require one consistent bijection across every observed state access."""
    if left.keys() != right.keys():
        raise ValueError("State snapshot coverage differs")
    for label, a in left.items():
        name = label.split(":")[0]
        if not name.endswith(("/indices", "/slot_table")):
            continue
        b = right[label]
        if a.shape != b.shape or a.dtype != b.dtype or a.is_floating_point():
            raise ValueError(f"Invalid slot metadata: {label}")
        for x, y in zip(a.reshape(-1).tolist(), b.reshape(-1).tolist()):
            if x < 0 or y < 0:
                if x != y:
                    raise ValueError("Padding slot changed into a live slot")
                continue
            if mapping.get(x, y) != y or reverse.get(y, x) != x:
                raise ValueError("Inconsistent or aliased physical-slot mapping")
            mapping[x], reverse[y] = y, x


def explain_state_difference(
    label: str, left: dict, right: dict, conv_width: int
) -> str | None:
    """Classify only source-defined unused storage; retain every raw mismatch."""
    if not 2 <= conv_width <= 6:
        raise ValueError("Unsupported convolution width")
    name = label.split(":")[0]
    if name.endswith(("/indices", "/slot_table")):
        # check_slot_mapping must already have validated the complete captures.
        return "bijective_physical_slot_renaming"
    if "/conv/" not in name or not name.endswith("/values"):
        return None
    prefix, suffix = name.split("/conv/", 1)
    a, b = left[label], right[label]
    if a.shape != b.shape or a.dtype != b.dtype or a.ndim != 3:
        raise ValueError("Unsupported convolution state layout")
    n, _, storage = a.shape
    history = conv_width - 1
    if storage < history:
        raise ValueError("Convolution state smaller than its history")
    validity = [
        _field(side, name.rsplit("/", 1)[0] + "/valid").reshape(-1)
        for side in (left, right)
    ]
    if not torch.equal(*validity) or validity[0].numel() != n:
        raise ValueError("Convolution slot validity differs")
    columns = torch.arange(storage).reshape(1, -1)
    active = columns < history
    if prefix.startswith("prefill/"):
        if suffix == "input_state/values":
            initial = [
                _field(side, prefix + "/conv/has_initial_state").reshape(-1)
                for side in (left, right)
            ]
            if not torch.equal(*initial) or initial[0].numel() != n:
                raise ValueError("Convolution initial-state contract differs")
            active = active & initial[0].reshape(-1, 1)
        elif suffix != "output_state/values":
            return None
        # causal_conv1d_fn fixes state_len = KERNEL_WIDTH - 1. It does not
        # initialize the extra speculative history columns in the allocation.
    elif prefix.startswith("verify/") and suffix == "input_state/values":
        selectors = [
            _field(side, prefix + "/conv/num_accepted_tokens").reshape(-1)[:n]
            for side in (left, right)
        ]
        if not torch.equal(*selectors) or selectors[0].numel() != n:
            raise ValueError("Convolution acceptance selectors differ")
        offset = selectors[0] - 1
        valid = validity[0]
        if ((offset < 0) | (offset + history > storage))[valid].any():
            raise ValueError("Convolution selector outside stored history")
        # Both the convolution and the rolling-state copy read within this
        # window; output storage is always compared in full, including q8.
        active = (columns >= offset[:, None]) & (columns < offset[:, None] + history)
    else:
        return None
    mask = (active & validity[0].reshape(-1, 1))[:, None, :].expand_as(a)
    aa, bb = a[mask].contiguous(), b[mask].contiguous()
    if not torch.equal(aa.view(torch.uint8), bb.view(torch.uint8)):
        return None
    return "unused_convolution_storage"
