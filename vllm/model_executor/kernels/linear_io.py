# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared linear tensor views, independent of provider registration order."""

import torch


def flatten_linear_input(x: torch.Tensor, features: int | None = None) -> torch.Tensor:
    """Keep existing matrix views; flatten only additional batch dimensions."""
    features = x.shape[-1] if features is None else features
    # Keep reshape's existing error for an ambiguous zero-width input.
    if x.ndim == 2 and features != 0 and x.shape[-1] == features:
        return x
    return x.reshape(-1, features)


def restore_linear_output(
    output: torch.Tensor, x: torch.Tensor, features: int | None = None
) -> torch.Tensor:
    features = output.shape[-1] if features is None else features
    if x.ndim == 2 and output.shape == (x.shape[0], features):
        return output
    return output.reshape(*x.shape[:-1], features)
