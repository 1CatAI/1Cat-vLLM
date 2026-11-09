# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Target sampling outcomes: sampled output, completed logits, or None fallback."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ComputedTargetLogits:
    """Projection already executed; non-gather ranks can legitimately hold None."""

    logits: torch.Tensor | None
