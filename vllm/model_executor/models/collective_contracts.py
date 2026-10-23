# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model contracts for retained hyperconnection collective compositions."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HcCollectiveContract:
    min_rows: int
    max_rows: int
    operators: tuple[str, ...]
    hidden_width: int = 10240
    world_size: int = 4
    fused_chain_cta_warps: int = 8

    def accepts_layout(self, branches):
        return bool(
            branches.is_cuda
            and branches.dtype == torch.float16
            and branches.ndim == 2
            and self.min_rows <= branches.shape[0] <= self.max_rows
            and branches.shape[1] == self.hidden_width
            and branches.is_contiguous()
        )


QWEN_HC_BATCH = HcCollectiveContract(2, 16, ("sm70_qwen38_hc_batch",))
QWEN_HC_SHARD = HcCollectiveContract(
    1, 1, ("sm70_qwen38_hc_down_allgather", "sm70_qwen38_hc_gate_mix")
)
