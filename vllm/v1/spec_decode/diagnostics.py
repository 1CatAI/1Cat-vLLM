# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common target diagnostics using initialization-captured policy."""

import torch


def trace_target_logits(policy, logits, input_batch, logger) -> None:
    if not policy.target_logits or logits is None:
        return
    positions = input_batch.positions[input_batch.logits_indices]
    if int(positions[0].item()) >= policy.target_min_position:
        top_values, top_ids = torch.topk(logits.float(), 2, dim=-1)
        logger.warning(
            "DFLASH_TARGET_LOGITS_TRACE inputs=%s positions=%s top1=%s top1_margin=%s",
            input_batch.input_ids[input_batch.logits_indices].tolist(),
            positions.tolist(),
            top_ids[:, 0].tolist(),
            (top_values[:, 0] - top_values[:, 1]).tolist(),
        )
