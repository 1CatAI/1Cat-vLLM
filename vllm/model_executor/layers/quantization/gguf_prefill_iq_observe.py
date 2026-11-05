# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Untimed owner-window descriptor census; no activation or prompt capture."""

from __future__ import annotations

import json

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class ObservedBaseline:
    def __init__(self, layer: int, rank: int, tokens: int) -> None:
        self.layer, self.rank, self.tokens = layer, rank, tokens

    def gemm(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        offsets: torch.Tensor,
        weights: torch.Tensor,
        stats: torch.Tensor,
        family: int,
        decoder: int,
        n: int,
        experts: int,
        group: int,
        gated: bool,
        tune: bool,
        indices: torch.Tensor | None,
    ) -> None:
        bounds = offsets.cpu().tolist()
        counts = [b - a for a, b in zip(bounds, bounds[1:])]
        logger.info(
            "PREFMOE_DESCRIPTOR %s",
            json.dumps(
                dict(
                    layer=self.layer,
                    rank=self.rank,
                    tokens=self.tokens,
                    role="gu" if indices is not None else "dn",
                    codec=decoder,
                    n=n,
                    k=input.shape[1],
                    descriptor_m=output.shape[0],
                    low=bounds[0],
                    high=bounds[-1],
                    counts=counts,
                ),
                separators=(",", ":"),
            ),
        )
        torch.ops._C.gguf_grouped_prefill_gemm_sm70_out(
            output,
            input,
            offsets,
            weights,
            stats,
            family,
            decoder,
            n,
            experts,
            group,
            gated,
            tune,
            indices,
        )
