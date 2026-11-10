# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact register-tiled packed index scores, owned by the model indexer."""

from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class IndexTiledScorer:
    """Load/admit once before graph capture; calls consume no environment."""

    def __init__(self, library: str | None, heads: int) -> None:
        if heads not in (4, 8):
            raise ValueError("V4.1 tiled index requires 4 or 8 heads per rank")
        if library is None:
            raise RuntimeError("V4.1 index library policy was not initialized")
        path = (
            Path(library)
            if library
            else (
                Path(__file__).resolve().parents[3]
                / "benchmarks/deepseek_v41/v41idx/build/v41idx.so"
            )
        )
        if torch.cuda.get_device_capability() != (7, 0):
            raise ValueError("V4.1 tiled index is qualified only on SM70")
        torch.ops.load_library(str(path))
        if not all(
            hasattr(torch.ops.v41idx, name)
            for name in ("version", "index_scores_tiled")
        ):
            raise RuntimeError(
                f"V4.1 tiled index library lacks required operations: {path}"
            )
        if torch.ops.v41idx.version() != 1:
            raise RuntimeError("V4.1 tiled index library ABI must be version 1")
        self.operation = torch.ops.v41idx.index_scores_tiled
        self.logged = False

    def scores(
        self,
        out: torch.Tensor,
        q: torch.Tensor,
        weights: torch.Tensor,
        keys: torch.Tensor,
        positions: torch.Tensor,
        heads: int,
        ratio: int,
    ) -> None:
        if not self.logged:
            logger.info("V4.1 tiled index scorer active: heads=%d", heads)
            self.logged = True
        self.operation(out, q, weights, keys, positions, heads, ratio)
