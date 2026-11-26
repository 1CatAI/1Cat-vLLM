# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact register-tiled packed index scores, owned by the model indexer."""

import time
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

    def report(self) -> None:
        """No counters: the tiled scorer has nothing to report."""


class IndexTensorCoreScorer:
    """Tensor-core decode scores (v41idx3), bitwise IndexTiledScorer's.

    The kernel adds four products at a time on the tensor cores. A guard in the
    kernel passes a (token, key) cell only where neither that add nor the tiled
    scorer's FMA chain can round, so both equal the exact sum, and recomputes
    every other head by the tiled scorer's chain. Four counters, allocated on
    ``device`` at construction so no graph's memory pool owns them (cells the
    guard did not pass, heads recomputed, calls with a live key, cells
    written), are read by ``report`` where Python runs: a prefill call, never
    inside a graph capture.
    """

    def __init__(
        self,
        library: str | None,
        heads: int,
        device: torch.device,
        report_seconds: float = 30.0,
    ) -> None:
        if heads != 8:
            raise ValueError("V4.1 tensor-core index requires 8 heads per rank")
        if library is None:
            raise RuntimeError("V4.1 index library policy was not initialized")
        path = (
            Path(library)
            if library
            else (
                Path(__file__).resolve().parents[3]
                / "benchmarks/deepseek_v41/v41idx3/build/v41idx3.so"
            )
        )
        if torch.cuda.get_device_capability() != (7, 0):
            raise ValueError("V4.1 tensor-core index is qualified only on SM70")
        torch.ops.load_library(str(path))
        if not all(
            hasattr(torch.ops.v41idx3, name) for name in ("version", "index_scores_tc")
        ):
            raise RuntimeError(
                f"V4.1 tensor-core index library lacks required operations: {path}"
            )
        if torch.ops.v41idx3.version() != 1:
            raise RuntimeError("V4.1 tensor-core index library ABI must be version 1")
        self.operation = torch.ops.v41idx3.index_scores_tc
        self.report_seconds = report_seconds
        self.counters = torch.zeros(4, dtype=torch.int64, device=device)
        self.read_at = time.monotonic()
        self.last: list[int] | None = None
        logger.info("V4.1 tensor-core index scorer active: heads=%d", heads)

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
        if self.counters.device != out.device:
            raise RuntimeError(
                f"V4.1 tensor-core index: counters on {self.counters.device}, "
                f"call on {out.device}"
            )
        self.operation(out, q, weights, keys, positions, heads, ratio, 0, self.counters)

    def report(self) -> None:
        """Log the counters when one moved, at most every report_seconds."""
        if self.counters.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        now = time.monotonic()
        if now - self.read_at < self.report_seconds:
            return
        self.read_at = now
        values = self.counters.tolist()
        if values != self.last:
            self.last = values
            logger.info(
                "V4.1 tensor-core index: %d cells over the guard, %d heads "
                "recomputed, %d calls, %d cells",
                *values,
            )
