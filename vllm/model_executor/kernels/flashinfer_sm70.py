# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin engine boundary for independently packaged FlashInfer SM70 kernels."""

import importlib
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)
_REQUIRED_VERSION = "0.7.2+1cat.sm70.3"
_DEFAULT_DIVISOR = math.sqrt(128)


def _load_provider():
    package = importlib.import_module("flashinfer")
    if package.__version__ != _REQUIRED_VERSION:
        raise RuntimeError(
            f"SM70 candidate requires flashinfer-python=={_REQUIRED_VERSION}; "
            f"loaded {package.__version__} from {package.__file__}"
        )
    provider = importlib.import_module("flashinfer.sm70")
    provider.prepare("qsa")
    return provider


class FlashInferQsaScorer:
    """Per-indexer workspace, warmed before capture; no performance admission."""

    def __init__(self, mode: str, workspace_limit_bytes: int = 16 * 1024 * 1024):
        if mode not in ("candidate", "force"):
            raise ValueError("FlashInfer QSA mode must be candidate or force")
        self.mode = mode
        self.workspace_limit_bytes = workspace_limit_bytes
        self.provider = _load_provider()
        self.plan: Any = None
        self.last_route = "unprepared"
        self._logged_rejections: set[str] = set()

    def _reject(self, reason):
        self.last_route = reason
        if self.mode == "force":
            raise RuntimeError(f"Forced FlashInfer SM70 QSA rejected: {reason}")
        if reason not in self._logged_rejections:
            logger.info("FlashInfer SM70 QSA candidate not selected: %s", reason)
            self._logged_rejections.add(reason)
        return None

    def validate_force(self, q, cache, table, requests, positions, lengths):
        if self.mode == "force" or q.shape[0] > 32:
            reason = self.provider.qsa_score_reason(
                q, cache, table, requests, positions, lengths
            )
            if reason is not None:
                self._reject(reason)

    def record_legacy_path(self, reason):
        self._reject(reason)

    def score(
        self,
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        compress,
        columns,
        divisor=_DEFAULT_DIVISOR,
    ):
        reason = self.provider.qsa_score_reason(
            q, cache, table, requests, positions, lengths
        )
        if reason is not None:
            return self._reject(reason)
        # One bounded workspace per layer. A larger layout requires a new plan;
        # captured storage is never silently discarded or replaced.
        required = 32 * columns * 4 + (32 + 32 * 32 + 32) * 4
        if required > self.workspace_limit_bytes:
            return self._reject("workspace_budget_exceeded")
        if self.plan is None:
            self.plan = self.provider.QsaScorePlan(32, 32, columns, q.device)
            logger.info("Prepared FlashInfer SM70 QSA: %s", self.plan.provenance())
        if columns > self.plan.max_columns:
            return self._reject("prepared_column_capacity_exceeded")
        result = self.plan.run(
            q,
            cache,
            table,
            requests,
            positions,
            lengths,
            compress=compress,
            columns=columns,
            divisor=divisor,
        )
        self.last_route = "flashinfer_sm70_qsa"
        return result
