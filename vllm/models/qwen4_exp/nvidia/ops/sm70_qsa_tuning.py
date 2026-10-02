# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured SM70 QSA tuning; legacy overrides expire after one released cycle."""

from dataclasses import dataclass

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class QsaTuning:
    score_tile_mb: int = 64
    cublas_min_rows: int = 512
    cublas_min_score_elements: int = 1024**2
    xqa_page4_min_rows: int = 64


# Keep the qualified SM70 indexer defaults and the conservative 64-row XQA
# crossover. These constants do not change the scorer, selection or precision.
# Historical benchmark overrides remain replayable during the compatibility
# cycle, including its former 4096-row XQA policy.
SM70_QSA_TUNING = QsaTuning()


def legacy_qsa_tuning(name: str, default: int) -> int:
    value = getattr(envs, name)
    if value is None:
        return default
    logger.warning_once(
        "%s is deprecated and will be removed after one released compatibility "
        "cycle. SM70 QSA selects measured tuning from SM70_QSA_TUNING; "
        "the legacy override is honored for this cycle.",
        name,
    )
    return int(value)
