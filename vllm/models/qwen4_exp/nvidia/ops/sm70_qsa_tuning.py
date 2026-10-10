# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured SM70 QSA tuning; legacy overrides expire after one released cycle."""

from vllm import envs
from vllm.config.sm70_sparse import SM70_QSA_TUNING, QsaTuning  # noqa: F401
from vllm.logger import init_logger

logger = init_logger(__name__)


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
