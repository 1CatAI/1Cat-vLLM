# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logging subscriber and compatibility access to diagnostic preparation."""

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100.plan import diagnostics
from vllm.v1.attention.backends.flash_v100.plan.events import (
    DiagnosticMessage,
    diagnostic_messages,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def log_diagnostic(event: DiagnosticMessage) -> None:
    logger.info(event.message, *event.args)


diagnostic_messages.subscribe(log_diagnostic)


def __getattr__(name: str):
    return getattr(diagnostics, name)
