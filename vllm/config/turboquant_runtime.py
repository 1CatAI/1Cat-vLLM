# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant provider and observation inputs captured during engine setup."""

import os
from collections.abc import Callable
from typing import ClassVar

from vllm.config.diagnostic_dump import TensorDumpConfig
from vllm.config.execution_policy_base import ExecutionPolicy
from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


def read_turboquant_legacy(name):
    raw = os.getenv(name)
    if name in (
        "VLLM_SM70_TURBOQUANT_FLASH_V100_PREFILL",
        "VLLM_SM70_TURBOQUANT_FLASH_V100_DECODE",
        "VLLM_SM70_TURBOQUANT_RESERVE_WORKSPACE",
    ):
        return raw != "0"
    if name.endswith(("_DIR", "_PATH")):
        return raw
    default = 1 if name.endswith("_DUMP_LIMIT") else 0
    converter = float if name.endswith("_THRESHOLD") else int
    if raw is None:
        return converter(default)
    try:
        return converter(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s env %s=%r",
            "float" if converter is float else "integer",
            name,
            raw,
        )
        return converter(default)


@config
class TurboQuantRuntimePolicy(ExecutionPolicy):
    flash_prefill: bool | None = None
    """Permit the existing Flash-V100 dense prefill capability probe."""
    flash_decode: bool | None = None
    """Permit Flash-V100 decode only for the existing supported packed key formats."""
    reserve_workspace: bool | None = None
    """Reserve continuation scratch before capture."""
    continuation_workspace_tokens: int | None = None
    """Nonpositive retains the model maximum-length fallback."""

    legacy_reader: ClassVar[Callable[[str], object] | None] = staticmethod(
        read_turboquant_legacy
    )
    aliases: ClassVar[dict[str, str]] = {
        "flash_prefill": "VLLM_SM70_TURBOQUANT_FLASH_V100_PREFILL",
        "flash_decode": "VLLM_SM70_TURBOQUANT_FLASH_V100_DECODE",
        "reserve_workspace": "VLLM_SM70_TURBOQUANT_RESERVE_WORKSPACE",
        "continuation_workspace_tokens": (
            "VLLM_SM70_TURBOQUANT_CONTINUATION_WORKSPACE_TOKENS"
        ),
    }

    def resolve(self):
        super().resolve()
        self.hash_fields = ("flash_prefill", "flash_decode")


@config
class TurboQuantDiagnostics(ExecutionPolicy):
    prefill_limit: int | None = None
    """Engine-wide prefill compare budget; nonpositive disables comparisons."""
    decode_limit: int | None = None
    """Engine-wide decode compare budget; nonpositive disables comparisons."""
    dump_limit: int | None = None
    """Per-stage tensor-dump budget, independent of compare limits."""
    dump_threshold: float | None = None
    """Minimum maximum difference required for a tensor dump."""
    dump_directory: str | None = None
    """Initialized dump destination; empty disables payload creation."""
    log_path: str | None = None
    """Initialized JSONL destination; preserves comparison record fields."""

    legacy_reader: ClassVar[Callable[[str], object] | None] = staticmethod(
        read_turboquant_legacy
    )
    aliases: ClassVar[dict[str, str]] = {
        "prefill_limit": "VLLM_SM70_TURBOQUANT_COMPARE_FLASH_V100_PREFILL",
        "decode_limit": "VLLM_SM70_TURBOQUANT_COMPARE_FLASH_V100_DECODE",
        "dump_limit": "VLLM_SM70_TURBOQUANT_COMPARE_DUMP_LIMIT",
        "dump_threshold": "VLLM_SM70_TURBOQUANT_COMPARE_DUMP_THRESHOLD",
        "dump_directory": "VLLM_SM70_TURBOQUANT_COMPARE_DUMP_DIR",
        "log_path": "VLLM_SM70_TURBOQUANT_COMPARE_LOG_PATH",
    }

    def dump_channels(self):
        return {
            "turboquant_" + stage: TensorDumpConfig(
                directory=self.dump_directory, max_dumps=self.dump_limit
            )
            for stage in ("prefill", "decode")
        }

    def compile_ignored_aliases(self):
        return set(self.aliases.values()) if self.sources else set()
