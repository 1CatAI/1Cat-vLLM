# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialized sparse-attention policy and qualified tuning defaults."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from pydantic import Field

from vllm.config.execution_policy_base import ExecutionPolicy
from vllm.config.utils import config, hash_factors, resolve_legacy_fields


@dataclass(frozen=True)
class QsaTuning:
    score_tile_mb: int = 64
    cublas_min_rows: int = 512
    cublas_min_score_elements: int = 1024**2
    xqa_page4_min_rows: int = 64


SM70_QSA_TUNING = QsaTuning()


def read_sparse_legacy(name):
    from vllm import envs

    raw = envs.environment_variables[name]()
    defaults = {
        "VLLM_SM70_QSA_INDEXER_SCORE_TILE_MB": SM70_QSA_TUNING.score_tile_mb,
        "VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_ROWS": SM70_QSA_TUNING.cublas_min_rows,
        "VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS": (
            SM70_QSA_TUNING.cublas_min_score_elements
        ),
        "VLLM_SM70_QSA_XQA_PAGE4_MIN_ROWS": SM70_QSA_TUNING.xqa_page4_min_rows,
    }
    if name in defaults:
        return defaults[name] if raw is None else int(raw)
    if name == "VLLM_SM70_QSA_MTP_TOPK":
        return raw  # Registered bool(int(...)) dialect and error behavior.
    return raw is None or raw == "1"


@config
class Sm70SparseConfig(ExecutionPolicy):
    """Per-engine sparse policy; operators still guard dynamic tensor layouts."""

    legacy_reader: ClassVar[Callable[[str], object] | None] = staticmethod(
        read_sparse_legacy
    )

    indexer_decode_cublas: bool = True
    """Share paged index keys between query heads and rows when eligible."""
    decode_bmm: bool = True
    """Gather packed FP8 keys for eligible FP16 sparse decode matmuls."""
    prefill_bmm: bool = True
    """Use bounded batched matmuls for eligible FP16 sparse prefill."""
    active: bool = Field(default=False, init=False)
    """Whether the engine metadata describes sparse indexed attention."""
    reason: str | None = Field(default=None, init=False)
    """Startup qualification; calls also validate dynamic tensor layouts."""
    errors: dict[str, str] = Field(default_factory=dict, init=False)
    """Malformed legacy inputs remain isolated from engines that do not use QSA."""

    qsa_indexer_cublas: bool | None = None
    """Use the qualified tiled cuBLAS scorer; exact legacy equality to 1."""
    qsa_mtp_topk: bool | None = None
    """Retain the M5/M10 sparse top-k compaction route."""
    qsa_score_tile_mb: int | None = None
    """Score workspace budget; retain the qualified 64 MiB default."""
    qsa_cublas_min_rows: int | None = None
    """Minimum query rows for the cuBLAS scorer."""
    qsa_cublas_min_score_elements: int | None = None
    """Minimum total score elements for the cuBLAS scorer."""
    qsa_xqa_page4: bool | None = None
    """Enable the qualified page-four XQA route."""
    qsa_xqa_page4_min_rows: int | None = None
    """Retain the calibrated page-four row crossover."""
    qsa_grouped_page4: bool | None = None
    """Enable the existing grouped page-four verifier."""
    qsa_grouped_pad_fix: bool | None = None
    """Retain the grouped verifier's padding correction semantics."""

    aliases: ClassVar[dict[str, str]] = {
        "qsa_indexer_cublas": "VLLM_SM70_QSA_INDEXER_CUBLAS",
        "qsa_mtp_topk": "VLLM_SM70_QSA_MTP_TOPK",
        "qsa_score_tile_mb": "VLLM_SM70_QSA_INDEXER_SCORE_TILE_MB",
        "qsa_cublas_min_rows": "VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_ROWS",
        "qsa_cublas_min_score_elements": (
            "VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS"
        ),
        "qsa_xqa_page4": "VLLM_SM70_QSA_XQA_PAGE4",
        "qsa_xqa_page4_min_rows": "VLLM_SM70_QSA_XQA_PAGE4_MIN_ROWS",
        "qsa_grouped_page4": "VLLM_SM70_QSA_GROUPED_PAGE4",
        "qsa_grouped_pad_fix": "VLLM_SM70_QSA_GROUPED_PAD_FIX",
    }

    def resolve(self):
        pending = {
            field: alias
            for field, alias in self.aliases.items()
            if field not in self.sources
        }
        resolve_legacy_fields(
            self, pending, reader=read_sparse_legacy, deferred_errors=self.errors
        )

    def validate_active(self):
        if self.active and self.errors:
            raise ValueError(next(iter(self.errors.values())))

    def value(self, field):
        if field in self.errors:
            raise ValueError(self.errors[field])
        return getattr(self, field)

    def compute_hash(self):
        return hash_factors(
            {
                "qsa": super().compute_hash(),
                "indexer_decode_cublas": self.indexer_decode_cublas,
                "decode_bmm": self.decode_bmm,
                "prefill_bmm": self.prefill_bmm,
            }
            if self.active
            else {}
        )


def sparse_policy(config=None) -> Sm70SparseConfig:
    from vllm.config.execution_policy import capture_execution_policy

    return capture_execution_policy(
        "kernel_config.sm70_sparse", Sm70SparseConfig, config
    )
