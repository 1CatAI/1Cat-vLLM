# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialized sparse-attention policy and qualified tuning defaults."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from pydantic import Field

from vllm.config.execution_policy_base import DeferredExecutionPolicy
from vllm.config.utils import config, hash_factors


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
        "VLLM_SM70_INDEXER_PREFILL_TILE_MB": 192,
        "VLLM_SM70_INDEXER_DECODE_CUBLAS_MIN_KEYS": 1024,
    }
    if name in defaults:
        return defaults[name] if raw is None else int(raw)
    if name == "VLLM_SM70_QSA_MTP_TOPK" or name.startswith("VLLM_SM70_DSV4_"):
        return raw  # Registered bool(int(...)) dialect and error behavior.
    return raw is None or raw == "1"


@config
class Sm70SparseConfig(DeferredExecutionPolicy):
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

    indexer_fused_logits: bool | None = None
    """Fuse paged decode dequantization and scoring."""

    indexer_relu: bool | None = None
    """Retain per-head ReLU scoring; the factored experiment remains explicit."""

    indexer_prefill_cublas: bool | None = None
    """Use cuBLAS for eligible prefill index scores."""

    indexer_prefill_tile_mb: int | None = None
    """Bound the prefill score tile and decode gather workspace in MiB."""

    indexer_decode_cublas_enabled: bool | None = None
    """Enable the shape-qualified decode scorer in addition to its public gate."""

    indexer_decode_min_keys: int | None = None
    """Minimum dynamic key bound for the decode cuBLAS route."""

    mla_splitk_swa: bool | None = None
    """Use split-K for sliding-window-only decode."""

    mla_splitk_c4: bool | None = None
    """Use split-K with compression ratio four."""

    mla_splitk_c128: bool | None = None
    """Use split-K with compression ratio 128."""

    mla_qk_dsplit: bool | None = None
    """Retain the independent split-QK numerical implementation."""

    aliases: ClassVar[dict[str, str]] = {
        "indexer_fused_logits": "VLLM_SM70_INDEXER_FUSED_LOGITS",
        "indexer_relu": "VLLM_SM70_INDEXER_RELU",
        "indexer_prefill_cublas": "VLLM_SM70_INDEXER_PREFILL_CUBLAS",
        "indexer_prefill_tile_mb": "VLLM_SM70_INDEXER_PREFILL_TILE_MB",
        "indexer_decode_cublas_enabled": "VLLM_SM70_INDEXER_DECODE_CUBLAS",
        "indexer_decode_min_keys": "VLLM_SM70_INDEXER_DECODE_CUBLAS_MIN_KEYS",
        "mla_splitk_swa": "VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_SWA",
        "mla_splitk_c4": "VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_C4",
        "mla_splitk_c128": "VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_C128",
        "mla_qk_dsplit": "VLLM_SM70_DSV4_SPARSE_MLA_QK_DSPLIT",
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

    def qualify(self, family: str | None):
        """Limit hashes and deferred validation to the model-declared family."""
        self.hash_fields = tuple(
            field
            for field in self.aliases
            if (family == "qsa" and field.startswith("qsa_"))
            or (family == "indexer" and not field.startswith("qsa_"))
        )

    def validate_active(self):
        if self.active:
            for field in (
                self.hash_fields if self.hash_fields is not None else self.aliases
            ):
                self.value(field)

    def compute_hash(self):
        if not self.active:
            return hash_factors({})
        factors: dict[str, object] = {"policy": super().compute_hash()}
        if self.hash_fields is None or any(
            not field.startswith("qsa_") for field in self.hash_fields
        ):
            factors.update(
                indexer_decode_cublas=self.indexer_decode_cublas,
                decode_bmm=self.decode_bmm,
                prefill_bmm=self.prefill_bmm,
            )
        return hash_factors(factors)


def sparse_policy(config=None) -> Sm70SparseConfig:
    from vllm.config.execution_policy import capture_execution_policy

    return capture_execution_policy(
        "kernel_config.sm70_sparse", Sm70SparseConfig, config
    )
