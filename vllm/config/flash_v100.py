# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 initialization declarations; execution consumes parsed choices.

Native entries preserve the append-only vector ABI in flash_v100_policy.h.
The compatibility raw inputs are provenance, never an execution-time getter.
"""

import os
from typing import ClassVar

from pydantic import Field

from vllm.config.execution_policy_base import ExecutionPolicy
from vllm.config.utils import config, hash_factors

NATIVE_FIELDS = (
    ("xqa_padded_smem", "VLLM_FLASH_V100_XQA_PADDED_SMEM", "first_ne0_on", None),
    ("xqa_g6_dual_cta", "VLLM_FLASH_V100_XQA_G6_DUAL_CTA", "first_eq1_off", None),
    ("e4m3_batch_xqa", "VLLM_FLASH_V100_E4M3_BATCH_XQA", "first_ne0_on", None),
    (
        "e4m3_batch_xqa_optimized",
        "VLLM_FLASH_V100_E4M3_BATCH_XQA_OPTIMIZED",
        "first_ne0_on",
        None,
    ),
    (
        "e4m3_page800_fastpath",
        "VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH",
        "first_ne0_on",
        None,
    ),
    (
        "e4m3_page800_fastpath_trace",
        "VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH_TRACE",
        "first_eq1_off",
        None,
    ),
    (
        "xqa_e5m2_g6_dual_cta",
        "VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA",
        "first_ne0_on",
        None,
    ),
    (
        "xqa_e5m2_g6_split_reduce",
        "VLLM_FLASH_V100_XQA_E5M2_G6_SPLIT_REDUCE",
        "first_ne0_on",
        None,
    ),
    (
        "xqa_e5m2_partition_page_ids",
        "VLLM_FLASH_V100_XQA_E5M2_PARTITION_PAGE_IDS",
        "first_ne0_on",
        None,
    ),
    ("xqa_e5m2_pair_load", "VLLM_FLASH_V100_XQA_E5M2_PAIR_LOAD", "first_ne0_on", None),
    (
        "xqa_e5m2_batch_wide_load",
        "VLLM_FLASH_V100_XQA_E5M2_BATCH_WIDE_LOAD",
        "first_ne0_on",
        None,
    ),
    (
        "dflash2_fixed_interleaved",
        "VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED",
        "first_ne0_on",
        None,
    ),
    (
        "dflash2_stage_page_ids",
        "VLLM_FLASH_V100_DFLASH2_STAGE_PAGE_IDS",
        "first_ne0_on",
        None,
    ),
    (
        "xqa_e5m2_p1024_begin",
        "VLLM_FLASH_V100_XQA_E5M2_P1024_BEGIN",
        "positive_atoi",
        61633,
    ),
    (
        "decode_fp8_xqa_min_seq_len",
        "VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN",
        "positive_atoi",
        16384,
    ),
    (
        "xqa_e5m2_g6_dual_cta_trace",
        "VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA_TRACE",
        "first_eq1_off",
        None,
    ),
    ("xqa_mtp5_dual_cta", "VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", "first_eq1_on", None),
    (
        "xqa_g6_dual_cta_dense",
        "VLLM_FLASH_V100_XQA_G6_DUAL_CTA_DENSE",
        "first_eq1_off",
        None,
    ),
    ("xqa_g6_p1024_auto", "VLLM_FLASH_V100_XQA_G6_P1024_AUTO", "first_ne0_on", None),
    (
        "xqa_g6_p1024_auto_trace",
        "VLLM_FLASH_V100_XQA_G6_P1024_AUTO_TRACE",
        "first_eq1_off",
        None,
    ),
    (
        "xqa_g6_p1024_sawtooth",
        "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH",
        "first_ne0_on",
        None,
    ),
    (
        "xqa_e4m3_g6_p64_p256_auto",
        "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO",
        "first_ne0_on",
        None,
    ),
    (
        "xqa_e4m3_g6_p64_p256_auto_trace",
        "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO_TRACE",
        "first_eq1_off",
        None,
    ),
    (
        "xqa_e4m3_g6_p256_begin",
        "VLLM_FLASH_V100_XQA_E4M3_G6_P256_BEGIN",
        "positive_atoi",
        12288,
    ),
    (
        "xqa_e4m3_g6_dual_cta_begin",
        "VLLM_FLASH_V100_XQA_E4M3_G6_DUAL_CTA_BEGIN",
        "positive_atoi",
        32768,
    ),
    (
        "xqa_e4m3_g6_wave_partitions",
        "VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS",
        "first_ne0_on",
        None,
    ),
    (
        "xqa_e4m3_g6_merged_wave_launch",
        "VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH",
        "first_ne0_on",
        None,
    ),
    (
        "xqa_e4m3_g6_p512_begin",
        "VLLM_FLASH_V100_XQA_E4M3_G6_P512_BEGIN",
        "positive_atoi",
        49152,
    ),
    (
        "xqa_e4m3_g6_p896_begin",
        "VLLM_FLASH_V100_XQA_E4M3_G6_P896_BEGIN",
        "positive_atoi",
        98304,
    ),
    (
        "xqa_e4m3_g6_p1664_begin",
        "VLLM_FLASH_V100_XQA_E4M3_G6_P1664_BEGIN",
        "positive_atoi",
        196608,
    ),
    ("decode_partition_size", "VLLM_FLASH_V100_DECODE_PARTITION_SIZE", "present", None),
    ("xqa_g6_qk_pipeline", "VLLM_FLASH_V100_XQA_G6_QK_PIPELINE", "first_ne0_on", None),
    (
        "xqa_g6_qk_pipeline_warps",
        "VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_WARPS",
        "warps",
        8,
    ),
    (
        "xqa_g6_qk_pipeline_trace",
        "VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_TRACE",
        "first_eq1_off",
        None,
    ),
    (
        "xqa_g6_p1024_sawtooth_trace",
        "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_TRACE",
        "first_eq1_off",
        None,
    ),
    (
        "xqa_g6_p1024_sawtooth_p1024_mid_seq_len",
        "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_MID_SEQ_LEN",
        "positive_atoi",
        111104,
    ),
    (
        "xqa_g6_p1024_sawtooth_p256_long_seq_len",
        "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P256_LONG_SEQ_LEN",
        "positive_atoi",
        147841,
    ),
    (
        "xqa_g6_p1024_sawtooth_p1024_final_seq_len",
        "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_FINAL_SEQ_LEN",
        "positive_atoi",
        258176,
    ),
    ("xqa_split_reduce", "VLLM_FLASH_V100_XQA_SPLIT_REDUCE", "first_eq1_off", None),
    (
        "xqa_batch_context_routing",
        "VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING",
        "first_ne0_on",
        None,
    ),
    (
        "xqa_batch_context_routing_trace",
        "VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING_TRACE",
        "first_eq1_off",
        None,
    ),
    ("xqa_block16_layout", "VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT", "layout", 0),
    (
        "xqa_block16_layout_require",
        "VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_REQUIRE",
        "first_eq1_off",
        None,
    ),
    (
        "xqa_block16_layout_trace",
        "VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_TRACE",
        "first_eq1_off",
        None,
    ),
    ("xqa_block784_index", "VLLM_FLASH_V100_XQA_BLOCK784_INDEX", "first_ne0_on", None),
    (
        "xqa_block784_index_trace",
        "VLLM_FLASH_V100_XQA_BLOCK784_INDEX_TRACE",
        "first_eq1_off",
        None,
    ),
    (
        "xqa_aligned_padded_smem",
        "VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM",
        "first_eq1_off",
        None,
    ),
    (
        "xqa_aligned_padded_smem_trace",
        "VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM_TRACE",
        "first_eq1_off",
        None,
    ),
    ("xqa_split_reduce_d_tile", "VLLM_FLASH_V100_XQA_SPLIT_REDUCE_D_TILE", "tile", 8),
    ("dense_d256_wmma_qk", "VLLM_FLASH_V100_DENSE_D256_WMMA_QK", "exact_ne0_on", None),
    (
        "dense_d256_low_smem",
        "VLLM_FLASH_V100_DENSE_D256_LOW_SMEM",
        "exact_ne0_off",
        None,
    ),
    (
        "prefill_d256_bm32_all_p",
        "VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P",
        "exact_ne0_on",
        None,
    ),
    (
        "prefill_d256_bm32_pair_scratch",
        "VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH",
        "exact_ne0_on",
        None,
    ),
    (
        "prefill_d256_low_smem",
        "VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM",
        "exact_ne0_on",
        None,
    ),
    (
        "prefill_d256_bm32_phase",
        "VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE",
        "exact_ne0_on",
        None,
    ),
    (
        "prefill_contig_fast",
        "VLLM_FLASH_V100_PREFILL_CONTIG_FAST",
        "exact_ne0_off",
        None,
    ),
    (
        "prefill_d256_scalar_qk",
        "VLLM_FLASH_V100_PREFILL_D256_SCALAR_QK",
        "exact_ne0_off",
        None,
    ),
    ("prefill_d256_bm32", "VLLM_FLASH_V100_PREFILL_D256_BM32", "exact_ne0_off", None),
    (
        "prefill_d256_output_stride_268",
        "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268",
        "exact_ne0_on",
        None,
    ),
    (
        "prefill_d256_output_stride_268",
        "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268",
        "exact_ne0_off",
        None,
    ),
    (
        "prefill_d256_software_pipeline",
        "VLLM_FLASH_V100_PREFILL_D256_SOFTWARE_PIPELINE",
        "exact_ne0_off",
        None,
    ),
    (
        "prefill_d256_sw_pipeline_qk",
        "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK",
        "exact_ne0_on",
        None,
    ),
    (
        "prefill_d256_sw_pipeline_pv",
        "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV",
        "exact_ne0_on",
        None,
    ),
    ("prefill_scalar_pv", "VLLM_FLASH_V100_PREFILL_SCALAR_PV", "nonempty_ne0", None),
    ("e4m3_scalar_fast", "VLLM_FLASH_V100_E4M3_SCALAR_FAST", "scalar_alias", None),
    ("tp2_e4m3_scalar_fast", "VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST", "unused", None),
)


def native_value(rule, raw, default=None):
    import ctypes

    import regex as re

    if rule == "unused":
        return 0
    if rule == "present":
        return raw is not None
    if rule in ("scalar_alias", "first_eq1_on"):
        return raw is None or (raw == "1" if rule == "scalar_alias" else raw[:1] == "1")
    if rule == "first_ne0_on":
        return raw is None or raw[:1] != "0"
    if rule == "first_eq1_off":
        return raw is not None and raw[:1] == "1"
    if rule == "exact_ne0_on":
        return raw is None or raw != "0"
    if rule == "exact_ne0_off":
        return raw is not None and raw != "0"
    if rule == "nonempty_ne0":
        return bool(raw) and raw[:1] != "0"
    match = re.match(r"[ \t\n\r\v\f]*([+-]?[0-9]+)", raw or "")
    # Match the worker libc's atoi: ASCII whitespace, strtol saturation followed
    # by an int conversion. Python int accepts a broader whitespace/size range.
    limit = 1 << (8 * ctypes.sizeof(ctypes.c_long) - 1)
    digits = match.group(1) if match else "0"
    negative = digits.startswith("-")
    digits = digits.lstrip("+-").lstrip("0") or "0"
    parsed = int(digits) if len(digits) <= 19 else limit
    parsed = min(limit - 1, max(-limit, -parsed if negative else parsed))
    value = ctypes.c_int(parsed).value
    if rule == "optional_atoi":
        return None if raw is None else value
    if rule == "score_block":
        if raw is None:
            return 0  # Use the translation unit's qualified block size.
        valid = re.fullmatch(r"[ \t\n\r\v\f]*[+-]?[0-9]+", raw)
        value = parsed if valid else -1
        return value if 8192 <= value <= 131072 and value % 8192 == 0 else -1
    if rule == "positive_atoi":
        return default if raw is None else max(1, value)
    if rule == "warps":
        return 6 if value == 6 else 8
    if rule == "layout":
        return value if value in (1, 2) else 0
    if rule == "tile":
        return value if value in (8, 16, 32) else 8
    raise ValueError(f"Unknown native Flash-V100 parser: {rule}")


@config
class CapturedFlashOptions:
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Per-field source, transferred with the worker configuration."""
    legacy_inputs: dict[str, str | None] = Field(default_factory=dict, init=False)
    """Original input for provider-specific parser projections and explanation."""
    errors: dict[str, str] = Field(default_factory=dict, init=False)
    """Invalid qualified values raise only at their retained consumer checkpoint."""
    error_causes: dict[str, str] = Field(default_factory=dict, init=False)
    """Preserve the chained parser error without serializing exception objects."""
    bindings: ClassVar[dict[str, tuple[str, str, object]]] = {}
    legacy_aliases: ClassVar[dict[str, tuple[str, ...]]] = {}

    def resolve(self):
        from vllm import envs
        from vllm.envs_metadata import EnvVar

        for aliases in self.legacy_aliases.values():
            for name in aliases:
                if name not in self.legacy_inputs:
                    self.legacy_inputs[name] = os.getenv(name)
                    variable = envs.environment_variables.get(name)
                    if isinstance(variable, EnvVar):
                        variable.warn_if_deprecated()
        for field, (name, parser, default) in self.bindings.items():
            if field in self.sources:
                continue
            raw = os.getenv(name)
            self.legacy_inputs[field] = raw
            variable = envs.environment_variables.get(name)
            if isinstance(variable, EnvVar):
                variable.warn_if_deprecated()
            if getattr(self, field) is not None:
                self.sources[field] = "typed"
                continue
            self.sources[field] = name if raw is not None else "default"
            value = raw if raw is not None else default
            try:
                if parser == "registered":
                    value = envs.environment_variables[name]()
                elif parser == "eq1":
                    value = value == "1"
                elif parser == "ne0":
                    value = value != "0"
                elif parser.startswith("native:"):
                    if parser == "native:scalar_alias" and raw is None:
                        raw = self.legacy_inputs[self.legacy_aliases[field][0]]
                    value = native_value(parser.removeprefix("native:"), raw, default)
                elif parser in ("int", "threshold", "partition"):
                    assert isinstance(value, str)
                    try:
                        value = int(value)
                    except ValueError as exc:
                        if parser == "threshold":
                            raise ValueError(
                                f"{name} must be an integer, got {value!r}"
                            ) from exc
                        if parser == "partition":
                            raise ValueError(
                                f"{name} must be one of (256, 512, 1024), got {value!r}"
                            ) from exc
                        raise
                    if parser == "threshold":
                        value = max(1, value)
                    if parser == "partition" and value not in (256, 512, 1024):
                        raise ValueError(
                            f"{name} must be one of (256, 512, 1024), got {value}"
                        )
                setattr(self, field, value)
            except ValueError as exc:
                self.errors[field] = str(exc)
                if exc.__cause__ is not None:
                    self.error_causes[field] = str(exc.__cause__)

    def value(self, field):
        if field in self.errors:
            if field in self.error_causes:
                raise ValueError(self.errors[field]) from ValueError(
                    self.error_causes[field]
                )
            raise ValueError(self.errors[field])
        return getattr(self, field)

    def native_input(self, field):
        if self.sources[field] == "typed":
            value = getattr(self, field)
            return str(int(value)) if isinstance(value, bool) else str(value)
        return self.legacy_inputs[field]

    def compile_ignored_aliases(self):
        return {
            name
            for field, (name, _, _) in self.bindings.items()
            if field in self.sources
        } | {
            name
            for aliases in self.legacy_aliases.values()
            for name in aliases
            if name in self.legacy_inputs
        }


@config
class FlashV100Options(CapturedFlashOptions):
    legacy_aliases: ClassVar[dict[str, tuple[str, ...]]] = {
        "e4m3_scalar_fast": ("VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST",),
    }

    tail_cudagraphs: bool | None = None
    """Scalar-tail attention also serves ordinary E4M3 decode, beyond DFlash."""
    scalar_tail_manifest: str | None = None
    """Optional experimental scalar-tail provider selected at loading."""

    def resolve(self):
        super().resolve()
        if self.bfla_pool is not None:
            self.bfla_pool = self.bfla_pool.lower()

    native_inputs: tuple[str | None, ...] = Field(default=(), init=False)
    """Versioned native binding input, prepared once in each worker."""
    native_effective: tuple[int, ...] = Field(default=(), init=False)
    """Parsed native calculation results for hashing and explanation."""
    python_policy: dict[str, bool | int | str | None] = Field(
        default_factory=dict, init=False
    )
    """Independent package's parsed Python policy, projected from its owners."""
    prefill_native_effective: tuple[int, ...] = Field(default=(), init=False)
    """FA2 prefill ABI 1: immutable policy, with diagnostic projection."""
    cache_dtype: str = Field(default="auto", init=False)
    """Qualified storage format, used to exclude unused native format policy."""

    def finalize(self, graph, diagnostics, cache_dtype="auto"):
        if self.native_inputs:
            return
        self.resolve()
        diagnostics.resolve()
        graph.resolve()
        self.cache_dtype = cache_dtype
        assert self.prefill_score_block_tokens is not None
        assert self.prefill_serial_tail is not None
        assert self.prefill_exact_tail is not None
        assert self.prefill_direct_tail is not None
        self.prefill_native_effective = (
            int(self.prefill_qk_algorithm is not None),
            self.prefill_qk_algorithm or 0,
            self.prefill_score_block_tokens,
            int(self.prefill_serial_tail),
            int(self.prefill_exact_tail),
            int(diagnostics.prefill_dump_tail),
            int(self.prefill_direct_tail),
        )
        inputs = {}
        scalar_alias = self.legacy_aliases["e4m3_scalar_fast"][0]
        inputs[scalar_alias] = self.legacy_inputs[scalar_alias]
        for owner in (self, diagnostics):
            for field, (name, _, _) in owner.bindings.items():
                inputs[name] = owner.native_input(field)
        for field, name in graph.aliases.items():
            source = graph.sources[field]
            if source in (name, "default"):
                inputs[name] = graph.legacy_inputs.get(field)
            else:
                value = getattr(graph, field)
                inputs[name] = (
                    None
                    if value is None
                    else str(int(value))
                    if isinstance(value, bool)
                    else str(value)
                )
        self.native_inputs = tuple(inputs[name] for _, name, _, _ in NATIVE_FIELDS)
        values = []
        for _, name, rule, default in NATIVE_FIELDS:
            raw = inputs[name]
            if rule == "scalar_alias" and raw is None:
                raw = inputs["VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST"]
            values.append(int(native_value(rule, raw, default)))
        self.native_effective = tuple(values)
        partition = graph.decode_partition
        error, cause = graph.decode_partition_error or (None, None)

        def raw_or(name, default):
            raw = inputs.get("VLLM_FLASH_V100_" + name)
            return default if raw is None else raw

        self.python_policy = {
            "dynamic_partitions": self.decode_dynamic_partitions,
            "staged_pv": self.xqa_staged_pv,
            "share_workspace": self.share_decode_workspace,
            "partition_size": partition,
            "partition_error": error,
            "partition_cause": cause,
            "scalar_fast": raw_or(
                "E4M3_SCALAR_FAST", raw_or("TP2_E4M3_SCALAR_FAST", "1")
            )
            == "1",
            "batch_xqa": raw_or("E4M3_BATCH_XQA", "1") == "1",
            "padded_smem": raw_or("XQA_PADDED_SMEM", "1") != "0",
            "dual_cta": raw_or("XQA_G6_DUAL_CTA", "0") == "1",
        }

    def hash_values(self):
        fields = {}
        for field in self.bindings:
            if field in (
                "prefill_qk_algorithm",
                "prefill_score_block_tokens",
                "prefill_serial_tail",
                "prefill_exact_tail",
                "prefill_direct_tail",
            ) and (
                not self.fa2_d256_prefill
                or not self.prefill_d256_gqa_arch_128k_experimental
                or self.prefill_d256_gqa_v37
            ):
                continue
            if field == "share_decode_workspace":
                continue
            if field.startswith("bfla_") and not self.bfla_prefill:
                continue
            if field.startswith("prefill_split_kv_") and not self.prefill_split_kv:
                continue
            if field.startswith("xqa_e5m2_") and self.cache_dtype != "fp8_e5m2":
                continue
            if (
                field.startswith("xqa_e4m3_") or field.startswith("e4m3_")
            ) and self.cache_dtype not in ("fp8", "fp8_e4m3"):
                continue
            fields[field] = getattr(self, field)
        # Native parser projections can intentionally differ from the Python
        # predicate for malformed/noncanonical legacy strings.
        fields["native"] = tuple(
            (field, value)
            for (field, _, _, _), value in zip(NATIVE_FIELDS, self.native_effective)
            if "trace" not in field
            and field != "tp2_e4m3_scalar_fast"
            and ("e5m2" not in field or self.cache_dtype == "fp8_e5m2")
            and ("e4m3" not in field or self.cache_dtype in ("fp8", "fp8_e4m3"))
        )
        fields["package"] = {
            field: value
            for field, value in self.python_policy.items()
            if field != "share_workspace"
            and (
                field not in ("scalar_fast", "batch_xqa")
                or self.cache_dtype in ("fp8", "fp8_e4m3")
            )
        }
        fields["errors"] = {
            field: error for field, error in self.errors.items() if field in fields
        }
        return fields

    prefill_qk_algorithm: int | None = None
    """Optional 79T cuBLAS algorithm; legacy atoi parsing and enum preserved."""
    prefill_score_block_tokens: int | None = None
    """79T workspace block size; 0 uses the build default, -1 defers invalid env."""
    prefill_serial_tail: bool | None = None
    """Keep the serial-tail default and the exact legacy string comparison."""
    prefill_exact_tail: bool | None = None
    """Return-affecting exact-tail experiment; legacy presence enables it."""
    prefill_direct_tail: bool | None = None
    """Return-affecting direct-tail experiment; legacy presence enables it."""

    bfla_threshold: float | None = None
    """Retained VLLM_FLASH_V100_BFLA_THRESHOLD input."""
    bfla_keep_mass: float | None = None
    """Retained VLLM_FLASH_V100_BFLA_KEEP_MASS input."""
    bfla_min_keep_blocks: int | None = None
    """Retained VLLM_FLASH_V100_BFLA_MIN_KEEP_BLOCKS input."""
    bfla_spec_stride: int | None = None
    """Retained VLLM_FLASH_V100_BFLA_SPEC_STRIDE input."""
    bfla_spec_prob: float | None = None
    """Retained VLLM_FLASH_V100_BFLA_SPEC_PROB input."""
    bfla_pool: str | None = None
    """Retained VLLM_FLASH_V100_BFLA_POOL input."""
    bfla_local_blocks: int | None = None
    """Retained VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS input."""
    bfla_spec_seed: int | None = None
    """Retained VLLM_FLASH_V100_BFLA_SPEC_SEED input."""
    fa2_d256_prefill: bool | None = None
    """Retained VLLM_FLASH_V100_FA2_D256_PREFILL input."""
    decode_xqa_q4_min_seq_len: int | None = None
    """Retained VLLM_FLASH_V100_DECODE_XQA_Q4_MIN_SEQ_LEN input."""
    decode_fp8_xqa_min_seq_len: int | None = None
    """Retained VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN input."""
    decode_dynamic_partitions: bool | None = None
    """Retained VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS input."""
    xqa_g6_p1024_sawtooth: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH input."""
    smallq_decode_xqa_min_seq_len: int | None = None
    """Retained VLLM_FLASH_V100_SMALLQ_DECODE_XQA_MIN_SEQ_LEN input."""
    prefill_d256_gqa_v37: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_GQA_V37 input."""
    prefill_dense_splitkv3: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3 input."""
    prefill_d256_gqa_arch_128k_experimental: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL input."""
    prefill_dense_splitkv3_min_kv: int | None = None
    """Retained VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3_MIN_KV input."""
    prefill_dense_splitkv3_q8000_experimental: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3_Q8000_EXPERIMENTAL input."""
    kernel_block_size16: bool | None = None
    """Retained VLLM_FLASH_V100_KERNEL_BLOCK_SIZE16 input."""
    enable_paged_prefill: bool | None = None
    """Retained VLLM_FLASH_V100_ENABLE_PAGED_PREFILL input."""
    prefill_contig_dense_min_q: int | None = None
    """Retained VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_Q input."""
    prefill_contig_dense_min_kv: int | None = None
    """Retained VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_KV input."""
    prefill_contig_dense_allow_copy: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_ALLOW_COPY input."""
    prefill_gather_dense_min_q: int | None = None
    """Retained VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_Q input."""
    prefill_gather_dense_min_kv: int | None = None
    """Retained VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_KV input."""
    prefill_split_kv_tokens: int | None = None
    """Retained VLLM_FLASH_V100_PREFILL_SPLIT_KV_TOKENS input."""
    prefill_split_kv_min_q: int | None = None
    """Retained VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_Q input."""
    prefill_split_kv_max_q: int | None = None
    """Retained VLLM_FLASH_V100_PREFILL_SPLIT_KV_MAX_Q input."""
    prefill_split_kv_min_kv: int | None = None
    """Retained VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_KV input."""
    bfla_min_q: int | None = None
    """Retained VLLM_FLASH_V100_BFLA_MIN_Q input."""
    bfla_min_kv: int | None = None
    """Retained VLLM_FLASH_V100_BFLA_MIN_KV input."""
    bfla_mask_block_n: int | None = None
    """Retained VLLM_FLASH_V100_BFLA_MASK_BLOCK_N input."""
    decode_use_paged_prefill: bool | None = None
    """Retained VLLM_FLASH_V100_DECODE_USE_PAGED_PREFILL input."""
    decode_use_bhmd_out: bool | None = None
    """Retained VLLM_FLASH_V100_DECODE_USE_BHMD_OUT input."""
    decode_use_scalar_paged: bool | None = None
    """Retained VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED input."""
    e4m3_grouped_fp32: bool | None = None
    """Retained VLLM_FLASH_V100_E4M3_GROUPED_FP32 input."""
    disable_paged_prefill: bool | None = None
    """Retained VLLM_FLASH_V100_DISABLE_PAGED_PREFILL input."""
    prefill_split_kv: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_SPLIT_KV input."""
    bfla_prefill: bool | None = None
    """Retained VLLM_FLASH_V100_BFLA_PREFILL input."""
    prefill_contig_dense: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_CONTIG_DENSE input."""
    prefill_gather_dense: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_GATHER_DENSE input."""
    prefill_use_paged_cache: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_USE_PAGED_CACHE input."""
    prefill_use_triton: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_USE_TRITON input."""
    allow_triton_fallback: bool | None = None
    """Retained VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK input."""
    smallq_decode_max_model_len: int | None = None
    """Retained VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN input."""
    decode_dense_reference: bool | None = None
    """Retained VLLM_FLASH_V100_DECODE_DENSE_REFERENCE input."""
    decode_dense_cache: bool | None = None
    """Retained VLLM_FLASH_V100_DECODE_DENSE_CACHE input."""
    decode_use_wmma_wrapper: bool | None = None
    """Retained VLLM_FLASH_V100_DECODE_USE_WMMA_WRAPPER input."""
    decode_use_xqa: bool | None = None
    """Retained VLLM_FLASH_V100_DECODE_USE_XQA input."""
    fp8_prefill_bridge: bool | None = None
    """Retained VLLM_FLASH_V100_FP8_PREFILL_BRIDGE input."""
    smallq_decode_use_xqa: bool | None = None
    """Retained VLLM_FLASH_V100_SMALLQ_DECODE_USE_XQA input."""
    prefill_prefix_decode_rows: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_PREFIX_DECODE_ROWS input."""
    xqa_mtp5_partition_size: int | None = None
    """Retained VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE input."""
    xqa_mtp5_dual_cta: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA input."""
    dflash2_batched_grouped_verify: bool | None = None
    """Retained VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY input."""
    xqa_padded_smem: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_PADDED_SMEM input."""
    xqa_g6_dual_cta: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_DUAL_CTA input."""
    e4m3_batch_xqa_optimized: bool | None = None
    """Retained VLLM_FLASH_V100_E4M3_BATCH_XQA_OPTIMIZED input."""
    e4m3_page800_fastpath: bool | None = None
    """Retained VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH input."""
    xqa_e5m2_g6_dual_cta: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA input."""
    xqa_e5m2_g6_split_reduce: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_E5M2_G6_SPLIT_REDUCE input."""
    xqa_e5m2_partition_page_ids: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_E5M2_PARTITION_PAGE_IDS input."""
    xqa_e5m2_pair_load: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_E5M2_PAIR_LOAD input."""
    xqa_e5m2_batch_wide_load: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_E5M2_BATCH_WIDE_LOAD input."""
    dflash2_fixed_interleaved: bool | None = None
    """Retained VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED input."""
    dflash2_stage_page_ids: bool | None = None
    """Retained VLLM_FLASH_V100_DFLASH2_STAGE_PAGE_IDS input."""
    xqa_e5m2_p1024_begin: int | None = None
    """Retained VLLM_FLASH_V100_XQA_E5M2_P1024_BEGIN input."""
    xqa_g6_dual_cta_dense: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_DUAL_CTA_DENSE input."""
    xqa_g6_p1024_auto: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_P1024_AUTO input."""
    xqa_e4m3_g6_p256_begin: int | None = None
    """Retained VLLM_FLASH_V100_XQA_E4M3_G6_P256_BEGIN input."""
    xqa_e4m3_g6_dual_cta_begin: int | None = None
    """Retained VLLM_FLASH_V100_XQA_E4M3_G6_DUAL_CTA_BEGIN input."""
    xqa_e4m3_g6_merged_wave_launch: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH input."""
    xqa_e4m3_g6_p896_begin: int | None = None
    """Retained VLLM_FLASH_V100_XQA_E4M3_G6_P896_BEGIN input."""
    xqa_e4m3_g6_p1664_begin: int | None = None
    """Retained VLLM_FLASH_V100_XQA_E4M3_G6_P1664_BEGIN input."""
    xqa_g6_qk_pipeline: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_QK_PIPELINE input."""
    xqa_g6_qk_pipeline_warps: int | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_WARPS input."""
    xqa_g6_p1024_sawtooth_p1024_mid_seq_len: int | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_MID_SEQ_LEN input."""
    xqa_g6_p1024_sawtooth_p256_long_seq_len: int | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P256_LONG_SEQ_LEN input."""
    xqa_g6_p1024_sawtooth_p1024_final_seq_len: int | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_FINAL_SEQ_LEN input."""
    xqa_split_reduce: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_SPLIT_REDUCE input."""
    xqa_block16_layout: int | None = None
    """Retained VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT input."""
    xqa_block16_layout_require: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_REQUIRE input."""
    xqa_block784_index: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_BLOCK784_INDEX input."""
    xqa_aligned_padded_smem: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM input."""
    xqa_split_reduce_d_tile: int | None = None
    """Retained VLLM_FLASH_V100_XQA_SPLIT_REDUCE_D_TILE input."""
    dense_d256_wmma_qk: bool | None = None
    """Retained VLLM_FLASH_V100_DENSE_D256_WMMA_QK input."""
    dense_d256_low_smem: bool | None = None
    """Retained VLLM_FLASH_V100_DENSE_D256_LOW_SMEM input."""
    prefill_d256_bm32_all_p: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P input."""
    prefill_d256_bm32_pair_scratch: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH input."""
    prefill_d256_low_smem: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM input."""
    prefill_d256_bm32_phase: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE input."""
    prefill_contig_fast: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_CONTIG_FAST input."""
    prefill_d256_scalar_qk: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_SCALAR_QK input."""
    prefill_d256_bm32: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_BM32 input."""
    prefill_d256_output_stride_268: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268 input."""
    prefill_d256_software_pipeline: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_SOFTWARE_PIPELINE input."""
    prefill_d256_sw_pipeline_qk: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK input."""
    prefill_d256_sw_pipeline_pv: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV input."""
    prefill_scalar_pv: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_SCALAR_PV input."""
    e4m3_scalar_fast: bool | None = None
    """Retained VLLM_FLASH_V100_E4M3_SCALAR_FAST input."""
    share_decode_workspace: bool | None = None
    """Retained VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE input."""
    xqa_staged_pv: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_STAGED_PV input."""

    bindings: ClassVar[dict[str, tuple[str, str, object]]] = {
        "prefill_qk_algorithm": (
            "PREFIX_QK_CUBLAS_ALGO_RUNTIME",
            "native:optional_atoi",
            None,
        ),
        "prefill_score_block_tokens": (
            "VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS",
            "native:score_block",
            None,
        ),
        "prefill_serial_tail": (
            "PREFIX_TORCH_SERIAL_TAIL",
            "native:exact_ne0_on",
            None,
        ),
        "prefill_exact_tail": ("PREFIX_TORCH_EXACT_TAIL", "native:present", None),
        "prefill_direct_tail": ("PREFIX_TORCH_DIRECT_TAIL", "native:present", None),
        "tail_cudagraphs": ("VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS", "registered", None),
        "scalar_tail_manifest": (
            "VLLM_SM70_DFLASH2_SCALAR_ATTENTION_MANIFEST",
            "registered",
            None,
        ),
        "bfla_threshold": ("VLLM_FLASH_V100_BFLA_THRESHOLD", "registered", None),
        "bfla_keep_mass": ("VLLM_FLASH_V100_BFLA_KEEP_MASS", "registered", None),
        "bfla_min_keep_blocks": (
            "VLLM_FLASH_V100_BFLA_MIN_KEEP_BLOCKS",
            "registered",
            None,
        ),
        "bfla_spec_stride": ("VLLM_FLASH_V100_BFLA_SPEC_STRIDE", "registered", None),
        "bfla_spec_prob": ("VLLM_FLASH_V100_BFLA_SPEC_PROB", "registered", None),
        "bfla_pool": ("VLLM_FLASH_V100_BFLA_POOL", "registered", None),
        "bfla_local_blocks": ("VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS", "registered", None),
        "bfla_spec_seed": ("VLLM_FLASH_V100_BFLA_SPEC_SEED", "registered", None),
        "fa2_d256_prefill": ("VLLM_FLASH_V100_FA2_D256_PREFILL", "registered", None),
        "decode_xqa_q4_min_seq_len": (
            "VLLM_FLASH_V100_DECODE_XQA_Q4_MIN_SEQ_LEN",
            "threshold",
            "32768",
        ),
        "decode_fp8_xqa_min_seq_len": (
            "VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN",
            "threshold",
            "16384",
        ),
        "decode_dynamic_partitions": (
            "VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS",
            "ne0",
            "1",
        ),
        "xqa_g6_p1024_sawtooth": ("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", "ne0", "1"),
        "smallq_decode_xqa_min_seq_len": (
            "VLLM_FLASH_V100_SMALLQ_DECODE_XQA_MIN_SEQ_LEN",
            "int",
            "4096",
        ),
        "prefill_d256_gqa_v37": (
            "VLLM_FLASH_V100_PREFILL_D256_GQA_V37",
            "registered",
            None,
        ),
        "prefill_dense_splitkv3": (
            "VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3",
            "registered",
            None,
        ),
        "prefill_d256_gqa_arch_128k_experimental": (
            "VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL",
            "registered",
            None,
        ),
        "prefill_dense_splitkv3_min_kv": (
            "VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3_MIN_KV",
            "registered",
            None,
        ),
        "prefill_dense_splitkv3_q8000_experimental": (
            "VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3_Q8000_EXPERIMENTAL",
            "registered",
            None,
        ),
        "kernel_block_size16": (
            "VLLM_FLASH_V100_KERNEL_BLOCK_SIZE16",
            "registered",
            None,
        ),
        "enable_paged_prefill": ("VLLM_FLASH_V100_ENABLE_PAGED_PREFILL", "ne0", None),
        "prefill_contig_dense_min_q": (
            "VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_Q",
            "registered",
            None,
        ),
        "prefill_contig_dense_min_kv": (
            "VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_KV",
            "registered",
            None,
        ),
        "prefill_contig_dense_allow_copy": (
            "VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_ALLOW_COPY",
            "registered",
            None,
        ),
        "prefill_gather_dense_min_q": (
            "VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_Q",
            "registered",
            None,
        ),
        "prefill_gather_dense_min_kv": (
            "VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_KV",
            "registered",
            None,
        ),
        "prefill_split_kv_tokens": (
            "VLLM_FLASH_V100_PREFILL_SPLIT_KV_TOKENS",
            "registered",
            None,
        ),
        "prefill_split_kv_min_q": (
            "VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_Q",
            "registered",
            None,
        ),
        "prefill_split_kv_max_q": (
            "VLLM_FLASH_V100_PREFILL_SPLIT_KV_MAX_Q",
            "registered",
            None,
        ),
        "prefill_split_kv_min_kv": (
            "VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_KV",
            "registered",
            None,
        ),
        "bfla_min_q": ("VLLM_FLASH_V100_BFLA_MIN_Q", "registered", None),
        "bfla_min_kv": ("VLLM_FLASH_V100_BFLA_MIN_KV", "registered", None),
        "bfla_mask_block_n": ("VLLM_FLASH_V100_BFLA_MASK_BLOCK_N", "registered", None),
        "decode_use_paged_prefill": (
            "VLLM_FLASH_V100_DECODE_USE_PAGED_PREFILL",
            "eq1",
            None,
        ),
        "decode_use_bhmd_out": ("VLLM_FLASH_V100_DECODE_USE_BHMD_OUT", "ne0", None),
        "decode_use_scalar_paged": (
            "VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED",
            "ne0",
            None,
        ),
        "e4m3_grouped_fp32": ("VLLM_FLASH_V100_E4M3_GROUPED_FP32", "registered", None),
        "disable_paged_prefill": ("VLLM_FLASH_V100_DISABLE_PAGED_PREFILL", "eq1", "0"),
        "prefill_split_kv": ("VLLM_FLASH_V100_PREFILL_SPLIT_KV", "registered", None),
        "bfla_prefill": ("VLLM_FLASH_V100_BFLA_PREFILL", "registered", None),
        "prefill_contig_dense": (
            "VLLM_FLASH_V100_PREFILL_CONTIG_DENSE",
            "registered",
            None,
        ),
        "prefill_gather_dense": (
            "VLLM_FLASH_V100_PREFILL_GATHER_DENSE",
            "registered",
            None,
        ),
        "prefill_use_paged_cache": (
            "VLLM_FLASH_V100_PREFILL_USE_PAGED_CACHE",
            "eq1",
            "0",
        ),
        "prefill_use_triton": ("VLLM_FLASH_V100_PREFILL_USE_TRITON", "ne0", "0"),
        "allow_triton_fallback": ("VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK", "eq1", "0"),
        "smallq_decode_max_model_len": (
            "VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN",
            "int",
            "0",
        ),
        "decode_dense_reference": (
            "VLLM_FLASH_V100_DECODE_DENSE_REFERENCE",
            "eq1",
            "0",
        ),
        "decode_dense_cache": ("VLLM_FLASH_V100_DECODE_DENSE_CACHE", "eq1", "0"),
        "decode_use_wmma_wrapper": (
            "VLLM_FLASH_V100_DECODE_USE_WMMA_WRAPPER",
            "eq1",
            "0",
        ),
        "decode_use_xqa": ("VLLM_FLASH_V100_DECODE_USE_XQA", "eq1", "1"),
        "fp8_prefill_bridge": ("VLLM_FLASH_V100_FP8_PREFILL_BRIDGE", "ne0", "1"),
        "smallq_decode_use_xqa": ("VLLM_FLASH_V100_SMALLQ_DECODE_USE_XQA", "eq1", "1"),
        "prefill_prefix_decode_rows": (
            "VLLM_FLASH_V100_PREFILL_PREFIX_DECODE_ROWS",
            "registered",
            None,
        ),
        "xqa_mtp5_partition_size": (
            "VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE",
            "partition",
            "1024",
        ),
        "xqa_mtp5_dual_cta": ("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", "eq1", "1"),
        "dflash2_batched_grouped_verify": (
            "VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY",
            "registered",
            None,
        ),
        "xqa_padded_smem": (
            "VLLM_FLASH_V100_XQA_PADDED_SMEM",
            "native:first_ne0_on",
            None,
        ),
        "xqa_g6_dual_cta": (
            "VLLM_FLASH_V100_XQA_G6_DUAL_CTA",
            "native:first_eq1_off",
            None,
        ),
        "e4m3_batch_xqa_optimized": (
            "VLLM_FLASH_V100_E4M3_BATCH_XQA_OPTIMIZED",
            "native:first_ne0_on",
            None,
        ),
        "e4m3_page800_fastpath": (
            "VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH",
            "native:first_ne0_on",
            None,
        ),
        "xqa_e5m2_g6_dual_cta": (
            "VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA",
            "native:first_ne0_on",
            None,
        ),
        "xqa_e5m2_g6_split_reduce": (
            "VLLM_FLASH_V100_XQA_E5M2_G6_SPLIT_REDUCE",
            "native:first_ne0_on",
            None,
        ),
        "xqa_e5m2_partition_page_ids": (
            "VLLM_FLASH_V100_XQA_E5M2_PARTITION_PAGE_IDS",
            "native:first_ne0_on",
            None,
        ),
        "xqa_e5m2_pair_load": (
            "VLLM_FLASH_V100_XQA_E5M2_PAIR_LOAD",
            "native:first_ne0_on",
            None,
        ),
        "xqa_e5m2_batch_wide_load": (
            "VLLM_FLASH_V100_XQA_E5M2_BATCH_WIDE_LOAD",
            "native:first_ne0_on",
            None,
        ),
        "dflash2_fixed_interleaved": (
            "VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED",
            "native:first_ne0_on",
            None,
        ),
        "dflash2_stage_page_ids": (
            "VLLM_FLASH_V100_DFLASH2_STAGE_PAGE_IDS",
            "native:first_ne0_on",
            None,
        ),
        "xqa_e5m2_p1024_begin": (
            "VLLM_FLASH_V100_XQA_E5M2_P1024_BEGIN",
            "native:positive_atoi",
            61633,
        ),
        "xqa_g6_dual_cta_dense": (
            "VLLM_FLASH_V100_XQA_G6_DUAL_CTA_DENSE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_g6_p1024_auto": (
            "VLLM_FLASH_V100_XQA_G6_P1024_AUTO",
            "native:first_ne0_on",
            None,
        ),
        "xqa_e4m3_g6_p256_begin": (
            "VLLM_FLASH_V100_XQA_E4M3_G6_P256_BEGIN",
            "native:positive_atoi",
            12288,
        ),
        "xqa_e4m3_g6_dual_cta_begin": (
            "VLLM_FLASH_V100_XQA_E4M3_G6_DUAL_CTA_BEGIN",
            "native:positive_atoi",
            32768,
        ),
        "xqa_e4m3_g6_merged_wave_launch": (
            "VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH",
            "native:first_ne0_on",
            None,
        ),
        "xqa_e4m3_g6_p896_begin": (
            "VLLM_FLASH_V100_XQA_E4M3_G6_P896_BEGIN",
            "native:positive_atoi",
            98304,
        ),
        "xqa_e4m3_g6_p1664_begin": (
            "VLLM_FLASH_V100_XQA_E4M3_G6_P1664_BEGIN",
            "native:positive_atoi",
            196608,
        ),
        "xqa_g6_qk_pipeline": (
            "VLLM_FLASH_V100_XQA_G6_QK_PIPELINE",
            "native:first_ne0_on",
            None,
        ),
        "xqa_g6_qk_pipeline_warps": (
            "VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_WARPS",
            "native:warps",
            8,
        ),
        "xqa_g6_p1024_sawtooth_p1024_mid_seq_len": (
            "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_MID_SEQ_LEN",
            "native:positive_atoi",
            111104,
        ),
        "xqa_g6_p1024_sawtooth_p256_long_seq_len": (
            "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P256_LONG_SEQ_LEN",
            "native:positive_atoi",
            147841,
        ),
        "xqa_g6_p1024_sawtooth_p1024_final_seq_len": (
            "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_FINAL_SEQ_LEN",
            "native:positive_atoi",
            258176,
        ),
        "xqa_split_reduce": (
            "VLLM_FLASH_V100_XQA_SPLIT_REDUCE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_block16_layout": (
            "VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT",
            "native:layout",
            0,
        ),
        "xqa_block16_layout_require": (
            "VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_REQUIRE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_block784_index": (
            "VLLM_FLASH_V100_XQA_BLOCK784_INDEX",
            "native:first_ne0_on",
            None,
        ),
        "xqa_aligned_padded_smem": (
            "VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM",
            "native:first_eq1_off",
            None,
        ),
        "xqa_split_reduce_d_tile": (
            "VLLM_FLASH_V100_XQA_SPLIT_REDUCE_D_TILE",
            "native:tile",
            8,
        ),
        "dense_d256_wmma_qk": (
            "VLLM_FLASH_V100_DENSE_D256_WMMA_QK",
            "native:exact_ne0_on",
            None,
        ),
        "dense_d256_low_smem": (
            "VLLM_FLASH_V100_DENSE_D256_LOW_SMEM",
            "native:exact_ne0_off",
            None,
        ),
        "prefill_d256_bm32_all_p": (
            "VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P",
            "native:exact_ne0_on",
            None,
        ),
        "prefill_d256_bm32_pair_scratch": (
            "VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH",
            "native:exact_ne0_on",
            None,
        ),
        "prefill_d256_low_smem": (
            "VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM",
            "native:exact_ne0_on",
            None,
        ),
        "prefill_d256_bm32_phase": (
            "VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE",
            "native:exact_ne0_on",
            None,
        ),
        "prefill_contig_fast": (
            "VLLM_FLASH_V100_PREFILL_CONTIG_FAST",
            "native:exact_ne0_off",
            None,
        ),
        "prefill_d256_scalar_qk": (
            "VLLM_FLASH_V100_PREFILL_D256_SCALAR_QK",
            "native:exact_ne0_off",
            None,
        ),
        "prefill_d256_bm32": (
            "VLLM_FLASH_V100_PREFILL_D256_BM32",
            "native:exact_ne0_off",
            None,
        ),
        "prefill_d256_output_stride_268": (
            "VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268",
            "native:exact_ne0_on",
            None,
        ),
        "prefill_d256_software_pipeline": (
            "VLLM_FLASH_V100_PREFILL_D256_SOFTWARE_PIPELINE",
            "native:exact_ne0_off",
            None,
        ),
        "prefill_d256_sw_pipeline_qk": (
            "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK",
            "native:exact_ne0_on",
            None,
        ),
        "prefill_d256_sw_pipeline_pv": (
            "VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV",
            "native:exact_ne0_on",
            None,
        ),
        "prefill_scalar_pv": (
            "VLLM_FLASH_V100_PREFILL_SCALAR_PV",
            "native:nonempty_ne0",
            None,
        ),
        "e4m3_scalar_fast": (
            "VLLM_FLASH_V100_E4M3_SCALAR_FAST",
            "native:scalar_alias",
            None,
        ),
        "share_decode_workspace": (
            "VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE",
            "ne0",
            "1",
        ),
        "xqa_staged_pv": ("VLLM_FLASH_V100_XQA_STAGED_PV", "eq1", "0"),
    }


@config
class FlashV100Diagnostics(CapturedFlashOptions):
    prefill_dump_tail: bool | None = None
    """Retain the native 79T tail observation points and output format."""

    legacy_aliases: ClassVar[dict[str, tuple[str, ...]]] = {
        "route_summary": ("VLLM_FLASH_V100_DEBUG_ROUTE_SUMMARY",),
    }

    def resolve(self):
        already_resolved = bool(self.sources)
        super().resolve()
        if already_resolved or self.sources["route_summary"] == "typed":
            return
        from vllm import envs

        if "VLLM_SM70_DEBUG" in os.environ:
            self.route_summary = (
                "routing" in envs.environment_variables["VLLM_SM70_DEBUG"]()
            )
            self.sources["route_summary"] = "VLLM_SM70_DEBUG"
        elif self.legacy_inputs[self.legacy_aliases["route_summary"][0]] == "1":
            self.route_summary = True
            self.sources["route_summary"] = "VLLM_FLASH_V100_DEBUG_ROUTE_SUMMARY"

    trace_decode_active: bool | None = None
    """Retained VLLM_FLASH_V100_TRACE_DECODE_ACTIVE input."""
    route_summary: bool | None = None
    """Retained VLLM_FLASH_V100_ROUTE_SUMMARY input."""
    compare_bhmd_out_dir: str | None = None
    """Retained VLLM_FLASH_V100_COMPARE_BHMD_OUT_DIR input."""
    compare_triton_out_dir: str | None = None
    """Retained VLLM_FLASH_V100_COMPARE_TRITON_OUT_DIR input."""
    compare_triton_tensor_dump_dir: str | None = None
    """Retained VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_DIR input."""
    compare_bhmd_out_max_calls: int | None = None
    """Retained VLLM_FLASH_V100_COMPARE_BHMD_OUT_MAX_CALLS input."""
    compare_triton_out_max_calls: int | None = None
    """Retained VLLM_FLASH_V100_COMPARE_TRITON_OUT_MAX_CALLS input."""
    compare_triton_tensor_dump_max_tokens: int | None = None
    """Retained VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_MAX_TOKENS input."""
    prefill_chunk_profile: bool | None = None
    """Retained VLLM_FLASH_V100_PREFILL_CHUNK_PROFILE input."""
    debug_prefill_compare: bool | None = None
    """Retained VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE input."""
    draft_graph_debug: bool | None = None
    """Retained VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG input."""
    draft_graph_debug_limit: int | None = None
    """Retained VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG_LIMIT input."""
    dflash_prefix_dump: bool | None = None
    """Retained VLLM_FLASH_V100_DFLASH_PREFIX_DUMP input."""
    e4m3_page800_fastpath_trace: bool | None = None
    """Retained VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH_TRACE input."""
    xqa_e5m2_g6_dual_cta_trace: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA_TRACE input."""
    xqa_g6_p1024_auto_trace: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_P1024_AUTO_TRACE input."""
    xqa_e4m3_g6_p64_p256_auto_trace: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO_TRACE input."""
    xqa_g6_qk_pipeline_trace: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_TRACE input."""
    xqa_g6_p1024_sawtooth_trace: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_TRACE input."""
    xqa_batch_context_routing_trace: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING_TRACE input."""
    xqa_block16_layout_trace: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_TRACE input."""
    xqa_block784_index_trace: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_BLOCK784_INDEX_TRACE input."""
    xqa_aligned_padded_smem_trace: bool | None = None
    """Retained VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM_TRACE input."""

    bindings: ClassVar[dict[str, tuple[str, str, object]]] = {
        "prefill_dump_tail": ("PREFIX_TORCH_DUMP_TAIL", "native:present", None),
        "trace_decode_active": ("VLLM_FLASH_V100_TRACE_DECODE_ACTIVE", "eq1", "0"),
        "route_summary": ("VLLM_FLASH_V100_ROUTE_SUMMARY", "eq1", "0"),
        "compare_bhmd_out_dir": ("VLLM_FLASH_V100_COMPARE_BHMD_OUT_DIR", "raw", None),
        "compare_triton_out_dir": (
            "VLLM_FLASH_V100_COMPARE_TRITON_OUT_DIR",
            "raw",
            None,
        ),
        "compare_triton_tensor_dump_dir": (
            "VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_DIR",
            "raw",
            None,
        ),
        "compare_bhmd_out_max_calls": (
            "VLLM_FLASH_V100_COMPARE_BHMD_OUT_MAX_CALLS",
            "int",
            "0",
        ),
        "compare_triton_out_max_calls": (
            "VLLM_FLASH_V100_COMPARE_TRITON_OUT_MAX_CALLS",
            "int",
            "0",
        ),
        "compare_triton_tensor_dump_max_tokens": (
            "VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_MAX_TOKENS",
            "int",
            "64",
        ),
        "prefill_chunk_profile": (
            "VLLM_FLASH_V100_PREFILL_CHUNK_PROFILE",
            "registered",
            None,
        ),
        "debug_prefill_compare": ("VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE", "eq1", "0"),
        "draft_graph_debug": ("VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG", "eq1", "0"),
        "draft_graph_debug_limit": (
            "VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG_LIMIT",
            "int",
            "12",
        ),
        "dflash_prefix_dump": ("VLLM_FLASH_V100_DFLASH_PREFIX_DUMP", "eq1", "0"),
        "e4m3_page800_fastpath_trace": (
            "VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH_TRACE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_e5m2_g6_dual_cta_trace": (
            "VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA_TRACE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_g6_p1024_auto_trace": (
            "VLLM_FLASH_V100_XQA_G6_P1024_AUTO_TRACE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_e4m3_g6_p64_p256_auto_trace": (
            "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO_TRACE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_g6_qk_pipeline_trace": (
            "VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_TRACE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_g6_p1024_sawtooth_trace": (
            "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_TRACE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_batch_context_routing_trace": (
            "VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING_TRACE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_block16_layout_trace": (
            "VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_TRACE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_block784_index_trace": (
            "VLLM_FLASH_V100_XQA_BLOCK784_INDEX_TRACE",
            "native:first_eq1_off",
            None,
        ),
        "xqa_aligned_padded_smem_trace": (
            "VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM_TRACE",
            "native:first_eq1_off",
            None,
        ),
    }


@config
class FlashV100Policy(ExecutionPolicy):
    """Execution policy owned by attention_config.flash_v100."""

    def bind_consumers(self, graph, trace, cache_dtype):
        self.options.finalize(graph, trace.flash_v100, cache_dtype)

    def compile_ignored_aliases(self):
        return self.options.compile_ignored_aliases()

    def explain(self):
        options = self.options
        return {
            "evidence": "initialized policy; native dispatch counts are separate",
            "values": {field: getattr(options, field) for field in options.bindings},
            "sources": dict(options.sources),
            "deferred_errors": dict(options.errors),
            "native_abi": 1,
            "native": [
                {"field": field, "parser": rule, "effective": value}
                for (field, _, rule, _), value in zip(
                    NATIVE_FIELDS, options.native_effective
                )
            ],
            "package_policy": dict(options.python_policy),
            "prefill_native": {
                "abi": 1,
                "values": list(options.prefill_native_effective),
                "resources": "worker runtime_resources.sm70_prefill",
                "shared_resource": "physical-device execution gate for kernel globals",
            },
            "resources": "worker runtime_resources.flash_v100",
        }

    options: FlashV100Options = Field(default_factory=FlashV100Options)
    """Backend, package and native choices bound to this attention owner."""

    def resolve(self) -> None:
        super().resolve()
        self.options.resolve()

    def compute_hash(self) -> str:
        return hash_factors(
            {"base": super().compute_hash(), "options": self.options.hash_values()}
            if self.active
            else {}
        )

    enabled: bool | None = None
    """Retain the platform's Flash-V100 backend qualification switch."""

    bfla_keep_ratio: float | None = None
    """Retained fraction for block-filtered prefill attention."""

    grouped_verify: bool | None = None
    """Enable the qualified grouped speculative attention operator."""

    grouped_verify_min_model_len: int | None = None
    """Minimum model context for grouped verification."""

    smallq_max_q: int | None = None
    """Largest query length admitted by small-query decode."""

    aliases: ClassVar[dict[str, str]] = {
        "bfla_keep_ratio": "VLLM_FLASH_V100_BFLA_KEEP_RATIO",
        "grouped_verify": "VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY",
        "grouped_verify_min_model_len": (
            "VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY_MIN_MODEL_LEN"
        ),
        "smallq_max_q": "VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q",
        "enabled": "VLLM_SM70_FLASH_ATTN_V100",
    }
