# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Typed execution policies captured once and serialized with their owners."""

from collections.abc import Callable
from contextlib import suppress
from typing import ClassVar, Literal

import torch
from pydantic import Field

from vllm.config.collective import CollectiveNativeConfig
from vllm.config.execution_policy_base import ExecutionPolicy as BaseExecutionPolicy
from vllm.config.flash_v100 import FlashV100Policy
from vllm.config.sm70_native import Sm70NativeConfig
from vllm.config.utils import config, hash_factors


def read_execution_legacy(name: str):
    import os

    from vllm import envs

    if name in (
        "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO",
        "VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS",
    ):
        return os.getenv(name, "1") != "0"
    if name in (
        "VLLM_SM70_DFLASH2_BF16_EMULATION",
        "VLLM_SM70_ENABLE_LM_HEAD_FASTPATH",
        "VLLM_SM70_LM_HEAD_TOP1_TC",
        "VLLM_GLM53_PP_MHC_MATERIALIZE",
    ):
        if name != "VLLM_SM70_DFLASH2_BF16_EMULATION":
            return os.getenv(name, "0").strip().lower() in ("1", "true", "yes", "on")
        raw = envs.environment_variables[name]()
        return ("1" if raw is None else raw).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    if name == "VLLM_QWEN35_MTP_KEEP_QUANT":
        return os.getenv(name, "0") == "1"
    if name in ("VLLM_SM70_GLM53_EXACT_KDA_GEMV", "VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS"):
        return os.getenv(name, "1") != "0"
    if name in ("VLLM_SM70_GLM53_TP8_CUBLASLT", "VLLM_SM70_GLM53_TP8_FUSED_FG_B"):
        return os.getenv(name, "0") != "0"
    if name in (
        "VLLM_SM70_GLM_MHC_PRE_THREADS",
        "VLLM_SM70_FP8_DENSE_TUNE_MAX_M",
        "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M",
        "VLLM_SM70_F16_DENSE_MAX_M",
    ):
        import regex as re

        # The native compatibility entry uses atoi, then accepts four sizes.
        is_threads = name == "VLLM_SM70_GLM_MHC_PRE_THREADS"
        raw = os.getenv(
            name,
            "256"
            if is_threads
            else "64"
            if name == "VLLM_SM70_F16_DENSE_MAX_M"
            else "16",
        )
        match = re.match(r"\s*([+-]?[0-9]+)", raw)
        value = int(match.group(1)) if match else 0
        if is_threads:
            return value if value in (128, 256, 512, 1024) else 256
        return max(value, 0)
    return envs.environment_variables[name]()


@config
class ExecutionPolicy(BaseExecutionPolicy):
    """Existing execution policies retain their initialization parser dialects."""

    legacy_reader: ClassVar[Callable[[str], object] | None] = staticmethod(
        read_execution_legacy
    )
    """Retain the existing execution family's initialization-only parsing."""


@config
class GraphPolicy(ExecutionPolicy):
    """Execution policy owned by compilation_config.runtime."""

    legacy_inputs: dict[str, str | None] = Field(default_factory=dict, init=False)
    """Original graph inputs for existing native parser projections."""
    decode_partition: int | None = Field(default=None, init=False)
    """Parsed partition override; None retains dynamic context selection."""
    decode_partition_error: tuple[str, str | None] | None = Field(
        default=None, init=False
    )
    """Captured error for the original attention admission checkpoint."""

    def resolve(self) -> None:
        import os

        if not self.legacy_inputs:
            self.legacy_inputs = {
                field: os.getenv(alias) for field, alias in self.aliases.items()
            }
        super().resolve()
        self.decode_partition = None
        self.decode_partition_error = None
        raw_decode = self.decode_partition_size
        if raw_decode is not None:
            try:
                self.decode_partition = int(raw_decode)
            except ValueError as exc:
                self.decode_partition_error = (
                    "VLLM_FLASH_V100_DECODE_PARTITION_SIZE must be one of "
                    f"(256, 512, 1024), got {raw_decode!r}",
                    str(exc),
                )
            else:
                if self.decode_partition not in (256, 512, 1024):
                    self.decode_partition_error = (
                        "VLLM_FLASH_V100_DECODE_PARTITION_SIZE must be one of "
                        f"(256, 512, 1024), got {self.decode_partition}",
                        None,
                    )
        self.mtp_partition_error = None
        self.mtp_partition = None
        raw = self.mtp_context_partition_size
        if raw is not None:
            try:
                self.mtp_partition = int(raw)
            except ValueError as exc:
                self.mtp_partition_error = (
                    "VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE must be one of "
                    f"(256, 512, 1024), got {raw!r}",
                    str(exc),
                )
            else:
                if self.mtp_partition not in (256, 512, 1024):
                    self.mtp_partition_error = (
                        "VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE must be one of "
                        f"(256, 512, 1024), got {self.mtp_partition}",
                        None,
                    )

    mtp_context_partition_size: str | int | None = None
    """Partition override for the existing MTP context buckets."""
    mtp_partition: int | None = Field(default=None, init=False)
    """Parsed override; request/context bucket selection remains dynamic."""
    mtp_partition_error: tuple[str, str | None] | None = Field(default=None, init=False)
    """Deferred partition error, consumed only at the original admission point."""

    def mtp_partition_hint(self):
        if self.mtp_partition_error is not None:
            message, cause = self.mtp_partition_error
            if cause is not None:
                raise ValueError(message) from ValueError(cause)
            raise ValueError(message)
        return self.mtp_partition

    def decode_partition_hint(self):
        if self.decode_partition_error is not None:
            message, cause = self.decode_partition_error
            if cause is not None:
                raise ValueError(message) from ValueError(cause)
            raise ValueError(message)
        return self.decode_partition

    def compute_hash(self):
        values = {
            field: getattr(self, field)
            for field in (
                self.hash_fields if self.hash_fields is not None else self.aliases
            )
        }
        for field in ("decode_partition_size", "mtp_context_partition_size"):
            if values.get(field) is not None:
                # An invalid input retains its original error spelling.
                with suppress(ValueError):
                    values[field] = int(values[field])
        return hash_factors(values if self.active else {})

    gdn_spec_piecewise: bool | None = None
    """Retain the SM70 aligned-cache speculative decode capture restriction."""
    eager_profile_run: bool | None = None
    """Skip compiled execution during the existing SM70 profiling step."""

    mtp_context_buckets: str | tuple[int, ...] | None = None
    """Explicit verification context buckets; empty disables, None uses defaults."""
    dsv4_context_buckets: str | tuple[int, ...] | None = None
    """Compressed-index decode buckets; parsed at graph initialization."""
    fp8_context_buckets: str | tuple[int, ...] | None = None
    """FP8 decode buckets; parsed at graph initialization."""
    batch_context_routing: bool | None = None
    """Select existing FP8 batch-context graph variants."""
    decode_partition_size: str | int | None = None
    """Raw legacy partition override; native admission retains its validation."""
    e4m3_batch_xqa: bool | None = None
    """Enable existing E4M3 batched XQA selection."""
    e4m3_p64_p256_auto: bool | None = None
    """Use the existing E4M3 context-dependent partition selection."""
    e4m3_wave_partitions: bool | None = None
    """Allow existing long-context wave graph variants."""
    e4m3_p512_begin: str | int | None = None
    """Legacy wave threshold, clamped at the graph initialization checkpoint."""
    decode_only_capture: bool | None = None
    """Retain the opt-in mixed/piecewise capture suppression."""

    aot_compile: bool | None = None
    """Save and reload ahead-of-time compiled artifacts."""

    mega_aot: bool | None = None
    """Use one combined ahead-of-time artifact when supported by the compiler."""

    breakable: bool | None = None
    """Use graph segments separated by eager custom operations."""

    sm70_breakable: bool | None = None
    """Legacy device-qualified request for segmented graphs."""

    compile_graph: bool | None = None
    """Use the retained compile graph numerical and capture policy."""

    decode_graph_no_compile: bool | None = None
    """Capture decode-only graphs without compiler tracing."""

    decode_capture_size: int | None = None
    """Largest batch captured by the no-compile decode policy."""

    eliminate_noops: bool | None = None
    """Enable no-op elimination for the retained compile policy."""

    dual_compile: bool | None = None
    """Trace prefill and decode separately while sharing weights."""

    split_draft_graphs: bool | None = None
    """Capture MTP draft graphs independently of verifier width."""

    estimate_graph_memory: bool | None = None
    """Use the existing graph memory admission estimator."""

    aliases: ClassVar[dict[str, str]] = {
        "gdn_spec_piecewise": "VLLM_SM70_QWEN_GDN_SPEC_DECODE_PIECEWISE",
        "eager_profile_run": "VLLM_SM70_FLASH_V100_0DOT3_EAGER_PROFILE_RUN",
        "mtp_context_partition_size": "VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE",
        "aot_compile": "VLLM_USE_AOT_COMPILE",
        "mega_aot": "VLLM_USE_MEGA_AOT_ARTIFACT",
        "breakable": "VLLM_USE_BREAKABLE_CUDAGRAPH",
        "sm70_breakable": "VLLM_SM70_USE_BREAKABLE_CUDAGRAPH",
        "compile_graph": "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH",
        "decode_graph_no_compile": "VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE",
        "decode_capture_size": "VLLM_SM70_FLASH_V100_DECODE_GRAPH_CAPTURE_SIZE",
        "eliminate_noops": "VLLM_SM70_FLASH_V100_0DOT3_ELIMINATE_NOOPS",
        "dual_compile": "VLLM_SM70_QWEN38_DUAL_COMPILE",
        "split_draft_graphs": "VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS",
        "estimate_graph_memory": "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
        "mtp_context_buckets": "VLLM_SM70_MTP_CONTEXT_BUCKETS",
        "dsv4_context_buckets": "VLLM_SM70_DSV4_DECODE_CONTEXT_BUCKETS",
        "fp8_context_buckets": "VLLM_SM70_FP8_KV_DECODE_CONTEXT_BUCKETS",
        "batch_context_routing": "VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING",
        "decode_partition_size": "VLLM_FLASH_V100_DECODE_PARTITION_SIZE",
        "e4m3_batch_xqa": "VLLM_FLASH_V100_E4M3_BATCH_XQA",
        "e4m3_p64_p256_auto": "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO",
        "e4m3_wave_partitions": "VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS",
        "e4m3_p512_begin": "VLLM_FLASH_V100_XQA_E4M3_G6_P512_BEGIN",
        "decode_only_capture": "VLLM_SM70_FLASH_V100_0DOT3_DECODE_ONLY_CAPTURE",
    }


@config
class LayerExecutionPolicy(ExecutionPolicy):
    """Execution policy owned by kernel_config.layer_execution."""

    quant_backend: Literal["auto", "marlin", "turbomind"] | None = None
    """Shared pre-Ampere quantization backend; format flags retain their gates."""
    gptq_turbomind: bool | None = None
    """Retained opt-in GPTQ weight-only provider."""
    compressed_tensors_turbomind: bool | None = None
    """Retained opt-in integer compressed-tensors provider."""
    mxfp4_turbomind: bool | None = None
    """MXFP4 loader admission; native tuning stays with its existing owner."""

    batch_fastpath: bool | None = None
    """Retain the existing batch fastpath model strategy."""

    hc_mtp_batch: bool | None = None
    """Retain the existing hc mtp batch model strategy."""

    hc_cooperative: bool | None = None
    """Retain the existing hc cooperative model strategy."""

    hc_full_unroll: bool | None = None
    """Retain the existing hc full unroll model strategy."""

    native: Sm70NativeConfig = Field(default_factory=Sm70NativeConfig)
    """B's native ABI owner for FP16 projection selectors and tuning."""

    dense_log_enabled: bool = Field(default=False, init=False)
    """Legacy logger integer boolean, separate from truth-word admission."""
    dense_log_error: str | None = Field(default=None, init=False)
    """Legacy logger parse error, raised only at its original checkpoint."""

    provider_errors: dict[str, str] = Field(default_factory=dict, init=False)
    """Keep unused provider parse failures behind their original admission gates."""

    deferred_fields: ClassVar[tuple[str, ...]] = (
        "quant_backend",
        "gptq_turbomind",
        "compressed_tensors_turbomind",
        "mxfp4_turbomind",
        "online_qpn8",
        "tp_local_topk20",
        "mtp_dense_fastpath",
        "shared_gate_fusion",
        "compact_topk20",
        "chunked_topk20_chunks",
        "glm_exact_kda_gemv",
        "topk_topp_b8_b16_warps8",
        "topk_topp_warps8",
    )

    def value(self, field):
        if field in self.provider_errors:
            raise ValueError(self.provider_errors[field])
        return getattr(self, field)

    def resolve(self, *, dflash=None) -> None:
        from vllm.config.utils import resolve_legacy_fields

        pending = {
            field: self.aliases[field]
            for field in self.deferred_fields
            if field not in self.sources
        }
        resolve_legacy_fields(
            self,
            pending,
            reader=read_execution_legacy,
            deferred_errors=self.provider_errors,
        )
        super().resolve()
        from vllm import envs

        if "dense_log_enabled" not in self.sources:
            if self.sources.get("lm_head_dense") == "typed":
                self.dense_log_enabled = bool(self.lm_head_dense)
            else:
                try:
                    self.dense_log_enabled = envs.environment_variables[
                        "VLLM_SM70_ENABLE_LM_HEAD_FASTPATH"
                    ]()
                except ValueError as error:
                    self.dense_log_error = str(error)
            self.sources["dense_log_enabled"] = self.sources["lm_head_dense"]
        if (
            self.native.f16_dense_max_m is not None
            and self.sources.get("dense_max_m") != "typed"
        ):
            self.dense_max_m = self.native.f16_dense_max_m
            self.sources["dense_max_m"] = "typed:native"
        overrides = {"VLLM_SM70_F16_DENSE_MAX_M": self.dense_max_m}
        if dflash is not None:
            overrides.update(dflash.native_overrides())
        self.native.resolve("f16", overrides)

    def compute_hash(self) -> str:
        factors: dict[str, object] = {"policy": super().compute_hash()}
        errors = {
            field: error
            for field, error in self.provider_errors.items()
            if self.hash_fields is None or field in self.hash_fields
        }
        if errors:
            factors["provider_errors"] = errors
        if self.hash_fields is None or "dense_f16" in self.hash_fields:
            factors["native"] = self.native.hash_options()
        return hash_factors(factors)

    topk_topp_b8_b16_warps8: bool | None = None
    """Use eight warps at the retained ordinary-sampling vocabulary/row gates."""
    topk_topp_warps8: bool | None = None
    """Enable the retained eight-warp schedule for verifier sampling rows."""

    glm_pp_mhc_materialize: bool | None = None
    """Materialize completed mHC states at the existing PP boundary."""

    greedy_token_fastpath: bool | None = None
    """Admit the retained runner greedy-token path at its dynamic eligibility gate."""
    tp_local_topk20: bool | None = None
    """Use local TP top-k20 only after the compact-sampler gate passes."""

    moe_dense_allowlist: str | None = None
    """Optional model-qualified projection suffixes for dense expert preparation."""
    shared_gate_fusion: bool | None = None
    """Fuse only the shared-expert projection shapes accepted by their adapter."""

    mtp_dense_fastpath: bool | None = None
    """Permit the model adapter's MTP FP16 projection allowlist."""
    mtp_dense_allowlist: str | None = None
    """Optional suffix list; the model adapter retains the historical defaults."""
    mtp_share_io_weights: bool | None = None
    """Share target embeddings and output weights on the qualified MTP model."""
    mtp_keep_quant: bool | None = None
    """Retain draft quantization despite a checkpoint exclusion rule."""

    online_qpn8: bool | None = None
    """Prepare the qualified checkpoint-FP16 weights in online QPN8 layout."""

    compact_topk20: bool | None = None
    """Admit the exact ordinary top-k20 sampler at its retained metadata gate."""

    chunked_topk20_chunks: int | None = None
    """Chunk count consumed only by the admitted SM70 248320-vocabulary sampler."""

    glm_exact_kda_gemv: bool | None = None
    """Retain the GLM exact KDA projection provider and legacy nonzero dialect."""

    batch_gemm_layouts: bool | None = None
    """Prepare compatible larger-batch dense weight layouts."""

    fp16_gemv: bool | None = None
    """Prepare exact FP16 single-token projection operators."""

    fused_gdn_input: bool | None = None
    """Allow fused input projections with their existing shape gates."""

    fused_hc: bool | None = None
    """Allow fused FP16 hyperconnection projections."""

    gemma_compile_native: bool | None = None
    """Preserve the native compiled Gemma normalization route."""

    lm_head_top1: bool | None = None
    """Enable the existing local logits top-one projection shortcut."""

    dsv4_fp13_gemv: bool | None = None
    """Preserve the qualified DeepSeek packed FP13 projection route."""
    dsv4_fp16_gemv: bool | None = None
    """Preserve the exact DeepSeek FP16 projection fallback."""

    dense_f16: bool | None = None
    """Prepare the retained small FP16 projection layout."""
    dense_allowlist: str | None = None
    """Comma-separated projection suffixes; None retains model defaults."""
    dense_max_m: int | None = None
    """Legacy dense row threshold used by diagnostics and native dispatch."""

    lm_head_dense: bool | None = None
    """Enable the retained packed FP16 dense logits provider."""

    lm_head_top1_tc: bool | None = None
    """Enable the retained packed Tensor Core top-one provider."""

    gemma_long_prefill_fused: bool | None = None
    """Enable exact mixed-dtype Gemma normalization at the existing row bound."""

    gemma_eager: bool | None = None
    """Use the eager Gemma normalization custom-op boundary."""

    shared_moe_overlap: bool | None = None
    """Overlap the shared expert with routed expert execution."""

    glm_cublaslt: bool | None = None
    """Enable the qualified cuBLASLt projection provider."""

    glm_fused_fg_b: bool | None = None
    """Allow the qualified fused recurrent gate projection."""

    mhc_native_verify: bool | None = None
    """Use the native eight-token hyperconnection normalization."""

    mhc_fused_post_dot: bool | None = None
    """Fuse hyperconnection post-processing with its dot product."""

    mhc_pre_threads: int | None = None
    """Threads for native multi-token hyperconnection normalization."""

    aliases: ClassVar[dict[str, str]] = {
        "quant_backend": "VLLM_SM70_QUANT_BACKEND",
        "gptq_turbomind": "VLLM_SM70_GPTQ_TURBOMIND",
        "compressed_tensors_turbomind": "VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND",
        "mxfp4_turbomind": "VLLM_SM70_MXFP4_TURBOMIND",
        "topk_topp_b8_b16_warps8": "VLLM_SM70_TOPK_TOPP_B8_B16_8_WARPS",
        "topk_topp_warps8": "VLLM_SM70_TOPK_TOPP_8_WARPS",
        "glm_pp_mhc_materialize": "VLLM_GLM53_PP_MHC_MATERIALIZE",
        "greedy_token_fastpath": "VLLM_SM70_GREEDY_TOKEN_FASTPATH",
        "tp_local_topk20": "VLLM_SM70_TP_LOCAL_TOPK20_SAMPLER",
        "moe_dense_allowlist": "VLLM_SM70_MOE_DENSE_ALLOWLIST",
        "shared_gate_fusion": "VLLM_SM70_QWEN3NEXT_SHARED_GATE_FUSION",
        "mtp_dense_fastpath": "VLLM_SM70_MTP_DENSE_F16_FASTPATH",
        "mtp_dense_allowlist": "VLLM_SM70_MTP_DENSE_F16_ALLOWLIST",
        "mtp_share_io_weights": "VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS",
        "mtp_keep_quant": "VLLM_QWEN35_MTP_KEEP_QUANT",
        "online_qpn8": "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8",
        "compact_topk20": "VLLM_SM70_COMPACT_TOPK20_SAMPLER",
        "chunked_topk20_chunks": "VLLM_SM70_CHUNKED_TOPK20_CHUNKS",
        "glm_exact_kda_gemv": "VLLM_SM70_GLM53_EXACT_KDA_GEMV",
        "batch_fastpath": "VLLM_SM70_QWEN38_BATCH_FASTPATH",
        "hc_mtp_batch": "VLLM_SM70_MTP_HC_BATCH",
        "hc_cooperative": "VLLM_SM70_MTP_HC_COOPERATIVE",
        "hc_full_unroll": "VLLM_SM70_MTP_HC_FULL_UNROLL",
        "batch_gemm_layouts": "VLLM_SM70_BATCH_GEMM_LAYOUTS",
        "fp16_gemv": "VLLM_SM70_QWEN38_FP16_GEMV",
        "fused_gdn_input": "VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16",
        "fused_hc": "VLLM_SM70_QWEN38_FUSED_HC_FP16",
        "gemma_compile_native": "VLLM_SM70_GEMMA_RMS_NORM_COMPILE_NATIVE",
        "lm_head_top1": "VLLM_SM70_LM_HEAD_TOP1",
        "dense_f16": "VLLM_SM70_ENABLE_DENSE_F16_FASTPATH",
        "dsv4_fp13_gemv": "VLLM_SM70_DSV4_FP13_GEMV",
        "dsv4_fp16_gemv": "VLLM_SM70_DSV4_FP16_GEMV",
        "dense_allowlist": "VLLM_SM70_F16_DENSE_ALLOWLIST",
        "dense_max_m": "VLLM_SM70_F16_DENSE_MAX_M",
        "lm_head_dense": "VLLM_SM70_ENABLE_LM_HEAD_FASTPATH",
        "lm_head_top1_tc": "VLLM_SM70_LM_HEAD_TOP1_TC",
        "gemma_long_prefill_fused": "VLLM_SM70_GEMMA_LONG_PREFILL_FUSED",
        "gemma_eager": "VLLM_SM70_GEMMA_RMS_NORM_EAGER",
        "shared_moe_overlap": "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP",
        "glm_cublaslt": "VLLM_SM70_GLM53_TP8_CUBLASLT",
        "glm_fused_fg_b": "VLLM_SM70_GLM53_TP8_FUSED_FG_B",
        "mhc_native_verify": "VLLM_SM70_GLM53_MHC_NATIVE_VERIFY",
        "mhc_fused_post_dot": "VLLM_SM70_GLM53_MHC_FUSED_POST_DOT_Q8",
        "mhc_pre_threads": "VLLM_SM70_GLM_MHC_PRE_THREADS",
    }


@config
class CommunicationPolicy(ExecutionPolicy):
    """Execution policy owned by parallel_config.communication."""

    native: CollectiveNativeConfig = Field(default_factory=CollectiveNativeConfig)
    """Immutable per-communicator native selection and launch parameters."""

    def resolve(self, native_overrides=None, *, layers=None, trace=None):
        super().resolve()
        overrides = dict(native_overrides or {})
        source = self.sources.get("tp8_hierarchical", "")
        if source == "typed" or source.startswith("default:"):
            overrides["sm70_tp8_hierarchical_custom_ar"] = int(
                bool(self.tp8_hierarchical)
            )
        self.native.resolve_for_owners(overrides, layers=layers, trace=trace)
        if self.native.sources.get("sm70_tp8_hierarchical_custom_ar") == "typed":
            self.tp8_hierarchical = self.native.registry_bool(
                "sm70_tp8_hierarchical_custom_ar", "0"
            )
            self.sources["tp8_hierarchical"] = "typed:native"

    def compute_hash(self):
        return hash_factors(
            {"policy": super().compute_hash(), "native": self.native.compute_hash()}
        )

    top1_custom_ar: bool | None = None
    """Provision the existing compact greedy-token collective."""

    symm_mem: bool | None = None
    """Select symmetric-memory collectives when eligible."""

    flashinfer: bool | None = None
    """Select the existing FlashInfer collective provider."""

    awq_overlap_side_stream: bool | None = None
    """Use the existing side stream for AWQ producer/reducer overlap."""

    awq_overlap_tile_numel: int | None = None
    """Elements per overlapping AWQ reduction tile."""

    awq_overlap_reducer_blocks: int | None = None
    """Reducer blocks for overlapping AWQ communication."""

    awq_overlap_kernel_reducer_blocks: int | None = None
    """Reducer grid override inside the overlapping AWQ kernel."""

    long_prefill_norm: bool | None = None
    """Retain the opt-in TP4 reduce-scatter, Gemma norm and all-gather route."""
    awq_tile_ar: bool | None = None
    """Enable the existing AWQ down-projection tile collective."""
    awq_tile_overlap: bool | None = None
    """Enable the existing producer/reducer overlap route."""
    awq_tile_numel: int | None = None
    """Tile capacity consumed by the existing communication workspace."""
    awq_tile_mode: str | None = None
    """Existing tile runtime mode, retaining invalid-mode ordinary fallback."""
    awq_engine_blocks: int | None = None
    """Existing single-engine collective grid."""
    awq_producer_blocks: int | None = None
    """Existing overlapped GEMM producer grid."""
    awq_reducer_blocks: int | None = None
    """Existing overlapped collective reducer grid."""

    tp4_push: bool | None = None
    """Allow the existing four-rank push collective."""

    tp8_hierarchical: bool | None = None
    """Allow the existing hierarchical eight-rank collective."""

    tp8_push: bool | None = None
    """Allow the push stage of the hierarchical eight-rank collective."""

    moe_add_allreduce: bool | None = None
    """Fuse expert residual addition with collective reduction."""

    mq_max_chunks: int | None = None
    """Capacity of the host message-queue broadcast buffer."""

    pp_layer_partition: str | None = None
    """Explicit comma-separated pipeline layer counts, or automatic."""

    aliases: ClassVar[dict[str, str]] = {
        "top1_custom_ar": "VLLM_SM70_TOP1_CUSTOM_AR",
        "symm_mem": "VLLM_ALLREDUCE_USE_SYMM_MEM",
        "flashinfer": "VLLM_ALLREDUCE_USE_FLASHINFER",
        "awq_overlap_side_stream": "VLLM_SM70_AWQ_MLP_DOWN_TILE_OVERLAP_SIDE_STREAM",
        "awq_overlap_tile_numel": "VLLM_SM70_AWQ_MLP_DOWN_TILE_OVERLAP_TILE_NUMEL",
        "awq_overlap_reducer_blocks": (
            "VLLM_SM70_AWQ_MLP_DOWN_TILE_OVERLAP_REDUCER_BLOCKS"
        ),
        "awq_overlap_kernel_reducer_blocks": (
            "VLLM_SM70_AWQ_MLP_DOWN_TILE_OVERLAP_KERNEL_REDUCER_BLOCKS"
        ),
        "tp4_push": "VLLM_SM70_TP4_PUSH_ALLREDUCE",
        "tp8_hierarchical": "VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR",
        "tp8_push": "VLLM_SM70_TP8_HIERARCHICAL_PUSH_AR",
        "moe_add_allreduce": "VLLM_SM70_MOE_ADD_ALLREDUCE",
        "mq_max_chunks": "VLLM_MQ_BROADCASTER_MAX_CHUNKS",
        "pp_layer_partition": "VLLM_PP_LAYER_PARTITION",
        "long_prefill_norm": "VLLM_SM70_TP4_LONG_PREFILL_FUSED_NORM",
        "awq_tile_ar": "VLLM_SM70_AWQ_MLP_DOWN_TILE_AR",
        "awq_tile_overlap": "VLLM_SM70_AWQ_MLP_DOWN_TILE_OVERLAP",
        "awq_tile_numel": "VLLM_SM70_AWQ_MLP_DOWN_TILE_AR_TILE_NUMEL",
        "awq_tile_mode": "VLLM_SM70_AWQ_MLP_DOWN_TILE_AR_MODE",
        "awq_engine_blocks": "VLLM_SM70_AWQ_MLP_DOWN_TILE_AR_ENGINE_BLOCKS",
        "awq_producer_blocks": "VLLM_SM70_AWQ_MLP_DOWN_TILE_AR_PRODUCER_BLOCKS",
        "awq_reducer_blocks": "VLLM_SM70_AWQ_MLP_DOWN_TILE_AR_REDUCER_BLOCKS",
    }


@config
class PlePlacementPolicy(ExecutionPolicy):
    """Execution policy owned by offload_config.ple."""

    hybrid: bool | None = None
    """Retain local embedding tables for decode alongside CPU offload."""

    cpu: bool | None = None
    """Run embedding table lookup in the dedicated CPU worker."""

    disk: bool | None = None
    """Read offloaded embedding rows from mapped checkpoint storage."""

    aliases: ClassVar[dict[str, str]] = {
        "hybrid": "VLLM_SM70_QWEN38_HYBRID_PLE",
        "cpu": "VLLM_PLE_CPU_OFFLOAD",
        "disk": "VLLM_PLE_DISK_OFFLOAD",
    }


POLICY_OWNERS = {
    "compilation_config.runtime": "GraphPolicy",
    "kernel_config.layer_execution": "LayerExecutionPolicy",
    "kernel_config.sm70_sparse": "Sm70SparseConfig",
    "parallel_config.communication": "CommunicationPolicy",
    "offload_config.ple": "PlePlacementPolicy",
    "attention_config.flash_v100": "FlashV100Policy",
    "attention_config.sm70_triton": "Sm70TritonAttentionPolicy",
    "attention_config.flash_v100.turboquant": "TurboQuantRuntimePolicy",
}


# These owners resolve at their existing model/speculation checkpoints, after
# platform defaults. Runtime contexts borrow them without resolving them again.
BOUND_POLICY_OWNERS = (
    *POLICY_OWNERS,
    "kernel_config.gdn.projection",
    "kernel_config.sm70_moe.unquantized",
    "kernel_config.sm70_moe.routing",
    "speculative_config.sampling_policy",
)


@torch.compiler.assume_constant_result
def _standalone_policy(cls):
    """Legacy standalone initialization, folded before tracing tensor work.

    Engine callers never take this path: they borrow their already-resolved
    owner, so compiling two engines cannot reuse a standalone policy object.
    """
    policy = cls()
    policy.resolve()
    return policy


def capture_execution_policy(owner: str, cls, cfg=None):
    """Initialization adapter for standalone callers and worker-owned objects."""
    if cfg is None:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if is_forward_context_available():
            policies = get_forward_context().runtime_resources.get(
                "execution_policies", {}
            )
            if owner in policies:
                return policies[owner]
            parent, _, field = owner.rpartition(".")
            if parent in policies:
                return getattr(policies[parent], field)
        from vllm.config import get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
    if cfg is None:
        return _standalone_policy(cls)
    for part in owner.split("."):
        cfg = getattr(cfg, part, None)
    return cfg


def graph_policy(cfg=None) -> GraphPolicy:
    return capture_execution_policy("compilation_config.runtime", GraphPolicy, cfg)


def layer_policy(cfg=None) -> LayerExecutionPolicy:
    return capture_execution_policy(
        "kernel_config.layer_execution", LayerExecutionPolicy, cfg
    )


def communication_policy(cfg=None) -> CommunicationPolicy:
    return capture_execution_policy(
        "parallel_config.communication", CommunicationPolicy, cfg
    )


def ple_policy(cfg=None) -> PlePlacementPolicy:
    return capture_execution_policy("offload_config.ple", PlePlacementPolicy, cfg)


def flash_v100_policy(cfg=None) -> FlashV100Policy:
    return capture_execution_policy("attention_config.flash_v100", FlashV100Policy, cfg)


def flash_v100_options(cfg=None):
    from vllm.config.flash_v100 import FlashV100Options

    return capture_execution_policy(
        "attention_config.flash_v100.options", FlashV100Options, cfg
    )
