# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Typed execution policies captured once and serialized with their owners."""

from typing import ClassVar

import torch
from pydantic import Field

from vllm.config.sm70_native import Sm70NativeConfig
from vllm.config.utils import config, hash_factors


def read_execution_legacy(name: str):
    import os

    from vllm import envs

    if name in (
        "VLLM_SM70_DFLASH2_BF16_EMULATION",
        "VLLM_SM70_ENABLE_LM_HEAD_FASTPATH",
        "VLLM_SM70_LM_HEAD_TOP1_TC",
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
class ExecutionPolicy:
    """Common provenance; computation never consumes legacy names."""

    aliases: ClassVar[dict[str, str]] = {}
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization source for each field; excluded from calculation hashes."""
    active: bool = Field(default=True, init=False)
    """Whether the owning feature can affect this engine's computation."""

    hash_fields: tuple[str, ...] | None = Field(default=None, init=False)
    """Effective computation fields; resource and unused feature choices are omitted."""

    def resolve(self) -> None:
        from vllm.config.sm70_runtime import resolve_legacy_fields

        pending = {
            field: alias
            for field, alias in self.aliases.items()
            if field not in self.sources
        }
        resolve_legacy_fields(self, pending, reader=read_execution_legacy)

    def compute_hash(self) -> str:
        return hash_factors(
            {
                field: getattr(self, field)
                for field in (
                    self.hash_fields if self.hash_fields is not None else self.aliases
                )
            }
            if self.active
            else {}
        )


@config
class GraphPolicy(ExecutionPolicy):
    """Execution policy owned by compilation_config.runtime."""

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
    }


@config
class LayerExecutionPolicy(ExecutionPolicy):
    """Execution policy owned by kernel_config.layer_execution."""

    native: Sm70NativeConfig = Field(default_factory=Sm70NativeConfig)
    """B's native ABI owner for FP16 projection selectors and tuning."""

    dense_log_enabled: bool = Field(default=False, init=False)
    """Legacy logger integer boolean, separate from truth-word admission."""
    dense_log_error: str | None = Field(default=None, init=False)
    """Legacy logger parse error, raised only at its original checkpoint."""

    def resolve(self, *, dflash=None) -> None:
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
        if self.hash_fields is None or "dense_f16" in self.hash_fields:
            factors["native"] = self.native.hash_options()
        return hash_factors(factors)

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
        "tp4_push": "VLLM_SM70_TP4_PUSH_ALLREDUCE",
        "tp8_hierarchical": "VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR",
        "tp8_push": "VLLM_SM70_TP8_HIERARCHICAL_PUSH_AR",
        "moe_add_allreduce": "VLLM_SM70_MOE_ADD_ALLREDUCE",
        "mq_max_chunks": "VLLM_MQ_BROADCASTER_MAX_CHUNKS",
        "pp_layer_partition": "VLLM_PP_LAYER_PARTITION",
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


@config
class FlashV100Policy(ExecutionPolicy):
    """Execution policy owned by attention_config.flash_v100."""

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
    }


POLICY_OWNERS = {
    "compilation_config.runtime": "GraphPolicy",
    "kernel_config.layer_execution": "LayerExecutionPolicy",
    "parallel_config.communication": "CommunicationPolicy",
    "offload_config.ple": "PlePlacementPolicy",
    "attention_config.flash_v100": "FlashV100Policy",
}


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
        from vllm.config import get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
    if cfg is None:
        return _standalone_policy(cls)
    config_name, field = owner.split(".")
    return getattr(getattr(cfg, config_name), field)


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
