# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configured route capabilities, not a claim that a request hit a kernel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

from vllm import envs
from vllm.envs_metadata import EnvVar
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _row(reason: str | None = None, **values: Any) -> dict[str, Any]:
    return {"enabled": reason is None, "reason": reason, **values}


def _switches(defaults: dict[str, str]) -> dict[str, Any]:
    return {name: getattr(envs, name) for name in defaults}


def _switches_match(values: dict[str, Any], defaults: dict[str, str]) -> bool:
    return all(
        (
            int(value) >= int(defaults[name])
            if int(defaults[name]) > 1
            else str(int(value) if isinstance(value, bool) else value) == defaults[name]
        )
        for name, value in values.items()
    )


def _is_sm70(cfg: VllmConfig) -> bool:
    from vllm.config.vllm import _participating_cuda_device_ids
    from vllm.platforms import current_platform

    devices = _participating_cuda_device_ids(cfg)
    return bool(devices) and all(
        current_platform.is_device_capability((7, 0), device_id=i) for i in devices
    )


def _flash_next_batch_report(cfg: VllmConfig) -> dict[str, Any]:
    """Explain qualified defaults and packed-copy cost, without claiming hits."""
    controls = {
        name: {
            "enabled": bool(getattr(envs, name)),
            "reason": None if getattr(envs, name) else "user_override",
            "description": cast(EnvVar, getter).metadata.description,
        }
        for name, getter in envs.environment_variables.items()
        if "Flash-Next qualified batch"
        in cast(EnvVar, getter).metadata.acceleration_paths
    }
    norm = cfg.kernel_config.sm70_rmsnorm_gated_exact
    if norm is not None:
        controls["VLLM_SM70_RMSNORM_GATED_EXACT"].update(
            enabled=bool(norm), reason=None if norm else "resolved_policy_disabled"
        )
    text = cfg.model_config.hf_text_config
    tp = cfg.parallel_config.tensor_parallel_size
    layers = int(getattr(text, "num_hidden_layers", 0))
    draft_layers = (
        int(getattr(text, "mtp_num_hidden_layers", 0))
        if getattr(cfg.speculative_config, "method", None) == "mtp"
        else 0
    )
    # These are sizes of existing packed buffers, not another admission gate.
    # Different local geometries are checked by their weight loaders; omit an
    # estimate rather than assuming that they allocate the TP4 reference packs.
    reference_layout = (
        tp == 4
        and getattr(text, "hidden_size", None) == 2560
        and getattr(text, "hc_count", None) == 4
        and getattr(text, "hc_lowrank", None) == 320
    )
    copies: dict[str, int] = {}
    if reference_layout:
        batch = envs.VLLM_SM70_QWEN38_BATCH_FASTPATH
        gdn_layers = list(getattr(text, "layer_types", ())).count("linear_attention")
        if batch or envs.VLLM_SM70_QWEN38_GDN_INPUT_BATCH:
            copies["gdn_input"] = gdn_layers * (4096 + 32) * 2560 * 2
        if batch or (draft_layers and envs.VLLM_SM70_MTP_HC_BATCH):
            copies["hc_target"] = layers * 2 * (96 * 10240 + 2560 * 320) * 2
            copies["hc_draft"] = draft_layers * 2 * (96 * 10240 + 2560 * 320) * 2
        if draft_layers and envs.VLLM_SM70_MTP_ROUTER_BATCH:
            copies["router"] = (layers + draft_layers) * 512 * 2560 * 2
        if draft_layers and envs.VLLM_SM70_MTP_SHARED_BATCH:
            copies["shared_expert"] = (layers + draft_layers) * 320 * 2560 * 2
    return {
        "scope": "configured_capabilities",
        "status": "runtime_guarded",
        "controls": controls,
        "packed_weight_memory": {
            "scope": "estimated_additional_bytes_per_rank",
            "reason": None
            if reference_layout
            else "estimate_requires_reference_layout",
            "components": copies,
            "total_bytes": sum(copies.values()) if reference_layout else None,
            "excludes": "allocator overhead, graphs, temporary workspaces and KV cache",
            "capacity_note": (
                "Packed weights reduce memory available to KV and graph/workspace "
                "peaks. An explicit KV byte budget does not shrink automatically. "
                "The TP4 reference added about 1.25 GiB/rank at load; C4 with "
                "1.5 GiB KV and prefill budget 8192 exhausted memory, while the "
                "matched budget-4096 profile completed. Reserve peak headroom "
                "or disable the packed-copy controls below."
            ),
            "mitigation": (
                "Set QWEN38_BATCH_FASTPATH, QWEN38_GDN_INPUT_BATCH, MTP_HC_BATCH, "
                "MTP_ROUTER_BATCH and MTP_SHARED_BATCH to 0 (VLLM_SM70_ prefix) "
                "to remove the corresponding packed copies."
            ),
        },
    }


def _native_capabilities(page_size: int) -> dict[str, bool]:
    # Register FA2 operators before probing availability.
    import vllm.vllm_flash_attn._vllm_fa2_C  # noqa: F401

    # isort: split
    # Keep the companion import stable with and without extracted build files.
    from flash_attn_v100 import (  # type: ignore[attr-defined]
        flash_attn_grouped_e4m3_fp32_available,
    )

    # isort: split
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _get_sm70_d256_gqa_architecture_q8192_op,
    )
    from vllm.v1.attention.ops.sm70_e4m3_long import (
        BUILTIN_MANIFEST,
        builtin_long_attention,
        long_attention_enabled,
        long_attention_page_supported,
    )
    from vllm.v1.attention.ops.sm70_e4m3_scalar import scalar_tail_attention_available

    return {
        "grouped_fp32": bool(flash_attn_grouped_e4m3_fp32_available()),
        "long_operator": builtin_long_attention() is not None,
        "long_enabled": long_attention_enabled(),
        "page_supported": long_attention_page_supported(page_size, BUILTIN_MANIFEST),
        "scalar": scalar_tail_attention_available(),
        "q8000": _get_sm70_d256_gqa_architecture_q8192_op() is not None,
    }


def _dflash_reason(cfg: VllmConfig) -> str | None:
    from vllm.config.vllm import _is_sm70_dflash2_verifier_contract

    if _is_sm70_dflash2_verifier_contract(
        cfg.model_config, cfg.speculative_config, cfg.parallel_config
    ):
        return None
    spec = cfg.speculative_config
    if spec is None:
        return "contract_mismatch:method=None≠dflash"
    if spec.method != "dflash":
        return f"contract_mismatch:method={spec.method}≠dflash"
    if spec.num_speculative_tokens != 7:
        return (
            f"contract_mismatch:num_speculative_tokens={spec.num_speculative_tokens}≠7"
        )
    draft = getattr(spec, "draft_model_config", None)
    draft_hf = getattr(draft, "hf_config", None)
    draft_config = getattr(draft_hf, "dflash_config", None)
    selector = None
    if isinstance(draft_config, Mapping):
        selector = draft_config.get("selector_top_k")
    text = getattr(cfg.model_config, "hf_text_config", None)
    requirements = (
        ("dtype", cfg.model_config.dtype, "torch.float16"),
        ("hidden_size", getattr(text, "hidden_size", None), 5120),
        ("num_attention_heads", getattr(text, "num_attention_heads", None), 24),
        ("num_key_value_heads", getattr(text, "num_key_value_heads", None), 4),
        ("head_dim", getattr(text, "head_dim", None), 256),
        ("selector_top_k", selector, 16),
        ("pipeline_parallel_size", cfg.parallel_config.pipeline_parallel_size, 1),
    )
    for name, actual, expected in requirements:
        if str(actual) != str(expected):
            return f"contract_mismatch:{name}={actual}≠{expected}"
    return "contract_mismatch:dflash2_verifier=False≠True"


def build_report(cfg: VllmConfig) -> dict[str, Any]:
    if cfg.model_config is None:
        # PLE/component processes create a config without a target model.
        # They must not probe target operators or enforce a serving profile.
        return {
            "profile": None,
            "sm70": _is_sm70(cfg),
            "scope": "component_config",
            "expected_acceleration": [],
            "paths": {},
        }
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.config.vllm import (
        _SM70_BATCH_GEMM_DEFAULTS,
        _SM70_DFLASH2_VERIFIER_DEFAULTS,
        _is_sm70_qwen38_decode_compile_contract,
    )

    from .profile import load_profile

    sm70 = _is_sm70(cfg)
    paths: dict[str, dict[str, Any]] = {}
    report = {
        "profile": "qwen38_27b_nvfp4_dflash2",
        "sm70": sm70,
        "scope": "configured_capabilities",
        "expected_acceleration": load_profile()["expected_acceleration"],
        "paths": paths,
    }
    # Configuration policy is resolved once per engine. Actual kernel selection
    # still needs each loaded layer's local layout and native capabilities.
    policy = getattr(cfg.kernel_config, "sm70_nvfp4", None)
    if policy is not None:
        report["linear_kernel_policy"] = {
            "scope": "ct_nvfp4_linear",
            "status": "runtime_guarded",
            "configuration": asdict(policy),
            "qpn2_reason": (
                "configuration_not_resolved"
                if not policy.resolved
                else "disabled_by_configuration_or_legacy_override"
                if not policy.qpn2
                else None
            ),
            "default_qualification_reason": (
                None
                if policy.qualified
                else "draft_selector_state_contract_not_quality_qualified"
            ),
        }
    if not sm70:
        names = set(load_profile()["expected_acceleration"]) | {
            "qwen38_decode",
            "e4m3_grouped_fp32",
            "long_context",
            "scalar_tail",
            "q8000_prefill",
            "compile_graph",
            "compile_cache",
        }
        paths.update({name: _row("not_applicable") for name in sorted(names)})
        return report

    tp = cfg.parallel_config.tensor_parallel_size
    paths["profile_hardware"] = _row(
        None if tp == 4 else f"contract_mismatch:tensor_parallel_size={tp}≠4", tp=tp
    )
    for name, defaults, reason in (
        ("dflash2_verifier", _SM70_DFLASH2_VERIFIER_DEFAULTS, _dflash_reason(cfg)),
        ("batch_gemm", _SM70_BATCH_GEMM_DEFAULTS, None),
    ):
        values = _switches(defaults)
        paths[name] = _row(
            reason or (None if _switches_match(values, defaults) else "user_override"),
            switches=values,
        )

    decode_names = (
        "VLLM_SM70_QWEN38_FP16_GEMV",
        "VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16",
        "VLLM_SM70_QWEN38_FUSED_HC_FP16",
        "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP",
        "VLLM_SM70_MOE_ADD_ALLREDUCE",
    )
    decode_values = {name: getattr(envs, name) for name in decode_names}
    decode_contract = _is_sm70_qwen38_decode_compile_contract(
        cfg.model_config, cfg.speculative_config, cfg.parallel_config
    )
    if decode_contract:
        report["profile"] = "qwen4exp_fp16_decode"
        report["expected_acceleration"] = [
            "qwen38_decode",
            "batch_gemm",
            "compile_graph",
        ]
        report["flash_next_batch"] = _flash_next_batch_report(cfg)
    paths["qwen38_decode"] = _row(
        "not_applicable"
        if not decode_contract
        else (
            None
            if all(decode_values[name] for name in decode_names[:3])
            else "user_override"
        ),
        switches=decode_values,
    )

    page_size = int(cfg.cache_config.block_size or 0)
    try:
        native = _native_capabilities(page_size)
    except (ImportError, AttributeError, RuntimeError, OSError) as exc:
        native = dict.fromkeys(
            (
                "grouped_fp32",
                "long_operator",
                "long_enabled",
                "page_supported",
                "scalar",
                "q8000",
            ),
            False,
        )
        report["operator_probe_error"] = type(exc).__name__ + ": " + str(exc)
    dtype = cfg.cache_config.cache_dtype
    grouped_reason = (
        "kv_dtype"
        if dtype != "fp8_e4m3"
        else (
            "user_override"
            if not envs.VLLM_FLASH_V100_E4M3_GROUPED_FP32
            else (
                None
                if native["grouped_fp32"]
                else "operator_missing:grouped_e4m3_fp32_revision4"
            )
        )
    )
    paths["e4m3_grouped_fp32"] = _row(
        grouped_reason,
        kv_cache_dtype=dtype,
        switches={
            "VLLM_FLASH_V100_E4M3_GROUPED_FP32": envs.VLLM_FLASH_V100_E4M3_GROUPED_FP32
        },
    )
    long_reason = grouped_reason or (
        "operator_missing:sm70_grouped_long_fwd"
        if not native["long_operator"]
        else (
            "user_override"
            if not native["long_enabled"]
            else (None if native["page_supported"] else "page_size")
        )
    )
    paths["long_context"] = _row(long_reason, block_size=page_size)
    paths["scalar_tail"] = _row(
        long_reason
        or (
            "user_override"
            if (
                not envs.VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS
                or envs.VLLM_FLASH_V100_DECODE_PARTITION_SIZE
            )
            else (
                "page_size"
                if page_size < 1024
                else (
                    None
                    if native["scalar"]
                    else "operator_missing:sm70_scalar_attention_fwd"
                )
            )
        ),
        block_size=page_size,
        switches={
            "VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS": envs.VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS,
            "VLLM_FLASH_V100_DECODE_PARTITION_SIZE": (
                envs.VLLM_FLASH_V100_DECODE_PARTITION_SIZE
            ),
        },
    )
    budget = cfg.scheduler_config.max_num_batched_tokens
    paths["q8000_prefill"] = _row(
        "budget<8000"
        if budget < 8000
        else (
            "user_override"
            if envs.VLLM_FLASH_V100_PREFILL_D256_GQA_V37
            else (
                None
                if native["q8000"]
                else "operator_missing:sm70_d256_gqa_architecture_q8192_fwd"
            )
        ),
        max_num_batched_tokens=budget,
        switches={
            "VLLM_FLASH_V100_PREFILL_D256_GQA_V37": (
                envs.VLLM_FLASH_V100_PREFILL_D256_GQA_V37
            )
        },
    )
    compilation = cfg.compilation_config
    graph = (
        not cfg.model_config.enforce_eager
        and compilation.mode not in (None, CompilationMode.NONE)
        and compilation.cudagraph_mode != CUDAGraphMode.NONE
    )
    paths["compile_graph"] = _row(
        None if graph else "user_override",
        mode=getattr(compilation.mode, "name", None),
        cudagraph_mode=getattr(compilation.cudagraph_mode, "name", None),
    )
    from torch._inductor import config as inductor_config

    from vllm.compilation.compiler_interface import is_compile_cache_enabled

    cache_config = compilation.inductor_compile_config
    cache_reason = None
    if envs.VLLM_DISABLE_COMPILE_CACHE:
        cache_reason = "compile_cache_disabled"
    elif cfg.model_config.enforce_eager or compilation.mode in (
        None,
        CompilationMode.NONE,
    ):
        cache_reason = "compilation_disabled"
    elif not is_compile_cache_enabled(cache_config):
        cache_reason = "inductor_cache_disabled"
    paths["compile_cache"] = _row(
        cache_reason,
        mode=getattr(compilation.mode, "name", None),
        switches={
            "VLLM_DISABLE_COMPILE_CACHE": envs.VLLM_DISABLE_COMPILE_CACHE,
            "VLLM_USE_AOT_COMPILE": envs.VLLM_USE_AOT_COMPILE,
            "force_disable_caches": cache_config.get("force_disable_caches", False),
            "torch_force_disable_caches": inductor_config.force_disable_caches,
        },
    )
    return report


def log_and_validate(cfg: VllmConfig) -> dict[str, Any]:
    report = build_report(cfg)
    if getattr(cfg, "is_speculative_draft", False):
        report["scope"] = "internal_draft_config"
        report["expected_acceleration"] = []
    # Diagnostic state must not become part of additional_config's compile hash.
    cfg.sm70_acceleration_report = report
    failures = []
    if report["sm70"] or envs.VLLM_SM70_REQUIRE_PROFILE_ACCELERATION:
        for name in report["expected_acceleration"]:
            row = report["paths"][name]
            if not row["enabled"]:
                failures.append(f"{name}: {row['reason']}")
                logger.warning(
                    "SM70 profile acceleration is disabled: %s (%s). "
                    "Compare final CLI overrides with the release profile; "
                    "for operator_missing, reinstall the complete SM70 wheel.",
                    name,
                    row["reason"],
                )
        logger.info("SM70 acceleration status: %s", report)
    report["expected_failures"] = failures
    if envs.VLLM_SM70_REQUIRE_PROFILE_ACCELERATION and failures:
        raise ValueError(
            "SM70 profile acceleration requirement failed: " + "; ".join(failures)
        )
    return report
