# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audited input domains at retained configuration boundaries.

Entries describe consumers; they do not remove reads from inventory or exempt
files from checks. New dynamic readers must declare their domain and lifecycle.
The associated contract tests prove that engine calls supply resolved policies.
"""

# (file, lexical scope) -> (lifecycle, domain / admission evidence).
DYNAMIC_DOMAINS = {
    ("vllm/config/diagnostic_dump.py", "TensorDumpConfig.resolve"): (
        "initialization",
        "DUMP_BINDINGS or the explicit channel bindings",
    ),
    (
        "vllm/config/diagnostic_sampling.py",
        "SamplingDiagnosticsConfig.__post_init__.read",
    ): ("initialization", "SamplingDiagnosticsConfig.aliases"),
    ("vllm/config/execution_policy.py", "read_execution_legacy"): (
        "initialization",
        "POLICY_OWNERS aliases and EXTRA_BINDINGS",
    ),
    ("vllm/config/execution_policy.py", "GraphPolicy.resolve"): (
        "initialization",
        "GraphPolicy.aliases",
    ),
    ("vllm/config/flash_v100.py", "CapturedFlashOptions.resolve"): (
        "initialization",
        "bindings and legacy_aliases on the concrete Flash policy",
    ),
    ("vllm/config/gdn.py", "GdnConfig.resolve"): (
        "initialization",
        "GDN_LEGACY_FIELDS, GDN_TEXT_FLAGS and GDN_FALLBACK_ALIASES",
    ),
    ("vllm/config/gdn.py", "GdnConfig.apply_platform_defaults"): (
        "initialization",
        "GDN_LEGACY_FIELDS; original platform checkpoint",
    ),
    ("vllm/config/gdn.py", "GdnProfileConfig.resolve"): (
        "initialization",
        "the max_logs / max_per_stage tuples, only when enabled",
    ),
    ("vllm/config/gdn_projection.py", "legacy_projection_value"): (
        "initialization_or_standalone",
        "GdnProjectionConfig.aliases; prepared engine providers pass their policy",
    ),
    ("vllm/config/gdn_schedule.py", "GdnScheduleConfig.resolve"): (
        "initialization",
        "GDN_SCHEDULE_FIELDS",
    ),
    ("vllm/config/gdn_state.py", "GdnStateConfig.resolve"): (
        "initialization",
        "GDN_STATE_FIELDS",
    ),
    ("vllm/config/gdn_state.py", "GdnStateTraceConfig.resolve"): (
        "initialization",
        "explicit state-diagnostic field/name tuples",
    ),
    ("vllm/config/legacy_inputs.py", "LegacyInputs.capture"): (
        "initialization",
        (
            "caller-supplied registered provider alias declarations; "
            "captured is serialized"
        ),
    ),
    ("vllm/config/policy_defaults.py", "PolicyDefaults.value"): (
        "initialization",
        "POLICY_OWNERS plus EXTRA_BINDINGS and DFlash2 fields",
    ),
    ("vllm/config/sm70_dflash2.py", "sm70_dflash2_enabled"): (
        "standalone_compatibility",
        "SM70_DFLASH2_LEGACY_FIELDS; resolved engine owner returns before legacy read",
    ),
    ("vllm/config/sm70_dflash2.py", "Sm70DFlash2Config.resolve"): (
        "initialization",
        "SM70_DFLASH2_LEGACY_FIELDS and model-qualified defaults",
    ),
    ("vllm/config/sm70_dflash2.py", "DFlashDiagnosticsConfig.__post_init__.read"): (
        "initialization",
        "DFlashDiagnosticsConfig.aliases",
    ),
    ("vllm/config/sm70_dflash2.py", "proposer_diagnostic_flag"): (
        "standalone_compatibility",
        "DFlash/Sampling diagnostic aliases; engine trace is supplied",
    ),
    ("vllm/config/sm70_native.py", "Sm70NativeConfig.capture_inputs"): (
        "initialization",
        "NATIVE_FIELDS append-only vector ABI",
    ),
    ("vllm/config/sm70_native.py", "CollectiveNativeConfig.resolve"): (
        "initialization",
        "CollectiveNativeConfig.aliases",
    ),
    ("vllm/config/sm70_runtime.py", "SpecDecodeTraceConfig.__post_init__"): (
        "initialization",
        "SpecDecodeTraceConfig.legacy_fields",
    ),
    ("vllm/config/sm70_runtime.py", "RuntimeTraceConfig.__post_init__.read_flag"): (
        "initialization",
        "RuntimeTraceConfig.layer_aliases",
    ),
    ("vllm/config/sm70_sparse.py", "read_sparse_legacy"): (
        "initialization",
        "Sm70SparseConfig.aliases",
    ),
    (
        "vllm/config/speculative_sampling.py",
        "SpeculativeSamplingPolicy.resolve_fields.reader",
    ): ("initialization", "SpeculativeSamplingPolicy.aliases"),
    (
        "vllm/config/speculative_sampling.py",
        "SpeculativeSamplingPolicy.resolve_fields",
    ): ("initialization", "pending draft_apply_top_p; two historical parser dialects"),
    ("vllm/config/turboquant_runtime.py", "read_turboquant_legacy"): (
        "initialization",
        "TurboQuantRuntimePolicy.aliases",
    ),
    ("vllm/config/utils.py", "resolve_legacy_fields"): (
        "initialization",
        "caller-supplied field-to-registered-name declaration",
    ),
    ("vllm/config/utils.py", "get_from_deprecated_env_if_set"): (
        "upstream_configuration",
        "registered deprecated names passed by upstream config constructors",
    ),
    ("vllm/config/utils.py", "set_from_deprecated_env_if_set"): (
        "upstream_configuration",
        "registered deprecated names passed by upstream config constructors",
    ),
    ("vllm/envs_metadata.py", "warn_deprecated_once"): (
        "process_notice",
        "explicit registered deprecated names; process-wide once/name ownership",
    ),
    ("vllm/envs_metadata.py", "EnvVar.warn_if_deprecated"): (
        "process_notice",
        "name bound by bind_env_names, never a policy selector",
    ),
    ("vllm/models/qwen4_exp/common/ple.py", "env_gib_bytes"): (
        "standalone_compatibility",
        (
            "PLE HOST_GIB, HOST_RESERVE_GIB and VRAM_RESERVE_GIB; engine "
            "uses offload_config.ple"
        ),
    ),
    ("vllm/models/qwen4_exp/common/ple.py", "_placement_gib_bytes"): (
        "standalone_compatibility",
        "three PLE budget aliases; configured owner returns before legacy fallback",
    ),
    ("vllm/models/qwen4_exp/nvidia/ops/sm70_qsa_tuning.py", "legacy_qsa_tuning"): (
        "standalone_compatibility",
        "QSA tuning aliases; qsa.py engine uses Sm70SparseConfig",
    ),
    ("vllm/v1/attention/backends/flash_v100/config.py", "registered"): (
        "standalone_compatibility",
        "registered Flash aliases; engine options are bound before execution",
    ),
    ("vllm/v1/attention/backends/flash_v100/config.py", "raw"): (
        "standalone_compatibility",
        "registered Flash aliases; engine options are bound before execution",
    ),
    ("vllm/v1/attention/backends/flash_v100/config.py", "env_is_set"): (
        "standalone_compatibility",
        "registered Flash aliases; engine options are bound before execution",
    ),
    ("csrc/custom_all_reduce_policy.h", "legacy"): (
        "standalone_compatibility",
        "PolicyField vector; active configured scope returns before getenv",
    ),
}


for _scope in ("legacy_policy", "policy_atoi", "policy_exact_one"):
    DYNAMIC_DOMAINS[("csrc/sm70_policy.h", _scope)] = (
        "standalone_compatibility",
        "sm70_policy_fields.inc; active PreparedPolicy returns before getenv",
    )
for _scope in (
    "sm70_marlin_try_get_geometry_env",
    "sm70_marlin_try_get_split_k_env",
    "sm70_marlin_try_get_metadata_env",
    "sm70_marlin_env_is_set",
):
    DYNAMIC_DOMAINS[("csrc/quantization/marlin/sm70_marlin_common.cuh", _scope)] = (
        "standalone_compatibility",
        (
            "SM70_MARLIN_{DENSE,MOE}_{CTA_GEOMETRY,SPLIT_K,METADATA_CACHE}; "
            "configured scope returns prepared overrides first"
        ),
    )
DYNAMIC_DOMAINS[("vllm/config/sm70_runtime.py", "RuntimeTraceConfig.__post_init__")] = (
    "initialization",
    "RuntimeTraceConfig.legacy_layer_aliases source provenance",
)

# These are upstream process/environment transport, introspection or unrelated
# connectors. Their names are dynamic by design and are outside 1Cat strategies.
PROCESS_DOMAINS = {
    "vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py": {
        "_warn_deprecated_env_vars": "MORIIO connector aliases"
    },
    "vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py": {
        "set_p2p_nccl_context": "P2P NCCL setup variables"
    },
    "vllm/entrypoints/serve/instrumentator/server_info.py": {
        "_get_vllm_env_vars": "VLLM environment introspection, no policy dispatch"
    },
    "vllm/platforms/cuda.py": {
        ("NvmlCudaPlatform.device_id_to_physical_device_id"): (
            "platform device_control_env_var"
        )
    },
    "vllm/platforms/interface.py": {
        "Platform.device_id_to_physical_device_id": "platform device_control_env_var"
    },
    "vllm/platforms/rocm.py": {
        "RocmPlatform.device_count": "platform device_control_env_var"
    },
    "vllm/ray/ray_env.py": {
        "get_env_vars_to_copy": "Ray worker startup propagation set"
    },
    "vllm/tracing/otel.py": {
        "get_span_exporter": "OTEL exporter protocol",
        "propagate_trace_to_env": "OTEL span context carrier",
    },
    "vllm/usage/usage_lib.py": {
        "_detect_cloud_provider": "cloud vendor identifiers",
        "UsageMessage._report_usage_once": "usage-statistics fields",
    },
    "vllm/utils/ompmultiprocessing.py": {
        "OMPProcessManager.configure_omp_envs": "OMP thread startup parameters"
    },
    "vllm/utils/system_utils.py": {
        "set_env_var": "upstream process initialization API",
        "update_environment_variables": "upstream process initialization API",
    },
    "vllm/v1/engine/core.py": {
        ("EngineCoreActorMixin._set_cuda_visible_devices"): (
            "Ray actor CUDA device visibility"
        )
    },
    "vllm/v1/engine/utils.py": {
        "set_device_control_env_var": "platform device visibility",
        "CoreEngineActorManager.__init__": "actor startup environment",
        "get_device_indices": "platform device visibility",
    },
    "vllm/v1/executor/ray_executor.py": {
        "RayDistributedExecutor._init_workers_ray": "Ray worker startup propagation set"
    },
    "vllm/v1/executor/ray_executor_v2.py": {
        "RayWorkerProc.initialize_worker": "Ray worker startup propagation set"
    },
    "vllm/v1/executor/vllm_net_devices.py": {
        "set_worker_gpu_nic_mapping": "worker GPU/NIC mapping"
    },
}


def dynamic_domain(site):
    key = site["path"], site["scope"]
    if key in DYNAMIC_DOMAINS:
        lifecycle, domain = DYNAMIC_DOMAINS[key]
        return dict(lifecycle=lifecycle, domain=domain)
    domain = PROCESS_DOMAINS.get(site["path"], {}).get(site["scope"])
    if domain:
        return dict(lifecycle="upstream_process_boundary", domain=domain)
    return None


# Parameter boundaries are explicit, not inferred from a spelling such as DEBUG
# or DDTREE. A dormant declaration is not evidence of algorithm deprecation.
PARAMETER_BOUNDARIES = {
    "VLLM_SM70_DFLASH2_DIRECT_ATTENTION_OUTPUT": (
        "registered_dormant",
        (
            "Registered experiment; no in-tree consumer. Retain compatibility; no "
            "negative algorithm evidence."
        ),
    ),
    "VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER": (
        "historical_notice",
        (
            "Already retired candidate ordering; "
            "docs/design/sm70_dflash2_retired_candidate_order.md"
        ),
    ),
    "VLLM_DFLASH_DISABLE_AUX_OUTPUTS": (
        "deferred_ddtree",
        (
            "GPUModelRunner.__init__: use_dflash_ddtree admission; ordinary DFlash "
            "uses Runner V2."
        ),
    ),
    "VLLM_DFLASH_DUMP_FIRST_PASS": (
        "deferred_ddtree",
        (
            "Legacy DFlashProposer selected only for use_dflash_ddtree; ordinary "
            "DFlash uses Runner V2."
        ),
    ),
}
for _name in (
    "PREFIX_BATCHED_TAIL_QK_ALGO_RUNTIME",
    "PREFIX_TAIL_OVERLAP_MODE",
    "PREFIX_TAIL_OVERLAP_START_BLOCK",
    "PREFIX_TAIL_WAVE_CONTINUOUS_QK",
    "PREFIX_TAIL_WAVE_READY_AFTER",
):
    PARAMETER_BOUNDARIES[_name] = (
        "standalone_benchmark",
        (
            "csrc/attention/sm70_79t/prefill.cu: !PREFIX_TORCH_EXTENSION main; absent "
            "from normal FA2 build."
        ),
    )
for _name in ("TM_LOG_FIRST_RANK_ONLY", "TM_LOG_LEVEL", "TM_SRC_FULL_PATH"):
    PARAMETER_BOUNDARIES[_name] = (
        "process_logging",
        (
            "TurboMind process logger/check formatting; no engine computation policy "
            "or workspace."
        ),
    )
for _name in (
    "VLLM_1CAT_DISABLE_QWEN35_MTP_DEFAULTS",
    "VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS",
    "VLLM_1CAT_ENABLE_QWEN35_MTP_DEFAULTS",
    "VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS",
):
    PARAMETER_BOUNDARIES[_name] = (
        "engine_argument_initialization",
        (
            "EngineArgs._maybe_apply_sm70_mtp_defaults; preserves early CLI/model "
            "eligibility checkpoint."
        ),
    )
for _name in (
    "VLLM_SM70_CUSTOM_AR_LIBRARY",
    "VLLM_SM70_FA2_D256_LIBRARY",
    "VLLM_SM70_FP8_QPN8_LIBRARY",
    "VLLM_SM70_GLM53_FP16_GEMV_LIBRARY",
    "VLLM_SM70_NVFP4_QPN2_PREFILL_LIBRARY",
    "VLLM_SM70_NVFP4_QPN_M1_LIBRARY",
    "VLLM_SM70_QSA_TOPK_LIBRARY",
    "VLLM_SM70_SAMPLER_LIBRARY",
):
    PARAMETER_BOUNDARIES[_name] = (
        "process_library_loading",
        (
            "Optional extension search path; native registrations are process-wide "
            "and bind before execution."
        ),
    )
# Reviewed deferred parameters, not a prefix-based exemption. Shared diagnostic
# aliases such as TARGET_FORWARD_NVTX/PROFILER_STEP have typed owners instead.
for _suffix in (
    "ATTN_COMPACT_BATCH",
    "COMPACT_DRAFTER_CONTEXT",
    "CONV_KERNEL",
    "DEBUG",
    "DISABLE_GDN_FAST_BUILD",
    "DISABLE_GDN_FAST_BUILD_CACHE",
    "ENABLE_GDN_FAST_BUILD",
    "ENABLE_GDN_FAST_BUILD_CACHE",
    "ENABLE_HYBRID_TREE_STATE",
    "ENGINE_PROFILE",
    "FAST_BUILD_DEBUG",
    "FORCE_MAMBA_COMPACT",
    "FUSED_GDN",
    "GDN_FAST_BUILD_TRITON",
    "GDN_SHARED_COMMON",
    "GPU_SAMPLER",
    "LINEAR_GDN",
    "MAMBA_COMPACT_BATCH",
    "PATH_PROBE",
    "PATH_PROBE_LAYER",
    "PATH_PROBE_MAX_REPORTS",
    "PATH_PROBE_NODE_LIMIT",
    "PROFILE",
    "QLA_GDN",
    "SERIAL_GDN",
    "SKIP_MAMBA_COMPACT",
    "STOCHASTIC_TOPK_LOGITS",
    "TRACE_JSONL",
    "TRACE_KV_CACHE_DIFF",
    "TRITON_BRANCH_ATTN",
    "TRITON_BRANCH_ATTN_STRICT",
    "TRITON_SAMPLER",
    "VERIFY_ROW_TRACE",
    "VERIFY_ROW_TRACE_CONTEXT",
    "VERIFY_ROW_TRACE_TOPK",
    "WORKER_PROFILE",
):
    PARAMETER_BOUNDARIES["VLLM_DFLASH_DDTREE_" + _suffix] = (
        "deferred_ddtree",
        (
            "Retained tree-specific consumer; scope and source locations remain in "
            "the inventory. DDTree is outside Phase D."
        ),
    )


def parameter_boundary(name):
    boundary = PARAMETER_BOUNDARIES.get(name)
    if boundary is None:
        return None
    lifecycle, evidence = boundary
    return dict(lifecycle=lifecycle, evidence=evidence)


# The standalone API remains usable without VllmConfig. These exact consumers
# are covered by poisoned-getter/engine-isolation tests; this list is not a
# directory exemption, and newly introduced consumers still fail the check.
COMPATIBILITY_CONSUMERS = {
    "flash-attention-v100/flash_attn_v100/legacy_policy.py": {
        "dynamic_partitions",
        "partition_size",
        "batch_xqa",
        "scalar_fast",
        "share_workspace",
        "dual_cta",
        "padded_smem",
    },
    "vllm/config/diagnostic_sampling.py": {"legacy_rejection_profile"},
    "vllm/config/sm70_dflash2.py": {"dflash2_bf16_emulation"},
    "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py": {
        "_sm70_flashqla_original_prefill_enabled",
        "_sm70_flashqla_indexed_prefill_enabled",
        "_sm70_flashqla_direct_output_enabled",
        "_sm70_flashqla_decode_warmup_enabled",
        "_sm70_assert_standard_core_not_active_spec",
    },
    "vllm/v1/attention/backends/gdn_attn.py": {
        "_sm70_flashqla_original_prefill_enabled",
        "build_gdn_spec_decode_state_contract",
        "select_gdn_state_block_ids",
    },
    "vllm/v1/sample/rejection_sampler.py": {
        "_token_matching_sampling_enabled",
        "_combined_bonus_sampling_enabled",
    },
    "vllm/v1/spec_decode/llm_base_proposer.py": {"_can_use_sparse_topk_draft_proposal"},
    "vllm/model_executor/layers/layernorm.py": {"RMSNormGated.__init__"},
    "vllm/models/qwen4_exp/common/ple.py": {
        "ple_host_budget_bytes",
        "ple_host_reserve_bytes",
        "ple_vram_reserve_bytes",
    },
    "vllm/sm70_decode_trace.py": {"sm70_decode_event_trace_enabled", "_should_log"},
    "csrc/attention/sm70_79t/register.cpp": {"policy_value"},
    "flash-attention-v100/include/flash_v100_policy.h": {"value"},
    "csrc/sm70_turbomind/ops/awq_sm70_gemm.cu": {"glm_mhc_pre_threads"},
}

CONFIG_INITIALIZERS = {
    "VllmConfig.__post_init__",
    "VllmConfig.__post_init__.enable_quant_fp8_custom_op_for_blocked_weights",
    "VllmConfig._set_compile_ranges",
    "RuntimeTraceConfig.__post_init__",
    "FlashV100Diagnostics.resolve",
    "LayerExecutionPolicy.resolve",
    "GdnProfileConfig.resolve",
    "GdnStateTraceConfig.resolve",
    "KernelConfig.resolve_sm70_rmsnorm_gated",
    "resolve_legacy_fields",
}


def consumer_lifecycle(name, site, boundary):
    if site["kind"] == "native_bound":
        return "bound_native_policy"
    if site.get("lifecycle"):
        return site["lifecycle"]
    path, scope = site["path"], site["scope"]
    domain = dynamic_domain(site)
    if domain:
        return domain["lifecycle"]
    if path.startswith("vllm/config/") and scope in CONFIG_INITIALIZERS:
        return "initialization"
    if scope in COMPATIBILITY_CONSUMERS.get(path, ()):
        return "standalone_compatibility"
    if boundary:
        lifecycle = boundary["lifecycle"]
        if (
            lifecycle == "engine_argument_initialization"
            and scope == "EngineArgs._maybe_apply_sm70_mtp_defaults"
        ):
            return lifecycle
        if lifecycle == "process_logging" and scope in ("Logger", "StripSrcPrefix"):
            return lifecycle
        if lifecycle == "process_library_loading" and (
            path == "vllm/_sm70/loader.py"
            or (path, scope)
            in {
                ("vllm/_custom_ops.py", "_maybe_load_sm70_custom_ar_library"),
                (
                    "vllm/v1/attention/backends/flash_v100/ops.py",
                    "get_sm70_splitd_d256_ops",
                ),
                ("vllm/models/qwen4_exp/nvidia/ops/qsa.py", ""),
            }
        ):
            return lifecycle
        if lifecycle == "deferred_ddtree":
            if "ddtree" in scope.lower() or "/spec/" in path or "/ddtree_" in path:
                return lifecycle
            if (path, scope) in {
                ("vllm/v1/core/sched/scheduler.py", "Scheduler.__init__"),
                ("vllm/v1/engine/core.py", "EngineCore.step"),
                ("vllm/v1/engine/core.py", "EngineCore.post_step"),
                ("vllm/v1/worker/gpu_model_runner.py", "GPUModelRunner.__init__"),
                (
                    "vllm/v1/worker/gpu_model_runner.py",
                    "GPUModelRunner._warmup_sm70_aux_kernels",
                ),
                ("vllm/v1/worker/gpu_model_runner.py", "GPUModelRunner._sample"),
                (
                    "vllm/v1/spec_decode/dflash.py",
                    "DFlashProposer._maybe_dump_first_pass",
                ),
            }:
                return lifecycle
    # Metadata profiling is a shared initialized diagnostic; the two tree-only
    # legacy callers retain their independent adapter.
    if (
        name == "VLLM_DFLASH_DDTREE_METADATA_PROFILE"
        and scope == "_dflash_ddtree_metadata_profile_enabled"
    ):
        return "deferred_ddtree"
    return "unclassified"


PARAMETER_BOUNDARIES.update(
    {
        "FLASH_QLA_SM70_PREBUILT_EXTENSION_PATH": (
            "process_library_loading",
            (
                "FlashQLA _load_ext: explicit DSO selection; cached process "
                "registration before provider binding."
            ),
        ),
        "FLASH_QLA_SM70_VERBOSE_BUILD": (
            "process_library_loading",
            "FlashQLA _load_ext/setup bundler: JIT/build log verbosity only.",
        ),
        "FLASH_QLA_SM70_ORIGINAL_FUSED_FWD": (
            "standalone_compatibility",
            (
                "Independent FlashQLA package override. Engine original-TileLang "
                "provider explicitly sets sm70_original=True and bypasses the getter."
            ),
        ),
        "FLASH_QLA_SM70_DDTREE_COLUMN_GROUPS_PER_BLOCK": (
            "deferred_ddtree",
            (
                "Native gdn_decode_mixed_qkv_ddtree_state only; ordinary decode uses "
                "GdnPolicy."
            ),
        ),
    }
)
COMPATIBILITY_CONSUMERS.update(
    {
        "flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py": {"fused_gdr_fwd"},
        "flash_qla/ops/gated_delta_rule/chunk/sm70/csrc/gdn_forward.cu": {
            "get_column_groups_per_block"
        },
    }
)
DYNAMIC_DOMAINS[
    ("flash_qla/ops/gated_delta_rule/chunk/sm70/fused_fwd.py", "_load_ext")
] = (
    "process_library_loading",
    "FlashQLA prebuilt extension path and fallback compiler verbosity",
)
COMPATIBILITY_CONSUMERS["csrc/quantization/marlin/sm70_marlin_common.cuh"] = {
    "sm70_marlin_dense_auto_env_is_set",
}
COMPATIBILITY_CONSUMERS["csrc/moe/marlin_moe_wna16/sm70_marlin_gemm.cuh"] = {
    "sm70_marlin_moe_auto_env_is_set",
}
