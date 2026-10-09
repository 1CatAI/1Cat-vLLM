# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Platform default checkpoints; model qualification is owned by adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ParallelConfig, VllmConfig

import os

import torch

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.logger import init_logger

logger = init_logger(__name__)

_SM70_NOMTP_CUDAGRAPH_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32)


_SM70_MTP_CUDAGRAPH_REQUEST_SIZES = (1, 2, 3, 4, 6, 8, 12, 16)


_SM70_SPECULATIVE_AUX_CUDAGRAPH_CAPTURE_SIZES = (1, 2, 4, 8, 9, 18)


_SM70_BATCH_GEMM_DEFAULTS = {
    # Common dense-operator policy, independent of model name, checkpoint
    # quantization, speculative method/width and service concurrency. Local
    # operators retain their dtype/layout/shape checks and small-M routes.
    "VLLM_SM70_BATCH_GEMM_LAYOUTS": "1",
    "VLLM_SM70_AWQ_WARMUP_MAX_M": "64",
    "VLLM_SM70_FP8_DENSE_TUNE_MAX_M": "64",
    "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M": "64",
}


def _participating_cuda_device_ids(cfg: VllmConfig) -> tuple[int, ...]:
    """Local device assignment used by UniProc/Multiproc and GPUWorker.

    Visibility is not participation: unused devices must not change engine
    defaults. Ray/custom placement is not known here, so preserve the legacy
    device-zero decision for those executors instead of guessing their ranks.
    """
    from vllm.platforms import current_platform

    if not current_platform.is_cuda() or current_platform.device_count() == 0:
        return ()
    parallel = cfg.parallel_config
    backend = parallel.distributed_executor_backend
    if backend == "external_launcher":
        return (int(os.environ.get("LOCAL_RANK", "0")),)
    if backend not in (None, "uni", "mp") or parallel.data_parallel_backend == "ray":
        return (0,)
    start = 0
    if parallel.world_size == 1:
        device = cfg.device_config.device
        if isinstance(device, torch.device) and device.index is not None:
            start = device.index
    if parallel.nnodes_within_dp == 1:
        dp_rank = parallel.data_parallel_rank_local
        if dp_rank is None:
            dp_rank = parallel.data_parallel_index
        start += (
            dp_rank * parallel.tensor_parallel_size * parallel.pipeline_parallel_size
        )
    return tuple(range(start, start + parallel.local_world_size))


def _any_participating_device_is_capability(
    cfg: VllmConfig, capability: tuple[int, int]
) -> bool:
    from vllm.platforms import current_platform

    return any(
        current_platform.is_device_capability(capability, device_id=device_id)
        for device_id in _participating_cuda_device_ids(cfg)
    )


def _any_participating_device_is_pre_ampere(cfg: VllmConfig) -> bool:
    """Whether any participating CUDA device is Volta or Turing.

    The SM70 Flash-V100 baseline is a pre-Ampere tuning, not a Volta tuning:
    both capabilities take the same kernels, the same fp16 accumulation
    contract and the same compile graph. Gating it on exactly (7, 0) leaves a
    Turing-only deployment unconfigured, and that does not merely run slower.
    """
    return _any_participating_device_is_capability(
        cfg, (7, 0)
    ) or _any_participating_device_is_capability(cfg, (7, 5))


def _apply_sm70_batch_gemm_defaults(*, is_sm70: bool, defaults) -> tuple[str, ...]:
    """Enable shared SM70 batch operators without overriding explicit settings."""
    if not is_sm70:
        return ()
    applied = []
    for env_name, env_value in _SM70_BATCH_GEMM_DEFAULTS.items():
        if env_name not in defaults:
            defaults[env_name] = env_value
            applied.append(env_name)
    return tuple(applied)


def checkpoint_kv_quant_allowed(cfg: VllmConfig) -> bool:
    """May the checkpoint's own metadata select a quantized KV cache here?

    A checkpoint that declares ``kv_cache_quant_algo`` or ``kv_cache_scheme``
    describes how its weights were produced. With ``--kv-cache-dtype auto``
    vLLM reads that as permission to also store the KV cache in FP8. On
    Volta and Turing there is no FP8 hardware: the cache is unpacked in
    software and decode attention loses its tensor-core route (measured on
    4x V100 with Qwen3.8-27B: +4.82 ms per decode round, 4.5x the cost of
    the FP8 weights the checkpoint ships with), and on Turing the FP8 cast
    is not compiled at all. So the directive is honored only when every
    participating device is Ampere or newer. An explicit ``--kv-cache-dtype``
    never reaches this policy.
    """
    return not _any_participating_device_is_pre_ampere(cfg)


def _apply_sm70_qwen38_disk_ple_defaults(
    parallel_config: ParallelConfig, *, defaults
) -> None:
    """Keep PLE disk-backed and complete its late-bound parallel config."""
    defaults["VLLM_SM70_QWEN38_HYBRID_PLE"] = "0"
    defaults["VLLM_PLE_CPU_OFFLOAD"] = "1"
    defaults["VLLM_PLE_DISK_OFFLOAD"] = "1"
    # ParallelConfig is validated before these model-aware defaults are
    # applied, so initialize the endpoint that its validator would have
    # created for an explicit PLE configuration.
    parallel_config.ensure_ple_offload_ipc_path()


def _ple_disk_cascade_cuda_supported() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_cuda()


def _qwen4exp_ple_cascade_requested(cfg: VllmConfig, *, defaults) -> bool:
    """Resolve disk tiers from storage, dtype and worker topology capabilities."""
    policy = cfg.kernel_config
    policy.ple_disk_cascade_active = False
    reason = None
    model = cfg.model_config
    text = getattr(model, "hf_text_config", None)
    layers = getattr(text, "ple_layer_ids", None)
    if not policy.ple_disk_cascade:
        reason = "disabled by KernelConfig"
    elif model is None or not layers:
        reason = "no PLE layers"
    elif not _ple_disk_cascade_cuda_supported():
        reason = "requires CUDA resident tiers"
    elif model.dtype != torch.float16:
        reason = "requires FP16 embedding output"
    elif (
        defaults.value("VLLM_SM70_QWEN38_HYBRID_PLE")
        or defaults.value("VLLM_PLE_DISK_OFFLOAD")
        or defaults.value("VLLM_PLE_CPU_OFFLOAD")
    ):
        reason = "existing explicit PLE placement takes precedence"
    elif cfg.load_config.load_format not in ("auto", "safetensors", "gguf"):
        reason = "requires file-backed safetensors or GGUF shards"
    elif cfg.parallel_config.prefill_context_parallel_size != 1:
        reason = "PLE worker does not yet support prefill context-parallel groups"
    elif (
        cfg.parallel_config.nnodes != 1
        or cfg.parallel_config.data_parallel_backend != "mp"
        or cfg.parallel_config.data_parallel_size_local
        != cfg.parallel_config.data_parallel_size
        or cfg.parallel_config.use_ubatching
        or cfg.weight_transfer_config is not None
    ):
        reason = "requires local multiprocessing workers without DBO or weight transfer"
    else:
        from vllm.model_executor.models.runtime_defaults import ple_storage_rejection

        reason = ple_storage_rejection(cfg)
    policy.ple_disk_cascade_reason = reason
    policy.ple_disk_cascade_active = reason is None
    return policy.ple_disk_cascade_active


def _apply_qwen4exp_ple_cascade_defaults(parallel_config: ParallelConfig) -> None:
    """Prepare the worker endpoint without changing process environment."""
    parallel_config.ensure_ple_offload_ipc_path()


def _sm70_nomtp_cudagraph_capture_sizes(max_num_seqs: int) -> list[int]:
    # B32 is the largest concurrency with end-to-end SM70 graph validation.
    # Keep larger scheduler capacities usable through the regular piecewise
    # path without forcing an unvalidated, memory-heavy full-graph capture.
    max_graph_reqs = min(max(int(max_num_seqs), 1), 32)
    capture_sizes = {
        size for size in _SM70_NOMTP_CUDAGRAPH_CAPTURE_SIZES if size <= max_graph_reqs
    }
    capture_sizes.update((1, 2, max_graph_reqs))
    return sorted(capture_sizes)


def _sm70_max_cudagraph_capture_size(
    capture_sizes: list[int], max_num_batched_tokens: int
) -> int:
    # The generic sizing later drops capture sizes above max_num_batched_tokens.
    # A cap above the largest remaining size would then read as a user setting
    # that contradicts the capture sizes and fail config validation.
    return max(size for size in capture_sizes if size <= max_num_batched_tokens)


def _sm70_mtp_cudagraph_capture_sizes(
    max_num_seqs: int,
    decode_query_len: int,
) -> list[int]:
    """Return exact SM70 MTP verifier token shapes for production requests."""
    max_graph_reqs = min(max(int(max_num_seqs), 1), 16)
    request_sizes = {
        size for size in _SM70_MTP_CUDAGRAPH_REQUEST_SIZES if size <= max_graph_reqs
    }
    request_sizes.add(max_graph_reqs)
    return [decode_query_len * size for size in sorted(request_sizes)]


def _sm70_speculative_cudagraph_capture_sizes(
    max_num_seqs: int,
    decode_query_len: int,
) -> list[int]:
    """Return bounded auxiliary and verifier shapes without a TP contract."""
    # DFlash verification has the same B32 attention/layout coverage as
    # ordinary decode. A 16-request cap leaves C32 outside CUDA Graph replay.
    max_graph_reqs = min(max(int(max_num_seqs), 1), 32)
    request_sizes = {
        size for size in _SM70_MTP_CUDAGRAPH_REQUEST_SIZES if size <= max_graph_reqs
    }
    request_sizes.add(max_graph_reqs)
    verifier_sizes = [decode_query_len * size for size in request_sizes]
    return sorted(
        set(_SM70_SPECULATIVE_AUX_CUDAGRAPH_CAPTURE_SIZES) | set(verifier_sizes)
    )


def apply_runtime_policy_defaults(self):
    from vllm.config.policy_defaults import PolicyDefaults
    from vllm.model_executor.models.runtime_defaults import (
        _is_sm70_dflash2_verifier_contract,
        _is_sm70_qwen38_decode_compile_contract,
    )
    from vllm.platforms import current_platform

    defaults = PolicyDefaults(self)
    for env_name in _apply_sm70_batch_gemm_defaults(
        is_sm70=(
            current_platform.is_cuda()
            and _any_participating_device_is_capability(self, (7, 0))
        ),
        defaults=defaults,
    ):
        logger.info_once(
            "Auto-setting %s=%s for SM70 batch GEMM. "
            "Local operators select compatible layouts and shapes. "
            "Set it explicitly to override.",
            env_name,
            defaults[env_name],
        )

    sm70_glm5_dflash_tp8_pp1_verifier = self.apply_model_runtime_defaults(
        defaults,
        "collectives",
        is_sm70=(
            current_platform.is_cuda()
            and _any_participating_device_is_capability(self, (7, 0))
        ),
    )

    if (
        self.model_config is not None
        and self.scheduler_config.enable_chunked_prefill
        and self.model_config.dtype == torch.float32
        and current_platform.get_device_capability() == (7, 5)
    ):
        logger.warning_once(
            "Turing devices tensor cores do not support float32 matmul. "
            "To workaround this limitation, vLLM will set 'ieee' input "
            "precision for chunked prefill triton kernels."
        )

    if defaults.value("VLLM_SM70_USE_BREAKABLE_CUDAGRAPH"):
        if current_platform.is_cuda() and _any_participating_device_is_capability(
            self, (7, 0)
        ):
            if "VLLM_USE_BREAKABLE_CUDAGRAPH" not in defaults:
                defaults["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
                logger.info_once(
                    "Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1 because "
                    "VLLM_SM70_USE_BREAKABLE_CUDAGRAPH=1 was requested on "
                    "SM70/V100. Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to opt "
                    "out."
                )
            elif defaults.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "0":
                logger.warning_once(
                    "VLLM_SM70_USE_BREAKABLE_CUDAGRAPH=1 was requested on "
                    "SM70/V100, but explicit VLLM_USE_BREAKABLE_CUDAGRAPH=0 "
                    "takes precedence."
                )
        else:
            logger.warning_once(
                "Ignoring VLLM_SM70_USE_BREAKABLE_CUDAGRAPH=1 because the "
                "current platform is not SM70/V100."
            )

    if self.model_config is not None and self.model_config.enforce_eager:
        logger.warning(
            "Enforce eager set, disabling torch.compile and CUDAGraphs. "
            "This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none"
        )
        self.compilation_config.mode = CompilationMode.NONE
        self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    if defaults.get("TORCH_COMPILE_DISABLE") == "1":
        logger.warning(
            "TORCH_COMPILE_DISABLE is set, disabling torch.compile. "
            "This is equivalent to setting -cc.mode=none"
        )
        self.compilation_config.mode = CompilationMode.NONE

    self.apply_model_runtime_defaults(defaults, "breakable", is_sm70=False)

    if defaults.value("VLLM_USE_BREAKABLE_CUDAGRAPH"):
        logger.warning_once(
            "VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's "
            "torch.compile pipeline. Equivalent to -cc.mode=none."
        )
        self.compilation_config.mode = CompilationMode.NONE

    sm70_compile_disabled_by_user = (
        (self.model_config is not None and self.model_config.enforce_eager)
        or defaults.get("TORCH_COMPILE_DISABLE") == "1"
        or defaults.value("VLLM_USE_BREAKABLE_CUDAGRAPH")
    )
    sm70_no_compile_decode_graph_requested = defaults.value(
        "VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE"
    )

    attention_backend = self.attention_config.backend
    attention_backend_name = getattr(attention_backend, "name", attention_backend)
    sm70_flash_v100_backend = attention_backend is None or attention_backend_name in (
        "FLASH_ATTN_V100",
        "FLASHINFER_SM70",
    )
    sm70_flash_v100_baseline = (
        current_platform.is_cuda()
        and _any_participating_device_is_pre_ampere(self)
        and defaults.value("VLLM_SM70_FLASH_ATTN_V100")
        and sm70_flash_v100_backend
    )
    if sm70_flash_v100_baseline:
        if (
            self.model_config is not None
            and self.model_config.multimodal_config is not None
            and not self.model_config.multimodal_config.language_model_only
        ):
            from vllm.config.multimodal import ImageDummyOptions, VideoDummyOptions

            limit_per_prompt = self.model_config.multimodal_config.limit_per_prompt
            if not limit_per_prompt:
                limit_per_prompt.update(
                    {
                        "image": ImageDummyOptions(count=1, width=None, height=None),
                        "video": VideoDummyOptions(
                            count=0,
                            num_frames=None,
                            width=None,
                            height=None,
                        ),
                    }
                )
                logger.info_once(
                    "Using SM70 Flash-V100 multimodal default: image=1, "
                    "video=0. Set --limit-mm-per-prompt to override."
                )
            elif "video" not in limit_per_prompt:
                limit_per_prompt["video"] = VideoDummyOptions(
                    count=0,
                    num_frames=None,
                    width=None,
                    height=None,
                )
                logger.info_once(
                    "Using SM70 Flash-V100 multimodal default: video=0 "
                    "for partial --limit-mm-per-prompt override."
                )
        self.apply_model_runtime_defaults(defaults, "prefill", is_sm70=True)
        self.kernel_config.gdn.apply_platform_defaults()
        sm70_baseline_env_defaults = {
            "VLLM_SM70_GEMMA_RMS_NORM_COMPILE_NATIVE": "1",
        }
        if (
            not sm70_compile_disabled_by_user
            and not sm70_no_compile_decode_graph_requested
        ):
            sm70_baseline_env_defaults["VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH"] = "1"
        elif "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH" not in defaults:
            logger.info_once(
                "Not auto-setting "
                "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1 for the "
                "SM70 Flash-V100 baseline because no-compile decode graph "
                "is requested or compile was explicitly disabled."
            )
        for env_name, env_value in sm70_baseline_env_defaults.items():
            if env_name not in defaults:
                defaults[env_name] = env_value
                logger.info_once(
                    "Auto-setting %s=%s for the SM70 Flash-V100 "
                    "baseline. Set it explicitly to override.",
                    env_name,
                    env_value,
                )
        if (
            not sm70_compile_disabled_by_user
            and not sm70_no_compile_decode_graph_requested
            and defaults.value("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH")
        ):
            for env_name in self.apply_model_runtime_defaults(
                defaults,
                "projections",
                is_sm70=all(
                    current_platform.is_device_capability((7, 0), device_id=i)
                    for i in _participating_cuda_device_ids(self)
                ),
            ):
                logger.info_once(
                    "Auto-setting %s=1 for shape-checked SM70 Qwen4Exp "
                    "FP16 decode operators. Set it explicitly to override.",
                    env_name,
                )
        if (
            _is_sm70_qwen38_decode_compile_contract(
                self.model_config,
                self.speculative_config,
                self.parallel_config,
            )
            and defaults.value("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH")
            and (
                defaults.value("VLLM_SM70_QWEN38_FP16_GEMV")
                or defaults.value("VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16")
                or defaults.value("VLLM_SM70_QWEN38_FUSED_HC_FP16")
            )
            and "VLLM_SM70_QWEN38_DUAL_COMPILE" not in defaults
        ):
            defaults["VLLM_SM70_QWEN38_DUAL_COMPILE"] = "1"
            logger.info_once(
                "Auto-enabling the SM70 Qwen3.8 dual-compile lane: "
                "large prefill and FULL decode graphs share one model."
            )
        if (
            _is_sm70_qwen38_decode_compile_contract(
                self.model_config,
                self.speculative_config,
                self.parallel_config,
            )
            and defaults.value("VLLM_SM70_QWEN38_DUAL_COMPILE")
            # The disk offload worker needs local multiprocessing endpoints.
            # Independently admitted projection operators retain their guards.
            and self.parallel_config.pipeline_parallel_size == 1
            and self.parallel_config.data_parallel_backend == "mp"
            and self.parallel_config.data_parallel_size_local
            == self.parallel_config.data_parallel_size
            and not any(
                name in defaults
                for name in (
                    "VLLM_SM70_QWEN38_HYBRID_PLE",
                    "VLLM_PLE_CPU_OFFLOAD",
                    "VLLM_PLE_DISK_OFFLOAD",
                )
            )
        ):
            _apply_sm70_qwen38_disk_ple_defaults(
                self.parallel_config, defaults=defaults
            )
            logger.info_once(
                "Auto-enabling disk-mmap PLE for the SM70 Qwen3.8 "
                "dual-compile lane: bounded result staging, no resident table."
            )
    if self.speculative_config is not None:
        policy = self.speculative_config.sm70_dflash2
        policy.resolve(
            qualified=(
                current_platform.is_cuda()
                and _any_participating_device_is_capability(self, (7, 0))
                and _is_sm70_dflash2_verifier_contract(
                    self.model_config, self.speculative_config, self.parallel_config
                )
            )
        )
        if (
            self.model_config is not None
            and self.model_config.quantization == "fp8"
            and self.kernel_config.sm70_fp8.qpn8 is None
            and (policy.qualified or "target_fp8_qpn8" in policy.explicit_fields)
        ):
            self.kernel_config.sm70_fp8.qpn8 = policy.target_fp8_qpn8
    if self.model_config is not None and self.model_config.quantization == "fp8":
        # Resolve after the per-engine verifier defaults.
        self.kernel_config.sm70_fp8.resolve()

    sm70_flash_0dot3_compile_graph = defaults.value(
        "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH"
    )
    sm70_flash_no_compile_graph = (
        defaults.value("VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE")
        and not sm70_flash_0dot3_compile_graph
    )
    sm70_flash_no_compile_graph_explicit = (
        "VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE" in defaults
    )
    if sm70_flash_0dot3_compile_graph:
        if sm70_compile_disabled_by_user:
            logger.warning_once(
                "Ignoring VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1 "
                "because enforce_eager, TORCH_COMPILE_DISABLE, or "
                "VLLM_USE_BREAKABLE_CUDAGRAPH explicitly disables the "
                "compile path."
            )
        elif (
            current_platform.is_cuda()
            and _any_participating_device_is_capability(self, (7, 0))
            and defaults.value("VLLM_SM70_FLASH_ATTN_V100")
        ):
            # None means unspecified; explicit modes take precedence.
            if self.compilation_config.mode is None:
                self.compilation_config.mode = CompilationMode.VLLM_COMPILE
            if self.compilation_config.cudagraph_mode is None:
                self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_AND_PIECEWISE
                )
            if self.compilation_config.cudagraph_capture_sizes is None:
                cudagraph_capture_sizes = _sm70_nomtp_cudagraph_capture_sizes(
                    self.scheduler_config.max_num_seqs
                )
                if (
                    self.speculative_config is not None
                    and self.speculative_config.num_speculative_tokens
                ):
                    decode_query_len = (
                        self.speculative_config.num_speculative_state_tokens() + 1
                    )
                    smallq_env = "VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q"
                    if smallq_env not in defaults and decode_query_len > defaults.value(
                        "VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q"
                    ):
                        defaults[smallq_env] = str(decode_query_len)
                        logger.info_once(
                            "Auto-setting %s=%s so SM70 Flash-V100 "
                            "speculative verifier graph capture uses the "
                            "graph-safe small-query decode branch.",
                            smallq_env,
                            decode_query_len,
                        )
                    if (
                        defaults.value("VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS")
                        and self.speculative_config.method == "mtp"
                    ):
                        cudagraph_capture_sizes = _sm70_mtp_cudagraph_capture_sizes(
                            self.scheduler_config.max_num_seqs,
                            decode_query_len,
                        )
                        logger.info_once(
                            "Using split SM70 MTP verifier cudagraph token "
                            "shapes %s for Flash-V100 compile graph.",
                            tuple(cudagraph_capture_sizes),
                        )
                    else:
                        cudagraph_capture_sizes = (
                            _sm70_speculative_cudagraph_capture_sizes(
                                self.scheduler_config.max_num_seqs,
                                decode_query_len,
                            )
                        )
                        logger.info_once(
                            "Using bounded SM70 speculative cudagraph token "
                            "shapes %s for Flash-V100 compile graph.",
                            tuple(cudagraph_capture_sizes),
                        )
                elif cudagraph_capture_sizes != [1, 2]:
                    logger.info_once(
                        "Using SM70 no-MTP decode cudagraph request shapes %s.",
                        tuple(cudagraph_capture_sizes),
                    )
                self.compilation_config.cudagraph_capture_sizes = (
                    cudagraph_capture_sizes
                )
            if self.compilation_config.max_cudagraph_capture_size is None:
                self.compilation_config.max_cudagraph_capture_size = (
                    _sm70_max_cudagraph_capture_size(
                        self.compilation_config.cudagraph_capture_sizes,
                        self.scheduler_config.max_num_batched_tokens,
                    )
                )
            if self.compilation_config.use_inductor_graph_partition is None:
                self.compilation_config.use_inductor_graph_partition = False
            self.kernel_config.ir_op_priority.rms_norm = [
                "vllm_c",
                "native",
            ]
            self.kernel_config.ir_op_priority.fused_add_rms_norm = [
                "vllm_c",
                "native",
            ]
            logger.info_once(
                "Using vllm_c RMSNorm priority for SM70 Flash-V100 "
                "0.0.3 compile graph quality parity."
            )
            if defaults.value("VLLM_SM70_FLASH_V100_0DOT3_ELIMINATE_NOOPS"):
                self.compilation_config.pass_config.eliminate_noops = True
                logger.info_once(
                    "Using eliminate_noops=True for SM70 Flash-V100 "
                    "0.0.3 compile graph parity."
                )
            if "VLLM_MQ_BROADCASTER_MAX_CHUNKS" not in defaults:
                defaults["VLLM_MQ_BROADCASTER_MAX_CHUNKS"] = "64"
                logger.info_once(
                    "Auto-setting VLLM_MQ_BROADCASTER_MAX_CHUNKS=64 for "
                    "SM70 Flash-V100 0.0.3 compile graph startup."
                )
            if (
                not self.use_v2_model_runner
                and "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" not in defaults
            ):
                defaults["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
                logger.info_once(
                    "Disabling the legacy SM70 graph memory profiler; "
                    "V2 budgets its graph reserve before KV allocation."
                )
            if "VLLM_SM70_LM_HEAD_TOP1" not in defaults:
                defaults["VLLM_SM70_LM_HEAD_TOP1"] = "0"
                logger.info_once(
                    "Auto-setting VLLM_SM70_LM_HEAD_TOP1=0 for SM70 "
                    "Flash-V100 0.0.3 compile graph quality parity; "
                    "greedy decode keeps the local-logits top1 shortcut."
                )
            sm70_dflash2_graph_cache = self.apply_model_runtime_defaults(
                defaults, "graph_cache", is_sm70=True
            )
            if "VLLM_USE_AOT_COMPILE" not in defaults:
                defaults["VLLM_USE_AOT_COMPILE"] = "1"
                logger.info_once(
                    "Auto-setting VLLM_USE_AOT_COMPILE=1 for SM70 "
                    "Flash-V100 0.0.3 compile graph quality parity."
                )
            elif defaults.get("VLLM_USE_AOT_COMPILE") == "0":
                if sm70_dflash2_graph_cache:
                    logger.info_once(
                        "Using SM70 E4M3 DFlash2 compiled graph caches "
                        "without AOT FX-graph reload; CUDA graphs remain enabled."
                    )
                elif sm70_glm5_dflash_tp8_pp1_verifier:
                    logger.info_once(
                        "Using the quality-qualified regular torch.compile "
                        "path for SM70 GLM-5.3 DFlash2 TP8/PP1."
                    )
                else:
                    logger.warning_once(
                        "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1 with "
                        "explicit VLLM_USE_AOT_COMPILE=0 is a diagnostic-only "
                        "configuration: regular torch.compile reproduced "
                        "deterministic greedy token drift."
                    )
            elif sm70_dflash2_graph_cache and defaults.value("VLLM_USE_AOT_COMPILE"):
                logger.warning_once(
                    "Explicit VLLM_USE_AOT_COMPILE=1 selects AOT cache reload, "
                    "which failed complete-output parity for the SM70 E4M3 "
                    "DFlash2 release contract. Remove this override to reuse "
                    "compiled graph caches without AOT FX-graph reload."
                )
            self.compilation_config.inductor_compile_config["combo_kernels"] = True
            self.compilation_config.inductor_compile_config[
                "benchmark_combo_kernel"
            ] = True
            logger.info_once(
                "Using combo_kernels=True and benchmark_combo_kernel=True "
                "for SM70 Flash-V100 0.0.3 compile graph quality parity."
            )
            logger.info_once(
                "Using SM70 Flash-V100 0.0.3 compile CUDA graph policy: "
                "mode=%s, cudagraph_mode=%s, "
                "capture_sizes=%s.",
                self.compilation_config.mode.name,
                self.compilation_config.cudagraph_mode.name,
                tuple(self.compilation_config.cudagraph_capture_sizes),
            )
        else:
            logger.warning_once(
                "Ignoring VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1 "
                "because the current platform is not SM70 CUDA or "
                "VLLM_SM70_FLASH_ATTN_V100 is disabled."
            )
    if sm70_flash_no_compile_graph:
        if self.model_config is not None and self.model_config.enforce_eager:
            logger.warning_once(
                "Ignoring VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE=1 "
                "because enforce_eager disables CUDA graphs."
            )
        elif (
            current_platform.is_cuda()
            and _any_participating_device_is_capability(self, (7, 0))
            and defaults.value("VLLM_SM70_FLASH_ATTN_V100")
        ):
            capture_size = max(
                1,
                defaults.value("VLLM_SM70_FLASH_V100_DECODE_GRAPH_CAPTURE_SIZE"),
            )
            if self.compilation_config.mode is None:
                self.compilation_config.mode = CompilationMode.NONE
            if self.compilation_config.cudagraph_mode is None:
                self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
            if self.compilation_config.cudagraph_capture_sizes is None:
                self.compilation_config.cudagraph_capture_sizes = list(
                    range(1, capture_size + 1)
                )
            if self.compilation_config.max_cudagraph_capture_size is None:
                self.compilation_config.max_cudagraph_capture_size = (
                    _sm70_max_cudagraph_capture_size(
                        self.compilation_config.cudagraph_capture_sizes,
                        self.scheduler_config.max_num_batched_tokens,
                    )
                )
            logger.info_once(
                "Using SM70 Flash-V100 no-compile decode CUDA graph "
                "policy: mode=%s, cudagraph_mode=%s, "
                "capture_size=%d.",
                self.compilation_config.mode.name,
                self.compilation_config.cudagraph_mode.name,
                capture_size,
            )
        else:
            if sm70_flash_no_compile_graph_explicit:
                logger.warning_once(
                    "Ignoring "
                    "VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE=1 "
                    "because the current platform is not SM70 CUDA or "
                    "VLLM_SM70_FLASH_ATTN_V100 is disabled."
                )

    defaults.finish()
    if defaults.value("VLLM_BATCH_INVARIANT"):
        self.parallel_config.communication.symm_mem = False
        self.parallel_config.communication.sources["symm_mem"] = (
            "safety:batch_invariant"
        )
        self.compilation_config.runtime.aot_compile = False
        self.compilation_config.runtime.sources["aot_compile"] = (
            "safety:batch_invariant"
        )
        if self.compilation_config.runtime.sources.get("mega_aot") == "default":
            self.compilation_config.runtime.mega_aot = False
    if self.offload_config.ple.cpu or self.offload_config.ple.hybrid:
        self.parallel_config.ensure_ple_offload_ipc_path()

    return defaults
