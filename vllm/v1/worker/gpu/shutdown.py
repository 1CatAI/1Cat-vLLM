# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def log_loaded_attention_route_summaries(config=None) -> None:
    """Flush counters before multiprocessing workers bypass Python atexit."""
    module = sys.modules.get("vllm.v1.attention.backends.flash_v100.routing")
    if module is not None:
        resources = getattr(config, "_runtime_resources", {})
        module._log_route_summary(resources.get("diagnostics"))


def _clear_loaded_gpu_workspaces(config=None) -> None:
    """Clear fork-specific GPU caches without importing unused backends."""
    cleanup_functions = (
        (
            "vllm.v1.attention.backends.flash_v100.dense_prefill",
            "clear_flash_attn_v100_workspaces",
            (config,),
        ),
        (
            "vllm.model_executor.kernels.linear.scaled_mm.sm70_fp8",
            "clear_sm70_fp8_workspaces",
            (),
        ),
        (
            "vllm.model_executor.layers.quantization.sm70_turbomind",
            "clear_sm70_turbomind_workspaces",
            (),
        ),
        (
            "vllm.model_executor.layers.quantization.nvfp4_sm70_moe",
            "clear_sm70_nvfp4_moe_workspaces",
            (),
        ),
        (
            "vllm.model_executor.layers.quantization.utils.sm70_nvfp4_native",
            "clear_sm70_nvfp4_native_workspaces",
            (),
        ),
    )
    for module_name, function_name, arguments in cleanup_functions:
        module = sys.modules.get(module_name)
        if module is not None:
            getattr(module, function_name)(*arguments)


def free_before_shutdown(vllm_config: VllmConfig) -> None:
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
    from vllm.runtime_resources import release_runtime_resources
    from vllm.v1.worker.workspace import reset_workspace_manager

    cache_config = vllm_config.cache_config
    cache_config.num_gpu_blocks = None

    compilation_config = vllm_config.compilation_config
    compilation_config.static_forward_context.clear()

    _ROPE_DICT.clear()
    reset_workspace_manager()
    _clear_loaded_gpu_workspaces(vllm_config)
    release_runtime_resources(vllm_config)
