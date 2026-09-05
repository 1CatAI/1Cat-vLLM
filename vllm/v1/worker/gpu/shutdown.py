# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def _clear_loaded_gpu_workspaces() -> None:
    """Clear fork-specific GPU caches without importing unused backends."""
    cleanup_functions = (
        (
            "vllm.v1.attention.backends.flash_attn_v100",
            "clear_flash_attn_v100_workspaces",
        ),
        (
            "vllm.model_executor.layers.quantization.fp8",
            "clear_sm70_fp8_workspaces",
        ),
        (
            "vllm.model_executor.layers.quantization.sm70_turbomind",
            "clear_sm70_turbomind_workspaces",
        ),
        (
            "vllm.model_executor.layers.quantization.nvfp4_sm70_moe",
            "clear_sm70_nvfp4_moe_workspaces",
        ),
    )
    for module_name, function_name in cleanup_functions:
        module = sys.modules.get(module_name)
        if module is not None:
            getattr(module, function_name)()


def free_before_shutdown(vllm_config: VllmConfig) -> None:
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
    from vllm.v1.worker.workspace import reset_workspace_manager

    cache_config = vllm_config.cache_config
    cache_config.num_gpu_blocks = None

    compilation_config = vllm_config.compilation_config
    compilation_config.static_forward_context.clear()

    _ROPE_DICT.clear()
    reset_workspace_manager()
    _clear_loaded_gpu_workspaces()
