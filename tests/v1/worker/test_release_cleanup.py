# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.v1.worker.utils import clear_layer_kv_caches

pytestmark = pytest.mark.cpu_test


class _AttentionLayer(nn.Module):
    def __init__(self, kv_cache):
        super().__init__()
        self.kv_cache = kv_cache
        self.impl = SimpleNamespace(
            _k_scale_cache=torch.ones(1),
            _v_scale_cache=torch.ones(1),
        )


def test_clear_layer_kv_caches_detaches_tensor_and_scale_views():
    tensor_layer = _AttentionLayer(torch.ones(4))
    list_layer = _AttentionLayer([torch.ones(4)])

    clear_layer_kv_caches([tensor_layer, list_layer])

    assert isinstance(tensor_layer.kv_cache, torch.Tensor)
    assert tensor_layer.kv_cache.numel() == 0
    assert list_layer.kv_cache == []
    assert tensor_layer.impl._k_scale_cache is None
    assert tensor_layer.impl._v_scale_cache is None


def test_compile_wrapper_cleanup_is_idempotent():
    wrapper = object.__new__(TorchCompileWithNoGuardsWrapper)
    handle = MagicMock()
    wrapper._bytecode_hook_handle = handle

    wrapper.cleanup()
    wrapper.cleanup()

    handle.remove.assert_called_once_with()
    assert wrapper._bytecode_hook_handle is None


def test_worker_shutdown_flushes_route_summary_once(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100
    from vllm.v1.worker import gpu_worker

    logger = MagicMock()
    counts = {"decode_e4m3_grouped_fp32": 3}
    monkeypatch.setattr(flash_attn_v100, "logger", logger)
    monkeypatch.setattr(flash_attn_v100, "_route_counts", counts)
    monkeypatch.setattr(gpu_worker, "ensure_kv_transfer_shutdown", lambda: None)
    monkeypatch.setattr(gpu_worker, "ensure_ec_transfer_shutdown", lambda: None)
    worker = object.__new__(gpu_worker.Worker)
    worker.profiler = None
    worker._ple_offload_worker_handle = None

    worker.shutdown()
    flash_attn_v100._log_route_summary()  # The later atexit hook is harmless.

    logger.info.assert_called_once_with(
        "FLASH_ATTN_V100 route summary: %s",
        '{"decode_e4m3_grouped_fp32": 3}',
    )
    assert not counts


def test_sm70_workspace_cleanup_releases_all_tensor_owners():
    from vllm.model_executor.kernels.linear.scaled_mm import sm70_fp8

    from vllm.model_executor.layers.quantization import (
        nvfp4_sm70_moe,
        sm70_turbomind,
    )
    from vllm.model_executor.layers.quantization.utils import sm70_nvfp4_native
    from vllm.v1.attention.backends import flash_attn_v100
    from vllm.v1.worker.gpu.shutdown import _clear_loaded_gpu_workspaces

    tensor = torch.ones(1)
    flash_attn_v100._sm70_fa2_cu_seqlens_cache[(0, 0, 0, 0)] = (tensor, tensor)
    flash_attn_v100._fp8_prefill_bridge_workspaces[(0, 0, 0, 0)] = (
        tensor,
        tensor,
        tensor,
    )
    flash_attn_v100._fp8_prefill_bridge_tail_workspaces[(0, 0, tensor.dtype, 0, 0)] = (
        tensor,
        tensor,
    )
    flash_attn_v100._prefill_gather_dense_workspaces[(0, 0, tensor.dtype, 0, 0, 0)] = (
        tensor,
        tensor,
    )
    flash_attn_v100._prefill_dense_splitkv3_workspaces[(0, 0, tensor.dtype)] = (
        tensor,
        tensor,
        tensor,
    )
    sm70_fp8._sm70_fp8_prefill_dense_workspaces[(0, tensor.dtype)] = tensor
    sm70_fp8._sm70_fp8_qpn8_pp2_tp4_workspaces[(0, tensor.dtype)] = tensor
    sm70_turbomind._nvfp4_qpn4_dense_workspaces[(0, tensor.dtype)] = tensor
    nvfp4_sm70_moe._qwen38_raw_scale_workspaces[0] = tensor
    sm70_nvfp4_native._scale_workspaces[0] = tensor
    sm70_nvfp4_native._code_workspaces[(0, 1)] = tensor

    _clear_loaded_gpu_workspaces()

    assert not flash_attn_v100._sm70_fa2_cu_seqlens_cache
    assert not flash_attn_v100._fp8_prefill_bridge_workspaces
    assert not flash_attn_v100._fp8_prefill_bridge_tail_workspaces
    assert not flash_attn_v100._prefill_gather_dense_workspaces
    assert not flash_attn_v100._prefill_dense_splitkv3_workspaces
    assert not sm70_fp8._sm70_fp8_prefill_dense_workspaces
    assert not sm70_fp8._sm70_fp8_qpn8_pp2_tp4_workspaces
    assert not sm70_turbomind._nvfp4_qpn4_dense_workspaces
    assert not nvfp4_sm70_moe._qwen38_raw_scale_workspaces
    assert not sm70_nvfp4_native._scale_workspaces
    assert not sm70_nvfp4_native._code_workspaces


def test_mrv2_shutdown_drops_target_draft_graph_and_model_refs(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as model_runner_module
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    target_model = nn.Sequential(_AttentionLayer(torch.ones(4)))
    draft_model = nn.Sequential(_AttentionLayer(torch.ones(4)))
    draft_graph_manager = object()
    speculator = SimpleNamespace(
        model=draft_model,
        query_cudagraph_manager=draft_graph_manager,
    )

    runner = object.__new__(GPUModelRunner)
    runner.vllm_config = MagicMock()
    runner._ple_offload_connector = None
    runner.cudagraph_manager = object()
    runner.kv_caches = [torch.ones(4)]
    runner.attn_groups = [object()]
    runner.kv_cache_config = object()
    runner.model_state = SimpleNamespace(model=target_model)
    runner.speculator = speculator
    runner.model = target_model

    monkeypatch.setattr(
        model_runner_module.torch.accelerator, "synchronize", MagicMock()
    )
    monkeypatch.setattr(
        model_runner_module.torch.accelerator, "empty_cache", MagicMock()
    )
    monkeypatch.setattr(model_runner_module, "free_before_shutdown", MagicMock())
    monkeypatch.setattr(model_runner_module.gc, "collect", MagicMock())

    runner.shutdown()

    assert runner.cudagraph_manager is None
    assert speculator.query_cudagraph_manager is None
    assert runner.speculator is None
    assert not hasattr(runner, "model_state")
    assert not hasattr(runner, "model")
    assert runner.kv_caches == []
    assert runner.attn_groups == []
    assert target_model[0].kv_cache.numel() == 0
    assert draft_model[0].kv_cache.numel() == 0
