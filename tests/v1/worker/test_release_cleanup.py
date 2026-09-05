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


def test_sm70_workspace_cleanup_releases_all_tensor_owners():
    from vllm.model_executor.layers.quantization import (
        fp8,
        nvfp4_sm70_moe,
        sm70_turbomind,
    )
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
    fp8._sm70_fp8_prefill_dense_workspaces[(0, tensor.dtype)] = tensor
    fp8._sm70_fp8_qpn8_pp2_tp4_workspaces[(0, tensor.dtype)] = tensor
    sm70_turbomind._nvfp4_qpn4_dense_workspaces[(0, tensor.dtype)] = tensor
    nvfp4_sm70_moe._qwen38_raw_scale_workspaces[0] = tensor

    _clear_loaded_gpu_workspaces()

    assert not flash_attn_v100._sm70_fa2_cu_seqlens_cache
    assert not flash_attn_v100._fp8_prefill_bridge_workspaces
    assert not flash_attn_v100._fp8_prefill_bridge_tail_workspaces
    assert not flash_attn_v100._prefill_gather_dense_workspaces
    assert not flash_attn_v100._prefill_dense_splitkv3_workspaces
    assert not fp8._sm70_fp8_prefill_dense_workspaces
    assert not fp8._sm70_fp8_qpn8_pp2_tp4_workspaces
    assert not sm70_turbomind._nvfp4_qpn4_dense_workspaces
    assert not nvfp4_sm70_moe._qwen38_raw_scale_workspaces


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
