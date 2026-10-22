# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.model_executor.kernels.linear.mixed_precision import sm70_gguf


def test_blas_workspace_grows_to_the_largest_admitted_projection(monkeypatch):
    monkeypatch.setattr(sm70_gguf, "_affine_blas_workspaces", {})
    monkeypatch.setattr(sm70_gguf, "_affine_blas_views", {})
    weight = torch.empty(1)
    small, large, again = torch.nn.Module(), torch.nn.Module(), torch.nn.Module()

    assert sm70_gguf._bind_affine_blas_workspace(small, weight, 1536, 2560)
    first = sm70_gguf._get_affine_blas_workspace(weight)
    assert first.numel() == 1536 * 2560
    assert sm70_gguf._bind_affine_blas_workspace(large, weight, 2560, 2560)
    assert sm70_gguf._bind_affine_blas_workspace(again, weight, 2560, 1536)

    workspace = sm70_gguf._get_affine_blas_workspace(weight)
    # Sized by the admitted shapes, not the largest calibrated one, and every
    # bound view (including the one made before growing) shares that storage.
    assert workspace.numel() == 2560 * 2560
    for layer, shape in (
        (small, (1536, 2560)),
        (large, (2560, 2560)),
        (again, (2560, 1536)),
    ):
        view = layer.gguf_tm_blas_workspace
        assert tuple(view.shape) == shape
        assert view.data_ptr() == workspace.data_ptr()
    # Runtime borrowers never allocate.
    assert sm70_gguf._get_affine_blas_workspace(weight) is workspace


def test_blas_workspace_failure_preserves_prepared_views(monkeypatch):
    monkeypatch.setattr(sm70_gguf, "_affine_blas_workspaces", {})
    monkeypatch.setattr(sm70_gguf, "_affine_blas_views", {})
    weight = torch.empty(1)
    small, large = torch.nn.Module(), torch.nn.Module()
    assert sm70_gguf._bind_affine_blas_workspace(small, weight, 2, 3)
    original = small.gguf_tm_blas_workspace

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("test allocation failure")

    monkeypatch.setattr(torch, "empty", fail)
    assert not sm70_gguf._bind_affine_blas_workspace(large, weight, 4, 5)
    assert small.gguf_tm_blas_workspace is original
    assert (
        sm70_gguf._get_affine_blas_workspace(weight).data_ptr() == original.data_ptr()
    )
    assert not hasattr(large, "gguf_tm_blas_workspace")


def test_blas_workspace_engines_grow_independently(monkeypatch):
    from vllm.config import set_current_vllm_config
    from vllm.config.kernel import KernelConfig
    from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
        clear_layer_workspaces,
    )

    monkeypatch.setattr(sm70_gguf, "_affine_blas_workspaces", {})
    monkeypatch.setattr(sm70_gguf, "_affine_blas_views", {})
    engines = [
        SimpleNamespace(
            kernel_config=KernelConfig(),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        for _ in range(2)
    ]
    layers = [torch.nn.Module(), torch.nn.Module()]
    weight = torch.empty(1)
    for cfg, layer in zip(engines, layers):
        with set_current_vllm_config(cfg):
            assert sm70_gguf._bind_affine_blas_workspace(layer, weight, 2, 3)
    second = layers[1].gguf_tm_blas_workspace
    with set_current_vllm_config(engines[0]):
        assert sm70_gguf._bind_affine_blas_workspace(torch.nn.Module(), weight, 4, 5)
    assert layers[1].gguf_tm_blas_workspace is second
    assert layers[0].gguf_tm_blas_workspace.data_ptr() != second.data_ptr()
    assert not sm70_gguf._affine_blas_workspaces
    with set_current_vllm_config(engines[0]):
        clear_layer_workspaces()
        assert sm70_gguf._get_affine_blas_workspace(weight) is None
    with set_current_vllm_config(engines[1]):
        assert (
            sm70_gguf._get_affine_blas_workspace(weight).data_ptr() == second.data_ptr()
        )
