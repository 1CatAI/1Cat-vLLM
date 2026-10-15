# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
