# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm import sm70_fp8 as fp8
from vllm.model_executor.layers.quantization.utils import sm70_layer_workspaces as ws

_LAYER = "model.layers.0.mlp.down_proj"


class _Layer(torch.nn.Module):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix


@pytest.fixture(autouse=True)
def _fresh_workspaces():
    fp8.clear_sm70_fp8_workspaces()
    yield
    fp8.clear_sm70_fp8_workspaces()


def _cpu_impl(op_name: str, impl):
    if torch._C._dispatch_has_kernel_for_dispatch_key(f"vllm::{op_name}", "CPU"):
        return None
    library = torch.library.Library("vllm", "IMPL", "CPU")
    library.impl(op_name, impl)
    return library


def _export_save_rebind_reload(module, inputs, first, second):
    exported = torch.export.export(module, inputs)
    assert torch.equal(exported.module()(*inputs), torch.full((2, 4), first))
    artifact = io.BytesIO()
    torch.export.save(exported, artifact)
    # A new process binds the same layer to a workspace at another address.
    fp8.clear_sm70_fp8_workspaces()
    workspace = torch.full((32,), second)
    fp8._bind_sm70_fp8_prefill_workspace(_Layer(_LAYER), workspace)
    artifact.seek(0)
    reloaded = torch.export.load(artifact).module()
    assert torch.equal(reloaded(*inputs), torch.full((2, 4), second))
    return workspace


@pytest.mark.parametrize(
    ("op_name", "native_name", "impl"),
    [
        (
            "sm70_fp8_qpn8_dispatch",
            "fp8_qpn8_dispatch_sm70_out",
            ws._sm70_fp8_qpn8_dispatch,
        ),
        (
            "sm70_fp8_prefill_dispatch",
            "fp8_gemm_sm70_prefill_dispatch_out",
            ws._sm70_fp8_prefill_dispatch,
        ),
        (
            "sm70_nvfp4_qpn4_dispatch",
            "nvfp4_qpn4_dispatch_sm70_out",
            ws._sm70_nvfp4_qpn4_dispatch,
        ),
    ],
)
def test_export_reload_resolves_the_current_workspace(
    monkeypatch, op_name, native_name, impl
):
    """A serialized graph must not retain the previous process's scratch pointer."""
    first = torch.ones(32)
    fp8._bind_sm70_fp8_prefill_workspace(_Layer(_LAYER), first)
    pointers = []

    def native(out, pointer, *args):
        pointers.append(pointer)
        workspace = ws._layer_workspaces[_LAYER]
        assert pointer == workspace.data_ptr()
        out.fill_(workspace[0].item())

    monkeypatch.setattr(fp8.sm70_ops, native_name, native)
    extra = {
        "sm70_fp8_qpn8_dispatch": (16, 2, False, False),
        "sm70_fp8_prefill_dispatch": (128, 4, 4, False, 64),
        "sm70_nvfp4_qpn4_dispatch": (0.25, False, False),
    }[op_name]

    class Projection(torch.nn.Module):
        def forward(self, x, weight, scales):
            out = torch.empty_like(x)
            getattr(torch.ops.vllm, op_name)(out, _LAYER, x, weight, scales, *extra)
            return out

    inputs = (torch.zeros(2, 4), torch.zeros(4, 4, dtype=torch.uint8), torch.ones(1, 4))
    library = _cpu_impl(op_name, impl)
    try:
        second = _export_save_rebind_reload(Projection(), inputs, 1.0, 2.0)
        assert pointers == [first.data_ptr(), second.data_ptr()]
        assert first.data_ptr() != second.data_ptr()
    finally:
        if library is not None:
            library._destroy()


def test_ba_split_export_reload_resolves_the_current_workspace(monkeypatch):
    first = torch.ones(32)
    fp8._bind_sm70_fp8_prefill_workspace(_Layer(_LAYER), first)
    pointers = []

    def native(qkv, z, b, a, qkvz_staging, ba_staging, pointer, *args):
        pointers.append(pointer)
        workspace = ws._layer_workspaces[_LAYER]
        assert pointer == workspace.data_ptr()
        qkv.fill_(workspace[0].item())

    monkeypatch.setattr(fp8.sm70_ops, "fp8_qpn8_dispatch_ba_split_sm70_out", native)

    class Projection(torch.nn.Module):
        def forward(self, x, weight, scales):
            qkv = torch.empty_like(x)
            z = torch.empty_like(x)
            b = torch.empty_like(x)
            a = torch.empty_like(x)
            qkvz_staging = torch.empty_like(x)
            ba_staging = torch.empty_like(x)
            torch.ops.vllm.sm70_fp8_qpn8_dispatch_ba_split(
                qkv, z, b, a, qkvz_staging, ba_staging, _LAYER, x, weight, scales, x
            )
            return qkv

    inputs = (torch.zeros(2, 4), torch.zeros(4, 4, dtype=torch.uint8), torch.ones(1, 4))
    library = _cpu_impl(
        "sm70_fp8_qpn8_dispatch_ba_split", ws._sm70_fp8_qpn8_dispatch_ba_split
    )
    try:
        second = _export_save_rebind_reload(Projection(), inputs, 1.0, 2.0)
        assert pointers == [first.data_ptr(), second.data_ptr()]
    finally:
        if library is not None:
            library._destroy()


def test_bind_requires_a_prefix():
    with pytest.raises(RuntimeError, match="by layer prefix"):
        fp8._bind_sm70_fp8_prefill_workspace(_Layer(""), torch.ones(4))


def test_bind_rejects_a_second_workspace_for_one_layer():
    layer = _Layer(_LAYER)
    workspace = torch.ones(4)
    fp8._bind_sm70_fp8_prefill_workspace(layer, workspace)
    fp8._bind_sm70_fp8_prefill_workspace(layer, workspace)
    assert layer.sm70_fp8_prefill_exact_dense_workspace_ptr == workspace.data_ptr()
    with pytest.raises(RuntimeError, match="already bound"):
        fp8._bind_sm70_fp8_prefill_workspace(layer, torch.ones(4))


def test_online_hc_export_reload_resolves_current_workspace(monkeypatch):
    first = torch.ones(32)
    fp8._bind_sm70_fp8_prefill_workspace(_Layer(_LAYER), first)
    pointers = []

    def native(block_out, injection_out, down, lora, gate, partials, pointer, *args):
        pointers.append(pointer)
        workspace = ws._layer_workspaces[_LAYER]
        assert pointer == workspace.data_ptr()
        block_out.fill_(workspace[0].item())

    monkeypatch.setattr(fp8.sm70_ops, "fp8_qpn8_hc_dispatch_sm70_out", native)

    class Projection(torch.nn.Module):
        def forward(self, x, codes, scales):
            outputs = [torch.empty_like(x) for _ in range(6)]
            torch.ops.vllm.sm70_online_qpn8_hc_dispatch(
                *outputs, _LAYER, x, codes, scales, codes, scales
            )
            return outputs[0]

    inputs = (torch.zeros(2, 4), torch.zeros(4, 4, dtype=torch.uint8), torch.ones(1, 4))
    library = _cpu_impl(
        "sm70_online_qpn8_hc_dispatch", ws._sm70_online_qpn8_hc_dispatch
    )
    try:
        second = _export_save_rebind_reload(Projection(), inputs, 1.0, 2.0)
        assert pointers == [first.data_ptr(), second.data_ptr()]
    finally:
        if library is not None:
            library._destroy()
