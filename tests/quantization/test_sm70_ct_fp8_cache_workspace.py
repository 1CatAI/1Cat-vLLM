# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a16_fp8 as ct_fp8,
)


def test_export_reload_resolves_current_workspace(monkeypatch):
    """A serialized graph must not retain the previous process's scratch pointer."""
    workspace = torch.ones(32)
    previous_workspace = workspace
    pointers = []
    monkeypatch.setattr(
        ct_fp8, "_get_sm70_fp8_prefill_exact_dense_workspace", lambda _: workspace
    )

    def native(out, pointer, x, codes, scales, split_k, nacc, prefetch, gated):
        pointers.append(pointer)
        assert pointer == workspace.data_ptr()
        out.fill_(workspace[0].item())

    monkeypatch.setattr(ct_fp8.sm70_ops, "fp8_qpn8_dispatch_sm70_out", native)

    class Projection(torch.nn.Module):
        def forward(self, x, codes, scales):
            out = torch.empty_like(x)
            torch.ops.vllm.sm70_ct_fp8_qpn8_dispatch(
                out, x, codes, scales, 16, 2, False, False
            )
            return out

    inputs = (torch.zeros(2, 4), torch.zeros(4, 4, dtype=torch.uint8), torch.ones(1, 4))
    library = None
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "vllm::sm70_ct_fp8_qpn8_dispatch", "CPU"
    ):
        library = torch.library.Library("vllm", "IMPL", "CPU")
        library.impl("sm70_ct_fp8_qpn8_dispatch", ct_fp8._sm70_ct_fp8_qpn8_dispatch)
    try:
        exported = torch.export.export(Projection(), inputs)
        assert torch.equal(exported.module()(*inputs), torch.ones(2, 4))
        artifact = io.BytesIO()
        torch.export.save(exported, artifact)
        workspace = torch.full((32,), 2.0)
        assert previous_workspace.data_ptr() != workspace.data_ptr()
        artifact.seek(0)
        reloaded = torch.export.load(artifact).module()
        assert torch.equal(reloaded(*inputs), torch.full((2, 4), 2.0))
        assert pointers == [previous_workspace.data_ptr(), workspace.data_ptr()]
    finally:
        if library is not None:
            library._destroy()
