# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from test_gguf_head_layout_projection import source as affine_source
from test_gguf_turbomind_model import source

from vllm._sm70.runtime import _active_owner, bind_native_runtime
from vllm.config import KernelConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
)
from vllm.runtime_resources import release_runtime_resources

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("kind", [14, 23, 18])
@pytest.mark.parametrize("mixed", [False, True])
def test_prepared_projection_borrows_native_owner_outside_forward(kind, mixed):
    import vllm._C  # noqa: F401
    import vllm._moe_C  # noqa: F401

    config = SimpleNamespace(
        kernel_config=KernelConfig(sm70_turbomind_workspace_bytes=8 * 1024**2)
    )
    data = (
        affine_source(n=64, k=512, weight_type=kind)
        if kind == 14
        else source(kind, n=64, k=512)
    )
    with set_current_vllm_config(config):
        owner = bind_native_runtime()
        projections = prepare_gguf_projections(
            [(torch.from_numpy(data).cuda(), kind)] * (2 if mixed else 1),
            torch.float16,
            True,
            512,
        )
    assert all(p.kernel.native_ops.owner is owner for p in projections)
    assert _active_owner.get() is None
    process_storage = torch.ops._C.sm70_gemm_workspace_storage()
    stream = torch.cuda.Stream()
    rows = []
    for m in (1, 5, 20):
        x = torch.randn(m, 512, device="cuda", dtype=torch.float16) * 0.125
        with torch.cuda.stream(stream):
            actual = apply_prepared_gguf_projections(x, projections)
            with owner.activate():
                expected = apply_prepared_gguf_projections(x, projections)
                streams, budget, total = torch.ops._C.sm70_gemm_workspace_storage()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = apply_prepared_gguf_projections(x, projections)
            for seed in (7, 19):
                torch.manual_seed(seed)
                x.copy_(torch.randn_like(x) * 0.125)
                graph.replay()
                torch.testing.assert_close(
                    captured,
                    apply_prepared_gguf_projections(x, projections),
                    rtol=0,
                    atol=0,
                )
        assert _active_owner.get() is None
        assert torch.ops._C.sm70_gemm_workspace_storage() == process_storage
        assert streams > 0 and budget == 8 * 1024**2
        rows.append(dict(m=m, streams=streams, budget=budget, native_bytes=total))
    compiled = torch.compile(
        lambda x, ps: apply_prepared_gguf_projections(x, ps),
        backend="eager",
        fullgraph=True,
    )
    with owner.activate():
        torch.testing.assert_close(
            compiled(x, projections),
            apply_prepared_gguf_projections(x, projections),
            rtol=0,
            atol=0,
        )
    torch._dynamo.reset()
    print(
        json.dumps(
            dict(
                kind=kind,
                mixed=mixed,
                rows=rows,
                default_owner_unchanged=True,
                graph_exact=True,
            )
        )
    )
    del projections, graph
    release_runtime_resources(config)
