# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP router packing and graph replay preserve the original FP16 logits."""

import pytest
import torch

import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv


def test_pack_preserves_every_weight_bit():
    raw = torch.randint(-(2**15), 2**15, (512, 2560), dtype=torch.int16)
    packed = gemv._pack_router_batch_weight(raw.view(torch.float16))
    restored = packed.permute(0, 4, 3, 1, 2, 5).contiguous().view(512, 2560)
    assert torch.equal(restored.view(torch.int16), raw)


@pytest.mark.parametrize("rows", [5, 10])
def test_native_and_dispatch_changed_input_graph(rows, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")
    if not hasattr(torch.ops._C, "qwen38_router_batch_sm70_out"):
        pytest.skip("Requires a source build with batch router")
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    assert not torch.backends.cuda.matmul.allow_fp16_accumulation
    monkeypatch.setenv("VLLM_SM70_MTP_ROUTER_BATCH", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    envs.disable_envs_cache()
    torch.manual_seed(20260927)
    w = torch.randn(512, 2560, device="cuda", dtype=torch.float16) * 0.03
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    packed = gemv._pack_router_batch_weight(w)
    expected = torch.empty(rows, 512, device="cuda", dtype=torch.float16)
    actual = torch.empty_like(expected)

    def run():
        torch.mm(x, w.t(), out=expected)
        torch.ops._C.qwen38_router_batch_sm70_out(actual, x, packed)
        return gemv._qwen38_sm70_fp16_gemv(x, w, "model.layers.0.mlp.gate", packed)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dispatched = run()
    try:
        for scale in (0.0, 0.001, 0.03, 0.1, 1.0, 3.0, 30.0):
            x.normal_(0, scale)
            actual.fill_(float("nan"))
            graph.replay()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            assert torch.equal(dispatched.view(torch.int16), expected.view(torch.int16))
    finally:
        envs.disable_envs_cache()
