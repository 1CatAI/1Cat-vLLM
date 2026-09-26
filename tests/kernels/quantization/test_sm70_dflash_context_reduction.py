# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft context projection must ignore numerically different cached tactics."""

import pytest
import torch


@pytest.mark.parametrize("rows", range(1, 9))
def test_context_fc_reduction_survives_old_cache_and_captured_tail(monkeypatch, rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    # M16 stays outside the stable-context selector. Its two independent M8
    # tiles provide an oracle for the chosen legacy accumulation tree.
    monkeypatch.setenv("VLLM_SM70_AWQ_TUNE_SMALL_SHAPES", "0")
    monkeypatch.setenv("VLLM_SM70_AWQ_TP2_FAST_SELECTOR", "1")
    monkeypatch.setenv("VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC", "0")
    monkeypatch.setenv(
        "VLLM_SM70_AWQ_TP2_FAST_TARGETS",
        ";".join(
            f"sm70_f16_f16_f16_tnt_fff_{m}x1280x25600_1"
            f"|8,256,64,{10 if m == 16 else 16},0,1@s884_1x4x1"
            for m in [rows, 16]
        ),
    )
    torch.manual_seed(417)
    weight = torch.randn(1280, 25600, device="cuda", dtype=torch.float16) * 0.01
    packed, meta = torch.ops._C.sm70_f16_prepare(weight)
    ld = int(meta[0])
    x = torch.randn(16, 25600, device="cuda", dtype=torch.float16) * 0.1
    oracle = torch.empty(16, 1280, device="cuda", dtype=torch.float16)
    out = torch.empty(rows, 1280, device="cuda", dtype=torch.float16)

    def call():
        torch.ops._C.sm70_f16_gemm_out(out, x[:rows], packed, ld, False)

    torch.ops._C.sm70_f16_gemm_out(oracle, x, packed, ld, False)
    call()  # Populate the ordinary cache with the incompatible split-16 tree.
    monkeypatch.setenv("VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC", "1")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()  # First use of the fixed tree may itself be a captured tail.
    for amplitude in (0.1, 0.3, 0.01):
        x.normal_(0, amplitude)
        torch.ops._C.sm70_f16_gemm_out(oracle, x, packed, ld, False)
        for invoke in (call, graph.replay):
            out.fill_(float("nan"))
            invoke()
            assert torch.isfinite(out).all()
            assert torch.equal(out.view(torch.int16), oracle[:rows].view(torch.int16))
