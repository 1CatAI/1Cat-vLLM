# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-scaled concurrent decode against the accepted eight-row kernels."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("rows", [9, 15, 16, 17, 24, 31, 32])
@pytest.mark.parametrize(
    "k,n,split,gated",
    [(1536, 256, 12, False), (4096, 512, 16, False), (512, 512, 8, True)],
)
def test_block_decode_preserves_small_row_bits_and_graph_inputs(
    monkeypatch, rows, k, n, split, gated
):
    import vllm._C  # noqa: F401

    monkeypatch.setenv("VLLM_SM70_FP8_QPN8_M32_CHUNKED", "1")
    torch.manual_seed(1069)
    weight = (torch.randn(n, k, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    scales = torch.full((n // 128, k // 128), 0.125, device="cuda")
    codes, packed_scales = torch.ops._C.fp8_qpn8_prepare_sm70(weight, scales)
    x = torch.empty(rows, k, device="cuda", dtype=torch.float16)
    width = n // 2 if gated else n
    backing = x.new_full((rows + 2, width), float("nan"))
    output = backing[1:-1]
    expected = torch.empty_like(output)
    workspace = x.new_full((k, n), float("nan"))
    small = (
        torch.ops._C.fp8_qpn8_gated_pair_sm70_out
        if gated
        else torch.ops._C.fp8_qpn8_gemm_sm70_out
    )

    def call():
        torch.ops._C.fp8_qpn8_dispatch_sm70_out(
            output,
            workspace.data_ptr(),
            x,
            codes,
            packed_scales,
            split,
            2,
            False,
            gated,
        )

    x.normal_(0, 0.1)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for amplitude in (0.1, 0.3, 0.0):
        x.normal_(0, amplitude)
        for start in range(0, rows, 8):
            small(
                expected[start : start + 8],
                x[start : start + 8],
                codes,
                packed_scales,
                split,
                2,
                True,
                False,
            )
        for invoke in (call, graph.replay):
            output.fill_(float("nan"))
            invoke()
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
            assert torch.isnan(backing[[0, -1]]).all()
            # The old fallback would reconstruct the dense matrix here.
            assert torch.isnan(workspace).all()


@pytest.mark.parametrize(
    "gated,k,split,chains,prefetch,chunked",
    [
        (True, 512, 32, 2, False, True),
        (True, 512, 8, 2, False, False),
        (False, 1536, 12, 1, False, True),
        (False, 1536, 12, 2, True, True),
    ],
)
def test_rollback_and_unsupported_configs_keep_dense_fallback(
    monkeypatch, gated, k, split, chains, prefetch, chunked
):
    import vllm._C  # noqa: F401

    monkeypatch.setenv("VLLM_SM70_FP8_QPN8_M32_CHUNKED", str(int(chunked)))
    torch.manual_seed(10690)
    n = 512
    weight = (torch.randn(n, k, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    scales = torch.full((n // 128, k // 128), 0.125, device="cuda")
    codes, packed_scales = torch.ops._C.fp8_qpn8_prepare_sm70(weight, scales)
    x = torch.randn(17, k, device="cuda", dtype=torch.float16) * 0.1
    output = x.new_empty((17, n // 2 if gated else n))
    workspace = x.new_full((k, n), float("nan"))
    # These configurations were accepted by the prefill fallback before the
    # new block route, even when the small-row kernel cannot execute them.
    torch.ops._C.fp8_qpn8_dispatch_sm70_out(
        output,
        workspace.data_ptr(),
        x,
        codes,
        packed_scales,
        split,
        chains,
        prefetch,
        gated,
    )
    assert torch.isfinite(workspace).all()
    expected = x.double() @ (weight.double() * 0.125).t()
    if gated:
        gate, up = expected.chunk(2, dim=-1)
        expected = gate.sigmoid() * gate * up
    assert (output.double() - expected).norm() / expected.norm() < 0.005
