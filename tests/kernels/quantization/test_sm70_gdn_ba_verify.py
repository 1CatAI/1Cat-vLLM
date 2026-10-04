# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch


@pytest.mark.parametrize("amplitude", [0.0, 0.125, 1.0])
@torch.inference_mode()
def test_m8_qkvz_is_bitwise_and_ba_matches_dense64(amplitude):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    from vllm import _sm70_ops as ops

    torch.manual_seed(891)
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * amplitude
    weight = (torch.randn(4096, 5120, device="cuda") * 0.01).to(torch.float8_e4m3fn)
    scales = torch.ones(4096, 1, device="cuda")
    codes, packed_scales = ops.fp8_qpn8_prepare_sm70(weight, scales)
    ba = (torch.randn(24, 5120, device="cuda") * 0.01).to(torch.bfloat16).half()
    ordinary = x.new_empty((8, 4096))
    q, z = x.new_empty((8, 2560)), x.new_empty((8, 1536))
    b, a = x.new_empty((8, 12)), x.new_empty((8, 12))
    ops.fp8_qpn8_gemm_sm70_out(ordinary, x, codes, packed_scales, 16, 2, True, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.fp8_qpn8_dispatch_ba_split_sm70_out(
            q,
            z,
            b,
            a,
            x.new_empty((8, 4096)),
            x.new_empty((8, 24)),
            0,
            x,
            codes,
            packed_scales,
            ba,
        )
    for _ in range(4):
        graph.replay()
    assert torch.equal(q, ordinary[:, :2560])
    assert torch.equal(z, ordinary[:, 2560:])
    reference = x.double() @ ba.double().T
    torch.testing.assert_close(
        torch.cat((b, a), dim=1).double(), reference, atol=0.002, rtol=0.001
    )
