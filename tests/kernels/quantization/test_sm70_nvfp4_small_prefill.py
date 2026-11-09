# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-prefill stores preserve weights in both resident packed layouts."""

import pytest
import torch


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    "m,k,n", [(256, 128, 32), (512, 5120, 8704), (560, 4352, 5120), (1024, 128, 64)]
)
def test_small_prefill_matches_scalar_dequantized_basis(m, k, n, shared, gated):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    torch.manual_seed(123)
    q = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    packed = (q[:, ::2] | (q[:, 1::2] << 4)).contiguous()
    scale_bits = torch.randint(0, 127, (n, k // 16), device="cuda", dtype=torch.uint8)
    raw_scales = scale_bits.view(torch.float8_e4m3fn)
    codes, scales = torch.ops._C.nvfp4_qpn2_prepare_sm70(packed, raw_scales)
    # Representative checkpoint scale; the legacy converter materializes the
    # 16384-biased scale in FP16 before multiplying it by the FP4 code.
    global_scale = 0.0005
    reference_weight = torch.empty(k, n, device="cuda", dtype=torch.float16)
    torch.ops._C.nvfp4_qpn4_dequantize_sm70_out(
        reference_weight,
        codes.view(k, n // 2),
        scales.view(k // 16, n),
        global_scale,
        True,
    )
    columns = torch.arange(m, device="cuda") % k
    x = torch.zeros(m, k, device="cuda", dtype=torch.float16)
    x[torch.arange(m, device="cuda"), columns] = 1
    output = torch.empty(m, n // 2 if gated else n, device="cuda", dtype=x.dtype)
    if shared:
        effective = (raw_scales.float() * global_scale).half()
        tm_weight, tm_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
            q.T.contiguous(), effective.T.contiguous(), 16, False
        )
        tm_scales.mul_(16384)
        torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
            output,
            x,
            tm_weight,
            scales,
            global_scale,
            8,
            2,
            tm_scales,
            16,
            int(meta[0]),
            int(meta[1]),
            gated,
            256,
            True,
        )
    else:
        scratch = torch.empty_like(reference_weight)
        torch.ops._C.nvfp4_qpn4_prefill_sm70_out(
            output,
            scratch.data_ptr(),
            x,
            codes.view(k, n // 2),
            scales.view(k // 16, n),
            global_scale,
            True,
            gated,
        )
        # Exercise the real vector-store kernel, not a reimplementation.
        assert torch.equal(
            scratch.view(torch.uint8), reference_weight.view(torch.uint8)
        )
    expected = reference_weight.index_select(0, columns)
    if gated:
        gate, up = expected.chunk(2, dim=1)
        expected = (gate.float() / (1 + torch.exp(-gate.float()))).half() * up
        torch.testing.assert_close(output, expected, rtol=2e-3, atol=1e-3)
    else:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.isfinite().all()
