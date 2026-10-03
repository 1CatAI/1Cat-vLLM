# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The block-packed activation layout of the QPN2 kernels changes nothing
but the memory layout: the GEMM and the gated GEMM give bit-identical
outputs with and without it, for every row count the dispatcher admits.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _prepared(n: int, k: int, seed: int):
    from vllm import _sm70_ops as sm70_ops

    generator = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=generator
    )
    scales = torch.randint(
        0x28, 0x3D, (n, k // 16), dtype=torch.uint8, device="cuda", generator=generator
    )
    codes, qpn2_scales = sm70_ops.nvfp4_qpn2_prepare_sm70(
        packed, scales.view(torch.float8_e4m3fn)
    )
    return codes, qpn2_scales, generator


def _run(
    gated: bool,
    m: int,
    n: int,
    k: int,
    pack: bool,
    split_k: int = 16,
    chains: int = 2,
) -> torch.Tensor:
    from vllm import _sm70_ops as sm70_ops

    codes, scales, generator = _prepared(n, k, seed=m * 7 + n + split_k)
    x = torch.randn(m, k, dtype=torch.float16, device="cuda", generator=generator)
    out = torch.empty((m, n // 2 if gated else n), dtype=torch.float16, device="cuda")
    if gated:
        sm70_ops.nvfp4_qpn2_gated_sm70_out(
            out, x, codes, scales, 0.01, split_k, chains, activation_pack=pack
        )
    else:
        sm70_ops.nvfp4_qpn2_gemm_sm70_out(
            out, x, codes, scales, 0.01, split_k, chains, activation_pack=pack
        )
    torch.accelerator.synchronize()
    return out


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("m", [1, 2, 4, 5, 7, 8, 9, 15, 16, 17, 32])
def test_block_pack_is_bit_identical(gated: bool, m: int):
    if not hasattr(torch.ops._C, "nvfp4_qpn2_gemm_sm70_out"):
        pytest.skip("build without the SM70 QPN2 extension")
    from vllm import _sm70_ops as sm70_ops

    assert sm70_ops.has_qpn2_activation_pack("nvfp4_qpn2_gemm_sm70_out")
    n, k = (8704, 5120) if gated else (3584, 5120)
    split_k = 8 if gated else 16
    unpacked = _run(gated, m, n, k, False, split_k)
    packed = _run(gated, m, n, k, True, split_k)
    assert torch.equal(packed, unpacked)
    assert torch.isfinite(unpacked).all()


@pytest.mark.parametrize(
    ("n", "k", "split_k", "chains"),
    [
        (3584, 5120, 8, 1),
        (3584, 5120, 8, 2),
        (3584, 5120, 16, 1),
        (3584, 5120, 32, 1),
        (3584, 5120, 32, 2),
        (5120, 1536, 8, 2),
        (5120, 1536, 16, 2),
        (5120, 4352, 16, 2),
        (62080, 5120, 8, 1),
    ],
)
@pytest.mark.parametrize("m", [3, 8, 15, 32])
def test_block_pack_is_bit_identical_across_launch_configs(
    n: int, k: int, split_k: int, chains: int, m: int
):
    if not hasattr(torch.ops._C, "nvfp4_qpn2_gemm_sm70_out"):
        pytest.skip("build without the SM70 QPN2 extension")
    from vllm import _sm70_ops as sm70_ops

    assert sm70_ops.has_qpn2_activation_pack("nvfp4_qpn2_gemm_sm70_out")
    unpacked = _run(False, m, n, k, False, split_k, chains)
    packed = _run(False, m, n, k, True, split_k, chains)
    assert torch.equal(packed, unpacked)


def test_packing_work_reasons_are_reported():
    from vllm import _sm70_ops as sm70_ops

    reason = torch.ops._C.nvfp4_qpn2_activation_pack_reason_sm70
    codes = torch.zeros(1, device="cuda", dtype=torch.uint8)
    assert "reduction work" in reason(codes, 1536, 5120, 16)
    assert "projection CTAs" in reason(codes, 5120, 32, 16)
    assert reason(codes, 5120, 3584, 16) == ""
    assert sm70_ops.has_qpn2_activation_pack("nvfp4_qpn2_gemm_sm70_out")


def test_native_abi_detection_preserves_full_graph_compilation():
    from vllm import _sm70_ops as sm70_ops

    n, k = 3584, 5120
    codes, scales, generator = _prepared(n, k, seed=84322)
    x = torch.randn(8, k, dtype=torch.float16, device="cuda", generator=generator)
    out = torch.empty(8, n, dtype=x.dtype, device=x.device)

    def run(x, out):
        sm70_ops.nvfp4_qpn2_gemm_sm70_out(
            out, x, codes, scales, 0.01, 16, 2, activation_pack=True
        )
        return out

    expected = run(x, torch.empty_like(out)).clone()
    compiled = torch.compile(run, backend="eager", fullgraph=True)
    assert torch.equal(compiled(x, out), expected)
