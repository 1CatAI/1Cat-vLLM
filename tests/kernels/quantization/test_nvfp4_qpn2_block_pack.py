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

PACK_ENV = "VLLM_SM70_NVFP4_QPN2_PACK"


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
    pack: str,
    monkeypatch,
    split_k: int = 16,
    chains: int = 2,
) -> torch.Tensor:
    from vllm import _sm70_ops as sm70_ops

    monkeypatch.setenv(PACK_ENV, pack)
    codes, scales, generator = _prepared(n, k, seed=m * 7 + n + split_k)
    x = torch.randn(m, k, dtype=torch.float16, device="cuda", generator=generator)
    out = torch.empty((m, n // 2 if gated else n), dtype=torch.float16, device="cuda")
    if gated:
        sm70_ops.nvfp4_qpn2_gated_sm70_out(out, x, codes, scales, 0.01, split_k, chains)
    else:
        sm70_ops.nvfp4_qpn2_gemm_sm70_out(out, x, codes, scales, 0.01, split_k, chains)
    torch.accelerator.synchronize()
    return out


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("m", [1, 2, 4, 5, 7, 8, 9, 15, 16, 17, 32])
def test_block_pack_is_bit_identical(gated: bool, m: int, monkeypatch):
    if not hasattr(torch.ops._C, "nvfp4_qpn2_gemm_sm70_out"):
        pytest.skip("build without the SM70 QPN2 extension")
    n, k = (8704, 5120) if gated else (3584, 5120)
    split_k = 8 if gated else 16
    unpacked = _run(gated, m, n, k, "0", monkeypatch, split_k)
    packed = _run(gated, m, n, k, "1", monkeypatch, split_k)
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
    n: int, k: int, split_k: int, chains: int, m: int, monkeypatch
):
    if not hasattr(torch.ops._C, "nvfp4_qpn2_gemm_sm70_out"):
        pytest.skip("build without the SM70 QPN2 extension")
    unpacked = _run(False, m, n, k, "0", monkeypatch, split_k, chains)
    packed = _run(False, m, n, k, "1", monkeypatch, split_k, chains)
    assert torch.equal(packed, unpacked)
