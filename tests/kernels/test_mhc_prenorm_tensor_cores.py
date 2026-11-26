# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mHC prenorm projection on Volta/Turing tensor cores (triton_sm7x_tensor_cores)."""

import pytest
import torch

from tests.kernels.utils import requires_tensor_core_triton
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
from vllm.model_executor.kernels.mhc.triton import mhc_prenorm_tensor_cores

HIDDEN = 4096
HC_MULT = 4
MIXES = (2 + HC_MULT) * HC_MULT


@pytest.fixture(autouse=True)
def engine_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _residual(tokens: int, gen: torch.Generator) -> torch.Tensor:
    residual = torch.randn(tokens, HC_MULT, HIDDEN, generator=gen, device="cuda") * 2
    residual[0] *= 2000  # attention-sink row, |x| ~ 1e4 as in the last layers
    return residual.half()


def _fn(gen: torch.Generator) -> torch.Tensor:
    # DSv4 checkpoint scale: mean |w| ~ 0.03.
    return torch.randn(MIXES, HC_MULT * HIDDEN, generator=gen, device="cuda") * 0.03


@requires_tensor_core_triton
@pytest.mark.parametrize("tokens", [17, 64, 508, 1030])
def test_projection_matches_fp64(tokens):
    gen = _generator(tokens)
    x = _residual(tokens, gen).view(tokens, HC_MULT * HIDDEN)
    fn = _fn(gen)
    out = torch.empty(1, tokens, MIXES, device="cuda")
    sqrsum = torch.empty(1, tokens, device="cuda")

    mhc_prenorm_tensor_cores(x, fn, out, sqrsum)

    xd, wd = x.double(), fn.double()
    # Relative to sum |x * w|: the FP32 FMA routes stay near 1e-8, products
    # rounded to FP16 would reach ~1e-4.
    error = (out[0].double() - xd @ wd.T).abs() / (xd.abs() @ wd.abs().T)
    assert error.max() < 3e-8
    reference = (xd * xd).sum(-1)
    assert ((sqrsum[0].double() - reference).abs() / reference).max() < 1e-6


@requires_tensor_core_triton
def test_projection_rejects_unaligned_k_and_split_outputs():
    x = torch.zeros(32, 1000, dtype=torch.float16, device="cuda")
    fn = torch.zeros(MIXES, 1000, device="cuda")
    with pytest.raises(ValueError, match="K divisible"):
        mhc_prenorm_tensor_cores(
            x,
            fn,
            torch.empty(1, 32, MIXES, device="cuda"),
            torch.empty(1, 32, device="cuda"),
        )
    x = torch.zeros(32, HC_MULT * HIDDEN, dtype=torch.float16, device="cuda")
    fn = torch.zeros(MIXES, HC_MULT * HIDDEN, device="cuda")
    with pytest.raises(ValueError, match="exactly one split"):
        mhc_prenorm_tensor_cores(
            x,
            fn,
            torch.empty(2, 32, MIXES, device="cuda"),
            torch.empty(2, 32, device="cuda"),
        )


def _fp16_ulp(values: torch.Tensor) -> torch.Tensor:
    magnitude = values.float().abs().clamp_min(2.0**-14)
    return torch.exp2(torch.floor(torch.log2(magnitude)) - 10)


@requires_tensor_core_triton
@pytest.mark.parametrize("tokens", [17, 508])
def test_fused_post_pre_route_keeps_outputs(tokens):
    gen = _generator(tokens)
    x = (torch.randn(tokens, HIDDEN, generator=gen, device="cuda") * 2).half()
    residual = _residual(tokens, gen)
    post_mix = torch.rand(tokens, HC_MULT, 1, generator=gen, device="cuda") * 2
    comb_mix = torch.softmax(
        torch.randn(tokens, HC_MULT, HC_MULT, generator=gen, device="cuda"), -1
    )
    fn = _fn(gen)
    hc_scale = torch.tensor([0.084, 0.041, 0.234], device="cuda")
    hc_base = torch.randn(MIXES, generator=gen, device="cuda") * 3
    norm_weight = (1 + 0.1 * torch.randn(HIDDEN, generator=gen, device="cuda")).half()

    def run(tensor_cores: bool):
        return mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_mix,
            comb_mix,
            fn,
            hc_scale,
            hc_base,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
            norm_weight=norm_weight,
            norm_eps=1e-6,
            tensor_cores=tensor_cores,
        )

    expected = run(False)
    actual = run(True)

    # The post step is shared; only the projection feeding pre changes.
    assert torch.equal(actual[0], expected[0])
    for got, want in zip(actual[1:3], expected[1:3]):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    layer_input, reference = actual[3], expected[3]
    differs = layer_input != reference
    assert differs.float().mean() < 1e-3
    # An element where the four mixed streams cancel is small, but its rounding
    # noise scales with the streams: bound it by one FP16 step of its row.
    row_step = _fp16_ulp(reference.float().abs().amax(dim=-1, keepdim=True))
    assert ((layer_input.float() - reference.float()).abs() <= row_step).all()
