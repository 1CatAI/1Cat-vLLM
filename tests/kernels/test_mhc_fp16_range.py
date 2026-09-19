# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mHC under float16 with an attention-sink row: one residual channel far
above the rest, as the first token of DeepSeek V4 has in its last layers.

Three places lost that row. The large-batch pre-norm GEMM squared float16
values in float16, so beyond |x| = 255.9 the square sum was inf and the row's
mixes fell back to hc_base alone -- only for more than 16 tokens, the
small-batch kernel squares in float32. A residual the model needs above 65504
was stored as inf, which the next attention turned into NaN for every token
that still has the row in its window. And the unnormalized pre-mix sum of the
four streams was staged in float16 before the fused RMSNorm scaled it down.
"""

import pytest
import torch

from vllm.model_executor.kernels.mhc import tilelang as mhc_tilelang
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and has_tilelang()),
    reason="float16 mHC kernels need CUDA and tilelang",
)

HC_MULT = 4
HIDDEN = 4096
NUM_MIXES = HC_MULT * 2 + HC_MULT * HC_MULT
EPS = 1e-6
POST_MULT = 2.0
SINKHORN_REPEAT = 20
FP16_MAX = torch.finfo(torch.float16).max
# Both sides of the 16-token switch between the small- and large-batch kernels.
TOKEN_COUNTS = [6, 13, 16, 17, 26]


def _inputs(sink_residual: float, sink_x: float):
    torch.manual_seed(0)
    fn = 0.01 * torch.randn((NUM_MIXES, HC_MULT * HIDDEN), device="cuda")
    hc_scale = torch.full((3,), 0.5, device="cuda")
    hc_base = 0.1 * torch.randn(NUM_MIXES, device="cuda")
    norm_weight = torch.ones(HIDDEN, device="cuda", dtype=torch.float16)
    num_tokens = max(TOKEN_COUNTS)
    residual = torch.randn((num_tokens, HC_MULT, HIDDEN), device="cuda").half()
    x = torch.randn((num_tokens, HIDDEN), device="cuda").half()
    residual[0, :, 3077] = sink_residual
    x[0, 3077] = sink_x
    post_mix = torch.full((num_tokens, HC_MULT, 1), 1.2, device="cuda")
    comb_mix = torch.full((num_tokens, HC_MULT, HC_MULT), 0.3, device="cuda")
    return fn, hc_scale, hc_base, norm_weight, residual, x, post_mix, comb_mix


def _fused_and_reference(num_tokens: int, sink_residual: float, sink_x: float):
    fn, hc_scale, hc_base, norm_weight, residual, x, post_mix, comb_mix = _inputs(
        sink_residual, sink_x
    )
    outputs = mhc_tilelang.mhc_fused_post_pre_tilelang(
        x[:num_tokens].contiguous(),
        residual[:num_tokens].contiguous(),
        post_mix[:num_tokens].contiguous(),
        comb_mix[:num_tokens].contiguous(),
        fn,
        hc_scale,
        hc_base,
        EPS,
        EPS,
        EPS,
        POST_MULT,
        SINKHORN_REPEAT,
        norm_weight=norm_weight,
        norm_eps=EPS,
    )
    exact = torch.einsum(
        "ij,ih->jh", comb_mix[0].float(), residual[0].float()
    ) + post_mix[0].float() * x[0].float().unsqueeze(0)
    stored = mhc_post_torch(x[:1], residual[:1], post_mix[:1], comb_mix[:1])
    reference = mhc_pre_torch(
        stored,
        fn,
        hc_scale,
        hc_base,
        EPS,
        EPS,
        EPS,
        POST_MULT,
        SINKHORN_REPEAT,
        norm_weight=norm_weight,
        norm_eps=EPS,
    )
    return outputs, exact, stored, reference


@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("sink_residual", [200.0, 3000.0, 35000.0])
@torch.inference_mode()
def test_sink_row_mixes_do_not_depend_on_the_token_count(
    num_tokens: int, sink_residual: float
):
    (residual_out, post_mix, comb_mix, layer_input), _, stored, reference = (
        _fused_and_reference(num_tokens, sink_residual, sink_x=1.0)
    )
    ref_post, ref_comb, ref_input = reference

    torch.testing.assert_close(residual_out[0], stored[0], atol=0, rtol=1e-3)
    torch.testing.assert_close(
        post_mix[0].flatten(), ref_post.flatten(), atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        comb_mix[0].flatten(), ref_comb.flatten(), atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        layer_input[0].float(), ref_input[0].float(), atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@torch.inference_mode()
def test_residual_beyond_float16_saturates_and_stays_finite(num_tokens: int):
    (residual_out, post_mix, comb_mix, layer_input), exact, _, reference = (
        _fused_and_reference(num_tokens, sink_residual=60000.0, sink_x=9000.0)
    )
    assert exact.abs().max() > FP16_MAX  # the case under test really overflows

    for tensor in (residual_out, post_mix, comb_mix, layer_input):
        assert torch.isfinite(tensor).all()
    assert residual_out[0].float().abs().max() == FP16_MAX
    # The four saturated streams add up past float16 before the fused
    # RMSNorm brings them back; the layer input must come out exact.
    torch.testing.assert_close(
        layer_input[0].float(), reference[2][0].float(), atol=2e-3, rtol=2e-3
    )


@torch.inference_mode()
def test_torch_post_saturates_a_float16_store():
    x = torch.full((1, HIDDEN), 9000.0, device="cuda", dtype=torch.float16)
    residual = torch.full(
        (1, HC_MULT, HIDDEN), 60000.0, device="cuda", dtype=torch.float16
    )
    post_mix = torch.full((1, HC_MULT, 1), 1.2, device="cuda")
    comb_mix = torch.full((1, HC_MULT, HC_MULT), 0.3, device="cuda")

    stored = mhc_post_torch(x, residual, post_mix, comb_mix)

    assert stored.dtype == torch.float16
    assert stored.float().abs().max() == FP16_MAX
