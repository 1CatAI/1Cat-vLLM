# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA glue kernels must equal the eager op sequence they replace, bit for bit.

Shapes are GLM-5.3-Flash's: linear_attn_config num_heads 64, head_dim 128
(TP1), rows 1/5/6/8 as verification and decode steps produce them; 64 and 1024
rows as prefill chunks produce them.
"""

import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fla.ops.kda import FusedRMSNormGated
from vllm.model_executor.layers.fla.ops.kda_glue import (
    kda_gated_rmsnorm,
    kda_pack_recurrent_inputs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_HEADS = 64
HEAD_DIM = 128
PROJECTION = NUM_HEADS * HEAD_DIM
BETA_ROW = NUM_HEADS + 2 * HEAD_DIM  # in_proj_bfg_a row: b | f_a | g_a
ROWS = (1, 5, 6, 8)
NORM_ROWS = (*ROWS, 64, 1024)
# (dtype of x, out_dtype): the fp32 recurrent output goes out as the model dtype,
# fp16 and bf16 inputs as themselves or widened.
NORM_DTYPES = [
    (torch.float32, torch.float16),
    (torch.float32, torch.bfloat16),
    (torch.float16, None),
    (torch.bfloat16, None),
    (torch.float16, torch.float32),
]


def _norm(custom_ops: list[str]) -> FusedRMSNormGated:
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=custom_ops))
    with set_current_vllm_config(config):
        norm = FusedRMSNormGated(
            HEAD_DIM, activation="sigmoid", device="cuda", dtype=torch.float16
        )
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight))
    return norm


@pytest.mark.parametrize("rows", NORM_ROWS)
@pytest.mark.parametrize("scale", [1.0, 1000.0, 1e-3])
@pytest.mark.parametrize(("x_dtype", "out_dtype"), NORM_DTYPES)
def test_gated_rmsnorm_equals_eager_forward_native(
    rows: int, scale: float, x_dtype: torch.dtype, out_dtype: torch.dtype | None
) -> None:
    norm = _norm(["none", "+fused_rms_norm_gated"])
    generator = torch.Generator(device="cuda").manual_seed(rows)
    x = (
        torch.randn((1, rows, NUM_HEADS, HEAD_DIM), device="cuda", generator=generator)
        * scale
    ).to(x_dtype)
    gate = torch.randn(
        (rows, NUM_HEADS, HEAD_DIM),
        device="cuda",
        generator=generator,
    ).half()
    reference = norm.forward_native(x, gate, out_dtype=out_dtype)
    # The kernel itself, and the enabled CustomOp that dispatches to it.
    result = kda_gated_rmsnorm(
        x, gate, norm.weight, norm.eps, out_dtype=out_dtype or x_dtype
    )
    dispatched = norm(x, gate, out_dtype=out_dtype)
    for output in (result, dispatched):
        assert output.shape == reference.shape and output.dtype == reference.dtype
        assert torch.equal(output, reference)


# The kernel takes four rows per program and changes its warp count at 4096 rows:
# row counts that are not a multiple of four, and either side of the change.
@pytest.mark.parametrize("num_rows", [1, 3, 7, 4095, 4096, 4099])
@pytest.mark.parametrize(("x_dtype", "out_dtype"), NORM_DTYPES)
def test_gated_rmsnorm_row_counts_equal_eager_forward_native(
    num_rows: int, x_dtype: torch.dtype, out_dtype: torch.dtype | None
) -> None:
    norm = _norm(["none", "+fused_rms_norm_gated"])
    generator = torch.Generator(device="cuda").manual_seed(num_rows)
    x = torch.randn((num_rows, HEAD_DIM), device="cuda", generator=generator).to(
        x_dtype
    )
    gate = torch.randn((num_rows, HEAD_DIM), device="cuda", generator=generator).half()
    reference = norm.forward_native(x, gate, out_dtype=out_dtype)
    result = kda_gated_rmsnorm(
        x, gate, norm.weight, norm.eps, out_dtype=out_dtype or x_dtype
    )
    assert torch.equal(result, reference)


@pytest.mark.parametrize("rows", ROWS)
def test_pack_recurrent_inputs_equals_contiguous_copies(rows: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(rows)
    qkv = torch.randn((rows, 3 * PROJECTION), device="cuda", generator=generator).half()
    projected = torch.randn((rows, BETA_ROW), device="cuda", generator=generator).half()
    beta = projected[:, :NUM_HEADS]  # strided split, as in _project_inputs
    packed = kda_pack_recurrent_inputs(qkv, beta, NUM_HEADS, HEAD_DIM)
    q, k, v = (
        part.reshape(1, -1, NUM_HEADS, HEAD_DIM)
        for part in qkv.split(PROJECTION, dim=-1)
    )
    expected = (q.contiguous(), k.contiguous(), v.contiguous())
    for result, reference in zip(packed[:3], expected, strict=True):
        assert result.is_contiguous() and torch.equal(result, reference)
    assert packed[3].is_contiguous()
    assert torch.equal(packed[3], beta.unsqueeze(0).contiguous())


@pytest.mark.parametrize("rows", ROWS)
def test_layer_recurrent_inputs_equal_the_split_it_replaces(rows: int) -> None:
    """Glm5NextLinearAttention._recurrent_inputs hands the recurrent kernel what
    the q|k|v split and the beta slice held before, packed for rows > 1."""
    from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention

    layer = object.__new__(Glm5NextLinearAttention)
    layer.local_num_heads = NUM_HEADS
    layer.head_dim = HEAD_DIM
    layer.local_projection_size = PROJECTION
    generator = torch.Generator(device="cuda").manual_seed(100 + rows)
    qkv = torch.randn((rows, 3 * PROJECTION), device="cuda", generator=generator).half()
    projected = torch.randn((rows, BETA_ROW), device="cuda", generator=generator).half()
    beta = projected[:, :NUM_HEADS].unsqueeze(0)  # (1, T, H), as the layer holds it

    q, k, v, beta_rec = layer._recurrent_inputs(qkv, beta)

    expected = [
        part.reshape(1, -1, NUM_HEADS, HEAD_DIM)
        for part in qkv.split(PROJECTION, dim=-1)
    ]
    for result, reference in zip((q, k, v), expected, strict=True):
        assert result.shape == (1, rows, NUM_HEADS, HEAD_DIM)
        assert result.is_contiguous() and torch.equal(result, reference)
    assert beta_rec.shape == (1, rows, NUM_HEADS)
    assert beta_rec.is_contiguous() and torch.equal(beta_rec, beta)
