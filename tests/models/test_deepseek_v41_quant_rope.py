# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import math
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from vllm.models.deepseek_v41.quant_rope import apply_rope, attention_qdq, rope_table

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _power_two_scale(value):
    bits = value.view(torch.int32)
    exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0)
    return torch.ldexp(torch.ones_like(value), exponent)


def _reference(x, fp4, group, e4m3):
    data = x.float().view(-1, group)
    maximum = data.abs().amax(-1, keepdim=True)
    if not fp4:
        scale = _power_two_scale(maximum.clamp_min(1e-4) * (1 / 448))
        values = (data / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    else:
        scale = (
            (maximum.clamp_min(6 * 2**-9) / 6)
            .clamp_max(448)
            .to(torch.float8_e4m3fn)
            .float()
            if e4m3
            else _power_two_scale(maximum.clamp_min(6 * 2**-126) * (1 / 6))
        )
        normalized = data / scale
        grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
        distance = (normalized.abs()[..., None] - grid).abs()
        nearest = distance == distance.amin(-1, keepdim=True)
        index = torch.arange(8)
        priority = torch.where(nearest, index + (index % 2) * 8, 100)
        values = torch.copysign(grid[priority.argmin(-1)], normalized)
    return (values * scale).half().view_as(x)


@pytest.mark.parametrize(
    "fp4,group,e4m3", [(False, 128, False), (True, 16, True), (True, 32, False)]
)
@pytest.mark.parametrize("magnitude", [0.0, 2**-20, 0.03125, 1.37, 100.0])
def test_software_attention_qdq_preserves_exact_fp16_values(
    fp4, group, e4m3, magnitude
):
    x = (
        torch.randn(6, 512, generator=torch.Generator().manual_seed(41)) * magnitude
    ).half()
    expected = _reference(x, fp4, group, e4m3)
    actual = attention_qdq(x.cuda(), fp4=fp4, group=group, e4m3_scale=e4m3)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "fp4,group,e4m3", [(False, 128, False), (True, 16, True), (True, 32, False)]
)
def test_attention_qdq_ties_signed_zero_and_graph_replay(fp4, group, e4m3):
    values = (
        torch.tensor(
            [
                0.0,
                -0.0,
                0.25,
                -0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                6.0,
                -6.0,
                2**-24,
                -(2**-24),
                0.0,
                0.0,
            ]
        )
        .repeat(32)
        .view(1, 512)
        .half()
    )
    actual = values.cuda()
    attention_qdq(actual, fp4=fp4, group=group, e4m3_scale=e4m3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = attention_qdq(actual, fp4=fp4, group=group, e4m3_scale=e4m3)
    for multiplier in (1.0, -0.75):
        actual.copy_(values * multiplier)
        graph.replay()
        expected = _reference(values * multiplier, fp4, group, e4m3)
        torch.testing.assert_close(
            output.cpu().view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )


@pytest.mark.parametrize("compressed", [False, True])
def test_rope_table_matches_reviewed_official_function(compressed):
    path = Path("/home/ymzx/ds41-models/base/inference/model.py")
    if not path.exists():
        pytest.skip("Official checkpoint reference is available on 54633")
    tree = ast.parse(path.read_text())
    method = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "precompute_freqs_cis"
    )
    scope = dict(torch=torch, math=math, lru_cache=lru_cache)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    reference = scope["precompute_freqs_cis"](
        64,
        16384,
        65536 if compressed else 0,
        160000 if compressed else 10000,
        16,
        32,
        1,
    )
    expected = torch.cat((reference.real, reference.imag), dim=-1)
    torch.testing.assert_close(
        rope_table(compressed=compressed, device="cpu"), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("inverse", [False, True])
def test_interleaved_rope_fp16_boundary_and_replay(dim, inverse):
    table = rope_table(compressed=True)
    positions = torch.tensor([0, 1, 63, 127, 8191, 16383], device="cuda")
    source = torch.randn(6, 16, dim, generator=torch.Generator().manual_seed(41)).half()
    x = source.cuda()
    apply_rope(x, positions, table, inverse=inverse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = apply_rope(x, positions, table, inverse=inverse)
    x.mul_(-0.75)
    positions.copy_(positions.flip(0))
    graph.replay()
    p = positions.cpu()
    cosine, sine = table.cpu()[p].split(32, dim=-1)
    phase = torch.complex(cosine, -sine if inverse else sine)[:, None]
    expected = x.cpu().clone()
    tail = torch.view_as_complex(expected[..., -64:].float().unflatten(-1, (32, 2)))
    expected[..., -64:] = torch.view_as_real(tail * phase).flatten(-2).half()
    torch.testing.assert_close(output.cpu(), expected, rtol=0.001, atol=2e-6)
    torch.testing.assert_close(
        output[..., :-64].cpu(), expected[..., :-64], rtol=0, atol=0
    )
