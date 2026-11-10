# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.fp8_linear import V41Fp8LinearMethod, fp8_block32_linear
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="SM70 Tensor Cores required"
)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 5120, 576),
        (5, 8192, 1280),
        (5, 2048, 4096),
        (5, 5120, 15360),
        (64, 512, 5120),
    ],
)
def test_raw_fp8_block32_matches_bounded_fp16_tc_reference_and_replay(m, n, k):
    generator = torch.Generator(device="cuda").manual_seed(4132)
    raw = torch.randint(
        0, 127, (n, k), generator=generator, device="cuda", dtype=torch.uint8
    )
    raw |= (
        torch.randint(
            0, 2, (n, k), generator=generator, device="cuda", dtype=torch.uint8
        )
        << 7
    )
    scale = torch.randint(
        111,
        119,
        (n // 32, k // 32),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    scale[0, 0] = 0  # E8M0's smallest nonzero scale, not numeric zero.
    weight, scales = raw.view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu)
    x = torch.randn(m, k, generator=generator, device="cuda").half() * 0.1
    fp8_block32_linear(x, weight, scales)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fp8_block32_linear(x, weight, scales)
    x.mul_(0.75)
    scale[0, 1] += 1
    graph.replay()
    # Decode only three 32-row blocks of the matrix for the numerical oracle.
    selected = torch.cat(
        [
            torch.arange(start, start + 32, device="cuda")
            for start in (0, n // 2, n - 32)
        ]
    )
    reference_weight = raw[selected].cpu().view(torch.float8_e4m3fn).float()
    reference_scale = torch.exp2(scale[selected // 32].cpu().float() - 127)
    reference_weight = (
        reference_weight * reference_scale.repeat_interleave(32, -1)
    ).half()
    expected = (x.float() @ reference_weight.to("cuda").float().T).half()
    torch.testing.assert_close(actual[:, selected], expected, rtol=0.001, atol=0.0001)


@pytest.mark.parametrize("m", [1, 5, 6, 64])
def test_grouped_fp8_projection_uses_the_same_raw_rows_and_replays(m):
    raw = torch.randint(0, 126, (2048, 4096), device="cuda", dtype=torch.uint8)
    scales = torch.full((64, 128), 113, device="cuda", dtype=torch.uint8)
    layer = SimpleNamespace(
        weight=raw.view(torch.float8_e4m3fn),
        weight_scale_inv=scales.view(torch.float8_e8m0fnu),
    )
    x = (torch.randn(m, 2, 4096, device="cuda") * 0.1).half()
    method = V41Fp8LinearMethod()

    def check(actual):
        expected = torch.cat(
            [
                method.apply(layer, x[:, group].contiguous())[
                    :, group * 1024 : (group + 1) * 1024
                ]
                for group in range(2)
            ],
            dim=-1,
        )
        torch.testing.assert_close(
            actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )

    check(method.apply_grouped(layer, x))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = method.apply_grouped(layer, x)
    x.mul_(-0.5)
    scales[:, 1].add_(1)
    graph.replay()
    check(output)


@pytest.mark.parametrize(
    "n,k",
    [
        (512, 5120),
        (576, 5120),
        (1280, 5120),
        (8192, 1280),
        (1024, 4096),
        (5120, 2048),
        (5120, 576),
        (5120, 15360),
    ],
)
def test_short_fp8_batches_keep_identical_row_boundaries_with_bias(n, k):
    generator = torch.Generator(device="cuda").manual_seed(41325)
    raw = torch.randint(
        0, 126, (n, k), generator=generator, device="cuda", dtype=torch.uint8
    )
    raw |= (
        torch.randint(
            0, 2, (n, k), generator=generator, device="cuda", dtype=torch.uint8
        )
        << 7
    )
    scale = torch.randint(
        111,
        119,
        (n // 32, k // 32),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    scale[0, 0] = 0
    weight, scales = raw.view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu)
    x = (torch.randn(6, k, generator=generator, device="cuda") * 0.1).half()
    bias = (torch.randn(n, generator=generator, device="cuda") * 0.01).half()
    selected = torch.cat(
        [
            torch.arange(start, start + 32, device="cuda")
            for start in (0, n // 2, n - 32)
        ]
    )
    reference_weight = raw[selected].cpu().view(torch.float8_e4m3fn).float()
    reference_scale = torch.exp2(scale[selected // 32].cpu().float() - 127)
    reference_weight = (
        (reference_weight * reference_scale.repeat_interleave(32, -1)).half().to("cuda")
    )
    expected = (x.float() @ reference_weight.float().T + bias[selected].float()).half()
    complete = fp8_block32_linear(x, weight, scales, bias)
    torch.testing.assert_close(complete[:, selected], expected, rtol=0.001, atol=0.0001)
    for m in range(1, 7):
        output = fp8_block32_linear(x[:m], weight, scales, bias)
        torch.testing.assert_close(
            output.view(torch.int16), complete[:m].view(torch.int16), rtol=0, atol=0
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fp8_block32_linear(x, weight, scales, bias)
    x.mul_(-0.5)
    scale[:, 1].add_(1)
    bias.mul_(2)
    graph.replay()
    updated = fp8_block32_linear(x, weight, scales, bias)
    torch.testing.assert_close(
        output.view(torch.int16), updated.view(torch.int16), rtol=0, atol=0
    )
