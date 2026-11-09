# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw Q5_K WMMA keeps FP16 operands and identical M1..6 row arithmetic."""

import pytest
import torch
from test_gguf_native import load_native
from test_gguf_q5_batched import packed_q5

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_device_capability(70),
    reason="SM70 qualification required",
)


@pytest.mark.parametrize(
    "n,k",
    [
        (64, 256),
        (128, 768),
        (576, 5120),
        (8192, 1280),
        (5120, 2048),
        (32320, 5120),
        (6400, 6144),
    ],
)
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_original_q5_hmma_fp16_boundaries_batch_invariance_and_graph(n, k, m):
    native = load_native()
    raw = packed_q5(n, k, 0.00137)
    x = (torch.randn(m, k, device="cuda") * 0.1).half()

    def check(actual):
        dense = native.ggml_dequantize_upstream(raw, 13, n, k, torch.float16)
        expected = (x.float() @ dense.float().T).half()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        if m == 6:
            for row in range(m):
                separate = native.ggml_dense_q5_k_hmma(raw, x[row : row + 1])
                torch.testing.assert_close(
                    actual[row : row + 1], separate, rtol=0, atol=0
                )

    check(native.ggml_dense_q5_k_hmma(raw, x))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = native.ggml_dense_q5_k_hmma(raw, x)
    x.mul_(-0.75)
    raw.copy_(packed_q5(n, k, 2**-10))
    graph.replay()
    check(output)


def test_q5_hmma_enforces_workspace_before_allocating_partials():
    native = load_native()
    raw = packed_q5(64, 256, 0.00137)
    x = torch.ones(6, 256, device="cuda", dtype=torch.half)
    with pytest.raises(RuntimeError, match="workspace"):
        native.ggml_dense_q5_k_hmma(raw, x, 4, 1)


@pytest.mark.parametrize(
    "groups,width,k", [(1, 64, 256), (4, 128, 768), (2, 1024, 4096)]
)
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_grouped_hmma_is_bit_exact_to_selected_full_projection_rows(
    groups, width, k, m
):
    native = load_native()
    raw = packed_q5(groups * width, k, 0.00137)
    x = (torch.randn(m, groups, k, device="cuda") * 0.1).half()

    def check(actual):
        expected = torch.cat(
            [
                native.ggml_dense_q5_k_hmma(raw, x[:, group].contiguous())[
                    :, group * width : (group + 1) * width
                ]
                for group in range(groups)
            ],
            dim=-1,
        )
        torch.testing.assert_close(
            actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )

    check(native.ggml_dense_q5_k_hmma(raw, x, grouped=True))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = native.ggml_dense_q5_k_hmma(raw, x, grouped=True)
    x.copy_(x.flip(1) * -0.75)
    raw.copy_(packed_q5(groups * width, k, 2**-10))
    graph.replay()
    check(output)
