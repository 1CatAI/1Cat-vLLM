# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep original K64 WMMA arithmetic; vectorize only the raw decoder."""

import pytest
import torch
from test_gguf_native import load_native
from test_gguf_q5_batched import packed_q5


@pytest.mark.parametrize("scale", [2**-20, 0.00137, 0.5])
def test_every_decoded_weight_with_basis_inputs(scale):
    native = load_native()
    raw = packed_q5(64, 256, scale)
    expected = native.ggml_dequantize_upstream(raw, 13, 64, 256, torch.float16)
    basis = torch.eye(256, device="cuda", dtype=torch.float16)
    values = []
    for first in range(0, 256, 6):
        values.append(
            native.ggml_dense_q5_k_hmma(
                raw, basis[first : first + 6], vector_decode=True
            )
        )
    actual = torch.cat(values).T.contiguous()
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "n,k,groups",
    [
        (576, 5120, 1),
        (8192, 1280, 1),
        (5120, 2048, 1),
        (2048, 4096, 2),
        (6400, 6144, 1),
        (32320, 5120, 1),
    ],
)
@pytest.mark.parametrize("m", range(1, 7))
@pytest.mark.parametrize("scale", [2**-20, 0.00137])
def test_vector_and_legacy_are_bitwise_same_after_raw_graph_updates(
    n, k, groups, m, scale
):
    native = load_native()
    raw = packed_q5(n, k, scale)
    x = (torch.randn(m, groups, k, device="cuda") * 0.1).half()
    x = x[:, 0].contiguous() if groups == 1 else x
    for vector in (False, True):
        native.ggml_dense_q5_k_hmma(raw, x, grouped=groups > 1, vector_decode=vector)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        new = native.ggml_dense_q5_k_hmma(
            raw, x, grouped=groups > 1, vector_decode=True
        )
        old = native.ggml_dense_q5_k_hmma(
            raw, x, grouped=groups > 1, vector_decode=False
        )
    for multiplier in (-0.75, 1.0):
        x.mul_(multiplier)
        raw.copy_(packed_q5(n, k, scale * 2))
        graph.replay()
        torch.testing.assert_close(
            new.view(torch.int16), old.view(torch.int16), rtol=0, atol=0
        )
