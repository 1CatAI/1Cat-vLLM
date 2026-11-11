# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""markov_bias (the DSpark sampling-path bias) against a torch reference."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("rows", [1, 2, 5, 16])
def test_markov_bias_matches_reference(rows):
    from vllm.models.deepseek_v41.markov_fused import ROWS, markov_bias

    vocab, rank = 1000, 64
    g = torch.Generator(device="cuda").manual_seed(rows)
    weight = torch.randn(vocab, rank, device="cuda", generator=g).bfloat16()

    greedy = SimpleNamespace(
        vocab=vocab,
        rank=rank,
        blocks=-(-vocab // ROWS),
        weight=weight.view(torch.int16),
    )
    embedding = torch.randn(rows, rank, device="cuda", generator=g).half()
    out = markov_bias(greedy, embedding)
    reference = embedding.float() @ weight.float().t()
    torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-3)
