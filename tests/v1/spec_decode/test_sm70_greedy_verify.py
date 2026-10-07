# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.spec_decode.sm70_greedy_verify import greedy_verify

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


@pytest.mark.parametrize("num_reqs", [1, 4])
@pytest.mark.parametrize("match_rate", [0.0, 0.6, 1.0])
def test_greedy_verify_matches_rejection_sample(num_reqs, match_rate):
    torch.manual_seed(num_reqs * 10 + int(match_rate * 10))
    dev, steps, vocab = "cuda", 4, 5000
    tokens = steps + 1
    num_logits = num_reqs * tokens
    logits = torch.randn(num_logits, vocab, device=dev)
    target = logits.argmax(-1)
    draft = torch.randint(0, vocab, (num_logits,), device=dev)
    # Draft token at row i+1 is checked against the target argmax of row i.
    keep = torch.rand(num_logits, device=dev) < match_rate
    shifted = torch.roll(target, 1)
    draft = torch.where(keep, shifted, draft)
    cu = torch.arange(0, num_logits + 1, tokens, device=dev, dtype=torch.int32)
    idx = torch.arange(num_reqs, device=dev, dtype=torch.int32)
    expanded = idx.repeat_interleave(tokens)
    local_pos = torch.arange(tokens, device=dev, dtype=torch.int32).repeat(num_reqs)
    pos = torch.arange(num_logits, device=dev, dtype=torch.int64)
    temperature = torch.zeros(num_reqs, device=dev)
    seeds = torch.zeros(num_reqs, device=dev, dtype=torch.int64)
    ref, ref_n = rejection_sample(
        logits,
        None,
        draft,
        cu,
        pos,
        idx,
        expanded,
        local_pos,
        temperature,
        seeds,
        steps,
    )
    out, out_n = greedy_verify(target, draft, cu, steps)
    assert torch.equal(ref_n, out_n)
    for r in range(num_reqs):
        n = int(ref_n[r])
        assert torch.equal(ref[r, :n], out[r, :n])
