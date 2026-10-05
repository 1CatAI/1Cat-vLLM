# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.models.qwen4_exp.nvidia.sm70_ngram_input import prepare_ngram_input
from vllm.v1.worker.gpu.buffer_utils import UvaBuffer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("requests,padded", [(0, 4), (1, 1), (1, 4), (4, 4)])
@pytest.mark.parametrize("length", [2, 7])
@pytest.mark.parametrize("uva", [False, True])
def test_official_state_and_rollback_graph(requests, padded, length, uva):
    device = "cuda"
    eos = 248046
    expected_context = torch.empty((padded, length), dtype=torch.int32, device=device)
    context = torch.empty_like(expected_context)
    query = torch.arange(padded + 1, dtype=torch.int32, device=device) * 5
    query_out = torch.empty_like(query)
    mapping = torch.tensor([3, 1, 2, 0], dtype=torch.int32, device=device)
    computed = torch.tensor([31, 0, 2, 23], dtype=torch.int32, device=device)
    tokens = torch.arange(4 * 64, dtype=torch.int32, device=device).reshape(4, 64)
    if uva:
        backing = UvaBuffer((4, 64), torch.int32)
        backing.cpu.copy_(torch.arange(4 * 64, dtype=torch.int32).reshape(4, 64))
        tokens = backing.uva
    batch = SimpleNamespace(
        num_reqs=requests, num_reqs_after_padding=padded, idx_mapping=mapping
    )
    states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=computed),
        all_token_ids=SimpleNamespace(gpu=tokens),
    )
    owner = SimpleNamespace(
        ngram_context=expected_context,
        ngram_eos_token_id=eos,
        ngram_context_offsets=torch.arange(
            -length, 0, dtype=torch.int64, device=device
        ),
    )

    def run():
        prepare_ngram_input(
            context, query_out, query, mapping, computed, tokens, requests, eos
        )

    def check():
        expected = Qwen4ExpModelState._prepare_ngram_context(owner, batch, states)
        torch.testing.assert_close(context, expected, rtol=0, atol=0)
        torch.testing.assert_close(query_out, query, rtol=0, atol=0)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    mapping.copy_(mapping.roll(1))
    computed.sub_(1).clamp_min_(0)
    tokens.add_(100)
    query.add_(3)
    graph.replay()
    check()
