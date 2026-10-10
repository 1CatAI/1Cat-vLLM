# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_mqa_paged, qsa_store_cache_rows
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "m,length",
    [(m, length) for m in (1, 5, 20) for length in (8192, 262144)] + [(512, 8192)],
)
def test_host_indexer_writes_and_graph_scores_match_resident_history(m, length):
    torch.manual_seed(1167)
    page = 204
    blocks = (length // 4 + page - 1) // page + 1
    host = torch.empty((blocks, page, 1, 128), dtype=torch.float16, pin_memory=True)
    host.normal_()
    mapped = get_accelerator_view_from_cpu_tensor(host)
    resident = host.to("cuda")
    table = torch.randperm(blocks, device="cuda", dtype=torch.int32).view(1, -1)
    queries = torch.randn((m, 4, 128), dtype=torch.float16, device="cuda")
    requests = torch.zeros(m, dtype=torch.int32, device="cuda")
    positions = torch.full((m,), length - 1, dtype=torch.int64, device="cuda")
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    slots = torch.tensor([0, page - 1, page, blocks * page - 1], device="cuda")
    values = torch.randn((4, 1, 128), dtype=torch.float16, device="cuda")

    def score(cache):
        return qsa_mqa_paged(
            queries,
            cache,
            table,
            requests,
            positions,
            lengths,
            4,
            shared_key_scoring=True,
        )

    def check_scores(actual, expected, visible, expected_visible):
        torch.testing.assert_close(visible, expected_visible, rtol=0, atol=0)
        # Columns beyond the complete visible groups are intentionally unwritten;
        # the selector consumes their lengths, not allocator leftovers.
        mask = torch.arange(actual.shape[1], device="cuda")[None, :] < visible[:, None]
        torch.testing.assert_close(actual[mask], expected[mask], rtol=0, atol=0)

    qsa_store_cache_rows(mapped, slots, values)
    actual, visible = score(mapped)
    qsa_store_cache_rows(resident, slots, values)
    expected, expected_visible = score(resident)
    check_scores(actual, expected, visible, expected_visible)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_store_cache_rows(mapped, slots, values)
        replay_scores, replay_visible = score(mapped)
    for cut in (length, length - 7, length):
        queries.normal_()
        values.normal_()
        lengths.fill_(cut)
        positions.fill_(cut - 1)
        graph.replay()
        qsa_store_cache_rows(resident, slots, values)
        expected, expected_visible = score(resident)
        check_scores(replay_scores, expected, replay_visible, expected_visible)
        torch.accelerator.synchronize()
        assert torch.equal(host.flatten(0, 1)[slots.cpu()], values.cpu())
