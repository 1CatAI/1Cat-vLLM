# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


@pytest.mark.parametrize("num_reqs", [1, 2, 3, 4, 8])
def test_prepared_context_uses_all_requests_and_clears_stale_batch(num_reqs):
    spec = DFlash2Speculator.__new__(DFlash2Speculator)
    spec.num_query_per_req = 8
    tokens = num_reqs * 8
    spec.hidden_states = torch.zeros(tokens, 4)
    spec._context_target_positions = torch.zeros(tokens, dtype=torch.long)
    spec._context_compute_graphs = {tokens: Mock()}
    spec._context_store_graphs = {tokens: Mock()}
    spec._context_kv_graphs = {tokens: Mock()}
    spec._prepared_context_batch = None
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens=tokens,
        num_draft_tokens=num_reqs * 7,
        is_prefilling_np=np.zeros(num_reqs, dtype=bool),
        num_scheduled_tokens=np.full(num_reqs, 8),
        positions=torch.arange(tokens) + 8192,
    )
    hidden = torch.arange(tokens * 4).reshape(tokens, 4).float()
    spec.prepare_target_context(batch, hidden, None)
    assert spec._prepared_context_batch is batch
    assert torch.equal(spec._get_prepared_context_hidden(batch), hidden)
    assert torch.equal(spec._context_target_positions, batch.positions)
    spec._context_compute_graphs[tokens].replay.assert_called_once()

    spec._precompute_context_kv(
        spec.hidden_states, batch.positions, torch.arange(tokens)
    )
    spec._context_store_graphs[tokens].replay.assert_called_once()
    spec._context_kv_graphs[tokens].replay.assert_not_called()
    assert spec._prepared_context_batch is None
    assert spec._get_prepared_context_hidden(batch) is None

    # A later call must not reuse the preceding batch's projected K/V.
    spec._precompute_context_kv(
        spec.hidden_states, batch.positions, torch.arange(tokens)
    )
    spec._context_kv_graphs[tokens].replay.assert_called_once()


@pytest.mark.parametrize("case", ["prefill", "ragged", "tail", "uncaptured"])
def test_context_pipeline_rejects_nonuniform_or_uncaptured_target(case):
    spec = DFlash2Speculator.__new__(DFlash2Speculator)
    spec.num_query_per_req = 8
    graph = Mock()
    spec._context_compute_graphs = {32: graph}
    spec._prepared_context_batch = object()
    batch = SimpleNamespace(
        num_reqs=4,
        num_tokens=32,
        num_draft_tokens=28,
        is_prefilling_np=np.array([False, False, case == "prefill", False]),
        num_scheduled_tokens=np.array([8, 8, 8, 8]),
    )
    if case == "ragged":
        batch.num_scheduled_tokens[:] = [7, 9, 8, 8]
    elif case == "tail":
        batch.num_draft_tokens = 27
    elif case == "uncaptured":
        batch.num_tokens = 24
    spec.prepare_target_context(batch, torch.empty(32, 4), None)
    graph.replay.assert_not_called()
    assert spec._prepared_context_batch is None


def test_metadata_graph_dispatch_uses_captured_shape_including_padding():
    spec = DFlash2Speculator.__new__(DFlash2Speculator)
    spec._draft_metadata_graphs = {(1, 8): Mock(), (4, 32): Mock(), (8, 64): Mock()}
    for shape in [(8, 64), (1, 8), (4, 32), (8, 64)]:
        assert spec._refresh_draft_graph_metadata(*shape)
    assert not spec._refresh_draft_graph_metadata(4, 24)
    assert not spec._refresh_draft_graph_metadata(3, 24)
    assert spec._draft_metadata_graphs[8, 64].replay.call_count == 2
