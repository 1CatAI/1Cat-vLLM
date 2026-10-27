# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The M5 graph reads changing query/KV metadata and rejects moved buffers."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.models.deepseek_v41 import query_graph
from vllm.models.deepseek_v41.cache import V41CacheMetadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires a CUDA device"
)


def test_m5_replay_reads_updated_positions_and_kv_mapping(monkeypatch):
    # This numerical fixture has no collectives; the real TP4 model uses
    # the distributed capture context and is checked by full-model probes.
    stream = torch.cuda.Stream()
    monkeypatch.setattr(
        query_graph,
        "graph_capture",
        lambda device: nullcontext(SimpleNamespace(stream=stream)),
    )
    metadata = V41CacheMetadata(
        torch.tensor([[3]], device="cuda", dtype=torch.int32),
        torch.arange(5, device="cuda", dtype=torch.int64),
        torch.tensor([False], device="cuda"),
        block_size=16,
        causal=False,
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=cast(Any, {"query": metadata}),
        slot_mapping={"query": metadata.slot_mapping},
    )
    ids = torch.arange(5, device="cuda", dtype=torch.int64)
    positions = ids.clone()
    calls = []

    def forward(input_ids, positions, inputs_embeds):
        calls.append(1)
        views = get_forward_context().attn_metadata
        assert isinstance(views, dict)
        view = views["query"]
        assert isinstance(view, V41CacheMetadata)
        return (
            input_ids
            + positions
            + view.slot_mapping
            + view.block_table[0, 0]
            + torch.where(view.is_prefilling, 10, 0)
        )

    graph = query_graph.V41QueryGraph(forward)
    with override_forward_context(context):
        first = graph(ids, positions).clone()
        torch.testing.assert_close(first, ids * 3 + 3)
        ids.fill_(4)
        positions.add_(5)
        metadata.slot_mapping.add_(1)
        metadata.block_table.fill_(7)
        metadata.is_prefilling.fill_(True)
        output = graph(ids, positions)
        expected = ids + positions + metadata.slot_mapping + 17
        torch.testing.assert_close(output, expected)
        assert graph.captures == 1 and graph.replays == 2 and len(calls) == 2
        with pytest.raises(ValueError, match="input storage"):
            graph(ids.clone(), positions)


def test_greedy_graph_reads_fresh_gathered_rows_and_current_markov_anchor(monkeypatch):
    stream = torch.cuda.Stream()
    monkeypatch.setattr(
        query_graph,
        "graph_capture",
        lambda device: nullcontext(SimpleNamespace(stream=stream)),
    )
    columns = torch.arange(16, device="cuda")[None]

    def forward(hidden, anchor):
        previous = anchor.long()
        tokens = []
        for step in range(5):
            logits = hidden[step : step + 1] + torch.sin(
                columns * 0.7 + previous[:, None] * 0.4
            )
            previous = logits.argmax(-1)
            tokens.append(previous)
        return torch.stack(tokens, dim=1).reshape(-1)

    graph = query_graph.V41GreedyDraftGraph(forward)
    for anchor_id in (0, 13, 5, 15):
        hidden = torch.randn(5, 16, device="cuda")
        anchor = torch.tensor([anchor_id], device="cuda", dtype=torch.int32)
        expected = forward(hidden, anchor)
        torch.testing.assert_close(graph(hidden, anchor), expected, rtol=0, atol=0)
    assert graph.captures == 1 and graph.replays == 4
    with pytest.raises(ValueError, match="geometry changed"):
        graph(hidden.half(), anchor)


def test_context_graphs_replay_current_rows_positions_and_rejected_slots(monkeypatch):
    from vllm.models.deepseek_v41.attention_ops import insert_context_slots

    stream = torch.cuda.Stream()
    monkeypatch.setattr(
        query_graph,
        "graph_capture",
        lambda device: nullcontext(SimpleNamespace(stream=stream)),
    )
    caches = {
        name: torch.zeros(2, 16, 512, device="cuda", dtype=torch.float16)
        for name in ("a", "b", "c")
    }

    def forward(states, positions, slots):
        for name, mapping in slots.items():
            insert_context_slots(
                caches[name], (states + positions[:, None]).half(), mapping
            )

    graphs = query_graph.V41ContextGraphs(forward)
    for rows in (1, 6, 2, 5, 3, 4, 6, 1):
        states = torch.randn(rows, 512, device="cuda", dtype=torch.float16)
        positions = torch.arange(rows, device="cuda") + rows
        slots = {name: torch.arange(1, rows + 1, device="cuda") for name in caches}
        slots["b"].fill_(-1)
        graphs(states, positions, slots)
        for name, mapping in slots.items():
            if name != "b":
                torch.testing.assert_close(
                    caches[name].view(32, 512)[mapping],
                    (states + positions[:, None]).half(),
                )
        assert torch.count_nonzero(caches["b"]) == 0
        assert len(graphs.entries) <= 6
    assert graphs.captures == 6 and graphs.replays == 8
