# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for conservative MRV2 attention graph selection."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.ops.sm70_e4m3_long import MAX_CONTEXT
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
)


@pytest.fixture
def graph_pair():
    manager = ModelCudaGraphManager.__new__(ModelCudaGraphManager)
    ordinary = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 1, 8)
    bounded = replace(ordinary, attention_context_bucket=MAX_CONTEXT)
    manager._long_attention_graphs = {ordinary: bounded}
    manager.graphs = {ordinary: object(), bounded: object()}
    return manager, ordinary, bounded


def test_context_boundary_and_switch_back(graph_pair):
    manager, ordinary, bounded = graph_pair
    for upper, expected in (
        (1024, bounded),
        (MAX_CONTEXT, bounded),
        (MAX_CONTEXT + 1, ordinary),
        (262144, ordinary),
        (32768, bounded),
        (0, ordinary),
    ):
        assert (
            manager.select_attention_graph(ordinary, torch.tensor([upper])) == expected
        )


def test_device_hint_never_copied_to_host(graph_pair):
    manager, ordinary, _ = graph_pair
    # Accessing a device hint's values would fail: selection must inspect the
    # device first and use the full-context graph without requesting a copy.
    device_hint = SimpleNamespace(device=torch.device("cuda"))
    assert manager.select_attention_graph(ordinary, device_hint) == ordinary


def test_other_batch_shapes_and_missing_capture_fall_back(graph_pair):
    manager, ordinary, bounded = graph_pair
    other = replace(ordinary, num_tokens=16, num_reqs=2)
    assert manager.select_attention_graph(other, torch.tensor([1024, 1024])) == other
    assert manager.select_attention_graph(ordinary, torch.tensor([1024, 0])) == ordinary
    del manager.graphs[bounded]
    assert manager.select_attention_graph(ordinary, torch.tensor([1024])) == ordinary


def test_disabled_operator_preserves_original_binding(monkeypatch):
    from vllm.v1.attention.ops.sm70_e4m3_long import MANIFEST_ENV, wrap_long_attention

    monkeypatch.delenv(MANIFEST_ENV, raising=False)

    def original(*args, **kwargs):
        raise AssertionError("Binding inspection must not launch an operator")

    assert wrap_long_attention(original) is original


@pytest.mark.parametrize("query_len", [1, 6, 8])
def test_tail_shapes_use_bounded_graph(query_len):
    manager = ModelCudaGraphManager.__new__(ModelCudaGraphManager)
    ordinary = BatchExecutionDescriptor(CUDAGraphMode.FULL, query_len, 1, query_len)
    bounded = replace(ordinary, attention_context_bucket=MAX_CONTEXT)
    manager._long_attention_graphs = {ordinary: bounded}
    manager.graphs = {ordinary: object(), bounded: object()}
    assert (
        manager.select_attention_graph(ordinary, torch.tensor([MAX_CONTEXT])) == bounded
    )
    assert (
        manager.select_attention_graph(ordinary, torch.tensor([MAX_CONTEXT + 1]))
        == ordinary
    )
