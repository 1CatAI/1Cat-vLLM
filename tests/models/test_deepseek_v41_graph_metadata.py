# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Captured commits read updated prefill flags at a stable device address."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.cache import V41CacheBuilder
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires a CUDA device"
)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6])
def test_graph_replay_selects_current_prefix_after_prefill_and_rejection(
    monkeypatch, rows
):
    def initialize(self, *args, **kwargs):
        self.device = torch.device("cuda", 0)
        self.kv_cache_spec = SimpleNamespace(block_size=16)

    monkeypatch.setattr(AttentionMetadataBuilder, "__init__", initialize)
    builder = V41CacheBuilder()
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, rows], device="cuda", dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, rows], dtype=torch.int32),
        seq_lens=torch.tensor([rows], device="cuda", dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=rows,
        max_query_len=rows,
        max_seq_len=rows,
        block_table_tensor=torch.zeros(1, 16, device="cuda", dtype=torch.int32),
        slot_mapping=torch.arange(rows, device="cuda", dtype=torch.int64),
        is_prefilling=None,
        is_dummy_batch=True,
        causal=True,
    )
    metadata = builder.build_for_cudagraph_capture(common)
    assert metadata.is_prefilling.is_cuda
    pointer = metadata.is_prefilling.data_ptr()
    counts = torch.tensor([2], device="cuda", dtype=torch.int64)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = torch.where(metadata.is_prefilling, rows, counts)
    for prefill, accepted in (
        (False, min(2, rows)),
        (True, 1),
        (False, 1),
        (False, rows),
    ):
        common.is_dummy_batch = False
        common.is_prefilling = torch.tensor([prefill], dtype=torch.bool)
        current = builder.build(0, common)
        assert current.is_prefilling.data_ptr() == pointer
        counts.fill_(accepted)
        graph.replay()
        torch.accelerator.synchronize()
        assert selected.item() == (rows if prefill else accepted)


def test_decode_graph_waits_for_the_latest_prefill_workspace_record():
    from vllm.model_executor.layers.quantization.gguf_shared_prefill import (
        SharedOriginalPrefill,
    )
    from vllm.model_executor.layers.quantization.gguf_shared_runtime import (
        SharedExpertLayerSpec,
    )

    cache = SimpleNamespace(
        experts=4,
        device=torch.device("cuda", 0),
        specs=(SharedExpertLayerSpec(5120, 2304, 4, 2, (16, 16, 17)),),
    )
    prefill = SharedOriginalPrefill(cache, tile_experts=1)
    prefill.used = True
    producer = torch.cuda.Stream()
    values = torch.zeros(1, device="cuda", dtype=torch.int64)
    with torch.cuda.stream(producer):
        values.fill_(1)
        prefill.done.record()
    producer.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prefill.wait_decode()
        output = values + 1
    for value in (7, 11):
        with torch.cuda.stream(producer):
            torch.cuda._sleep(1_000_000)
            values.fill_(value)
            prefill.done.record()
        graph.replay()
        torch.accelerator.synchronize()
        assert output.item() == value + 1
