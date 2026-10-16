# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


class _RecordingDraft:
    def __init__(self, layer_names):
        self._layer_names = layer_names
        self.slots = "unset"

    def get_draft_kv_cache_layer_names(self):
        return self._layer_names

    def precompute_and_store_context_kv(self, hidden, positions, slots):
        self.slots = slots

    def map_draft_to_target(self, ids):
        return ids + 100


def _speculator(model) -> DSparkSpeculator:
    speculator = object.__new__(DSparkSpeculator)
    speculator.model = model
    speculator.draft_logits = None
    return speculator


def test_context_slots_are_passed_per_cache_layer():
    model = _RecordingDraft(["layer.a", "layer.b"])
    speculator = _speculator(model)
    hidden, positions = torch.zeros(3, 4), torch.zeros(3, dtype=torch.int64)
    first, shared = torch.tensor([1, 2, 3]), torch.tensor([7, 8, 9])

    speculator._precompute_context_kv(hidden, positions, [first, None])
    assert model.slots == {"layer.a": first}

    speculator._precompute_context_kv(hidden, positions, shared)
    assert model.slots == {"layer.a": shared, "layer.b": shared}

    speculator._precompute_context_kv(hidden, positions, None)
    assert model.slots is None


def test_greedy_step_maps_the_draft_argmax_to_target_ids():
    speculator = _speculator(_RecordingDraft([]))
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.7, 0.2, 0.1]])
    tokens = speculator._sample_step(
        logits, torch.zeros(2, dtype=torch.int32), torch.ones(2, dtype=torch.int64), 0
    )
    assert tokens.tolist() == [101, 100]


def test_dummy_batch_at_a_context_bound_keeps_query_offsets():
    buffers = InputBuffers(
        max_num_reqs=4, max_num_tokens=16, device=torch.device("cpu")
    )
    batch = InputBatch.make_dummy(2, 12, buffers, context_len=4096)
    assert batch.query_start_loc.tolist() == [0, 6, 12]
    assert batch.seq_lens.tolist() == [4096, 4096]
    assert batch.seq_lens_cpu_upper_bound.tolist() == [4096, 4096]
    assert batch.num_computed_tokens_np.tolist() == [4090, 4090]
    assert batch.capture_context_len == 4096

    plain = InputBatch.make_dummy(2, 12, buffers)
    assert plain.seq_lens.tolist() == [6, 6]
    assert plain.capture_context_len is None
    assert np.array_equal(plain.num_computed_tokens_np, np.zeros(2, dtype=np.int32))


def _manager_with_buckets(buckets):
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 6, 1, uniform_token_count=6)
    variants = tuple(
        BatchExecutionDescriptor(
            CUDAGraphMode.FULL, 6, 1, uniform_token_count=6, attention_context_bucket=b
        )
        for b in buckets
    )
    manager = object.__new__(ModelCudaGraphManager)
    manager._compressed_context_graphs = {desc: variants}
    manager._qsa_short_graphs = {}
    manager._long_attention_graphs = {}
    manager.graphs = {variant: SimpleNamespace() for variant in variants}
    return manager, desc, variants


def test_replay_selects_the_smallest_covering_context_bucket():
    manager, desc, (b2k, b4k, b16k) = _manager_with_buckets((2048, 4096, 16384))
    select = manager.select_attention_graph
    assert select(desc, torch.tensor([300], dtype=torch.int32)) == b2k
    assert select(desc, torch.tensor([2049], dtype=torch.int32)) == b4k
    assert select(desc, torch.tensor([16384], dtype=torch.int32)) == b16k
    # Beyond the largest bucket, or without a host bound: the full-length graph.
    assert select(desc, torch.tensor([16385], dtype=torch.int32)) == desc
    assert select(desc, torch.tensor([], dtype=torch.int32)) == desc


def test_replay_skips_buckets_that_were_not_captured():
    manager, desc, (b2k, b4k) = _manager_with_buckets((2048, 4096))
    del manager.graphs[b2k]
    assert manager.select_attention_graph(desc, torch.tensor([100])) == b4k
