# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding
from vllm.v1.ple_offload.protocol import _PLE_OFFLOAD_REQUEST_DECODER, PleOffloadRequest
from vllm.v1.ple_offload.worker import PleOffloadRunner


def mapped_layer(capacity):
    layer = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(layer)
    layer.head_dim = 4
    layer.ngram_embedding = SimpleNamespace(org_vocab_size=8)
    shard = torch.arange(32, dtype=torch.uint8).reshape(8, 4)
    layer._disk_shards = [shard]
    layer._disk_shard_pointers = [shard.data_ptr()]
    layer._disk_shard_size = 8
    layer._release_disk_pages = False
    layer._prefetch_cache_rows = capacity
    layer._prefetch_cache = OrderedDict()
    layer.compute_ngram_ids = Mock(side_effect=lambda ids, *_: ids.reshape(-1, 1))
    return layer, shard.numpy().copy()


def warm(layer, ids):
    layer.prefetch_rows(
        torch.tensor(ids), torch.tensor([0, len(ids)]), torch.tensor([[0, 0]])
    )


def test_old_demand_protocol_still_decodes():
    request = _PLE_OFFLOAD_REQUEST_DECODER.decode(
        msgspec.msgpack.encode(dict(dp_rank=1, num_tokens=5, num_reqs=1))
    )
    assert request.prefetch_ids is None and request.prefetch_context is None


def test_wrong_prediction_preserves_demand_and_bounds_cache():
    layer, reference = mapped_layer(2)
    warm(layer, [0, 1, 2])
    assert list(layer._prefetch_cache) == [1, 2]
    # A rejected draft warmed unrelated rows; exact demand still reads disk.
    actual = layer._gather_mapped_rows(np.array([6, 2, 6, 1]))
    np.testing.assert_array_equal(actual, reference[[6, 2, 6, 1]])
    assert sum(row.nbytes for row in layer._prefetch_cache.values()) <= 8


def test_cached_rows_avoid_mmap_reads(monkeypatch):
    layer, reference = mapped_layer(2)
    warm(layer, [1, 2])

    def unexpected_read(*_):
        pytest.fail("cached raw FP8 row was read from mmap again")

    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.ple_layer.ctypes.memmove", unexpected_read
    )
    np.testing.assert_array_equal(
        layer._gather_mapped_rows(np.array([2, 1, 2])), reference[[2, 1, 2]]
    )
    warm(layer, [1, 2])  # Duplicate hints also cause no I/O.


def test_zero_cache_budget_skips_ngram_and_io():
    layer, _ = mapped_layer(0)
    warm(layer, [1, 2])
    layer.compute_ngram_ids.assert_not_called()
    assert not layer._prefetch_cache


def test_prefetch_does_not_publish_or_overwrite_demand(monkeypatch):
    events: list[object] = []

    class Layer:
        def forward_impl(self, hidden, ids, starts, context, output_buffer):
            events.append("demand")
            output_buffer[: ids.numel(), 0].copy_(ids)
            return output_buffer[: ids.numel()]

        def prefetch_rows(self, ids, starts, context):
            events.append(("prefetch", ids.tolist(), starts.tolist(), context.tolist()))

    runner = PleOffloadRunner.__new__(PleOffloadRunner)
    runner._clamp_input_ids = False
    runner._layers = {"ple": Layer()}
    result = torch.empty(5, 1, dtype=torch.int32)
    flag = torch.zeros(1, dtype=torch.int32)
    runner._worker_targets = {
        0: {
            "ple": [
                SimpleNamespace(
                    copy_stream=None,
                    cpu_output_buffer=result,
                    sem=SimpleNamespace(flag_tensor=flag),
                )
            ]
        }
    }
    ids = torch.tensor([4, 5, 6, 7, 8], dtype=torch.int32)
    runner._input_bufs = {
        0: SimpleNamespace(
            input_ids_buf=ids,
            query_start_loc_buf=torch.tensor([0, 5]),
            ngram_context_buf=torch.tensor([[2, 3]]),
        )
    }
    runner._pinned_bufs = {0: {"ple": result}}
    runner.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=1)
    )
    published = []
    monkeypatch.setattr("vllm.v1.ple_offload.worker.wait_host_resets", lambda _: None)
    monkeypatch.setattr(
        "vllm.v1.ple_offload.worker.publish_host_flag",
        lambda _: published.append("publish"),
    )
    hint = PleOffloadRequest(0, 3, 1, [10, 11, 12], [[8, 9]])
    runner._handle_requests([hint, PleOffloadRequest(0, 5, 1)])
    assert events == ["demand", ("prefetch", [10, 11, 12], [0, 3], [[8, 9]])]
    assert published == ["publish"]
    assert torch.equal(result[:, 0], ids)
    runner._handle_requests([hint])
    assert published == ["publish"]  # Hints alone never signal completion.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_prefix_uses_accepted_history_and_stable_copy_stream():
    from vllm.v1.ple_offload.prefetch import _pack_prefix

    # Batch order differs from request-state order. State 2 accepted one token,
    # state 0 accepted four; unsampled state 1 must not send a hint.
    device = torch.device("cuda", 0)
    mapping = torch.tensor([2, 0, 1], dtype=torch.int32, device=device)
    computed = torch.tensor([5, 1, 3], dtype=torch.int32, device=device)
    sampled = torch.tensor([1, 4, 0], dtype=torch.int32, device=device)
    last = torch.tensor([105, 101, 203], dtype=torch.int32, device=device)
    history = torch.tensor(
        [
            [100, 101, 102, 103, 104, 105],
            [100, 101, 0, 0, 0, 0],
            [200, 201, 202, 203, 0, 0],
        ],
        dtype=torch.int32,
        device=device,
    )
    drafts = torch.tensor(
        [[204, 205, 206, 207], [106, 107, 108, 109], [102, 103, 104, 105]],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(3, 6, dtype=torch.int32, device=device)
    _pack_prefix[(3,)](
        mapping,
        computed,
        sampled,
        last,
        history,
        drafts,
        output,
        history.stride(0),
        drafts.stride(0),
        6,
        2,
        99,
    )
    copy_stream = torch.cuda.Stream(device=device)
    main_stream = torch.cuda.current_stream(device)
    host = torch.empty_like(output, device="cpu", pin_memory=True)
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(main_stream)
        host.copy_(output, non_blocking=True)
        done = torch.cuda.Event()
        done.record(copy_stream)
    # Mutating draft storage on the main stream must not alter the staged prefix.
    drafts.fill_(999)
    done.synchronize()
    assert host.tolist() == [
        [1, 203, 204, 205, 201, 202],
        [1, 105, 106, 107, 103, 104],
        [0, 101, 102, 103, 99, 100],
    ]


@pytest.mark.parametrize("full_graph", [False, True])
def test_prefetch_runs_between_second_and_third_draft_outside_graph(full_graph):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

    drafter = EagleSpeculator.__new__(EagleSpeculator)
    drafter.num_speculative_steps = 4
    drafter.input_buffers = SimpleNamespace(
        positions=torch.zeros(1), query_start_loc=torch.tensor([0, 1])
    )
    drafter.idx_mapping = torch.zeros(1, dtype=torch.int32)
    drafter.current_draft_step = torch.zeros(1, dtype=torch.int32)
    order: list[object] = []

    def draft(*args, **kwargs):
        order.append(int(drafter.current_draft_step.item()))

    drafter.generate_draft = draft
    drafter.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=draft)
    batch = SimpleNamespace(
        cg_mode=CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE, num_tokens=1
    )
    drafter.multi_step_decode(
        1, True, batch, None, draft_prefetch=lambda: order.append("prefetch")
    )
    assert order == [1, "prefetch", 2, 3]
