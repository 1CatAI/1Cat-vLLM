# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import numpy as np
import pytest
import torch
from torch import nn

import vllm.models.qwen4_exp.nvidia.ple_layer as ple_module
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding
from vllm.v1.ple_offload import draft_prefetch
from vllm.v1.ple_offload.draft_prefetch import (
    DraftRowPrefetcher,
    DraftTokenRing,
    ring_shape,
)

EOS = 248044


def ngram_module():
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.ngram_size = 3
    module.heads_per_ngram = 8
    module.ngram_heads = 16
    module.eos_token_id = EOS
    module.register_buffer("positions_buffer", torch.arange(64))
    module.register_buffer("padded_buffer", torch.empty(4, 64, dtype=torch.long))
    module.register_buffer(
        "layer_multipliers",
        torch.tensor([3229177723303, 2164907095717, 1840338581269]),
    )
    module.register_buffer(
        "ngram_heads_vocab_sizes",
        torch.tensor([20000003 + 14 * i for i in range(16)]),
    )
    module.register_buffer(
        "ngram_heads_offsets", torch.tensor([20000003 * i for i in range(16)])
    )
    return module


@pytest.mark.parametrize("split", [1, 2, 5])
def test_draft_row_ids_match_the_verify_window(monkeypatch, split):
    """Prefetching positions incrementally names exactly the verify rows."""
    monkeypatch.setattr(ple_module, "is_offload_process", lambda: True)
    module = ngram_module()
    generator = torch.Generator().manual_seed(7)
    for _ in range(16):
        window = torch.randint(0, 248050, (7,), generator=generator).tolist()
        for position in (0, 1, 3):
            window[position] = EOS if position % 2 else window[position]
        context, tokens = window[:2], window[2:]
        expected = module.compute_ngram_ids(
            torch.tensor(tokens, dtype=torch.int32),
            torch.tensor([0, 5], dtype=torch.int16),  # reference Torch path
            torch.tensor([context], dtype=torch.int32),
        )
        pieces = []
        for begin in range(0, 5, split):
            end = min(begin + split, 5)
            pieces.append(
                module.draft_prefetch_row_ids(
                    window[2 + begin : 2 + end], window[begin : begin + 2]
                )
            )
        np.testing.assert_array_equal(
            np.concatenate(pieces), expected.reshape(-1).numpy()
        )


class RecordingPrefetcher(DraftRowPrefetcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.advised: list[tuple[int, int, int]] = []
        self._madvise = lambda address, size, advice: self.advised.append(
            (address, size, advice)
        )


def ring_with(max_reqs=2, num_spec=4):
    return torch.zeros(ring_shape(max_reqs, num_spec), dtype=torch.int64)


def test_prefetch_advises_pages_of_requested_rows_only():
    ring = ring_with()
    base, row_bytes, total = 1 << 30, 90, 1000
    seen = []

    def row_ids(tokens, context):
        seen.append((list(tokens), list(context)))
        return np.array([t % total for t in tokens] + [total + 5, -1])

    prefetcher = RecordingPrefetcher(ring, 2, 4, [(row_ids, base, row_bytes, total)])
    window = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
    prefetcher.prefetch(window, 1, 3)
    assert seen == [([4, 5], [2, 3]), ([11, 12], [9, 10])]
    page = prefetcher._page
    rows = np.array([4, 5, 11, 12])
    expected = set(((base + rows * row_bytes) // page).tolist())
    expected |= set(((base + rows * row_bytes + row_bytes - 1) // page).tolist())
    assert sorted(a // page for a, _, _ in prefetcher.advised) == sorted(expected)
    assert all(
        size == page and advice == draft_prefetch._MADV_WILLNEED
        for _, size, advice in prefetcher.advised
    )
    assert prefetcher.positions == 2


def test_prefetch_thread_follows_rounds_and_positions():
    ring = ring_with(max_reqs=1, num_spec=4)
    calls = []

    def row_ids(tokens, context):
        calls.append((list(tokens), list(context)))
        return np.empty(0, dtype=np.int64)

    prefetcher = DraftRowPrefetcher(ring, 1, 4, [(row_ids, 4096, 90, 10)], poll_s=1e-4)
    header = ring[:8]
    body = ring[8:]

    def wait_for(count):
        deadline = time.monotonic() + 5
        while len(calls) < count:
            assert time.monotonic() < deadline, calls
            time.sleep(1e-3)

    prefetcher.start()
    try:
        body[:3] = torch.tensor([100, 101, 102])  # context0, context1, bonus
        header[:3] = torch.tensor([1, 1, 1])
        wait_for(1)
        body[3:7] = torch.tensor([103, 104, 105, 106])
        header[1] = 3  # bonus, d1, d2 known
        wait_for(2)
        header[1] = 5
        wait_for(3)
        # A new round restarts at the bonus position.
        body[:3] = torch.tensor([200, 201, 202])
        header[:2] = torch.tensor([2, 1])
        wait_for(4)
    finally:
        prefetcher.close()
    assert calls == [
        ([102], [100, 101]),
        ([103, 104], [101, 102]),
        ([105, 106], [103, 104]),
        ([202], [200, 201]),
    ]
    assert prefetcher.positions == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ring_publishes_window_from_eager_and_graph_work():
    device = torch.device("cuda", torch.accelerator.current_device_index())
    ring = DraftTokenRing(device, max_reqs=3, num_spec=4)
    try:
        ring.validate_registration()
        all_tokens = torch.arange(1000, 1000 + 4 * 32, device=device).view(4, 32)
        num_computed = torch.tensor([10, 1, 7, 0], device=device, dtype=torch.int32)
        idx_mapping = torch.tensor([2, 1, 0], device=device, dtype=torch.int32)
        ring.publish_round(idx_mapping, 2, all_tokens, num_computed, EOS)
        torch.accelerator.synchronize()
        host = ring.host
        assert host[:3].tolist() == [1, 1, 2]
        context = host[8 : 8 + 9].view(3, 3)
        assert context[0].tolist() == [1069, 1070, 1071]  # request 2, end 7
        assert context[1].tolist() == [EOS, 1032, 1033]  # request 1, end 1

        drafts = torch.zeros(8, 4, dtype=torch.int64, device=device)
        step = torch.zeros((), dtype=torch.int64, device=device)
        ring.publish_draft_step(drafts, step)  # warm up before capture
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ring.publish_draft_step(drafts, step)
        for value in range(4):
            drafts[:, : value + 1] = torch.arange(value + 1, device=device) + 500
            step.fill_(value)
            graph.replay()
            torch.accelerator.synchronize()
            assert host[1].item() == value + 2
            published = host[8 + 9 :].view(3, 4)
            assert published[0, : value + 1].tolist() == list(range(500, 501 + value))
        assert host[0].item() == 1  # graph work never advances the round
    finally:
        ring.close()
