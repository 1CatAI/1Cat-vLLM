# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref

import pytest
import torch

from vllm.v1.ple_offload.prefetch import ExactRowPrefetchCache, SampledKeyReader


def test_mailbox_owns_flag_after_registration_is_released():
    payload = torch.tensor([[1, 7, 8, 9]], dtype=torch.int32).share_memory_()
    flag = torch.full((16,), 2, dtype=torch.int32).share_memory_()
    flag_reference = weakref.ref(flag)
    reader = SampledKeyReader(payload, flag)
    del payload, flag
    gc.collect()
    # Check ownership before dereferencing: an unowned address may segfault.
    assert flag_reference() is not None
    assert reader.read() == [(7, 8, 9)]
    del reader
    gc.collect()
    assert flag_reference() is None


def test_reordered_reused_request_slots_preserve_owned_exact_rows():
    cache = ExactRowPrefetchCache(2)
    keys = [(7, 8, 9), (9, 10, 11)]
    rows = torch.arange(14, dtype=torch.uint8).reshape(2, 7)
    expected = rows.clone()
    cache.put(keys, rows)
    rows.zero_()  # Producer staging is reused while the cached bytes survive.
    out = torch.empty_like(rows)
    assert cache.copy(keys[::-1], out)
    assert torch.equal(out, expected.flip(0))
    # A recycled slot with a different token history must fall back.
    out.fill_(255)
    assert not cache.copy([(7, 8, 10), keys[0]], out)
    assert (out == 255).all()
    cache.put([(1, 2, 3)], torch.full((1, 7), 42, dtype=torch.uint8))
    assert cache.resident_bytes == 14
    assert not cache.copy([keys[1]], out[:1])
    with pytest.raises(RuntimeError, match="different row bytes"):
        cache.put([keys[0]], torch.full((1, 7), 99, dtype=torch.uint8))


def test_mailbox_rejects_torn_snapshot_and_accepts_skipped_generations():
    payload = torch.tensor(
        [[1, 7, 8, 9], [0, 0, 0, 0]], dtype=torch.int32
    ).share_memory_()
    flag = torch.zeros(16, dtype=torch.int32).share_memory_()
    reader = SampledKeyReader(payload, flag)
    assert reader.read() is None
    values = iter([2, 4])
    reader._load = lambda *_: next(values)
    assert reader.read() is None
    reader._load = lambda *_: 6
    assert reader.read() == [(7, 8, 9)]
    assert reader.read() is None
    reader._load = lambda *_: 7
    assert reader.read() is None
    payload[0, -1] = 10
    reader._load = lambda *_: 12
    assert reader.read() == [(7, 8, 10)]


def test_worker_uses_prefetched_rows_only_for_matching_decode_keys():
    from types import SimpleNamespace

    from vllm.v1.ple_offload.protocol import PleOffloadRequest
    from vllm.v1.ple_offload.worker import (
        PleOffloadInputBuffers,
        PleOffloadOutputTarget,
        PleOffloadRunner,
    )

    class Layer:
        calls = 0

        def forward_impl(self, hidden, ids, offsets, context, output_buffer):
            self.calls += 1
            result = output_buffer[: len(ids)]
            result.copy_((ids + context.sum(1))[:, None].expand(-1, 7))
            return result

    layer = Layer()
    runner = PleOffloadRunner.__new__(PleOffloadRunner)
    cache = ExactRowPrefetchCache(4)
    output = torch.empty(8, 7, dtype=torch.int32)
    flag = torch.zeros(16, dtype=torch.int32)
    runner._layers = {"layer": layer}
    reader = SimpleNamespace(read=lambda: [(1, 2, 3), (2, 3, 4)])
    runner._sampled_key_readers = {0: reader}
    runner._prefetch_caches = {(0, "layer"): cache}
    runner._prefetch_scratch = {(0, "layer"): torch.empty(8, 7, dtype=torch.int32)}
    runner._pinned_bufs = {0: {"layer": output}}
    runner._worker_targets = {
        0: {
            "layer": [
                PleOffloadOutputTarget(
                    0, None, SimpleNamespace(flag_tensor=flag), None, output
                )
            ]
        }
    }
    runner._input_bufs = {
        0: PleOffloadInputBuffers(
            torch.tensor([4, 3], dtype=torch.int32),
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([[2, 3], [1, 2]], dtype=torch.int32),
        )
    }
    runner._clamp_input_ids = False
    assert runner._service_sampled_keys()
    assert layer.calls == 1
    runner._handle_requests([PleOffloadRequest(0, 2, 2)])
    assert layer.calls == 1
    assert torch.equal(output[:2], torch.tensor([9, 6])[:, None].expand(-1, 7))
    flag.zero_()
    runner._input_bufs[0].input_ids_buf[0] = 5
    runner._handle_requests([PleOffloadRequest(0, 2, 2)])
    assert layer.calls == 2
    assert torch.equal(output[:2], torch.tensor([10, 6])[:, None].expand(-1, 7))
