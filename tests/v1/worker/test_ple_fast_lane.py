# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import numpy as np
import pytest
import torch
from torch import nn

import vllm._custom_ops  # noqa: F401
import vllm.models.qwen4_exp.nvidia.ple_layer as ple_module
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding
from vllm.transformers_utils.gguf_rows import PackedGGUFRowReader
from vllm.v1.ple_offload.fast_lane import request_shape

pytestmark = pytest.mark.skipif(
    not hasattr(torch.ops._C, "ple_fast_lane_start"),
    reason="native PLE fast lane is not built",
)

EOS = 248044
TABLE_ROWS = sum([251 + 2 * h for h in range(16)])
SIZES = [251 + 2 * h for h in range(16)]
OFFSETS = [sum(SIZES[:h]) for h in range(16)]
MULTIPLIERS = [3229177723303, 2164907095717, 1840338581269]
MAX_REQS, MAX_TOKENS = 4, 64


def table(rows=TABLE_ROWS):
    rng = np.random.default_rng(1160)
    data = rng.integers(0, 256, (rows, 5, 18), dtype=np.uint8)
    scales = rng.uniform(-2, 2, (rows, 5)).astype(np.float16)
    data[:, :, :2] = scales.view(np.uint8).reshape(rows, 5, 2)
    return data.reshape(rows, 90)


def ngram_module():
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.ngram_size, module.heads_per_ngram, module.ngram_heads = 3, 8, 16
    module.eos_token_id = EOS
    module.register_buffer("positions_buffer", torch.arange(MAX_TOKENS))
    module.register_buffer(
        "padded_buffer", torch.empty(MAX_REQS, MAX_TOKENS, dtype=torch.long)
    )
    module.register_buffer("layer_multipliers", torch.tensor(MULTIPLIERS))
    module.register_buffer("ngram_heads_vocab_sizes", torch.tensor(SIZES))
    module.register_buffer("ngram_heads_offsets", torch.tensor(OFFSETS))
    return module


class Lane:
    def __init__(self, data, cache_rows, clamp=True, targets=2):
        self.request = torch.zeros(
            request_shape(MAX_REQS, MAX_TOKENS), dtype=torch.long
        )
        self.results = [
            torch.full((MAX_TOKENS, 1440), 0xEE, dtype=torch.uint8)
            for _ in range(targets)
        ]
        self.flags = [torch.zeros(16, dtype=torch.int32) for _ in range(targets)]
        self.data = data
        self.seq = 0
        self.handle = torch.ops._C.ple_fast_lane_start(
            self.request,
            MAX_REQS,
            MAX_TOKENS,
            data.ctypes.data,
            90,
            data.shape[0],
            MULTIPLIERS,
            SIZES,
            OFFSETS,
            EOS,
            [r.data_ptr() for r in self.results],
            [f.data_ptr() for f in self.flags],
            cache_rows * 90,
            1000,
            clamp,
            None,
            0,
            0,
        )

    def stats(self):
        return torch.ops._C.ple_fast_lane_stats(self.handle).tolist()

    def submit(self, tokens, starts, context):
        header = 8
        self.request[header : header + len(starts)] = torch.tensor(starts)
        base = header + MAX_REQS + 1
        flat = [v for pair in context for v in pair]
        self.request[base : base + len(flat)] = torch.tensor(flat)
        ids = base + MAX_REQS * 2
        self.request[ids : ids + len(tokens)] = torch.tensor(tokens)
        self.request[1] = len(tokens)
        self.request[2] = len(starts) - 1
        self.seq += 1
        self.request[0] = self.seq  # x86 stores are ordered; the body precedes.
        deadline = time.monotonic() + 5
        while not all(int(f[0]) == 1 for f in self.flags):
            assert time.monotonic() < deadline, self.stats()
            time.sleep(1e-4)
        outputs = [r[: len(tokens)].clone() for r in self.results]
        for f in self.flags:
            f[0] = 0  # consumers release the result
        return outputs

    def close(self):
        torch.ops._C.ple_fast_lane_stop(self.handle)


def expected(module, reader, tokens, starts, context, clamp):
    ids = torch.tensor(tokens, dtype=torch.int32)
    if clamp:
        ids = ids.clamp_min(0)
    rows = module.compute_ngram_ids(
        ids,
        torch.tensor(starts, dtype=torch.int32),
        torch.tensor(context, dtype=torch.int32),
    )
    return torch.from_numpy(
        reader.lookup_packed_iq4nl(rows.numpy()).reshape(len(tokens), -1)
    )


@pytest.mark.parametrize("cache_rows", [0, 64, 1 << 16])
@pytest.mark.parametrize("clamp", [True, False])
def test_native_lane_matches_python_owner(monkeypatch, cache_rows, clamp):
    monkeypatch.setattr(ple_module, "is_offload_process", lambda: True)
    data = table()
    reader = PackedGGUFRowReader(data, 20, 160)
    module = ngram_module()
    lane = Lane(data, cache_rows, clamp=clamp)
    rng = np.random.default_rng(7)
    try:
        cases = [
            ([5, 9, 5, 9, 5], [0, 5], [[1, 2]]),  # repeats hit the cache
            ([3, -1, 7, -1, 4, 4], [0, 2, 6], [[EOS, 8], [9, EOS]]),
            ([11, 12, 13, 14], [0, 2, 2, 4], [[1, 1], [2, 2], [3, 3]]),
            ([20, 21, 22, 23, 24, 25, 26, 27], [0, 5], [[30, 31]]),  # padding
        ]
        for _ in range(12):
            reqs = int(rng.integers(1, MAX_REQS + 1))
            lengths = rng.integers(0, 6, reqs)
            starts = [0, *np.cumsum(lengths).tolist()]
            total = max(starts[-1], 1) + int(rng.integers(0, 3))
            tokens = rng.integers(-1, 200000, total).tolist()
            tokens = [EOS if t % 9 == 0 else t for t in tokens]
            context = rng.integers(0, 200000, (reqs, 2)).tolist()
            context = [[EOS if v % 4 == 0 else v for v in pair] for pair in context]
            cases.append((tokens, starts, context))
        for tokens, starts, context in cases:
            outputs = lane.submit(tokens, starts, context)
            want = expected(module, reader, tokens, starts, context, clamp)
            for out in outputs:
                assert torch.equal(out, want)
        stats = lane.stats()
        assert stats[0] == len(cases) and stats[6] == 0
        if cache_rows >= 1 << 16:
            assert stats[2] > 0  # hits
    finally:
        lane.close()


def test_native_lane_waits_for_consumers_and_reports_bad_rows(monkeypatch):
    monkeypatch.setattr(ple_module, "is_offload_process", lambda: True)
    data = table()
    lane = Lane(data, 0)
    try:
        lane.submit([1, 2], [0, 2], [[3, 4]])
        # A consumer that has not released blocks the next publication.
        lane.flags[1][0] = 1
        lane.request[0] = lane.seq + 1
        time.sleep(0.05)
        assert int(lane.flags[0][0]) == 0
        lane.flags[1][0] = 0
        deadline = time.monotonic() + 5
        while int(lane.flags[0][0]) != 1:
            assert time.monotonic() < deadline
            time.sleep(1e-4)
        lane.seq += 1
        for f in lane.flags:
            f[0] = 0
    finally:
        lane.close()

    module = ngram_module()
    rows = module.compute_ngram_ids(
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([[5, 6]], dtype=torch.int32),
    ).reshape(-1)
    bad = data.copy()
    blocks = bad.reshape(-1, 5, 18)
    blocks[rows[3], :, :2] = (
        np.full(5, 1024, dtype=np.float16).view(np.uint8).reshape(5, 2)
    )
    blocks[rows[3], :, 2:] = 0xFF
    lane = Lane(bad, 0)
    try:
        lane.request[8:10] = torch.tensor([0, 1])
        lane.request[13:15] = torch.tensor([5, 6])
        lane.request[21] = 7
        lane.request[1], lane.request[2] = 1, 1
        lane.request[0] = 1
        deadline = time.monotonic() + 5
        while lane.stats()[6] == 0:
            assert time.monotonic() < deadline
            time.sleep(1e-3)
        assert lane.stats()[6] == 3  # FP16 overflow, never published
        assert all(int(f[0]) == 0 for f in lane.flags)
    finally:
        lane.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_published_requests_are_served_natively(monkeypatch):
    from vllm.v1.ple_offload.fast_lane import FastLaneRequest

    monkeypatch.setattr(ple_module, "is_offload_process", lambda: True)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    data = table()
    reader = PackedGGUFRowReader(data, 20, 160)
    module = ngram_module()
    slot = FastLaneRequest(device, MAX_REQS, MAX_TOKENS)
    results = [torch.zeros((MAX_TOKENS, 1440), dtype=torch.uint8) for _ in range(2)]
    flags = [torch.zeros(16, dtype=torch.int32) for _ in range(2)]
    handle = torch.ops._C.ple_fast_lane_start(
        slot.host,
        MAX_REQS,
        MAX_TOKENS,
        data.ctypes.data,
        90,
        data.shape[0],
        MULTIPLIERS,
        SIZES,
        OFFSETS,
        EOS,
        [r.data_ptr() for r in results],
        [f.data_ptr() for f in flags],
        1 << 20,
        1000,
        True,
        None,
        0,
        0,
    )
    input_ids = torch.zeros(MAX_TOKENS, dtype=torch.int32, device=device)
    starts = torch.zeros(MAX_REQS + 1, dtype=torch.int32, device=device)
    context = torch.zeros((MAX_REQS, 2), dtype=torch.int32, device=device)
    try:
        rng = np.random.default_rng(11)
        for step in range(6):
            reqs = 1 + step % MAX_REQS
            lengths = rng.integers(1, 6, reqs)
            qsl = [0, *np.cumsum(lengths).tolist()]
            tokens = rng.integers(-1, 200000, qsl[-1]).tolist()
            ctx = rng.integers(0, 200000, (reqs, 2)).tolist()
            input_ids[: len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
            starts[: reqs + 1] = torch.tensor(qsl, dtype=torch.int32)
            context[:reqs] = torch.tensor(ctx, dtype=torch.int32)
            slot.publish(input_ids, starts, context, reqs, len(tokens))
            deadline = time.monotonic() + 5
            while not all(int(f[0]) == 1 for f in flags):
                assert time.monotonic() < deadline
                time.sleep(1e-4)
            want = expected(module, reader, tokens, qsl, ctx, True)
            for r in results:
                assert torch.equal(r[: len(tokens)], want)
            for f in flags:
                f[0] = 0
    finally:
        torch.ops._C.ple_fast_lane_stop(handle)
        slot.close()


def test_native_lane_follows_the_draft_ring(tmp_path):
    from vllm.v1.ple_offload.draft_prefetch import ring_shape

    path = tmp_path / "table.bin"
    table().tofile(path)
    data = np.memmap(path, dtype=np.uint8, mode="r", shape=(TABLE_ROWS, 90))
    ring = torch.zeros(ring_shape(MAX_REQS, 4), dtype=torch.long)
    request = torch.zeros(request_shape(MAX_REQS, MAX_TOKENS), dtype=torch.long)
    results = [torch.zeros((MAX_TOKENS, 1440), dtype=torch.uint8)]
    flags = [torch.zeros(16, dtype=torch.int32)]
    handle = torch.ops._C.ple_fast_lane_start(
        request,
        MAX_REQS,
        MAX_TOKENS,
        data.ctypes.data,
        90,
        TABLE_ROWS,
        MULTIPLIERS,
        SIZES,
        OFFSETS,
        EOS,
        [r.data_ptr() for r in results],
        [f.data_ptr() for f in flags],
        0,
        1000,
        True,
        ring,
        MAX_REQS,
        4,
    )

    def positions(count):
        deadline = time.monotonic() + 5
        while torch.ops._C.ple_fast_lane_stats(handle)[11].item() < count:
            assert time.monotonic() < deadline
            time.sleep(1e-3)

    try:
        context = ring[8 : 8 + MAX_REQS * 3].view(MAX_REQS, 3)
        drafts = ring[8 + MAX_REQS * 3 :].view(MAX_REQS, 4)
        context[:2] = torch.tensor([[1, 2, 3], [4, EOS, 6]])
        # Writer order: requests and count first, the round number last.
        ring[2], ring[1], ring[0] = 2, 1, 1  # two requests, bonus known
        positions(2)
        drafts[:2] = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14]])
        ring[1] = 3  # bonus, d1, d2
        positions(6)
        ring[1] = 9  # beyond the window is clamped to bonus + 4 drafts
        positions(10)
        ring[1], ring[0] = 1, 2  # a new round restarts at the bonus
        positions(12)
        time.sleep(0.02)
        stats = torch.ops._C.ple_fast_lane_stats(handle).tolist()
        assert stats[11] == 12 and stats[6] == 0
    finally:
        torch.ops._C.ple_fast_lane_stop(handle)
