# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise speculative tail history with the real kernels on the CPU interpreter."""

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "pool,spec,expected", [(4, 0, 4), (4, 1, 8), (4, 7, 12), (16, 7, 32)]
)
def test_model_tail_spec_retains_speculative_history(pool, spec, expected):
    import torch

    from vllm.models.glm5next.nvidia.attention import Glm5NextTailCache
    from vllm.v1.core.single_type_kv_cache_manager import KpoolTailManager

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=spec)
        if spec
        else None
    )
    owner = SimpleNamespace(_index_kpool=pool, head_dim=128, dtype=torch.float16)
    result = Glm5NextTailCache.get_kv_cache_spec(owner, config)
    assert result.block_size == result.sliding_window == expected
    assert result.real_page_size_bytes == expected * 2 * 128 * 2
    assert not result.prefix_cacheable
    assert result.max_num_blocks_per_req(config, 215040) == 1
    assert result.max_admission_blocks_per_request(1024, 215040) == 1
    assert KpoolTailManager.find_longest_cache_hit(
        [], 215040, [0, 1], None, result, False, 4608
    ) == ([], [])


def test_kpool_rejection_history_actual_kernels_cpu():
    # A fresh interpreter is required: changing TRITON_INTERPRET after the
    # kernels were imported by another test would silently retain GPU launches.
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1", HF_HUB_OFFLINE="1")
    completed = subprocess.run(
        [sys.executable, __file__, "--worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report == {
        "acceptance_phase_cases": 32,
        "continuous_multi_request_rounds": 16,
        "prefix_chunk_reuse_cases": 4,
        "plain_decode_cases": 4,
        "cuda_initialized": False,
    }


def _run_worker():
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from vllm.models.glm5next.nvidia.ops import kpool_compress as ops
    from vllm.v1.attention.backends.mla.indexer import compute_kpool_tail_slot_mapping

    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    # Native FP8 conversions are supported by the interpreter; the software
    # conversion uses inline PTX. Tail addressing and pooling execute the exact
    # same production kernel for either conversion policy.
    ops.needs_software_fp8 = lambda: False
    pool, head, page = 4, 128, 64
    ape = torch.zeros((pool, head), dtype=torch.float32)

    def payload(position, request, rejected=False):
        value = position + request * 40 + (300 if rejected else 0)
        channels = torch.arange(head, dtype=torch.float32)
        key = (value / 8 + channels / 256).to(torch.float16)
        gate = ((value % 7) / 8 + channels / 512).to(torch.float16)
        return key, gate

    def empty(history):
        raw = torch.full((4, 152064), 0x5A, dtype=torch.uint8)
        tail = torch.as_strided(
            raw.view(torch.float16),
            (4, 2, history, head),
            (152064 // 2, history * head, head, 1),
        )
        tail.zero_()
        return (
            torch.zeros((4, page, head + 4), dtype=torch.uint8),
            tail,
        )

    def tail_slots(positions, blocks, history):
        lengths = [len(row) for row in positions]
        flat = torch.tensor(sum(positions, []), dtype=torch.int64)
        starts = torch.tensor(
            [0] + list(itertools.accumulate(lengths)), dtype=torch.int32
        )
        return compute_kpool_tail_slot_mapping(
            torch.full_like(flat, -1),
            torch.tensor(blocks, dtype=torch.int32).view(-1, 1),
            starts,
            flat,
            flat.numel(),
            len(blocks),
            history,
        )

    def seed(tail, rows, blocks, identities):
        keys, gates = zip(
            *(payload(p, ident) for row, ident in zip(rows, identities) for p in row)
        )
        ops.kpool_seed_tail_cache(
            tail,
            torch.stack(keys),
            torch.stack(gates),
            tail_slots(rows, blocks, tail.shape[2]),
            pool,
            head,
        )

    def update(state, rows, blocks, identities, accepted=None):
        cache, tail = state
        width = max(map(len, rows))
        positions = torch.full((len(rows), width), -1, dtype=torch.int32)
        slots = torch.full_like(positions, -1)
        ring_slots = torch.full_like(positions, -1)
        keys = torch.zeros((len(rows), width, head), dtype=torch.float16)
        gates = torch.zeros_like(keys)
        actual_slots = tail_slots(rows, blocks, tail.shape[2])
        cursor = 0
        for row, (pos, block, identity) in enumerate(zip(rows, blocks, identities)):
            positions[row, : len(pos)] = torch.tensor(pos)
            ring_slots[row, : len(pos)] = actual_slots[cursor : cursor + len(pos)]
            cursor += len(pos)
            for token, position in enumerate(pos):
                keys[row, token], gates[row, token] = payload(
                    position, identity, accepted is not None and token > accepted[row]
                )
                if position % pool == pool - 1:
                    slots[row, token] = block * page + position // pool
        ops.kpool_decode_update_and_maybe_write_cache_batched(
            cache,
            tail,
            ring_slots,
            keys,
            gates,
            ape,
            slots,
            positions,
            pool,
            head,
            round_scale=False,
        )

    def committed_equal(spec, plain, block, last, identity):
        # Future speculative cache entries may differ; every completed,
        # committed pool must match independent one-token execution exactly.
        count = (last + 1) // pool
        left, right = spec[0][block].reshape(-1), plain[0][block].reshape(-1)
        assert torch.equal(left[: count * head], right[: count * head])
        scale_start = page * head
        assert torch.equal(
            left[scale_start : scale_start + count * 4],
            right[scale_start : scale_start + count * 4],
        )
        for position in range(last // pool * pool, last + 1):
            key, gate = payload(position, identity)
            assert torch.equal(spec[1][block, 0, position % spec[1].shape[2]], key)
            assert torch.equal(spec[1][block, 1, position % spec[1].shape[2]], gate)
        assert not spec[1][0].any() and not spec[0][0].any()
        assert not spec[1][3].any() and not spec[0][3].any()
        # Reconstruct the full padded allocation to catch writes that ignore
        # the physical page stride (including the prefill seeding kernel).
        raw = torch.as_strided(spec[1].view(torch.uint8), (4, 152064), (152064, 1))
        assert (raw[:, 2 * spec[1].shape[2] * head * 2 :] == 0x5A).all()

    # Every possible pool phase and accepted draft length, followed by a real
    # pool completion, reproduces the exact previously failing rollback path.
    for phase in range(pool):
        for accepted in range(8):
            spec, plain = empty(12), empty(4)
            start = 4 + phase
            for state in (spec, plain):
                seed(state[1], [list(range(start))], [1], [0])
            update(spec, [list(range(start, start + 8))], [1], [0], [accepted])
            for position in range(start, start + accepted + 1):
                update(plain, [[position]], [1], [0])
            committed_equal(spec, plain, 1, start + accepted, 0)
            for position in range(start + accepted + 1, start + accepted + 5):
                update(spec, [[position]], [1], [0])
                update(plain, [[position]], [1], [0])
                committed_equal(spec, plain, 1, position, 0)

    # Repeated rejection/partial/full acceptance, reordered request rows,
    # unequal query lengths and padding must not corrupt either request.
    spec, plain = empty(12), empty(4)
    starts = [6, 9]
    for state in (spec, plain):
        seed(state[1], [list(range(s)) for s in starts], [1, 2], [0, 1])
    for round_id in range(16):
        order = [0, 1] if round_id % 2 else [1, 0]
        widths = [8, 5]
        accept_counts = [round_id % 8, (7 - round_id) % 5]
        rows = [list(range(starts[r], starts[r] + widths[r])) for r in order]
        update(
            spec, rows, [r + 1 for r in order], order, [accept_counts[r] for r in order]
        )
        for request in range(2):
            for position in range(
                starts[request], starts[request] + accept_counts[request] + 1
            ):
                update(plain, [[position]], [request + 1], [request])
            starts[request] += accept_counts[request] + 1
            committed_equal(spec, plain, request + 1, starts[request] - 1, request)

    # Prefill chunking and reuse of a previously dirty request slot. The
    # slot's old tail need not be zeroed: seeding the live prefix suffices.
    for chunks in ([2], [1, 1, 5], [4, 4, 5], [3, 17, 2]):
        spec, plain = empty(12), empty(4)
        spec[1][1].fill_(999)
        plain[1][1].fill_(999)
        position = 0
        for size in chunks:
            for state in (spec, plain):
                seed(state[1], [list(range(position, position + size))], [1], [2])
            position += size
        update(spec, [list(range(position, position + 8))], [1], [2], [0])
        update(plain, [[position]], [1], [2])
        committed_equal(spec, plain, 1, position, 2)
        for following in range(position + 1, position + 5):
            update(spec, [[following]], [1], [2])
            update(plain, [[following]], [1], [2])
            committed_equal(spec, plain, 1, following, 2)

    # Non-speculative layout retains the original four-slot byte behaviour.
    for phase in range(pool):
        legacy, expanded = empty(4), empty(12)
        start = 4 + phase
        for state in (legacy, expanded):
            seed(state[1], [list(range(start))], [1], [0])
        for position in range(start, start + 12):
            update(legacy, [[position]], [1], [0])
            update(expanded, [[position]], [1], [0])
            committed_equal(expanded, legacy, 1, position, 0)

    assert not torch.cuda.is_initialized()
    print(
        json.dumps(
            dict(
                acceptance_phase_cases=32,
                continuous_multi_request_rounds=16,
                prefix_chunk_reuse_cases=4,
                plain_decode_cases=4,
                cuda_initialized=False,
            )
        )
    )


if __name__ == "__main__":
    assert sys.argv[1:] == ["--worker"]
    _run_worker()
