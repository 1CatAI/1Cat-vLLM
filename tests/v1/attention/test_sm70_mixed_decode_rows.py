# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident decode/verify rows inside a mixed prefill batch (SM70 Flash-V100)."""

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends import flash_attn_v100 as fa
from vllm.v1.attention.ops.sm70_e4m3_grouped import (
    grouped_e4m3_fp32_groups_allowed,
)

HEADS = 6
DIM = 256
PAGE = 848
COLS = 32


@pytest.fixture(autouse=True)
def _grouped_available(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "flash_attn_v100",
        SimpleNamespace(flash_attn_grouped_e4m3_fp32_available=lambda v=4: True),
    )
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)


def _fake_grouped(calls):
    def op(q, k, v, table, lengths, *, out, softmax_scale, k_scale, v_scale):
        assert table.shape[0] <= 16
        assert q.shape[0] == table.shape[0] * 8 == lengths.shape[0]
        calls.append(table.shape[0])
        for g in range(table.shape[0]):
            for r in range(8):
                row = g * 8 + r
                n = int(lengths[row])
                out[row] = (
                    (q[row].float() * n + float(table[g, 0])).half()
                    if n > 0
                    else torch.zeros_like(q[row])
                )

    return op


def _impl(kv_cache_dtype="fp8_e4m3", op=None):
    impl = object.__new__(fa.FlashAttnV100Impl)
    impl.kv_cache_dtype = kv_cache_dtype
    impl.use_smallq_decode_xqa = True
    impl.flash_attn_grouped_e4m3_fp32_paged = op
    impl.scale = 0.0625
    impl.smallq_decode_max_query_len = 16
    impl.use_decode_xqa = False
    impl.flash_attn_decode_paged_xqa = None
    impl._flash_v100_window_size = lambda causal: (-1, -1)
    return impl


def _batch(q_lens, seq_lens_device, host_extra=0):
    """A mixed batch: ``q_lens`` per request, cumulative query_start_loc."""
    qsl = [0]
    for n in q_lens:
        qsl.append(qsl[-1] + n)
    num_tokens = qsl[-1]
    query = torch.randn(num_tokens, HEADS, DIM).half()
    block_table = (
        torch.arange(len(q_lens), dtype=torch.int32)[:, None]
        .expand(len(q_lens), COLS)
        .contiguous()
        + 100
    )
    seq_lens = torch.tensor(seq_lens_device, dtype=torch.int32)
    metadata = SimpleNamespace(
        block_table=block_table,
        seq_lens=seq_lens,
        causal=True,
        is_dflash_selector_target=True,
    )
    qsl_cpu = torch.tensor(qsl, dtype=torch.int32)
    seq_cpu = seq_lens + host_extra
    return query, metadata, qsl_cpu, seq_cpu


def _expected(query, metadata, qsl, q_lens):
    out = torch.zeros_like(query)
    for r, n in enumerate(q_lens):
        for j in range(n):
            tok = int(qsl[r]) + j
            length = int(metadata.seq_lens[r]) - n + 1 + j
            out[tok] = (
                query[tok].float() * length + float(metadata.block_table[r, 0])
            ).half()
    return out


# Request 0 is a prefill chunk (not a row); 1, 2, 3 and 4 are decode/verify.
Q_LENS = [100, 8, 5, 1, 16]
SEQ_LENS = [4000, 9000, 70000, 123456, 262000]


def test_plan_layout():
    _, metadata, qsl, seq = _batch(Q_LENS, SEQ_LENS)
    plan = fa._mixed_decode_rows_plan(metadata, qsl, seq, 16, torch.device("cpu"))
    assert plan.rows == (1, 2, 3, 4)
    assert plan.num_groups == 1 + 1 + 1 + 2
    assert plan.max_query_len == 16
    # Tokens are ordered by request and then by position.
    assert plan.src_idx.tolist() == list(range(100, 100 + 8 + 5 + 1 + 16))
    assert plan.token_req.tolist() == [1] * 8 + [2] * 5 + [3] + [4] * 16
    # Each request starts a new group of eight rows.
    assert plan.group_req.tolist() == [1, 2, 3, 4, 4]
    assert plan.dst_idx.tolist()[:8] == list(range(8))
    assert plan.dst_idx.tolist()[8:13] == [8, 9, 10, 11, 12]
    assert plan.dst_idx.tolist()[13] == 16
    assert plan.dst_idx.tolist()[14:] == list(range(24, 40))
    # Cached on the shared per-step metadata.
    assert (
        fa._mixed_decode_rows_plan(metadata, qsl, seq, 16, torch.device("cpu")) is plan
    )


def test_uniform_or_empty_batches_have_no_plan():
    for q_lens, seq in (([8, 8], [900, 900]), ([100, 200], [1000, 2000])):
        _, metadata, qsl, seq_cpu = _batch(q_lens, seq)
        assert (
            fa._mixed_decode_rows_plan(metadata, qsl, seq_cpu, 16, torch.device("cpu"))
            is None
        )


def test_lengths_come_from_the_device_not_the_host_upper_bound():
    # Under async speculation the host shadow is only an upper bound.
    _, metadata, qsl, seq_cpu = _batch(Q_LENS, SEQ_LENS, host_extra=7)
    plan = fa._mixed_decode_rows_plan(metadata, qsl, seq_cpu, 16, torch.device("cpu"))
    lengths = plan.token_lengths(metadata.seq_lens)
    expected = []
    for r in (1, 2, 3, 4):
        expected += [SEQ_LENS[r] - Q_LENS[r] + 1 + j for j in range(Q_LENS[r])]
    assert lengths.tolist() == expected
    assert lengths.dtype == torch.int32
    grouped = plan.group_lengths(metadata.seq_lens)
    assert grouped.shape == (plan.num_groups * 8,)
    # Padding rows stay zero so they can never read the cache.
    assert int((grouped > 0).sum()) == len(expected)
    assert grouped[13:16].tolist() == [0, 0, 0]


@pytest.mark.parametrize("q_lens_extra", [0, 1])
def test_grouped_route_matches_per_token_reference(q_lens_extra):
    calls: list[int] = []
    impl = _impl(op=_fake_grouped(calls))
    q_lens = Q_LENS + [8] * 20 * q_lens_extra
    seq = SEQ_LENS + [5000 + i for i in range(20)] * q_lens_extra
    query, metadata, qsl, seq_cpu = _batch(q_lens, seq, host_extra=3)
    out = torch.full_like(query, 7)
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    k = torch.empty((4, PAGE, 1, DIM), dtype=torch.uint8)
    consumed = impl._run_prefill_prefix_decode_rows(
        layer, query, k, k, metadata, out, qsl, seq_cpu, (-1, -1)
    )
    assert consumed == set(range(1, len(q_lens)))
    expected = _expected(query, metadata, qsl, q_lens)
    # The prefill request's rows are untouched.
    assert torch.equal(out[:100], torch.full_like(out[:100], 7))
    assert torch.equal(out[100:], expected[100:])
    assert max(calls) <= 16
    assert (
        sum(calls)
        == fa._mixed_decode_rows_plan(
            metadata, qsl, seq_cpu, 16, torch.device("cpu")
        ).num_groups
    )


def test_admission_failure_falls_back_without_writing():
    impl = _impl(kv_cache_dtype="fp8_e5m2", op=_fake_grouped([]))
    query, metadata, qsl, seq_cpu = _batch(Q_LENS, SEQ_LENS)
    plan = fa._mixed_decode_rows_plan(metadata, qsl, seq_cpu, 16, torch.device("cpu"))
    out = torch.full_like(query, 7)
    k = torch.empty((4, PAGE, 1, DIM), dtype=torch.uint8)
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    assert not impl._run_mixed_rows_grouped_e4m3(
        layer, query, k, k, metadata, out, plan
    )
    assert torch.equal(out, torch.full_like(out, 7))


def test_scalar_fallback_uses_device_row_lengths():
    seen = {}
    impl = _impl(kv_cache_dtype="fp8_e5m2", op=None)

    def scalar(q, kc, vc, table, lengths, **kwargs):
        seen["lengths"] = lengths.clone()
        seen["table"] = table.clone()
        kwargs["out"].copy_(q)

    impl._call_flash_attn_decode_paged = scalar
    query, metadata, qsl, seq_cpu = _batch(Q_LENS, SEQ_LENS, host_extra=5)
    out = torch.zeros_like(query)
    k = torch.empty((4, PAGE, 1, DIM), dtype=torch.uint8)
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    consumed = impl._run_prefill_prefix_decode_rows(
        layer, query, k, k, metadata, out, qsl, seq_cpu, (-1, -1)
    )
    assert consumed == {1, 2, 3, 4}
    expected = []
    for r in (1, 2, 3, 4):
        expected += [SEQ_LENS[r] - Q_LENS[r] + 1 + j for j in range(Q_LENS[r])]
    assert seen["lengths"].tolist() == expected
    assert seen["table"].shape[0] == len(expected)
    assert torch.equal(out[100:], query[100:])


def _groups_instance():
    return SimpleNamespace(
        kv_cache_dtype="fp8_e4m3",
        use_smallq_decode_xqa=True,
        flash_attn_grouped_e4m3_fp32_paged=object(),
        _flash_v100_window_size=lambda causal: (-1, -1),
    )


def _groups_args(groups=3):
    q = torch.empty((groups * 8, 6, 256), dtype=torch.float16)
    k = torch.empty((1, 848, 1, 256), dtype=torch.uint8)
    table = torch.zeros((groups, 310), dtype=torch.int32)
    lengths = torch.zeros(groups * 8, dtype=torch.int32)
    return q, k, table, lengths


@pytest.mark.parametrize(
    "change",
    [
        None,
        "unavailable",
        "e5m2",
        "rows",
        "too_many_groups",
        "page",
        "dtype",
        "lengths",
        "window",
        "causal",
        "capacity",
        "partition",
    ],
)
def test_group_admission(monkeypatch, change):
    instance = _groups_instance()
    q, k, table, lengths = _groups_args(17 if change == "too_many_groups" else 3)
    causal = True
    if change == "unavailable":
        instance.flash_attn_grouped_e4m3_fp32_paged = None
    elif change == "e5m2":
        instance.kv_cache_dtype = "fp8_e5m2"
    elif change == "rows":
        q = q[:-1]
        lengths = lengths[:-1]
    elif change == "page":
        k = k[:, :817]
    elif change == "dtype":
        q = q.float()
    elif change == "lengths":
        lengths = lengths.long()
    elif change == "window":
        instance._flash_v100_window_size = lambda causal: (4096, 0)
    elif change == "causal":
        causal = False
    elif change == "capacity":
        table = torch.zeros((3, 999), dtype=torch.int32)
    elif change == "partition":
        monkeypatch.setenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", "1024")
    out = torch.empty_like(q)
    assert grouped_e4m3_fp32_groups_allowed(
        instance, q, k, k, table, lengths, causal=causal, out=out
    ) is (change is None)
