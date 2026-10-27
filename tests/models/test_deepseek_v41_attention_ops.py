# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.models.deepseek_v41.attention_ops import (
    attend_draft_latents,
    attend_paged_latents,
    gather_index_keys,
    insert_context_slots,
    insert_latents,
    select_compressed_positions,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _paged(width, ratio, block=16, count=128):
    # Deliberately padded physical pages and permuted page ids.
    rows = block // ratio
    backing = torch.full(
        (count, rows * width + 32), float("nan"), dtype=torch.half, device="cuda"
    )
    cache = backing[:, : rows * width].view(count, rows, width)
    table = torch.randperm(count, device="cuda").int()[None]
    return backing, cache, table


@pytest.mark.parametrize("ratio", [1, 2])
def test_paged_index_cache_group_boundaries_and_replay(ratio):
    backing, cache, table = _paged(128, ratio)
    positions = torch.arange(63, 69, device="cuda")
    values = torch.arange(6 * 128, device="cuda").view(6, 128).half()
    valid = torch.ones(6, dtype=torch.bool, device="cuda")

    def run():
        insert_latents(
            cache, values, positions, valid, table, block_size=16, ratio=ratio
        )
        return gather_index_keys(
            cache, table, positions, valid, block_size=16, ratio=ratio, max_context=1024
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    values.add_(1)
    graph.replay()
    for i in range(6):
        p = int(positions[i])
        if (p + 1) % ratio == 0:
            torch.testing.assert_close(result[p // ratio], values[i], rtol=0, atol=0)
    assert not result[69 // ratio :].any()
    assert torch.isnan(backing[:, -32:]).all()


@pytest.mark.parametrize("ratio", [1, 2])
def test_top512_is_retained_above_short_bypass_threshold_and_masks_future(ratio):
    q = torch.zeros(6, 8, 128, device="cuda", dtype=torch.half)
    q[..., 0] = 1
    weights = torch.ones(6, 8, device="cuda", dtype=torch.half) / 8
    keys = torch.zeros(1024, 128, device="cuda", dtype=torch.half)
    keys[:, 0] = torch.arange(1024, device="cuda") / 1024
    positions = torch.tensor([0, 2, 510, 700, 1023, 1400], device="cuda")
    valid = torch.tensor([True, True, True, True, True, False], device="cuda")
    selected = select_compressed_positions(
        q, weights, keys, positions, valid, ratio, lambda x: x
    )
    assert selected.shape == (6, 512)
    for row, position in enumerate(positions.cpu().tolist()):
        n = min((position + 1) // ratio, 1024) if valid[row] else 0
        actual = selected[row][selected[row] >= 0]
        expected = torch.arange(max(n - 512, 0), n, device="cuda", dtype=torch.int32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_sparse_window_plus_compressed_cache_matches_fp32_softmax_with_sink(ratio):
    _, window, window_table = _paged(512, 1)
    _, compressed, compressed_table = _paged(512, ratio or 1)
    positions = torch.arange(253, 259, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(41)
    window_values = (
        torch.randn(259, 512, generator=generator, device="cuda").half() * 0.1
    )
    compressed_values = (
        torch.randn(259, 512, generator=generator, device="cuda").half() * 0.1
    )
    valid_all = torch.ones(259, dtype=torch.bool, device="cuda")
    insert_latents(
        window,
        window_values,
        torch.arange(259, device="cuda"),
        valid_all,
        window_table,
        block_size=16,
    )
    insert_latents(
        compressed,
        compressed_values,
        torch.arange(259, device="cuda"),
        valid_all,
        compressed_table,
        block_size=16,
        ratio=ratio or 1,
    )
    valid = torch.ones(6, device="cuda", dtype=torch.bool)
    valid[-1] = False
    topk = (
        torch.arange(10, device="cuda", dtype=torch.int32)[None]
        .expand(6, -1)
        .contiguous()
    )
    q = torch.randn(6, 16, 512, generator=generator, device="cuda").half()
    sink = torch.randn(16, generator=generator, device="cuda")
    kwargs = dict(window_block=16)
    if ratio:
        kwargs.update(
            compressed_cache=compressed,
            compressed_table=compressed_table,
            compressed_block=16,
            ratio=ratio,
            topk=topk,
        )
    actual = attend_paged_latents(
        q, positions, valid, window, window_table, sink, **kwargs
    )
    for i, p in enumerate(positions.cpu().tolist()):
        if not valid[i]:
            assert not actual[i].any()
            continue
        keys = [window_values[p - 127 : p + 1]]
        if ratio:
            keys.append(compressed_values[(topk[i].long() + 1) * ratio - 1])
        key = torch.cat(keys)
        scores = q[i].float() @ key.float().T * 512**-0.5
        probability = torch.softmax(torch.cat((scores, sink[:, None]), dim=1), dim=-1)[
            :, :-1
        ].half()
        expected = (probability.float() @ key.float()).half()
        torch.testing.assert_close(actual[i], expected, rtol=0.001, atol=2e-6)


@pytest.mark.parametrize("context", [3, 127, 259])
def test_draft_queries_are_noncausal_and_do_not_mutate_context(context):
    _, cache, table = _paged(512, 1)
    generator = torch.Generator(device="cuda").manual_seed(5141)
    values = torch.randn(context, 512, generator=generator, device="cuda").half() / 10
    logical = torch.arange(context, device="cuda")
    slots = table[0, logical // 16].long() * 16 + logical % 16
    insert_context_slots(cache, values, slots)
    # A rejected/padded context row must leave its physical slot unchanged.
    insert_context_slots(cache, torch.ones_like(values[-1:]), slots.new_full((1,), -1))
    before = cache.clone()
    q = torch.randn(5, 16, 512, generator=generator, device="cuda").half() / 10
    kv = torch.randn(5, 512, generator=generator, device="cuda").half() / 10
    positions = torch.arange(context, context + 5, device="cuda")
    valid = torch.tensor([True, True, True, True, False], device="cuda")
    sink = torch.randn(16, generator=generator, device="cuda")

    def run():
        return attend_draft_latents(q, kv, positions, valid, cache, table, sink, 16)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    kv.add_(0.01)
    graph.replay()
    keys = torch.cat((values[-128:], kv[:4]))
    for i in range(4):
        scores = q[i].float() @ keys.float().T * 512**-0.5
        probabilities = torch.softmax(
            torch.cat((scores, sink[:, None]), dim=1), dim=-1
        )[:, :-1].half()
        expected = (probabilities.float() @ keys.float()).half()
        torch.testing.assert_close(actual[i], expected, rtol=0.001, atol=2e-6)
    assert not actual[4].any()
    torch.testing.assert_close(cache, before, rtol=0, atol=0, equal_nan=True)
