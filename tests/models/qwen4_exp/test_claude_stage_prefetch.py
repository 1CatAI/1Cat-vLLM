# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


def page_order(layout, blocks, used):
    if layout == "contiguous":
        return torch.arange(used).int()
    if layout == "few_runs":
        width = (used + 2) // 3
        pages = torch.arange(used)
        return ((pages // width) * (width + 17) + pages % width).int()
    return torch.randperm(blocks)[:used].int()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("new_tokens", [1, 700, 8160])
@pytest.mark.parametrize("layout", ["fragmented", "contiguous", "few_runs"])
def test_prefetched_prefix_then_rest_equals_full_staging(dtype, new_tokens, layout):
    torch.manual_seed(new_tokens)
    dev = torch.device("cuda", 0)
    page, blocks, length = 16, 4096, 16 * 1500 + 9
    pool = get_accelerator_view_from_cpu_tensor(
        torch.zeros((blocks, 2, page, 1, 256), dtype=dtype)
    )
    state = HostQSAKV(
        blocks, page, 256, dev, hot_tokens=2048, rows=8, width=2051, history=pool
    )
    used = (length + page - 1) // page
    table = torch.full((1, 2048), -1, dtype=torch.int32)
    table[0, :used] = page_order(layout, blocks, used)
    table = table.to(dev)
    pos = torch.arange(length, device=dev)
    slots = table[0, pos // page].long() * page + pos % page
    k = (torch.randn(length, 1, 256, device=dev) * 0.5).half()
    state.write(k, k * 0.5, slots)
    lengths = torch.tensor([length], dtype=torch.int32, device=dev)
    plan = state.stage_plan(table, lengths, 131072)
    full = state.stage_history(table, lengths, 131072, plan, slot=0)
    full = [t.clone() for t in full[:2]]
    stream = torch.cuda.Stream()
    done, prefix = state.prefetch_prefix(plan, new_tokens, 131072, 1, stream, None)
    assert prefix == (length - new_tokens) // page
    torch.cuda.current_stream().wait_event(done)
    part = state.stage_history(
        table, lengths, 131072, plan, slot=1, staged_pages=prefix
    )
    torch.accelerator.synchronize()
    assert torch.equal(full[0], part[0]) and torch.equal(full[1], part[1])


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("new_tokens", [1, 700, 8160])
@pytest.mark.parametrize("layout", ["fragmented", "contiguous", "few_runs"])
def test_write_staged_equals_copying_back_this_steps_pages(dtype, new_tokens, layout):
    torch.manual_seed(new_tokens + 1)
    dev = torch.device("cuda", 0)
    page, blocks, length = 16, 4096, 16 * 1500 + 9
    pool = get_accelerator_view_from_cpu_tensor(
        torch.zeros((blocks, 2, page, 1, 256), dtype=dtype)
    )
    state = HostQSAKV(
        blocks, page, 256, dev, hot_tokens=2048, rows=8, width=2051, history=pool
    )
    used = (length + page - 1) // page
    table = torch.full((1, 2048), -1, dtype=torch.int32)
    table[0, :used] = page_order(layout, blocks, used)
    table = table.to(dev)
    pos = torch.arange(length, device=dev)
    slots = table[0, pos // page].long() * page + pos % page
    k = (torch.randn(length, 1, 256, device=dev) * 0.5).half()
    v = k * 0.5
    old = length - new_tokens
    state.write(k[:old], v[:old], slots[:old])  # earlier steps
    lengths = torch.tensor([length], dtype=torch.int32, device=dev)
    plan = state.stage_plan(table, lengths, 131072)
    # This step: prefix prefetched before the write, then the write.
    stream = torch.cuda.Stream()
    done, prefix = state.prefetch_prefix(plan, new_tokens, 131072, 1, stream, None)
    torch.cuda.current_stream().wait_event(done)
    state.write(k[old:], v[old:], slots[old:])
    upto = -(-old // page)
    part = state.stage_history(
        table, lengths, 131072, plan, slot=1, staged_pages=prefix, upto=upto
    )
    state.write_staged(k[old:], v[old:], slots[old:], plan, part[0], part[1])
    full = state.stage_history(table, lengths, 131072, plan, slot=0)
    torch.accelerator.synchronize()

    def tokens(history):  # [k, 2, page, 1, D] -> K/V rows of valid tokens
        return history.transpose(0, 1).reshape(2, -1, 256)[:, :length]

    assert torch.equal(tokens(full[0]), tokens(part[0]))
    if dtype == torch.uint8:
        assert torch.equal(full[1][:length], part[1][:length])


@pytest.mark.parametrize("stage_slots", [1, 2])
@pytest.mark.parametrize("length", [4096, 131088, 262144])
def test_stage_slot_reuse_waits_for_previous_reader(stage_slots, length, monkeypatch):
    from vllm.models.qwen4_exp.nvidia import qsa
    from vllm.models.qwen4_exp.nvidia.ops import host_kv

    # Independent workspace owner, with no buffers retained from other tests.
    monkeypatch.setattr(host_kv, "_STAGES", {})
    monkeypatch.setattr(
        qsa, "_PREFETCH", {"stream": None, "free": [None, None], "pending": {}}
    )
    page = 16
    blocks = (length + page - 1) // page
    table = torch.randperm(blocks, device="cuda", dtype=torch.int32)[None]
    lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
    states = []
    for owner in range(3):
        pool = get_accelerator_view_from_cpu_tensor(
            torch.full((blocks, 2, page, 1, 256), owner + 1, dtype=torch.uint8)
        )
        state = HostQSAKV(
            blocks,
            page,
            256,
            torch.device("cuda", 0),
            hot_tokens=16,
            rows=8,
            width=2051,
            history=pool,
        )
        state.host_scales.fill_(owner + 0.5)
        states.append(state)
    monkeypatch.setattr(qsa, "_TARGET_HOST_KV", states)
    capacity = 262144
    plan = states[0].stage_plan(table, lengths, capacity)
    assert plan is not None
    results = []
    for owner, state in enumerate(states):
        pending = qsa._PREFETCH["pending"].pop(id(state), None)
        prefix = 0
        if pending is not None:
            assert pending[0] is plan
            torch.cuda.current_stream().wait_event(pending[1])
            prefix = pending[2]
        staged = state.stage_history(
            table,
            lengths,
            capacity,
            plan,
            slot=owner % stage_slots,
            staged_pages=prefix,
        )
        # Delay the read so an unfenced side stream can overwrite this buffer.
        torch.cuda._sleep(5000000)
        results.append((staged[0].clone(), staged[1][:length].clone()))
        qsa._prefetch_next_prefix(owner, plan, 32, capacity, stage_slots)
    torch.accelerator.synchronize()
    for owner, (history, scales) in enumerate(results):
        assert torch.all(history == owner + 1)
        assert torch.all(scales == owner + 0.5)
    assert len(host_kv._STAGES) == stage_slots


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_multi_request_shared_prefix_dedup_prefetch_and_write_staged(dtype):
    torch.manual_seed(42)
    dev = torch.device("cuda", 0)
    page, blocks = 16, 512
    pool = get_accelerator_view_from_cpu_tensor(
        torch.zeros((blocks, 2, page, 1, 256), dtype=dtype)
    )
    state = HostQSAKV(
        blocks, page, 256, dev, hot_tokens=2048, rows=8, width=2051, history=pool
    )
    # 3 requests sharing 3 prefix blocks [10, 11, 12] (48 tokens):
    # - req 0: old=54 (3 shared + 6 tokens in block 20), new=20 -> len=74 (blocks 20,21)
    # - req 1: old=48 (3 shared), new=30 -> len=78 (pure-new blocks 30, 31)
    # - req 2: old=48 (3 shared), new=0 -> len=48
    table_cpu = torch.full((3, 8), -1, dtype=torch.int32)
    table_cpu[0, :5] = torch.tensor([10, 11, 12, 20, 21], dtype=torch.int32)
    table_cpu[1, :5] = torch.tensor([10, 11, 12, 30, 31], dtype=torch.int32)
    table_cpu[2, :3] = torch.tensor([10, 11, 12], dtype=torch.int32)
    table = table_cpu.to(dev)
    lengths = torch.tensor([74, 78, 48], dtype=torch.int32, device=dev)
    query_lens = [20, 30, 0]

    # Pre-step writes: shared 48 tokens + req 0's 6 partial tokens in block 20.
    shared_slots = torch.cat(
        [torch.arange(b * page, (b + 1) * page, device=dev) for b in (10, 11, 12)]
    )
    req0_old_slots = torch.arange(20 * page, 20 * page + 6, device=dev)
    old_slots = torch.cat([shared_slots, req0_old_slots])
    k_old = (torch.randn(old_slots.numel(), 1, 256, device=dev) * 0.4).half()
    state.write(k_old, k_old * 0.5, old_slots)

    plan = state.stage_plan(table, lengths, 131072, query_lens=query_lens)
    assert plan is not None
    # Unique blocks: [10, 11, 12] (pure_old=3), [20] (partial -> upto=4), [21, 30, 31]
    assert plan[1] == 7
    assert state.prefix_pages(plan, sum(query_lens)) == 3
    assert plan[6] == 4

    stream = torch.cuda.Stream()
    done, prefix = state.prefetch_prefix(plan, sum(query_lens), 131072, 1, stream, None)
    assert prefix == 3
    torch.cuda.current_stream().wait_event(done)

    # In-step writes for req 0 (positions 54..73) and req 1 (positions 48..77).
    pos0 = torch.arange(54, 74, device=dev)
    slots0 = table[0, pos0 // page].long() * page + pos0 % page
    pos1 = torch.arange(48, 78, device=dev)
    slots1 = table[1, pos1 // page].long() * page + pos1 % page
    new_slots = torch.cat([slots0, slots1])
    k_new = (torch.randn(new_slots.numel(), 1, 256, device=dev) * 0.4).half()
    v_new = k_new * 0.5
    state.write(k_new, v_new, new_slots)

    part = state.stage_history(
        table, lengths, 131072, plan, slot=1, staged_pages=prefix, upto=plan[6]
    )
    state.write_staged(k_new, v_new, new_slots, plan, part[0], part[1])
    full = state.stage_history(table, lengths, 131072, plan, slot=0)
    torch.accelerator.synchronize()

    stg_tbl = plan[2].cpu()
    for req, length in enumerate([74, 78, 48]):
        for tok in range(length):
            sb = int(stg_tbl[req, tok // page])
            off = tok % page
            assert torch.equal(full[0][sb, :, off], part[0][sb, :, off])
            if dtype == torch.uint8:
                assert torch.equal(full[1][sb * page + off], part[1][sb * page + off])
