# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("new_tokens", [1, 700, 8160])
def test_prefetched_prefix_then_rest_equals_full_staging(dtype, new_tokens):
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
    table[0, :used] = torch.randperm(blocks)[:used].int()
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
def test_write_staged_equals_copying_back_this_steps_pages(dtype, new_tokens):
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
    table[0, :used] = torch.randperm(blocks)[:used].int()
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
