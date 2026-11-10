# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context prefill through host QSA KV against an exact device reference.

Mirrors serving: 512-token chunks are written, then attended in 8-row slices
with causal 2051-wide sparse selections over a context longer than the
selection width, with an 8K hot cache. The reference reads identical history
bytes resident on the device, so the hot cache must not change any output.
"""

import pytest
import torch

pytest.importorskip("vllm._sm70_qsa_device_C")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
    pytest.skip("requires SM70", allow_module_level=True)

from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops.host_kv_attention import (  # noqa: E402
    host_qsa_attention,
)

WIDTH, PAGE, CHUNK, ROWS = 2051, 64, 512, 8


def _selection(position, generator):
    """The selector's layout: complete 4-token groups, then the open tail."""
    visible = position + 1
    complete, tail = visible // 4, visible % 4
    groups = torch.arange(complete)
    if complete > WIDTH // 4:
        groups = torch.randperm(complete, generator=generator)[: WIDTH // 4]
    chosen = (groups.sort().values[:, None] * 4 + torch.arange(4)).flatten()
    chosen = torch.cat([chosen, torch.arange(complete * 4, visible)[: WIDTH % 4]])
    assert tail <= 3
    out = torch.full((WIDTH,), -1, dtype=torch.int32)
    out[: chosen.numel()] = chosen.to(torch.int32)
    return out


def _workspace(rows, dev):
    splits = (WIDTH + 63) // 64
    return (
        torch.empty(rows * 6 * splits * 256, dtype=torch.float32, device=dev),
        torch.empty(rows * 6 * splits * 2, dtype=torch.float32, device=dev),
    )


def _run(mode, dtype, context, hot_tokens=8192, seed=0):
    """mode: native / triton (host paths) or reference (device history)."""
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(seed)
    blocks = context // PAGE + 8
    state = HostQSAKV(
        blocks,
        PAGE,
        256,
        dev,
        hot_tokens=hot_tokens,
        rows=ROWS,
        width=WIDTH,
        dtype=dtype,
        direct_device=(mode == "native"),
    )
    assert (state.device_history_workspace is not None) == (mode == "native")
    table = torch.randperm(blocks, generator=g).to(torch.int32)[None].to(dev)
    keys = torch.randn(context, 1, 256, generator=g).half().to(dev)
    values = torch.randn(context, 1, 256, generator=g).half().to(dev)
    queries = torch.randn(context, 6, 256, generator=g).half().to(dev)
    lengths = torch.tensor([context], dtype=torch.int32, device=dev)
    outputs = torch.empty_like(queries)
    for start in range(0, context, CHUNK):
        stop = min(start + CHUNK, context)
        tokens = torch.arange(start, stop, device=dev)
        slots = (table[0].long()[tokens // PAGE] * PAGE + tokens % PAGE).int()
        state.write(keys[start:stop], values[start:stop], slots)
        lengths.fill_(stop)
        if mode == "reference":
            torch.accelerator.synchronize()
            history, scales = state.history.clone(), state.scales.clone()
        for row in range(start, stop, ROWS):
            end = min(row + ROWS, stop)
            idx = torch.stack([_selection(p, g) for p in range(row, end)]).to(dev)
            requests = torch.zeros(end - row, dtype=torch.int32, device=dev)
            positions = torch.arange(row, end, device=dev, dtype=torch.int64)
            if mode == "reference":
                torch.ops.vllm_sm70_qsa_device.run(
                    queries[row:end],
                    history,
                    scales,
                    idx,
                    table,
                    requests,
                    positions,
                    lengths,
                    outputs[row:end],
                    None,
                    *_workspace(end - row, dev),
                )
                continue
            host_qsa_attention(
                queries[row:end],
                state,
                idx,
                table,
                requests,
                positions,
                lengths,
                outputs[row:end],
                None,
                launch_rows=32,
            )
    torch.accelerator.synchronize()
    return outputs


def _bad_rows(actual, reference, atol):
    worst = (actual.float() - reference.float()).abs().amax(dim=(1, 2))
    return (worst > atol).nonzero().flatten(), worst


@pytest.mark.parametrize("hot_tokens", [8192, 1024])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_native_prefill_matches_device_reference(dtype, hot_tokens):
    context = 7800
    reference = _run("reference", dtype, context, hot_tokens)
    native = _run("native", dtype, context, hot_tokens)
    assert torch.isfinite(native).all()
    bad, worst = _bad_rows(native, reference, 0)
    assert bad.numel() == 0, (
        f"{bad.numel()} rows differ, first {bad[:8].tolist()}, "
        f"max {worst.max().item():.4f}"
    )


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_triton_host_prefill_matches_device_reference(dtype):
    context = 7800
    reference = _run("reference", dtype, context)
    triton = _run("triton", dtype, context)
    bad, worst = _bad_rows(triton, reference, 2e-3)
    assert bad.numel() == 0, (
        f"{bad.numel()} rows differ, first {bad[:8].tolist()}, "
        f"max {worst.max().item():.4f}"
    )


def _attend(state, mode, queries, keys, values, table, context, g, outputs):
    dev = queries.device
    lengths = torch.tensor([context], dtype=torch.int32, device=dev)
    for start in range(0, context, CHUNK):
        stop = min(start + CHUNK, context)
        tokens = torch.arange(start, stop, device=dev)
        slots = (table[0].long()[tokens // PAGE] * PAGE + tokens % PAGE).int()
        state.write(keys[start:stop], values[start:stop], slots)
        lengths.fill_(stop)
        if mode == "reference":
            torch.accelerator.synchronize()
            history, scales = state.history.clone(), state.scales.clone()
        for row in range(start, stop, ROWS):
            end = min(row + ROWS, stop)
            idx = torch.stack([_selection(p, g) for p in range(row, end)]).to(dev)
            requests = torch.zeros(end - row, dtype=torch.int32, device=dev)
            positions = torch.arange(row, end, device=dev, dtype=torch.int64)
            if mode == "reference":
                torch.ops.vllm_sm70_qsa_device.run(
                    queries[row:end],
                    history,
                    scales,
                    idx,
                    table,
                    requests,
                    positions,
                    lengths,
                    outputs[row:end],
                    None,
                    *_workspace(end - row, dev),
                )
            else:
                host_qsa_attention(
                    queries[row:end],
                    state,
                    idx,
                    table,
                    requests,
                    positions,
                    lengths,
                    outputs[row:end],
                    None,
                    launch_rows=32,
                )


def _two_requests(mode, dtype, first, second, hot_tokens):
    """A long request, then another one reusing its freed blocks."""
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(11)
    blocks = first // PAGE + 8
    state = HostQSAKV(
        blocks,
        PAGE,
        256,
        dev,
        hot_tokens=hot_tokens,
        rows=ROWS,
        width=WIDTH,
        dtype=dtype,
        direct_device=(mode == "native"),
    )
    outs = []
    for context in (first, second):
        table = torch.randperm(blocks, generator=g).to(torch.int32)[None].to(dev)
        keys = torch.randn(context, 1, 256, generator=g).half().to(dev)
        values = torch.randn(context, 1, 256, generator=g).half().to(dev)
        queries = torch.randn(context, 6, 256, generator=g).half().to(dev)
        out = torch.empty_like(queries)
        _attend(state, mode, queries, keys, values, table, context, g, out)
        outs.append(out)
    torch.accelerator.synchronize()
    return outs


@pytest.mark.parametrize("mode", ["native", "triton"])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_second_request_on_reused_blocks_matches_reference(dtype, mode):
    reference = _two_requests("reference", dtype, 7800, 2500, 8192)
    actual = _two_requests(mode, dtype, 7800, 2500, 8192)
    atol = 0 if mode == "native" else 2e-3
    for name, a, r in zip(("first", "second"), actual, reference):
        bad, worst = _bad_rows(a, r, atol)
        assert bad.numel() == 0, (
            f"{name}: {bad.numel()} rows differ, first {bad[:8].tolist()}, "
            f"max {worst.max().item():.4f}"
        )


def _serving_geometry(max_model_len):
    """Kernel pages of 16 tokens, 816-token scheduler blocks from the history
    partition (IDs from 29), and a block table as wide as max_model_len."""
    page, per_block, state_blocks = 16, 51, 29
    history_blocks = -(-max_model_len // 816) + 1
    blocks = (state_blocks + history_blocks) * per_block
    width = max_model_len // page
    return page, per_block, state_blocks, blocks, width


@pytest.mark.parametrize("max_model_len", [32768, 65536, 262144])
@pytest.mark.parametrize("warmup", [False, True])
def test_serving_geometry_two_chunks_match_reference(max_model_len, warmup):
    dev = torch.device("cuda")
    page, per_block, state_blocks, blocks, width = _serving_geometry(max_model_len)
    g = torch.Generator().manual_seed(max_model_len)
    state = HostQSAKV(
        blocks,
        page,
        256,
        dev,
        hot_tokens=8192,
        rows=ROWS,
        width=WIDTH,
        dtype=torch.uint8,
        direct_device=True,
    )
    hist_copy = None
    if warmup:
        # Profiling/warmup traffic: dummy rows written through a zero table.
        dummy = torch.zeros(1, width, dtype=torch.int32, device=dev)
        n = 512
        k = torch.randn(n, 1, 256, generator=g).half().to(dev)
        state.write(k, k, torch.arange(n, dtype=torch.int32, device=dev))
        idx = torch.stack([_selection(p, g) for p in range(n - 8, n)]).to(dev)
        q = torch.randn(8, 6, 256, generator=g).half().to(dev)
        host_qsa_attention(
            q,
            state,
            idx,
            dummy,
            torch.zeros(8, dtype=torch.int32, device=dev),
            torch.arange(n - 8, n, device=dev),
            torch.tensor([n], dtype=torch.int32, device=dev),
            torch.empty_like(q),
            None,
            launch_rows=32,
        )
    # The request's blocks come from the history partition.
    logical_blocks = torch.arange(state_blocks, state_blocks + 2).repeat_interleave(
        per_block
    )
    table = torch.zeros(1, width, dtype=torch.int32)
    table[0, : logical_blocks.numel()] = (
        logical_blocks * per_block + torch.arange(per_block).repeat(2)
    ).int()
    table = table.to(dev)
    context = 1003
    keys = torch.randn(context, 1, 256, generator=g).half().to(dev)
    values = torch.randn(context, 1, 256, generator=g).half().to(dev)
    queries = torch.randn(context, 6, 256, generator=g).half().to(dev)
    lengths = torch.tensor([0], dtype=torch.int32, device=dev)
    worst = 0.0
    for start, stop in ((0, 512), (512, context)):
        t = torch.arange(start, stop, device=dev)
        slots = (table[0].long()[t // page] * page + t % page).int()
        state.write(keys[start:stop], values[start:stop], slots)
        lengths.fill_(stop)
        torch.accelerator.synchronize()
        hist_copy, scale_copy = state.history.clone(), state.scales.clone()
        for base in range(start, stop, 32):
            end32 = min(base + 32, stop)
            for row in range(base, end32, ROWS):
                end = min(row + ROWS, end32)
                idx = torch.stack([_selection(p, g) for p in range(row, end)]).to(dev)
                req = torch.zeros(end - row, dtype=torch.int32, device=dev)
                pos = torch.arange(row, end, device=dev, dtype=torch.int64)
                out = torch.empty(end - row, 6, 256, dtype=torch.float16, device=dev)
                ref = torch.empty_like(out)
                host_qsa_attention(
                    queries[row:end],
                    state,
                    idx,
                    table,
                    req,
                    pos,
                    lengths,
                    out,
                    None,
                    launch_rows=end32 - base,
                )
                torch.ops.vllm_sm70_qsa_device.run(
                    queries[row:end],
                    hist_copy,
                    scale_copy,
                    idx,
                    table,
                    req,
                    pos,
                    lengths,
                    ref,
                    None,
                    *_workspace(end - row, dev),
                )
                torch.accelerator.synchronize()
                worst = max(worst, (out.float() - ref.float()).abs().max().item())
    assert worst == 0.0, f"max diff {worst}"


def test_native_resolves_wide_prefill_slices_exactly():
    """Prefill slices wider than the Triton staging rows resolve in one pass."""
    dev = torch.device("cuda")
    page, blocks, context, slice_rows = 16, 2 * 51 + 51, 1003, 32
    g = torch.Generator().manual_seed(32)
    state = HostQSAKV(
        blocks,
        page,
        256,
        dev,
        hot_tokens=8192,
        rows=ROWS,
        width=WIDTH,
        dtype=torch.uint8,
        direct_device=True,
        resolve_rows=slice_rows,
    )
    assert state.resolve_rows == slice_rows and state.rows == ROWS
    table = torch.arange(blocks, dtype=torch.int32, device=dev)[None]
    keys = torch.randn(context, 1, 256, generator=g).half().to(dev)
    values = torch.randn(context, 1, 256, generator=g).half().to(dev)
    queries = torch.randn(context, 6, 256, generator=g).half().to(dev)
    lengths = torch.zeros(1, dtype=torch.int32, device=dev)
    for start, stop in ((0, 512), (512, context)):
        t = torch.arange(start, stop, device=dev)
        state.write(keys[start:stop], values[start:stop], t.int())
        lengths.fill_(stop)
        torch.accelerator.synchronize()
        hist, scales = state.history.clone(), state.scales.clone()
        for row in range(start, stop, slice_rows):
            end = min(row + slice_rows, stop)
            idx = torch.stack([_selection(p, g) for p in range(row, end)]).to(dev)
            req = torch.zeros(end - row, dtype=torch.int32, device=dev)
            pos = torch.arange(row, end, device=dev, dtype=torch.int64)
            out = torch.empty(end - row, 6, 256, dtype=torch.float16, device=dev)
            ref = torch.empty_like(out)
            host_qsa_attention(
                queries[row:end], state, idx, table, req, pos, lengths, out, None
            )
            torch.ops.vllm_sm70_qsa_device.run(
                queries[row:end],
                hist,
                scales,
                idx,
                table,
                req,
                pos,
                lengths,
                ref,
                None,
                *_workspace(end - row, dev),
            )
            torch.accelerator.synchronize()
            assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (
                f"rows {row}:{end} differ"
            )
