# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""History staged in windows must reproduce the single-pass prefill readers."""

import importlib

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.host_kv import merge_windows
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_device_capability(70),
    reason="SM70 QSA device history kernels",
)

PAGE, BLOCKS, ROWS, WIDTH = 64, 96, 256, 2048


def _inputs(fp8):
    importlib.import_module("vllm._sm70_qsa_device_C")
    torch.manual_seed(20261021)
    dev = "cuda"
    length = BLOCKS * PAGE
    if fp8:
        # Finite E4M3 codes only.
        history = torch.randint(
            0, 256, (BLOCKS, 2, PAGE, 1, 256), dtype=torch.uint8, device=dev
        ).bitwise_and_(0x77)
        scales = torch.rand(BLOCKS * PAGE * 2, device=dev) * 0.02 + 0.001
    else:
        history = (torch.randn(BLOCKS, 2, PAGE, 1, 256, device=dev) * 0.5).half()
        scales = torch.ones(2, device=dev)
    q = (torch.randn(ROWS, 6, 256, device=dev) * 2).half()
    gate = torch.randn(ROWS, 6, 256, device=dev).half()
    indices = torch.randint(0, length, (ROWS, WIDTH), dtype=torch.int32, device=dev)
    indices[:, -100:] = -1
    order = torch.randperm(BLOCKS, device=dev).int()
    table = order[None, :].contiguous()
    requests = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    positions = torch.full((ROWS,), length - 1, dtype=torch.long, device=dev)
    positions[: ROWS // 2] = torch.randint(0, length, (ROWS // 2,), device=dev)
    lengths = torch.tensor([length], dtype=torch.int32, device=dev)
    return history, scales, q, gate, indices, order, table, requests, positions, lengths


@pytest.mark.parametrize("fp8", [True, False])
@pytest.mark.parametrize("windows", [2, 3, 5])
def test_windowed_history_matches_single_pass(fp8, windows):
    ops = torch.ops.vllm_sm70_qsa_device
    (history, scales, q, gate, indices, order, table, requests, positions, lengths) = (
        _inputs(fp8)
    )
    step, splits = 64, WIDTH // 64
    partial = torch.empty(step * 6 * splits * 256, device="cuda")
    state = torch.empty(step * 6 * splits * 2, device="cuda")
    reference = torch.empty_like(q)
    if fp8:
        ops.run_rows(
            q,
            history,
            scales,
            indices,
            table,
            requests,
            positions,
            lengths,
            reference,
            gate,
            12,
        )
    else:
        ops.run_tiled(
            q,
            history,
            scales,
            indices,
            table,
            requests,
            positions,
            lengths,
            reference,
            gate,
            partial,
            state,
            step,
            1,
        )
    out = torch.empty_like(q)
    acc = torch.empty(ROWS, 6, 256, device="cuda")
    part = torch.empty_like(acc)
    acc_state = torch.empty(ROWS, 6, 2, device="cuda")
    part_state = torch.empty_like(acc_state)
    capacity = -(-BLOCKS // windows)
    starts = list(range(0, BLOCKS, capacity))
    for number, first in enumerate(starts):
        last = min(first + capacity, BLOCKS)
        pages = order[first:last].long()
        staged = history[pages].contiguous()
        staged_scales = (
            scales.view(BLOCKS, PAGE * 2)[pages].reshape(-1).contiguous()
            if fp8
            else scales
        )
        window = torch.full_like(table, -1)
        window[0, first:last] = torch.arange(
            last - first, dtype=torch.int32, device="cuda"
        )
        rows_out = acc if number == 0 else part
        rows_state = acc_state if number == 0 else part_state
        if fp8:
            ops.run_rows_window(
                q,
                staged,
                staged_scales,
                indices,
                window,
                requests,
                positions,
                lengths,
                rows_out,
                rows_state,
                12,
            )
        else:
            ops.run_tiled_window(
                q,
                staged,
                staged_scales,
                indices,
                window,
                requests,
                positions,
                lengths,
                partial,
                state,
                rows_out,
                rows_state,
                step,
                1,
            )
        if number:
            final = number == len(starts) - 1
            merge_windows(
                acc,
                acc_state,
                part,
                part_state,
                out if final else None,
                gate if final else None,
            )
    torch.accelerator.synchronize()
    # One FP16 rounding of the output (the merge reorders the softmax sum).
    torch.testing.assert_close(out, reference, rtol=0, atol=2e-3)
