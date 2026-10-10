# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Staged SM70 router and routed gate/up schedules are byte-identical."""

import pytest
import torch

import vllm._custom_ops  # noqa: F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0)
    or not hasattr(torch.ops._C, "qwen38_router_batch_unstaged_sm70_out"),
    reason="SM70 staged kernels required",
)

BLOCK_BYTES = {18: 98, 21: 110, 22: 82}


@pytest.mark.parametrize("m", [2, 5, 8, 9, 16])
def test_staged_router_matches_unstaged_bytes(m):
    torch.manual_seed(m)
    x = torch.randn(m, 2560, dtype=torch.float16, device="cuda")
    packed = torch.randn(64, 40, 2, 4, 8, 8, dtype=torch.float16, device="cuda") * 0.02
    staged = torch.empty(m, 512, dtype=torch.float16, device="cuda")
    reference = torch.empty_like(staged)
    torch.ops._C.qwen38_router_batch_sm70_out(staged, x, packed)
    torch.ops._C.qwen38_router_batch_unstaged_sm70_out(reference, x, packed)
    assert torch.equal(staged.view(torch.int16), reference.view(torch.int16))


def lattice_rows(kind, experts, n, k, generator):
    width = k // 256 * BLOCK_BYTES[kind]
    stride = (width + 7) // 8 * 8
    rows = torch.randint(
        0, 256, (experts, n, stride), dtype=torch.uint8, generator=generator
    )
    # Keep the FP16 block scales finite and modest.
    scales = (torch.rand(experts, n, k // 256, generator=generator) * 0.02).half()
    blocks = rows[:, :, :width].reshape(experts, n, k // 256, BLOCK_BYTES[kind])
    blocks[..., :2] = scales.view(torch.uint8).reshape(experts, n, k // 256, 2)
    rows[:, :, :width] = blocks.reshape(experts, n, width)
    rows[:, :, width:] = 0
    return rows.cuda()


@pytest.mark.parametrize("kind", [18, 21, 22])
@pytest.mark.parametrize("m", [1, 5, 20])
def test_staged_gate_up_matches_current_bytes(kind, m):
    generator = torch.Generator().manual_seed(kind * 100 + m)
    experts, n, k, top_k = 64, 160, 2560, 10
    gate = lattice_rows(kind, experts, n, k, generator)
    up = lattice_rows(kind, experts, n, k, generator)
    x = torch.randn(m, k, dtype=torch.float16, device="cuda")
    q8 = torch.empty((m, k // 32, 36), dtype=torch.uint8, device="cuda")
    torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
    ids = (
        torch.stack(
            [torch.randperm(experts, generator=generator)[:top_k] for _ in range(m)]
        )
        .int()
        .cuda()
    )
    staged = torch.empty((m, top_k, n // 32, 36), dtype=torch.uint8, device="cuda")
    reference = torch.empty_like(staged)
    torch.ops._C.gguf_dp4a_gate_up_sm70_out(staged, q8, ids, gate, up, kind, True, 0)
    torch.ops._C.gguf_dp4a_gate_up_sm70_out(
        reference, q8, ids, gate, up, kind, True, 16
    )
    assert torch.equal(staged, reference)
