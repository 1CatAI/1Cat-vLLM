# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fixed-K unrolled routed gate/up schedule is byte-identical."""

import pytest
import torch

import vllm._custom_ops  # noqa: F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 required",
)

BLOCK_BYTES = {18: 98, 21: 110, 22: 82}


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
def test_unrolled_gate_up_matches_current_bytes(kind, m):
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
    unrolled = torch.empty((m, top_k, n // 32, 36), dtype=torch.uint8, device="cuda")
    reference = torch.empty_like(unrolled)
    torch.ops._C.gguf_dp4a_gate_up_sm70_out(unrolled, q8, ids, gate, up, kind, True, 0)
    torch.ops._C.gguf_dp4a_gate_up_sm70_out(
        reference, q8, ids, gate, up, kind, True, 16
    )
    assert torch.equal(unrolled, reference)


def test_unrolled_gate_up_requires_k2560():
    generator = torch.Generator().manual_seed(7)
    gate = lattice_rows(22, 4, 32, 1280, generator)
    q8 = torch.zeros((1, 40, 36), dtype=torch.uint8, device="cuda")
    ids = torch.zeros((1, 2), dtype=torch.int32, device="cuda")
    out = torch.empty((1, 2, 1, 36), dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="row partition"):
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(out, q8, ids, gate, gate, 22, True, 0)
