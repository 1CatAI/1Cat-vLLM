# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4.sm70.qnorm_rope_kv_fp8_insert import (
    sm70_qnorm_rope_kv_fp8_insert,
)
from vllm.utils.math_utils import round_up

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HEAD_BYTES = NOPE_DIM + ROPE_DIM * 2 + 8
TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an exact SM70 CUDA device",
)


def _make_cos_sin_cache(max_position: int, device: str) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=device)
            / ROPE_DIM
        )
    )
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    frequencies = torch.outer(positions, inv_freq)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)


def _reference_kv_rope(
    kv: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    cos_sin = cos_sin_cache[positions].float()
    cos, sin = cos_sin.chunk(2, dim=-1)
    rope = kv[:, NOPE_DIM:].float().view(-1, ROPE_DIM // 2, 2)
    even = rope[..., 0]
    odd = rope[..., 1]
    rotated_even = torch.addcmul(-odd * sin, even, cos)
    rotated_odd = torch.addcmul(odd * cos, even, sin)
    rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(1)
    return rotated.to(torch.float16).to(torch.bfloat16).to(torch.float16)


def test_sm70_kv_rope_insert_is_repeatable_across_physical_blocks():
    torch.manual_seed(7)
    device = "cuda"
    num_tokens = 1024
    num_heads = 8
    block_size = 64
    sequence_blocks = num_tokens // block_size
    block_starts = (5, 37, 69, 101)
    num_blocks = block_starts[-1] + sequence_blocks + 1
    block_stride = round_up(block_size * HEAD_BYTES, TOKEN_DATA_BYTES)

    q_input = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.float16, device=device
    )
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = _make_cos_sin_cache(num_tokens, device)
    expected_rope = _reference_kv_rope(kv, positions, cos_sin_cache)

    gathered_outputs = []
    q_outputs = []
    for block_start in block_starts:
        storage = torch.full(
            (num_blocks * block_stride,), 0xA5, dtype=torch.uint8, device=device
        )
        cache = storage.as_strided(
            (num_blocks, block_size, HEAD_BYTES),
            (block_stride, HEAD_BYTES, 1),
        )
        slot_mapping = block_start * block_size + torch.arange(
            num_tokens, dtype=torch.int64, device=device
        )
        q = q_input.clone()
        q_out = sm70_qnorm_rope_kv_fp8_insert(
            q,
            kv,
            cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            eps=1e-6,
            block_size=block_size,
        )

        gathered = torch.empty(
            (1, num_tokens, HEAD_DIM), dtype=torch.float16, device=device
        )
        dequantize_and_gather_k_cache(
            gathered,
            cache,
            seq_lens=torch.tensor([num_tokens], dtype=torch.int32, device=device),
            gather_lens=None,
            block_table=torch.arange(
                block_start,
                block_start + sequence_blocks,
                dtype=torch.int32,
                device=device,
            ).unsqueeze(0),
            block_size=block_size,
            offset=0,
        )
        gathered_outputs.append(gathered[0].clone())
        q_outputs.append(q_out.clone())

    for gathered in gathered_outputs:
        torch.testing.assert_close(
            gathered[:, NOPE_DIM:], expected_rope, rtol=0, atol=0
        )
        torch.testing.assert_close(gathered, gathered_outputs[0], rtol=0, atol=0)
    for q_out in q_outputs[1:]:
        torch.testing.assert_close(q_out, q_outputs[0], rtol=0, atol=0)
