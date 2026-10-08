# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native FP16 writer contract, including NaN payloads and untouched slots."""

import math

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="SM70 native KV writer"
)


@pytest.mark.parametrize("layout", ["NHD", "HND", "paged"])
@pytest.mark.parametrize("heads,dim", [(1, 256), (2, 80)])
@torch.inference_mode()
def test_fp16_writer_preserves_all_half_bits_and_padding(layout, heads, dim):
    # Padded QKV input exercises a noncompact token stride. The last row is
    # excluded from writes, while every half encoding appears in live rows.
    tokens = math.ceil(65536 / (heads * dim)) + 1
    bits = (torch.arange(tokens * heads * dim) % 65536).to(torch.int16)
    inputs = torch.empty((tokens, 3, heads, dim), dtype=torch.int16)
    inputs[:, 0] = bits.reshape(tokens, heads, dim)
    inputs[:, 2] = bits.flip(0).reshape(tokens, heads, dim)
    gpu_inputs = inputs.cuda().view(torch.float16)
    key, value = gpu_inputs[:, 0], gpu_inputs[:, 2]
    slots = torch.arange(tokens, dtype=torch.int64) + 17
    slots[-1] = -1
    block_size = 16
    blocks = math.ceil((tokens + 17) / block_size)
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    if layout == "paged":
        key_shape = (blocks, heads, dim // 8, block_size, 8)
        value_shape = (blocks, heads, dim, block_size)
    else:
        key_shape = value_shape = (blocks, block_size, heads, dim)
    expected_k = torch.full(key_shape, 0x3555, dtype=torch.int16)
    expected_v = torch.full(value_shape, 0x3555, dtype=torch.int16)
    key_cache = expected_k.cuda().view(torch.float16)
    value_cache = expected_v.cuda().view(torch.float16)
    if layout == "HND":
        key_cache = key_cache.transpose(1, 2).contiguous().transpose(1, 2)
        value_cache = value_cache.transpose(1, 2).contiguous().transpose(1, 2)
    # Auto is a bit-preserving copy, independent of calibrated scale values.
    scale = torch.tensor([0.7], device="cuda", dtype=torch.float32)
    writer = ops.reshape_and_cache if layout == "paged" else ops.reshape_and_cache_flash
    writer(key, value, key_cache, value_cache, slots.cuda(), "auto", scale, scale)
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        block, offset = divmod(slot, block_size)
        if layout == "paged":
            expected_k[block, :, :, offset, :] = inputs[row, 0].reshape(heads, -1, 8)
            expected_v[block, :, :, offset] = inputs[row, 2]
        else:
            expected_k[block, offset] = inputs[row, 0]
            expected_v[block, offset] = inputs[row, 2]
    assert torch.equal(key_cache.cpu().view(torch.int16), expected_k)
    assert torch.equal(value_cache.cpu().view(torch.int16), expected_v)
