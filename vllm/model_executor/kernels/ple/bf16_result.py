# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode checkpoint BF16 bytes using integer loads and FP16 conversion.

The mmap owner and its row cache retain checkpoint bytes. Only the selected
rows cross the result transport; the consumer converts them to model FP16.
The graph sees uint8 inputs and does not require native BF16 operations.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _decode_bf16_bytes(PACKET, OUTPUT, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    lo = tl.load(PACKET + 2 * offsets, valid, other=0).to(tl.uint32)
    hi = tl.load(PACKET + 2 * offsets + 1, valid, other=0).to(tl.uint32)
    # A BF16 value is the upper half of its exact FP32 representation.
    # This uses integer loads/bitcasts and the ordinary FP32 -> FP16 cast.
    value = ((lo | (hi << 8)) << 16).to(tl.float32, bitcast=True)
    tl.store(OUTPUT + offsets, value.to(tl.float16), valid)


def _result_shape(packet: torch.Tensor) -> tuple[int, int]:
    if (
        packet.ndim != 2
        or packet.dtype != torch.uint8
        or not packet.is_contiguous()
        or packet.shape[1] <= 0
        or packet.shape[1] % 2
    ):
        raise ValueError("BF16 PLE result requires contiguous complete uint8 pairs")
    return packet.shape[0], packet.shape[1] // 2


def decode_bf16_result(packet: torch.Tensor) -> torch.Tensor:
    shape = _result_shape(packet)
    if not packet.is_cuda:
        raise ValueError("BF16 PLE result decoding requires a CUDA consumer")
    output = torch.empty(shape, dtype=torch.float16, device=packet.device)
    if output.numel():
        _decode_bf16_bytes[(triton.cdiv(output.numel(), 1024),)](
            packet, output, output.numel(), BLOCK=1024, num_warps=4
        )
    return output


def decode_bf16_result_fake(packet: torch.Tensor) -> torch.Tensor:
    return torch.empty(_result_shape(packet), dtype=torch.float16, device=packet.device)


direct_register_custom_op(
    op_name="ple_decode_bf16_result",
    op_func=decode_bf16_result,
    fake_impl=decode_bf16_result_fake,
)
