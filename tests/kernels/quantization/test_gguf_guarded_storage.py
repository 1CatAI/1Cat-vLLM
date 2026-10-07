# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_native import (
    empty_guarded_weight,
    packed_tp_span,
    pad_weight_tail,
)
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size


@pytest.mark.parametrize("weight_type,k", [(3, 160), (20, 160), (18, 2560), (12, 256)])
def test_guarded_expert_bank_has_one_zero_tail_and_reuses_storage(weight_type, k):
    block, size = quant_size(weight_type)
    shape = (4, 64, k // block * size)
    bank = empty_guarded_weight(shape, weight_type, torch.device("cpu"))
    source = torch.randint(0, 256, shape, dtype=torch.uint8)
    bank.copy_(source)
    result = pad_weight_tail(bank, weight_type, storage_has_zero_tail=True)
    assert result is bank
    torch.testing.assert_close(result, source, rtol=0, atol=0)
    tail = ((-k) % 512) // block * size
    assert bank.untyped_storage().nbytes() == bank.numel() + tail
    backing = torch.empty(0, dtype=torch.uint8).set_(
        bank.untyped_storage(), 0, (bank.numel() + tail,), (1,)
    )
    assert not backing[bank.numel() :].count_nonzero()


def test_guarded_flag_rejects_bank_without_required_storage_tail():
    bank = torch.zeros((4, 64, 100), dtype=torch.uint8)
    with pytest.raises(ValueError, match="does not own its zero storage tail"):
        pad_weight_tail(bank, 3, storage_has_zero_tail=True)


def test_q2_tp_boundary_blocks_preserve_weights_and_padded_projection():
    full_k = 640
    block, size = quant_size(42)
    data = np.random.default_rng(710).integers(
        0, 256, (64, full_k // block, size), dtype=np.uint8
    )
    data[:, :, :2] = (
        np.full((64, full_k // block), 0.03125, np.float16)
        .view(np.uint8)
        .reshape(64, -1, 2)
    )
    data = data.reshape(64, -1)
    full = dequantize(data, 42)
    inputs = torch.randn(5, full_k, generator=torch.Generator().manual_seed(711)).half()
    outputs = []
    bank_bytes = 0
    for rank in range(4):
        first, last, left, physical_k = packed_tp_span(full_k, 42, 4, rank)
        assert (left, physical_k) == (32 if rank % 2 else 0, 192)
        local = dequantize(data[:, first:last].copy(), 42)
        np.testing.assert_array_equal(
            local[:, left : left + 160], full[:, rank * 160 : (rank + 1) * 160]
        )
        padded = torch.nn.functional.pad(
            inputs[:, rank * 160 : (rank + 1) * 160], (left, physical_k - left - 160)
        )
        outputs.append(padded.double() @ torch.from_numpy(local).double().T)
        bank_bytes += 64 * (last - first)
    reference = inputs.double() @ torch.from_numpy(full).double().T
    torch.testing.assert_close(sum(outputs), reference, rtol=0, atol=0)
    assert bank_bytes == 4 * 64 * 54
    assert bank_bytes < 4 * 64 * 100  # Existing Q4_1 repacking of local K160.
