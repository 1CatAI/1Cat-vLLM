# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_native import (
    empty_guarded_weight,
    pad_weight_tail,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_size


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
