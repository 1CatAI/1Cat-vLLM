# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep TurboMind NVFP4 loading bounded without changing prepared bytes."""

import pytest
import torch

from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    _prepared_expert_bank,
)


def test_padded_expert_storage_is_allocated_separately():
    original = torch.zeros(3, 4, 4, dtype=torch.uint8)
    prepared = torch.empty(5, dtype=torch.int32)
    bank = _prepared_expert_bank(original, prepared)
    assert bank.shape == (3, 5)
    assert bank.data_ptr() != original.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "n,k,interleaved", [(512, 256, False), (320, 2560, True), (4096, 4096, False)]
)
def test_prepared_experts_remain_bitwise_equal(n, k, interleaved):
    from vllm.model_executor.layers.fused_moe.sm70.fp4_codec import Fp4MoECodec
    from vllm.model_executor.layers.quantization.sm70_turbomind import (
        unpack_mxfp4_weight,
    )

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("TurboMind preparation requires SM70")
    torch.manual_seed(32)
    experts = 3
    original = torch.randint(
        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(experts, k // 16, n, device="cuda", dtype=torch.float16)

    def prepare(e):
        return Fp4MoECodec.prepare_weights(
            "nvfp4",
            unpack_mxfp4_weight(original[e]),
            scales[e],
            16,
            interleave_gated_silu=interleaved,
        )

    expected = torch.stack([prepare(e)[0] for e in range(experts)])
    actual = None
    for e in range(experts):
        prepared, _, _ = prepare(e)
        if actual is None:
            actual = _prepared_expert_bank(original, prepared)
        actual[e].copy_(prepared)
    assert actual is not None
    assert actual.data_ptr() == original.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
