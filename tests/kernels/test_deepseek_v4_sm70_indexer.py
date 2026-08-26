# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.sm70.indexer import sm70_indexer_prefill_logits


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires NVIDIA V100/SM70",
)
def test_sm70_indexer_prefill_dequantizes_fp8_storage_as_uint8() -> None:
    torch.manual_seed(20260803)
    num_queries = 3
    num_heads = 4
    num_keys = 7
    head_dim = 128

    q = torch.randn(
        (num_queries, num_heads, head_dim), device="cuda", dtype=torch.float16
    )
    weights = torch.randn((num_queries, num_heads), device="cuda", dtype=torch.float32)
    # E4M3FN bit pattern 0x38 is exactly 1.0. Constructing the tensor through
    # uint8 also avoids requiring native FP8 conversion support on SM70.
    k_bits = torch.full((num_keys, head_dim), 0x38, device="cuda", dtype=torch.uint8)
    k_quant = k_bits.view(torch.float8_e4m3fn)
    scales = torch.linspace(0.5, 2.0, num_keys, device="cuda", dtype=torch.float32)

    actual = sm70_indexer_prefill_logits(q, k_quant, scales, weights)

    weighted_q = torch.sum(q.float() * weights[:, :, None], dim=1).half()
    k_reference = scales[:, None].expand(num_keys, head_dim).half()
    expected = torch.mm(weighted_q, k_reference.T, out_dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
