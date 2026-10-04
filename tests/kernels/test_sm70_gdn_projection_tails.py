# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
    _split_gdn_projection_tails,
)


@pytest.mark.parametrize("m", [5, 20])
def test_tail_split_preserves_half_payloads_and_qkv_alias(m):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    qkvz = torch.empty(m, 4096, device="cuda", dtype=torch.float16)
    ba = torch.empty(m, 24, device="cuda", dtype=torch.float16)
    z = torch.empty(m, 12, 128, device="cuda", dtype=torch.float16)
    bits = torch.arange(m * 4096, device="cuda", dtype=torch.int32).to(torch.int16)
    qkvz.view(torch.int16).copy_(bits.view(m, 4096))
    ba.view(torch.int16).copy_(bits[: m * 24].view(m, 24))
    qkv, b, a = _split_gdn_projection_tails(qkvz, ba, z)
    assert qkv.data_ptr() == qkvz.data_ptr() and qkv.stride() == (4096, 1)
    assert torch.equal(
        z.view(m, 1536).view(torch.int16), qkvz[:, 2560:].view(torch.int16)
    )
    assert torch.equal(b.view(torch.int16), ba[:, :12].view(torch.int16))
    assert torch.equal(a.view(torch.int16), ba[:, 12:].view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qkv, b, a = _split_gdn_projection_tails(qkvz, ba, z)
        qkv.zero_()
    for offset in (23456, -12345):
        qkvz.view(torch.int16).copy_((bits + offset).view(m, 4096))
        ba.view(torch.int16).copy_((bits[: m * 24] + offset).view(m, 24))
        expected_z = qkvz[:, 2560:].clone()
        expected_b, expected_a = ba[:, :12].clone(), ba[:, 12:].clone()
        graph.replay()
        assert torch.equal(
            z.view(m, 1536).view(torch.int16), expected_z.view(torch.int16)
        )
        assert torch.equal(b.view(torch.int16), expected_b.view(torch.int16))
        assert torch.equal(a.view(torch.int16), expected_a.view(torch.int16))
        assert torch.count_nonzero(qkv) == 0
