# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""4096-token prefill chunks take the native Q4096 79T specialization."""

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import dense_prefill


def _q4096_op():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    load_fa2_library(torch.device("cuda"))
    op = getattr(torch.ops._vllm_fa2_C, "sm70_d256_gqa_architecture_q4096_fwd", None)
    if op is None:
        pytest.skip("Q4096 architecture operator was not built")
    return op


@pytest.mark.parametrize(
    ("query_len", "kv_len", "eligible"),
    [
        (4096, 8192, True),
        (4064, 262144, True),
        (4000, 8192, True),
        (3999, 8192, False),
        (4097, 8192, False),
        # The first chunks of a prompt keep the faster exact Split-D kernel.
        (4096, 4096, False),
        (4064, 8160, False),
    ],
)
def test_q4096_eligibility(query_len, kv_len, eligible):
    _q4096_op()
    assert dense_prefill._sm70_79t_q4096_eligible(query_len, kv_len) is eligible


@pytest.mark.parametrize("query_len", [4096, 4064, 4000])
@torch.inference_mode()
def test_q4096_dispatch_pads_short_chunks_exactly(query_len):
    op = _q4096_op()
    torch.manual_seed(4096 + query_len)
    kv_len = 16384
    q = torch.randn(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    dense_prefill._run_sm70_d256_gqa_79t_q4096_dispatch(
        q, k, v, out, softmax_scale=0.0625, architecture_q4096_op=op
    )
    # Leading padding keeps each real row's bottom-right causal alignment, so
    # the result equals the native kernel on the zero-padded query.
    padded = torch.zeros(1, 4096, 6, 256, device="cuda", dtype=torch.float16)
    padded[:, 4096 - query_len :].copy_(q)
    padded_out = torch.empty_like(padded)
    op(padded, k, v, padded_out, 0.0625, True)
    torch.testing.assert_close(out, padded_out[:, 4096 - query_len :], rtol=0, atol=0)

    rows = torch.tensor([0, 255, 256, query_len - 257, query_len - 1], device="cuda")
    scores = torch.einsum("rhd,kd->hrk", q[0, rows].float(), k[0, :, 0].float()) / 16
    keys = torch.arange(kv_len, device="cuda")
    scores.masked_fill_(
        keys[None, None, :] > (kv_len - query_len + rows)[None, :, None], -torch.inf
    )
    reference = (scores.softmax(-1) @ v[0, :, 0].float()).permute(1, 0, 2)
    torch.testing.assert_close(out[0, rows].float(), reference, rtol=0.01, atol=0.03)
