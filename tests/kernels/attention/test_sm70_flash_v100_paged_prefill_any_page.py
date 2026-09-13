# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D256 BM32 phase paged prefill at KV page sizes other than 784.

The align-mode attention block is 784 tokens on fp16 KV, but MTP pads the Mamba
page and makes it 816. With ``VLLM_FLASH_V100_PREFILL_D256_BM32_ANY_PAGE=1``
the BM32 phase kernel accepts any page size that is a multiple of its 16-token
page slot; the output must match the page-784 path and a dense reference
exactly, and with the flag off page 784 must be unaffected.
"""

from __future__ import annotations

import pytest
import torch

FLAG = "VLLM_FLASH_V100_PREFILL_D256_BM32_ANY_PAGE"
NUM_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 256


def _paged_layout(k_lin, v_lin, page: int, seed: int):
    seq_len = k_lin.shape[0]
    num_blocks = (seq_len + page - 1) // page
    k = torch.zeros(
        num_blocks, page, NUM_KV_HEADS, HEAD_DIM, dtype=k_lin.dtype, device=k_lin.device
    )
    v = torch.zeros_like(k)
    k.view(-1, NUM_KV_HEADS, HEAD_DIM)[:seq_len] = k_lin
    v.view(-1, NUM_KV_HEADS, HEAD_DIM)[:seq_len] = v_lin
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(num_blocks, generator=gen).to(k_lin.device)
    block_table = torch.argsort(perm).to(torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=k_lin.device)
    return k[perm].contiguous(), v[perm].contiguous(), block_table, seq_lens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("any_page", [False, True])
@pytest.mark.parametrize("page", [816, 896])
@pytest.mark.parametrize("query_len", [32, 784])
@torch.inference_mode()
def test_bm32_phase_any_page_matches_page784_and_dense(
    monkeypatch: pytest.MonkeyPatch, any_page: bool, page: int, query_len: int
) -> None:
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 is SM70/V100 only")
    fi = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    if any_page:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)

    torch.manual_seed(1234)
    device = "cuda"
    seq_len = 6 * 784 + 100  # several pages, unaligned tail
    scale = HEAD_DIM**-0.5
    k_lin = torch.randn(
        seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    v_lin = torch.randn_like(k_lin)
    query = torch.randn(
        1, query_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )

    dense = fi.flash_attn_func(
        query, k_lin.unsqueeze(0), v_lin.unsqueeze(0), causal=True, softmax_scale=scale
    )
    outs = {}
    for p in (784, page):
        k, v, block_table, seq_lens = _paged_layout(k_lin, v_lin, p, seed=p)
        outs[p] = fi.flash_attn_prefill_paged(
            query, k, v, block_table, seq_lens, softmax_scale=scale, causal=True
        )
    torch.accelerator.synchronize()

    assert torch.isfinite(outs[page]).all()
    torch.testing.assert_close(outs[784], dense, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(outs[page], dense, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(outs[page], outs[784], atol=2e-3, rtol=1e-2)
