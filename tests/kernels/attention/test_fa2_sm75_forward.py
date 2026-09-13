# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Turing FA2 build (_vllm_fa2_C_sm75) computes the same attention as torch.

Runs only on a compute capability 7.5 device with the sm75 library installed;
it exercises the loader (library chosen for the device) and the fp16 forward
path for the head sizes the Qwen3.8 and Llama families use.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _turing_device_index() -> int | None:
    if not current_platform.is_cuda():
        return None
    for i in range(torch.accelerator.device_count()):
        if torch.cuda.get_device_capability(i) == (7, 5):
            return i
    return None


@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("seqlen", [1, 8, 333])
def test_sm75_varlen_forward_matches_torch(head_size: int, seqlen: int):
    index = _turing_device_index()
    if index is None:
        pytest.skip("needs a compute capability 7.5 device")
    device = torch.device(f"cuda:{index}")
    from vllm.vllm_flash_attn import flash_attn_interface as fai

    if fai._fa2_library_path((7, 5)) is None:
        pytest.skip("the sm75 FA2 library is not installed")
    torch.accelerator.set_device_index(index)
    torch.manual_seed(0)
    batch, heads, kv_heads = 2, 8, 2
    q = torch.randn(
        batch * seqlen, heads, head_size, device=device, dtype=torch.float16
    )
    k = torch.randn(
        batch * seqlen, kv_heads, head_size, device=device, dtype=torch.float16
    )
    v = torch.randn_like(k)
    cu = torch.arange(0, (batch + 1) * seqlen, seqlen, device=device, dtype=torch.int32)

    out = fai.flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=seqlen,
        cu_seqlens_q=cu,
        max_seqlen_k=seqlen,
        cu_seqlens_k=cu,
        causal=True,
        fa_version=2,
    )
    assert fai._fa2_loaded_capability == (7, 5)

    # torch reference: grouped heads expanded, causal SDPA per sequence.
    ref = torch.empty_like(q)
    rep = heads // kv_heads
    for b in range(batch):
        s = slice(b * seqlen, (b + 1) * seqlen)
        qb = q[s].transpose(0, 1).float()
        kb = k[s].repeat_interleave(rep, dim=1).transpose(0, 1).float()
        vb = v[s].repeat_interleave(rep, dim=1).transpose(0, 1).float()
        ref[s] = (
            torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, is_causal=True)
            .transpose(0, 1)
            .to(torch.float16)
        )
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
