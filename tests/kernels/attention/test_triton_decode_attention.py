# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.triton_decode_attention import (
    _GROUPED_TILES,
    _decode_grouped_tile,
    _device_smem_budget,
    _fwd_grouped_kernel_stage1,
    _grouped_block_dims,
    _grouped_stage1_smem_bytes,
    decode_attention_fwd,
)

DEVICE_TYPE = current_platform.device_type


_V100_SMEM = 98304
_TURING_SMEM = 65536


@pytest.mark.parametrize(
    ("is_hip", "dims", "smem_budget", "expected"),
    [
        # dims = (BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV)
        (True, (128, 0, 128), None, (16, 16)),
        (False, (512, 64, 512), _V100_SMEM, (16, 16)),  # MLA 576 on V100
        (False, (512, 64, 512), _TURING_SMEM, (16, 8)),  # MLA 576 on Turing
        (False, (512, 64, 512), None, (32, 16)),  # sm80+ is not modelled
        (False, (256, 32, 256), _V100_SMEM, (32, 16)),  # MLA 288
        (False, (128, 0, 128), _TURING_SMEM, (32, 16)),
        (False, (256, 0, 128), _TURING_SMEM, (32, 16)),
        (False, (128, 0, 512), _V100_SMEM, (32, 16)),
        (False, (128, 0, 512), _TURING_SMEM, (16, 16)),
        (False, (256, 0, 512), _V100_SMEM, (32, 16)),
        (False, (256, 0, 512), _TURING_SMEM, (16, 16)),
        (False, (1024, 0, 512), _V100_SMEM, (16, 8)),  # exactly 98304 B
        (False, (1024, 0, 512), None, (32, 16)),
    ],
)
def test_decode_grouped_tile_follows_smem_budget(is_hip, dims, smem_budget, expected):
    block_dmodel, block_dpe, block_dv = dims
    assert (
        _decode_grouped_tile(
            is_hip=is_hip,
            block_dmodel=block_dmodel,
            block_dpe=block_dpe,
            block_dv=block_dv,
            smem_budget=smem_budget,
        )
        == expected
    )


def test_decode_grouped_tile_without_fitting_tile_raises():
    with pytest.raises(ValueError, match="No grouped decode attention tile"):
        _decode_grouped_tile(
            is_hip=False,
            block_dmodel=1024,
            block_dpe=0,
            block_dv=512,
            smem_budget=_TURING_SMEM,
        )


# (D_QK, D_V, is_mla) of the grouped decode tests and the two MLA ranks.
_GROUPED_FORMS = [
    (128, 128, False),
    (128, 512, False),
    (192, 128, False),
    (192, 512, False),
    (576, 128, False),
    (576, 512, False),
    (576, 512, True),
    (288, 256, True),
]


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Needs the CUDA Triton compiler"
)
@pytest.mark.parametrize("block_n, block_h", _GROUPED_TILES)
@pytest.mark.parametrize("d_qk, d_v, is_mla", _GROUPED_FORMS)
def test_grouped_smem_formula_matches_compiled_kernel(
    d_qk, d_v, is_mla, block_n, block_h
):
    device = torch.device(DEVICE_TYPE, torch.accelerator.current_device_index())
    if _device_smem_budget(device) is None:
        pytest.skip("Triton does not stage the dot operands in fp32 on this GPU")
    dtype = torch.bfloat16
    heads, kv_heads, batch, cache, splits = 32, 8, 2, 64, 2
    q = torch.randn(batch, heads, d_qk, dtype=dtype, device=device)
    k = torch.randn(cache, kv_heads, d_qk, dtype=dtype, device=device)
    v = torch.randn(cache, kv_heads, d_v, dtype=dtype, device=device)
    att = torch.empty(batch, heads, splits, d_v + 1, dtype=torch.float32, device=device)
    req_to_token = torch.zeros(batch, 32, dtype=torch.int32, device=device)
    seq_len = torch.full((batch,), 16, device=device)
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    kv_group_num = heads // kv_heads
    block_dmodel, block_dpe, block_dv = _grouped_block_dims(d_qk, d_v, is_mla=is_mla)
    compiled = _fwd_grouped_kernel_stage1.warmup(
        q,
        k,
        v,
        0.1,
        req_to_token,
        seq_len,
        att,
        req_to_token.stride(0),
        q.stride(0),
        q.stride(1),
        k.stride(-3),
        k.stride(-2),
        v.stride(-3),
        v.stride(-2),
        att.stride(0),
        att.stride(1),
        att.stride(2),
        scale,
        scale,
        kv_group_num=kv_group_num,
        q_head_num=heads,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_DPE=block_dpe,
        BLOCK_DV=block_dv,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        NUM_KV_SPLITS=splits,
        PAGE_SIZE=1,
        logit_cap=0.0,
        num_warps=4,
        num_stages=2,
        Lk=d_qk,
        Lv=d_v,
        IS_MLA=is_mla,
        grid=(batch, triton.cdiv(heads, min(block_h, kv_group_num)), splits),
    )
    assert compiled.metadata.shared == _grouped_stage1_smem_bytes(
        block_n=block_n,
        block_h=block_h,
        block_dmodel=block_dmodel,
        block_dpe=block_dpe,
        block_dv=block_dv,
    )


def _grouped_tile_fits(d_qk: int, d_v: int) -> bool:
    device = torch.device(DEVICE_TYPE, torch.accelerator.current_device_index())
    try:
        block_dmodel, block_dpe, block_dv = _grouped_block_dims(d_qk, d_v, is_mla=False)
        _decode_grouped_tile(
            is_hip=current_platform.is_rocm(),
            block_dmodel=block_dmodel,
            block_dpe=block_dpe,
            block_dv=block_dv,
            smem_budget=_device_smem_budget(device),
        )
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("B", [3, 5])
@pytest.mark.parametrize("L", [1027, 1025])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D_QK", [128, 192, 576])
@pytest.mark.parametrize("D_V", [128, 512])
@pytest.mark.parametrize("CACHE_SIZE", [16384])
@pytest.mark.parametrize("PAGE_SIZE", [1, 16])
def test_decode_attention(B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE):
    if H_KV != H_Q and not _grouped_tile_fits(D_QK, D_V):
        pytest.skip("No grouped decode tile fits into this GPU's shared memory")
    assert CACHE_SIZE % PAGE_SIZE == 0
    dtype = torch.bfloat16
    seq_len = L  # This represents the number of tokens already in the sequence
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, CACHE_SIZE // PAGE_SIZE, (B, num_pages_per_batch, 1), device=DEVICE_TYPE
    )
    req_to_token = req_to_page * PAGE_SIZE
    req_to_token = req_to_token.expand(B, num_pages_per_batch, PAGE_SIZE)
    req_to_token = req_to_token + torch.arange(PAGE_SIZE, device=DEVICE_TYPE).view(
        1, 1, -1
    )
    req_to_token = req_to_token.view(B, -1)
    req_to_token = req_to_token[:, :seq_len].contiguous()

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)

    # k_buffer and v_buffer represent all previous tokens
    # Page size is 1.
    k_buffer = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE)
    v_buffer = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE)

    # o will have the same shape as q
    o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)

    lse = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)

    b_seq_len = torch.full((B,), seq_len, device=DEVICE_TYPE)

    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1),
        dtype=torch.float32,
        device=DEVICE_TYPE,
    )

    # Call the original implementation.
    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o,
        lse,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
    )

    # Page size can be larger than 1.
    k_buffer = k_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
    v_buffer = v_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)

    o1 = torch.zeros_like(o)
    lse1 = torch.zeros_like(lse)

    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o1,
        lse1,
        req_to_page,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,
    )

    assert torch.allclose(o, o1)


def _quantize_to_fp8(tensor: torch.Tensor):
    """Quantize a BF16 tensor to FP8 e4m3fn with per-tensor scale.

    Returns (fp8_tensor, scale) where:
        fp8_tensor ≈ tensor / scale  (stored as float8_e4m3fn)
        tensor ≈ fp8_tensor.to(float32) * scale  (dequantized)
    """
    amax = tensor.abs().amax()
    # float8_e4m3fn max representable value is 448.0
    scale = (amax / 448.0).clamp(min=1e-12).to(torch.float32)
    fp8_tensor = (
        (tensor.to(torch.float32) / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return fp8_tensor, scale


@pytest.mark.parametrize("B", [3])
@pytest.mark.parametrize("L", [1025])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D_QK", [128, 576])
@pytest.mark.parametrize("D_V", [128, 512])
@pytest.mark.parametrize("CACHE_SIZE", [16384])
@pytest.mark.parametrize("PAGE_SIZE", [1, 16])
def test_decode_attention_fp8(B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE):
    """Test FP8 KV cache path: quantize K/V to FP8, run kernel with scales,
    and compare against BF16 reference output."""
    assert CACHE_SIZE % PAGE_SIZE == 0
    dtype = torch.bfloat16
    seq_len = L
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, CACHE_SIZE // PAGE_SIZE, (B, num_pages_per_batch, 1), device=DEVICE_TYPE
    )
    req_to_token = req_to_page * PAGE_SIZE
    req_to_token = req_to_token.expand(B, num_pages_per_batch, PAGE_SIZE)
    req_to_token = req_to_token + torch.arange(PAGE_SIZE, device=DEVICE_TYPE).view(
        1, 1, -1
    )
    req_to_token = req_to_token.view(B, -1)
    req_to_token = req_to_token[:, :seq_len].contiguous()

    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)

    # Create BF16 K/V as reference
    k_bf16 = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE)
    v_bf16 = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE)

    # --- BF16 reference ---
    o_ref = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
    lse_ref = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
    )

    if PAGE_SIZE == 1:
        decode_attention_fwd(
            q,
            k_bf16,
            v_bf16,
            o_ref,
            lse_ref,
            req_to_token,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
        )
    else:
        k_paged = k_bf16.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
        v_paged = v_bf16.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)
        decode_attention_fwd(
            q,
            k_paged,
            v_paged,
            o_ref,
            lse_ref,
            req_to_page,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            page_size=PAGE_SIZE,
        )

    # --- FP8 path ---
    k_fp8, k_scale = _quantize_to_fp8(k_bf16)
    v_fp8, v_scale = _quantize_to_fp8(v_bf16)

    o_fp8 = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
    lse_fp8 = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
    attn_logits_fp8 = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
    )

    if PAGE_SIZE == 1:
        decode_attention_fwd(
            q,
            k_fp8,
            v_fp8,
            o_fp8,
            lse_fp8,
            req_to_token,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits_fp8,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    else:
        k_fp8_paged = k_fp8.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
        v_fp8_paged = v_fp8.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)
        decode_attention_fwd(
            q,
            k_fp8_paged,
            v_fp8_paged,
            o_fp8,
            lse_fp8,
            req_to_page,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits_fp8,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            page_size=PAGE_SIZE,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    # FP8 tolerances match test_mla_backends.py test_backend_correctness.
    torch.testing.assert_close(o_ref, o_fp8, atol=5e-1, rtol=1e-2)
