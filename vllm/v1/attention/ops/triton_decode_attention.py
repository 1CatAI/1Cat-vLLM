# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/sgl-project/sglang/blob/9f635ea50de920aa507f486daafba26a5b837574/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# which was originally adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

# Changes:
# - Add support for page size >= 1.

# Copyright 2025 vLLM Team
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size >= 1.
"""

import logging

import torch
from packaging import version

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.kv_codec import decode_kv_payload

is_hip_ = current_platform.is_rocm()


def _decode_grouped_block_size(
    *,
    is_hip: bool,
    is_mla: bool,
    block_dmodel: int,
    is_sm70: bool,
) -> int:
    if is_hip or (is_mla and block_dmodel >= 512 and is_sm70):
        return 16
    return 32


logger = logging.getLogger(__name__)

# Only print the following warnings when triton version < 3.2.0.
# The issue won't affect performance or accuracy.
if version.parse(triton.__version__) < version.parse("3.2.0"):
    logger.warning(
        "The following error message 'operation scheduled before its operands' "
        "can be ignored."
    )


@triton.jit
def _decode_payload(payload, scale, CODEC: tl.constexpr, OUT: tl.constexpr):
    if CODEC == 1:
        return (decode_kv_payload(payload, CODEC).to(tl.float32) * scale).to(OUT)
    elif payload.dtype.is_fp8():
        return (payload.to(tl.float32) * scale).to(OUT)
    return payload.to(OUT)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    k_scale,
    v_scale,
    Row_Scales,
    stride_buf_kpage,
    stride_buf_vpage,
    stride_scale_page,
    stride_scale_token,
    CODEC: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            page_valid = (kv_page_number >= 0) & (kv_page_number < PAGE_CAPACITY)
            token_valid = (offs_n < split_kv_end) & page_valid
            k_loc = (
                kv_page_number * stride_buf_kpage + offs_n % PAGE_SIZE * stride_buf_kbs
            )
            v_loc = (
                kv_page_number * stride_buf_vpage + offs_n % PAGE_SIZE * stride_buf_vbs
            )
            if CODEC == 3:
                row_scale = tl.load(
                    Row_Scales
                    + kv_page_number * stride_scale_page
                    + offs_n % PAGE_SIZE * stride_scale_token,
                    mask=token_valid,
                    other=0.0,
                )
            offs_buf_k = k_loc[:, None] + cur_kv_head * stride_buf_kh + offs_d[None, :]
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=token_valid[:, None] & (mask_d[None, :]),
                other=0.0,
            )
            k = _decode_payload(k, ks, CODEC, q.dtype)
            qk = tl.sum(q[None, :] * k, 1)
            if CODEC == 3:
                qk *= row_scale
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(token_valid, qk, float("-inf"))

            offs_buf_v = v_loc[:, None] + cur_kv_head * stride_buf_vh + offs_dv[None, :]
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=token_valid[:, None] & (mask_dv[None, :]),
                other=0.0,
            )
            v = _decode_payload(v, vs, CODEC, q.dtype)

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            safe_max = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
            re_scale = tl.exp(e_max - safe_max)
            p = tl.exp(qk - safe_max)
            acc *= re_scale
            pv = p * row_scale if CODEC == 3 else p
            acc += tl.sum(pv[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            tl.where(e_sum == 0, 0.0, acc / e_sum),
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
    k_scale,
    v_scale,
    cache_view=None,
):
    BLOCK = 64 if not is_hip_ else 8

    NUM_KV_SPLITS = num_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    num_warps = 4
    if kv_group_num != 1:
        num_warps = 1 if is_hip_ else 2

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_scale,
        v_scale,
        cache_view.key_scale
        if cache_view is not None and cache_view.key_scale is not None
        else k_scale,
        k_buffer.stride(0) if k_buffer.ndim == 4 else page_size * k_buffer.stride(-3),
        v_buffer.stride(0) if v_buffer.ndim == 4 else page_size * v_buffer.stride(-3),
        cache_view.key_scale.stride(0)
        if cache_view is not None and cache_view.codec.native_id == 3
        else 0,
        cache_view.key_scale.stride(1)
        if cache_view is not None and cache_view.codec.native_id == 3
        else 0,
        CODEC=cache_view.codec.native_id
        if cache_view is not None
        else (1 if k_buffer.dtype == torch.uint8 else 0),
        PAGE_CAPACITY=k_buffer.shape[0]
        if k_buffer.ndim == 4
        else k_buffer.shape[0] // page_size,
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    k_scale,
    v_scale,
    Row_Scales,
    stride_buf_kpage,
    stride_buf_vpage,
    stride_scale_page,
    stride_scale_token,
    CODEC: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    IS_MLA: tl.constexpr = False,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(
        Q + offs_q,
        mask=(mask_h[:, None]) & (mask_d[None, :]),
        other=0.0,
        cache_modifier=".ca",
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe,
            mask=(mask_h[:, None]) & (mask_dpe[None, :]),
            other=0.0,
            cache_modifier=".ca",
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]
        if BLOCK_DPE > 0:
            base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]

        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
                cache_modifier=".ca",
            )
            page_valid = (kv_page_number >= 0) & (kv_page_number < PAGE_CAPACITY)
            token_valid = (offs_n < split_kv_end) & page_valid
            k_loc = (
                kv_page_number * stride_buf_kpage + offs_n % PAGE_SIZE * stride_buf_kbs
            )
            v_loc = (
                kv_page_number * stride_buf_vpage + offs_n % PAGE_SIZE * stride_buf_vbs
            )
            if CODEC == 3:
                row_scale = tl.load(
                    Row_Scales
                    + kv_page_number * stride_scale_page
                    + offs_n % PAGE_SIZE * stride_scale_token,
                    mask=token_valid,
                    other=0.0,
                )

            # explicitly facilitate overlapping load/compute
            offs_buf_k = k_loc[None, :] + base_offs_k
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=token_valid[None, :] & (mask_d[:, None]),
                other=0.0,
                cache_modifier=".cg",
            )

            k = _decode_payload(k, ks, CODEC, q.dtype)
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = k_loc[None, :] + base_offs_kpe
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=token_valid[None, :] & (mask_dpe[:, None]),
                    other=0.0,
                    cache_modifier=".cg",
                )
                kpe = _decode_payload(kpe, ks, CODEC, qpe.dtype)
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            if CODEC == 3:
                qk *= row_scale
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & token_valid[None, :], qk, float("-inf"))

            if not IS_MLA:
                offs_buf_v = v_loc[:, None] + base_offs_v
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=token_valid[:, None] & (mask_dv[None, :]),
                    other=0.0,
                )
                v = _decode_payload(v, vs, CODEC, q.dtype)
            else:
                # MLA uses a single c_kv.
                # loading the same c_kv to interpret it as v is not necessary.
                # transpose the existing c_kv (aka k) for the dot product.
                v = tl.trans(k)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            safe_max = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
            re_scale = tl.exp(e_max - safe_max)
            p = tl.exp(qk - safe_max[:, None])
            acc *= re_scale[:, None]
            if CODEC == 3:
                pv = p * row_scale[None, :]
                # Preserve small weighted tail mass through the MMA input cast.
                # A binary scale is exact in FP32; bounded rows avoid overflowing
                # the input dtype for finite, large cache values.
                if v.dtype == tl.float16:
                    input_limit = 65504.0
                else:
                    input_limit = 3.3895313892515355e38
                factor = tl.where(
                    tl.max(tl.abs(pv), 1) <= input_limit / 2048.0, 2048.0, 1.0
                )
                pv *= factor[:, None]
                hi = pv.to(v.dtype)
                lo = (pv - hi.to(tl.float32)).to(v.dtype)
                acc += (tl.dot(hi, v) + tl.dot(lo, v)) / factor[:, None]
            else:
                acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            tl.where(e_sum[:, None] == 0, 0.0, acc / e_sum[:, None]),
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
    k_scale,
    v_scale,
    is_mla=False,
    cache_view=None,
):
    # with is_mla there is only a single c_kv in smem.
    # could increase BLOCK or num_stages.
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # Align tile dimensions with latent rank for MLA to avoid shape mismatch.
    if is_mla:
        if not is_hip_ and Lk == 576:
            BLOCK_DMODEL = 512
            BLOCK_DPE = 64
        elif not is_hip_ and Lk == 288:
            BLOCK_DMODEL = 256
            BLOCK_DPE = 32
        else:
            BLOCK_DMODEL = triton.next_power_of_2(Lv)
            BLOCK_DPE = triton.next_power_of_2(Lk - Lv) if Lk > Lv else 0
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    # Patch 0103: BLOCK=16 for HIP (original) and for MLA-large on SM70 (V100).
    # On SM70 at BLOCK=32 the stage-1 kernel needs 102,400 B of smem (Triton
    # stages dot operands as fp32 on sm70: peak = 4 B x [BLOCK_H*(BLOCK_DMODEL+
    # BLOCK_DPE) + BLOCK*BLOCK_DMODEL]) against the 98,304 B hardware limit ->
    # OutOfResources on the first decode. BLOCK=16 fits (69,632 B) and won the
    # tiling bench vs BN32/BH8 (spill catastrophe) and BN16/BH8 (loses at
    # saturated grids) at 128-user shapes. num_stages is a no-op here (no
    # cp.async on sm70 - KV loads cannot be multi-buffered); leave it alone.
    # Proof + bench: patches/0103-mla-triton-decode-smem/poc/.
    BLOCK = _decode_grouped_block_size(
        is_hip=is_hip_,
        is_mla=is_mla,
        block_dmodel=BLOCK_DMODEL,
        is_sm70=current_platform.is_device_capability(70),
    )

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if is_hip_:
        # https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html#mi300x-triton-kernel-performance-optimization
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1
    elif not is_hip_ and BLOCK_DMODEL >= 1024:
        # Avoid shared memory overflow on NVIDIA when BLOCK_DMODEL is large
        # like non-MLA D_QK=576, BLOCK_DMODEL=1024, BLOCK_H=16
        # exceeds 101376 bytes limit
        num_stages = 1

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_scale,
        v_scale,
        cache_view.key_scale
        if cache_view is not None and cache_view.key_scale is not None
        else k_scale,
        k_buffer.stride(0) if k_buffer.ndim == 4 else page_size * k_buffer.stride(-3),
        v_buffer.stride(0) if v_buffer.ndim == 4 else page_size * v_buffer.stride(-3),
        cache_view.key_scale.stride(0)
        if cache_view is not None and cache_view.codec.native_id == 3
        else 0,
        cache_view.key_scale.stride(1)
        if cache_view is not None and cache_view.codec.native_id == 3
        else 0,
        CODEC=cache_view.codec.native_id
        if cache_view is not None
        else (1 if k_buffer.dtype == torch.uint8 else 0),
        PAGE_CAPACITY=k_buffer.shape[0]
        if k_buffer.ndim == 4
        else k_buffer.shape[0] // page_size,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        IS_MLA=is_mla,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    o,
    lse,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    OUTPUT_FP16: tl.constexpr = 0,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            safe_max = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
            old_scale = tl.exp(e_max - safe_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - safe_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    result = tl.where(e_sum == 0, 0.0, acc / e_sum)
    if OUTPUT_FP16:
        result = result.to(tl.float16)
    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        result,
        mask=mask_d,
    )
    lse_val = e_max + tl.log(e_sum)
    tl.store(
        lse + cur_batch * stride_lse_bs + cur_head,
        lse_val,
    )


def _decode_softmax_reducev_fwd(
    logits,
    q,
    o,
    lse,
    v_buffer,
    b_seq_len,
    num_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        lse,
        b_seq_len,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
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
    page_size,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
    cache_view=None,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
        k_scale,
        v_scale,
        cache_view=cache_view,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
    )


def decode_attention_fwd_grouped(
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
    page_size,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
    is_mla=False,
    cache_view=None,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
        k_scale,
        v_scale,
        is_mla=is_mla,
        cache_view=cache_view,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
    )


def decode_attention_fwd(
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
    page_size=1,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
    is_mla=False,
    cache_view=None,
):
    assert num_kv_splits == attn_logits.shape[2]

    if k_scale is None:
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    if v_scale is None:
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

    if k_buffer.dtype == torch.int8 and cache_view is None:
        raise ValueError("INT8 decode requires its typed cache view and scales")
    if k_buffer.dtype == torch.float8_e4m3fn and not current_platform.supports_fp8():
        k_buffer = k_buffer.view(torch.uint8)
        v_buffer = v_buffer.view(torch.uint8)

    kv_group_num = q.shape[1] // v_buffer.shape[-2]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
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
            page_size,
            logit_cap,
            k_scale,
            v_scale,
            cache_view=cache_view,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
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
            page_size,
            logit_cap,
            k_scale,
            v_scale,
            is_mla=is_mla,
            cache_view=cache_view,
        )
