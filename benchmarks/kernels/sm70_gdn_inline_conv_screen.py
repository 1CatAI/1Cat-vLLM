# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline per-warp convolution while retaining the original BV2 recurrence.

Every warp preloads its old history. Four Q/K history owners wait for their
readers before overwriting shared history; V columns have unique owners.
Only these four CTAs poll, so this is not a blocking whole-grid barrier.
The independent research harness does not install this as a serving route.
"""

import hashlib
import importlib.util
from pathlib import Path

import torch

from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as gdn

HELPERS = r"""
@triton.jit
def conv_value(old0, old1, old2, current, w0, w1, w2, w3):
    value = tl.full(old0.shape, 0, tl.float32)
    value += old0 * w0
    value += old1 * w1
    value += old2 * w2
    value += current * w3
    value = value / (1 + tl.exp(-value))
    return value.to(tl.float16).to(tl.float32)

@triton.jit
def history_load(Hist, feature, slot, offset,
                 HS: tl.constexpr, HD: tl.constexpr, HT: tl.constexpr):
    return tl.load(Hist + slot*HS + feature*HD + offset*HT)

@triton.jit
def history_store(Hist, feature, slot, offset, value,
                  HS: tl.constexpr, HD: tl.constexpr, HT: tl.constexpr):
    tl.store(Hist + slot*HS + feature*HD + offset*HT, value)

@triton.jit
def ready_store(pointer, value):
    tl.inline_asm_elementwise(
        "st.release.gpu.global.u32 [$1], $2; mov.u32 $0, 0;",
        constraints="=r,l,r", args=[pointer, value], dtype=tl.uint32,
        is_pure=False, pack=1)

@triton.jit
def ready_load(pointer):
    return tl.inline_asm_elementwise(
        "ld.acquire.gpu.global.u32 $0, [$1];", constraints="=r,l",
        args=[pointer], dtype=tl.uint32, is_pure=False, pack=1)
"""

PRELOAD = r"""
    # All old-history values stay private for the complete recurrence.
    slot = tl.load(HistIndices)
    previous = tl.load(num_accepted_tokens) - 1
    q_feature = i_h*K + o_k
    k_feature = H*K + i_h*K + o_k
    v_feature = 2*H*K + i_hv*V + o_v
    q0 = history_load(Hist,q_feature,slot,previous,HS,HD,HT)
    q1 = history_load(Hist,q_feature,slot,previous+1,HS,HD,HT)
    q2 = history_load(Hist,q_feature,slot,previous+2,HS,HD,HT)
    k0 = history_load(Hist,k_feature,slot,previous,HS,HD,HT)
    k1 = history_load(Hist,k_feature,slot,previous+1,HS,HD,HT)
    k2 = history_load(Hist,k_feature,slot,previous+2,HS,HD,HT)
    v0 = history_load(Hist,v_feature,slot,previous,HS,HD,HT)
    v1 = history_load(Hist,v_feature,slot,previous+1,HS,HD,HT)
    v2 = history_load(Hist,v_feature,slot,previous+2,HS,HD,HT)
    qw0=tl.load(Conv+q_feature*WD+0*WT)
    qw1=tl.load(Conv+q_feature*WD+1*WT)
    qw2=tl.load(Conv+q_feature*WD+2*WT)
    qw3=tl.load(Conv+q_feature*WD+3*WT)
    kw0=tl.load(Conv+k_feature*WD+0*WT)
    kw1=tl.load(Conv+k_feature*WD+1*WT)
    kw2=tl.load(Conv+k_feature*WD+2*WT)
    kw3=tl.load(Conv+k_feature*WD+3*WT)
    vw0=tl.load(Conv+v_feature*WD+0*WT)
    vw1=tl.load(Conv+v_feature*WD+1*WT)
    vw2=tl.load(Conv+v_feature*WD+2*WT)
    vw3=tl.load(Conv+v_feature*WD+3*WT)
    # Bind publication to all shared-history reads, beyond a compiler barrier.
    checksum = tl.sum((q0.to(tl.uint16,bitcast=True) ^ q1.to(tl.uint16,bitcast=True)
        ^ q2.to(tl.uint16,bitcast=True) ^ k0.to(tl.uint16,bitcast=True)
        ^ k1.to(tl.uint16,bitcast=True)
        ^ k2.to(tl.uint16,bitcast=True)).to(tl.uint32),0)
    tl.debug_barrier()
    # 3 value heads x 64 BV2 readers per Q/K group. Only one owner waits.
    ready_store(Ready+i_h*192+(i_hv%3)*64+i_v,checksum | 1)
    owns_qk = (i_hv%3 == 0) & (i_v == 0)
    if owns_qk:
        lanes=tl.arange(0,256)
        pending=True
        while pending:
            flags=ready_load(Ready+i_h*192+tl.minimum(lanes,191))
            pending=tl.min(tl.where(lanes<192,flags,1),axis=0) == 0
        tl.store(Ready+i_h*192+lanes,0,lanes<192)
        history_store(Hist,q_feature,slot,0,q1,HS,HD,HT)
        history_store(Hist,q_feature,slot,1,q2,HS,HD,HT)
        history_store(Hist,k_feature,slot,0,k1,HS,HD,HT)
        history_store(Hist,k_feature,slot,1,k2,HS,HD,HT)
    history_store(Hist,v_feature,slot,0,v1,HS,HD,HT)
    history_store(Hist,v_feature,slot,1,v2,HS,HD,HT)
"""

CONVOLVE = r"""
        q_raw=tl.load(p_q,mask=mask_k,other=0)
        k_raw=tl.load(p_k,mask=mask_k,other=0)
        v_raw=tl.load(p_v,mask=mask_v,other=0)
        b_q=conv_value(q0,q1,q2,q_raw,qw0,qw1,qw2,qw3)
        b_k=conv_value(k0,k1,k2,k_raw,kw0,kw1,kw2,kw3)
        b_v=conv_value(v0,v1,v2,v_raw,vw0,vw1,vw2,vw3)
        q0,q1,q2=q1,q2,q_raw
        k0,k1,k2=k1,k2,k_raw
        v0,v1,v2=v1,v2,v_raw
        if owns_qk:
            history_store(Hist,q_feature,slot,2+i_t,q_raw,HS,HD,HT)
            history_store(Hist,k_feature,slot,2+i_t,k_raw,HS,HD,HT)
        history_store(Hist,v_feature,slot,2+i_t,v_raw,HS,HD,HT)
"""


def candidate_module(directory):
    source = Path(gdn.__file__).read_text()
    begin = source.index("@triton.heuristics(")
    end = source.index("\ndef fused_sigmoid_gating_delta_rule_update(", begin)
    kernel = (
        source[begin:end]
        .replace(
            "def fused_sigmoid_gating_delta_rule_update_kernel(",
            "def inline_conv_delta_kernel(",
            1,
        )
        .replace(
            "    A_log,\n",
            "    Conv,\n    Hist,\n    HistIndices,\n    Ready,\n"
            "    HS: tl.constexpr,\n    HD: tl.constexpr,\n    HT: tl.constexpr,\n"
            "    WD: tl.constexpr,\n    WT: tl.constexpr,\n    A_log,\n",
            1,
        )
    )
    marker = "    if T == 0:\n"
    zero = r"""
    padding=tl.arange(0,8)
    columns=i_v*BV+tl.arange(0,BV)
    tl.store(o+(bos+padding[:,None])*HV*V+i_hv*V+columns[None,:],0)
"""
    assert kernel.count(marker) == 1
    kernel = kernel.replace(marker, zero + marker)
    marker = "    for i_t in range(0, T):\n"
    assert kernel.count(marker) == 1
    kernel = kernel.replace(marker, PRELOAD + marker)
    old = (
        "        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)\n"
        "        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)\n"
        "        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)\n"
    )
    assert kernel.count(old) == 1
    kernel = kernel.replace(old, CONVOLVE)
    text = (
        "from vllm.triton_utils import triton, tl\n"
        "from vllm.model_executor.layers.fla.ops.op import exp\n\n" + HELPERS + kernel
    )
    generated = directory / "inline_conv_generated.py"
    generated.write_text(text)
    spec = importlib.util.spec_from_file_location("inline_conv_generated", generated)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inline_conv_delta_kernel, hashlib.sha256(text.encode()).hexdigest()


def launch(
    kernel,
    qkv,
    a,
    b,
    a_log,
    bias,
    conv,
    hist,
    hist_indices,
    ready,
    state,
    indices,
    out,
    cu,
    accepted,
):
    assert qkv.shape == (8, 2560) and qkv.dtype == torch.float16
    assert hist.shape == (1, 2560, 10) and ready.shape == (4, 192)
    assert ready.dtype == torch.uint32 and state.dtype == torch.float32
    kernel[(1, 64, 12)](
        Conv=conv,
        Hist=hist,
        HistIndices=hist_indices,
        Ready=ready,
        HS=hist.stride(0),
        HD=hist.stride(1),
        HT=hist.stride(2),
        WD=conv.stride(0),
        WT=conv.stride(1),
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=bias,
        beta=1.0,
        threshold=20.0,
        q=qkv,
        k=qkv,
        v=qkv,
        mixed_qkv=qkv,
        o=out,
        h0=state,
        ht=state,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        ddtree_parent_ids=None,
        scale=128**-0.5,
        N=1,
        T=8,
        B=1,
        H=4,
        HV=12,
        K=128,
        V=128,
        BK=128,
        BV=2,
        Q_OFFSET=0,
        K_OFFSET=512,
        V_OFFSET=1024,
        QKV_STRIDE=qkv.stride(0),
        stride_init_state_token=state.stride(0),
        stride_final_state_token=state.stride(0),
        stride_indices_seq=indices.stride(0),
        stride_indices_tok=indices.stride(1),
        stride_parent_ids_seq=1,
        stride_parent_ids_tok=1,
        INPLACE_FINAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=True,
        IS_KDA=False,
        MIXED_QKV=True,
        PRECOMPUTED_GATING=False,
        QUANTIZE_BETA=False,
        QUANTIZE_STATE_EACH_STEP=False,
        MATCH_RECURRENT_NUMERICS=True,
        num_warps=1,
        num_stages=3,
    )
