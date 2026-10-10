# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read a next-projection prefix during unchanged GDN recurrence work.

The ordinary installed kernel supplies the entire recurrence. Only two sets of
side-effecting L2 prefetch instructions are inserted at a selected token. This
does not alter BV2, snapshot layout, arithmetic or the input/output contract.
"""

import hashlib
import importlib.util
from pathlib import Path

from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as gdn


def candidate_module(directory):
    source = Path(gdn.__file__).read_text()
    begin = source.index("@triton.heuristics(")
    end = source.index("\ndef fused_sigmoid_gating_delta_rule_update(", begin)
    kernel = source[begin:end].replace(
        "def fused_sigmoid_gating_delta_rule_update_kernel(",
        "def gdn_weight_window_kernel(",
        1,
    )
    kernel = kernel.replace(
        "    A_log,\n",
        "    prefetch_codes,\n    PREFETCH_TOKEN: tl.constexpr,\n    A_log,\n",
        1,
    )
    marker = "    for i_t in range(0, T):\n"
    assert kernel.count(marker) == 1
    prefetch = """
        if i_t == PREFETCH_TOKEN:
            worker = i_nh * tl.num_programs(1) + i_v
            for batch in tl.static_range(2):
                line = worker * 32 + tl.arange(0, 32) + batch * 24576
                tile = line // 192
                local = line % 192
                group = (local // 16) * 8 + (local % 16) // 4
                address = prefetch_codes + (tile * 96 + group) * 512 + (line % 4) * 128
                tl.inline_asm_elementwise(
                    "{ .reg .pred p; setp.lt.u32 p,$2,30720; "
                    "@p prefetch.global.L2 [$1]; mov.u32 $0,0; }",
                    constraints="=r,l,r", args=[address, line],
                    dtype=tl.int32, is_pure=False, pack=1)
"""
    kernel = kernel.replace(marker, marker + prefetch, 1)
    prefix = """
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fla.ops.op import exp
"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gdn_weight_window.py"
    path.write_text(prefix + kernel)
    spec = importlib.util.spec_from_file_location("gdn_weight_window", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gdn_weight_window_kernel, hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def launch(kernel, codes, token, *values, **kwargs):
    import torch

    A_log, a, b, bias, qkv, h, hv, k, v, output = values
    assert (h, hv, k, v) == (4, 12, 128, 128)
    assert qkv.shape == (8, 2560) and qkv.stride(1) == 1
    assert output.dtype == qkv.dtype == torch.float16
    assert codes.numel() == 5120 * 1536 and codes.is_contiguous()
    assert kwargs["use_qk_l2norm_in_kernel"]
    assert kwargs["match_recurrent_numerics"] and kwargs["match_recurrent_schedule"]
    state = kwargs["initial_state"]
    assert state.dtype == torch.float32
    indices = kwargs["ssm_state_indices"]
    g, beta = kwargs["precomputed_g"], kwargs["precomputed_beta"]
    kernel[(1, 64, 12)](
        prefetch_codes=codes,
        PREFETCH_TOKEN=token,
        A_log=A_log,
        a=g,
        b=beta,
        dt_bias=bias,
        beta=1.0,
        threshold=20.0,
        q=qkv,
        k=qkv,
        v=qkv,
        mixed_qkv=qkv,
        o=output,
        h0=state,
        ht=state,
        cu_seqlens=kwargs["cu_seqlens"],
        ssm_state_indices=indices,
        num_accepted_tokens=kwargs["num_accepted_tokens"],
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
        PRECOMPUTED_GATING=True,
        QUANTIZE_BETA=False,
        QUANTIZE_STATE_EACH_STEP=False,
        MATCH_RECURRENT_NUMERICS=True,
        num_warps=1,
        num_stages=1,
    )
    return output, state
