# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional SM70 single-request GDN verification provider."""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def native_verifier():
    return verify if hasattr(torch.ops._C, "sm70_gdn_verify_out") else None


def verify(
    layer,
    requests,
    mixed_qkv,
    a,
    b,
    state,
    out,
    num_actual_tokens,
    direct_out,
    query_start,
    state_indices,
    slot_selectors,
):
    if not (
        requests == 1
        and mixed_qkv.dtype == torch.float16
        and mixed_qkv.stride(1) == 1
        and mixed_qkv.shape[0] <= 8
        and state.dtype == torch.float32
        and layer.head_k_dim == 128
        and layer.head_v_dim == 128
    ):
        return None
    logger.info_once("SM70 CUDA GDN single-request verification route hit.")
    tokens = mixed_qkv.shape[0]
    heads = layer.num_v_heads // layer.tp_size
    verify_out = (
        out[:num_actual_tokens].unsqueeze(0)
        if direct_out
        else mixed_qkv.new_empty((1, tokens, heads, 128))
    )
    torch.ops._C.sm70_gdn_verify_out(
        mixed_qkv,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        state,
        verify_out.view(-1, heads, 128),
        query_start[: requests + 1],
        state_indices,
        slot_selectors,
        layer.num_k_heads // layer.tp_size,
        heads,
        layer.head_k_dim**-0.5,
        1,
        None,
        None,
        None,
        tokens,
    )
    return verify_out, state
