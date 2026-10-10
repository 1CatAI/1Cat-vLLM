# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise-exact kernel fusion for the SM70 Qwen GDN MTP verify path.

Production runs every Qwen GDN layer eagerly inside the opaque
``vllm::qwen_gdn_full_forward`` op (the MTP full-forward quality guard), so
Inductor cannot fuse its small kernels. For a pure speculative-verify batch
the layer issues 26 small kernels between the in_proj_ba GEMM and out_proj.
The units below remove 22 of them without changing a single output bit:

* u1: do not materialize z, a and b (index_select / contiguous copies); the
  consumers read strided views of the projection outputs. The z view is used
  only with u4 and the a/b views only with u3.
* u2: the causal-conv1d update writes its output directly in the packed
  q|k|v layout that ``rearrange_mixed_qkv`` builds (removes 3 copies + cat).
  The convolution arithmetic is the production kernel's, unchanged.
* u3: the recurrent gated-delta kernel computes g and beta in-kernel with the
  exact expressions of ``fused_gdn_gating_kernel`` (FP32 values, no rounding
  boundary in between) and writes core_attn_out directly (removes the gating
  kernel and the output copy). Padded rows of core_attn_out keep the zeros of
  the preceding fill instead of uninitialized memory.
* u4: RMSNormGated (sigmoid gate, FP16 in/out) as one kernel that reproduces
  the eager ATen chain bit for bit (see the kernel docstring).

``ONECAT_GDN_FUSE`` selects the units: a comma list of u1..u4, or "all".
Unset or empty keeps the production path.
"""

import collections
import os
from typing import cast

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    _select_sm70_bv,
    _select_sm70_num_stages,
    _select_sm70_num_warps,
    _use_sm70_fla_recurrent_schedule,
)
from vllm.model_executor.layers.fla.ops.op import exp
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

try:
    from triton.language.extra.cuda import libdevice
except ImportError:  # pragma: no cover
    from triton.language.extra import libdevice

UNITS = ("u1", "u2", "u3", "u4")
logger = init_logger(__name__)
# STEP-48 W1 (opt-in): ONECAT_GDN_BV48=<n> caps the verify recurrence's value block
# at n (BV = min(automatic choice, n)). Every BV keeps the same per-row K reduction
# layout (4 elements per thread x 32 lanes, one warp), so the output and the written
# states are bit-identical. Unset keeps the automatic choice.
_GDN_BV48 = os.getenv("ONECAT_GDN_BV48", "")
HEAD_DIM = 128


def parse_units(raw: str | None) -> frozenset[str]:
    if raw is None:
        return frozenset()
    items = {item.strip().lower() for item in raw.split(",") if item.strip()}
    if "all" in items:
        return frozenset(UNITS)
    unknown = items - set(UNITS) - {"none", "0", "off"}
    if unknown:
        raise ValueError(f"ONECAT_GDN_FUSE: unknown units {sorted(unknown)}")
    return frozenset(items & set(UNITS))


_ENV_UNITS = parse_units(os.getenv("ONECAT_GDN_FUSE"))
_UNITS_OVERRIDE: frozenset[str] | None = None
# Route decisions taken in Python (eager calls and graph captures).
ROUTE_COUNTS: collections.Counter = collections.Counter()


def enabled_units() -> frozenset[str]:
    return _ENV_UNITS if _UNITS_OVERRIDE is None else _UNITS_OVERRIDE


def set_units_override(units: frozenset[str] | None) -> None:
    """Diagnostics: replace the environment selection (None restores it)."""
    global _UNITS_OVERRIDE
    _UNITS_OVERRIDE = units


# --------------------------------------------------------------------------
# u2: causal_conv1d_update with packed q|k|v output
# --------------------------------------------------------------------------
@triton.jit
def _causal_conv1d_update_packed_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # packed [total * q_dim | total * k_dim | total * v_dim]
    total_tokens,
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.int64,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    # Body copied from causal_conv1d.py::_causal_conv1d_update_kernel; only the
    # output addressing differs (packed q|k|v instead of in place).
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    # Output region of this program (BLOCK_N divides Q_DIM and K_DIM): q, k or v,
    # starting at feature region_start; the region holds total_tokens rows of
    # region_width features each.
    feat_block = tl.program_id(1) * BLOCK_N
    in_k = (feat_block >= Q_DIM).to(tl.int32)
    in_v = (feat_block >= Q_DIM + K_DIM).to(tl.int32)
    region_start = in_k * Q_DIM + in_v * K_DIM
    region_width = Q_DIM + in_k * (K_DIM - Q_DIM) + in_v * (V_DIM - K_DIM)

    if IS_APC_ENABLED:
        # Get the state from the initial_state_idx
        conv_state_init = tl.load(initial_state_idx + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
    else:
        conv_state_init = 0
        current_last_index = 0

    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
    ).to(tl.int64)

    if USE_PAD_SLOT:  # noqa
        if conv_states_input_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_VARLEN:
        query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
        query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
        # revise state_len and seqlen
        state_len = state_len - (seqlen - (query_end_index - query_start_index))
        seqlen = query_end_index - query_start_index
        x_offset = query_start_index * stride_x_token
    else:
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq

    if query_start_index == query_end_index:
        return

    if IS_SPEC_DECODING:
        conv_state_token_offset = (
            tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
        )
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        conv_states_ptrs = prior_tokens + 4 * stride_conv_state_tok  # [BLOCK_N]
        col4 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
            :, None
        ]
    )  # [BLOCK_M, BLOCK_N]
    mask = (
        (conv_states_input_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    # Get the state from the initial_state_idx
    # cache_idx
    conv_states_offset = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
    ).to(tl.int64)
    if USE_PAD_SLOT:  # noqa
        if conv_states_offset == pad_slot_id:
            return
    conv_state_ptrs_target = (
        conv_state_ptr
        + (conv_states_offset * stride_conv_state_seq)  # Offset from seq
        + (idx_feats * stride_conv_state_dim)
    )[None, :] + (  # [BLOCK_N,]
        idx_tokens * stride_conv_state_tok
    )[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + (4 * stride_w_width)  # [BLOCK_N] tensor
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + (5 * stride_w_width)  # [BLOCK_N] tensor
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim
    o_base = (
        o_ptr
        + region_start * total_tokens
        + query_start_index * region_width
        + (idx_feats - region_start)
    )

    # STEP 5: compute each token
    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = o_base + idx_token * region_width

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_update_packed(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: bool | str | None,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    out: torch.Tensor,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """causal_conv1d_update (varlen, speculative) writing q|k|v packed.

    Argument handling follows causal_conv1d.py::causal_conv1d_update for the
    query_start_loc / num_accepted_tokens case. ``out`` is a flat buffer of
    x.size(0) * (q_dim + k_dim + v_dim) elements in rearrange_mixed_qkv order.
    """
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    assert x.dtype == conv_state.dtype and out.dtype == x.dtype
    assert x.dim() == 2 and x.stride(1) == 1
    total, dim = x.shape
    assert dim == q_dim + k_dim + v_dim
    assert out.is_contiguous() and out.numel() == total * dim
    block_n = 256
    assert q_dim % block_n == 0 and k_dim % block_n == 0
    batch = conv_state_indices.size(0)
    seqlen = max_query_len
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()
    stride_w_dim, stride_w_width = weight.stride()
    stride_x_token, stride_x_dim = x.stride()
    stride_x_seq = 0
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0)
    state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (batch, triton.cdiv(dim, META["BLOCK_N"]))

    _causal_conv1d_update_packed_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        None,
        None,
        out,
        total,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=True,
        IS_APC_ENABLED=False,
        IS_SPEC_DECODING=True,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=block_n,
        Q_DIM=q_dim,
        K_DIM=k_dim,
        V_DIM=v_dim,
    )
    return out


# --------------------------------------------------------------------------
# u3: recurrent gated delta rule with in-kernel gating and direct output
# --------------------------------------------------------------------------
@triton.jit(do_not_specialize=["N", "T"])
def _fused_recurrent_gdn_verify_kernel(
    q,
    k,
    v,
    b_ptr,
    a_ptr,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    stride_b_tok,
    stride_a_tok,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    softplus_beta: tl.constexpr,
    softplus_threshold: tl.constexpr,
    zero_rows,  # rows of o to cover with ZERO_FILL (o's full first dimension)
    ZERO_FILL: tl.constexpr = False,
):
    # fla/ops/fused_recurrent.py::fused_recurrent_gated_delta_rule_fwd_kernel
    # for IS_VARLEN, IS_CONTINUOUS_BATCHING, IS_SPEC_DECODING, USE_INITIAL_STATE,
    # INPLACE_FINAL_STATE, USE_QK_L2NORM_IN_KERNEL, scalar beta, not KDA.
    # g and beta come from qwen_gdn_linear_attn.py::fused_gdn_gating_kernel's
    # expressions instead of FP32 buffers written by that kernel.
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if ZERO_FILL:  # noqa: SIM102 - keep the Triton constexpr branch separate
        if i_nh == N * HV:
            # STEP-47 u3z: rows past the last sequence (graph padding and the
            # rows beyond T) get the zeros the caller no longer pre-fills.
            o_vz = i_v * BV + tl.arange(0, BV)
            tail = tl.load(cu_seqlens + N).to(tl.int64)
            for row in range(tail, zero_rows):
                for hz in range(0, HV):
                    tl.store(
                        o + (row * HV + hz) * V + o_vz,
                        tl.zeros([BV], dtype=o.dtype.element_ty),
                        mask=o_vz < V,
                    )
            return
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int64),
        tl.load(cu_seqlens + i_n + 1).to(tl.int64),
    )
    all = T
    T = eos - bos

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b_ptr + bos * stride_b_tok + i_hv
    p_a = a_ptr + bos * stride_a_tok + i_hv

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    blk_A_log = tl.load(A_log + i_hv)
    blk_bias = tl.load(dt_bias + i_hv)

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    # Load state index and check for invalid entries.
    # Mamba/GDN state tables use PAD_SLOT_ID=-1; state slot 0 is a
    # valid live slot in the 0.0.3 MTP path.
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
    if state_idx < 0:
        if ZERO_FILL:
            # This sequence's rows stay as the (formerly zero-filled) buffer.
            for i_z in range(0, T):
                tl.store(
                    p_o + i_z * HV * V,
                    tl.zeros([BV], dtype=p_o.dtype.element_ty),
                    mask=mask_v,
                )
        return
    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BV, BK]
        # fused_gdn_gating_kernel: g = -exp(A_log) * softplus(a + dt_bias)
        blk_a = tl.load(p_a)
        x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
        softplus_x = tl.where(
            softplus_beta * x <= softplus_threshold,
            (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * x)),
            x,
        )
        b_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
        b_h *= exp(b_g)
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        # fused_gdn_gating_kernel: beta = sigmoid(b), FP32
        blk_b = tl.load(p_b)
        b_beta = tl.sigmoid(blk_b.to(tl.float32))
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        # Load state index and check for invalid entries.
        final_state_idx = tl.load(
            ssm_state_indices + i_n * stride_indices_seq + i_t
        ).to(tl.int64)
        if final_state_idx >= 0:
            p_ht = ht + final_state_idx * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_a += stride_a_tok
        p_b += stride_b_tok


def fused_recurrent_gdn_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    zero_fill_rows: int | None = None,
) -> torch.Tensor:
    """fused_recurrent_gated_delta_rule(q, k, v, g, beta, inplace_final_state,
    cu_seqlens, ssm_state_indices, num_accepted_tokens, qk-l2norm) with
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias, beta_dtype=float32)
    evaluated in-kernel; o is written to ``out`` ([T, HV, V], contiguous).
    q, k: [1, T, H, K]; v: [1, T, HV, V]; a, b: [T, HV] with unit inner stride.
    """
    B, T, H, K = cast(tuple[int, int, int, int], k.shape)
    V = v.size(-1)
    HV = v.shape[2]
    assert B == 1 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert a.stride(1) == 1 and b.stride(1) == 1
    assert out.is_contiguous() and out.shape[0] >= T and out.shape[-2:] == (HV, V)
    assert ssm_state_indices.ndim == 2
    N = len(cu_seqlens) - 1
    scale = K**-0.5
    sm70_schedule = _use_sm70_fla_recurrent_schedule(q.device)
    BK = triton.next_power_of_2(K)
    BV = (
        _select_sm70_bv(V, N, HV, q.device)
        if sm70_schedule
        else min(triton.next_power_of_2(V), 32)
    )
    if sm70_schedule and _GDN_BV48:
        BV = min(BV, int(_GDN_BV48))
        logger.info_once("ONECAT_FUSE47 b48 route: cap%d", int(_GDN_BV48))
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = _select_sm70_num_stages(T) if sm70_schedule else 3
    num_warps = _select_sm70_num_warps(BV, N, HV) if sm70_schedule else 1
    stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()
    # STEP-47 u3z: zero_fill_rows = the full row count of the (uninitialized)
    # buffer behind ``out``; one extra program zeroes the rows past the last
    # sequence and invalid-state sequences zero their own rows.
    zero_fill = zero_fill_rows is not None
    grid = (NK, NV, N * HV + (1 if zero_fill else 0))
    _fused_recurrent_gdn_verify_kernel[grid](
        q=q,
        k=k,
        v=v,
        b_ptr=b,
        a_ptr=a,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        stride_b_tok=b.stride(0),
        stride_a_tok=a.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        zero_rows=zero_fill_rows if zero_fill_rows is not None else 0,
        ZERO_FILL=zero_fill,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# --------------------------------------------------------------------------
# u4: RMSNormGated (sigmoid gate) in one kernel, bitwise equal to eager
# --------------------------------------------------------------------------
@triton.jit
def _gdn_rmsnorm_sigmoid_gate_exact_kernel(
    x_ptr,  # fp16 [rows, 128], contiguous
    z_ptr,  # fp16, row (t, h) at z_ptr + t * stride_zt + h * 128
    w_ptr,  # fp16 [128]
    out_ptr,  # fp16 [rows, 128], contiguous
    num_heads,
    stride_zt,
    eps,
):
    """RMSNormGated.forward_static (group_size None, norm_before_gate, sigmoid)
    as computed by the eager chain under torch 2.10 on CUDA:
      v = x*x; var = mean(v): ATen reduce_kernel, vec4 input path: lane j of 32
      sums ((v[4j] + v[4j+1]) + v[4j+2]) + v[4j+3], then shfl_down 16..1
      (Reduce.cuh:663), times 1/128; r = rsqrtf(var + eps);
      out = ((x * r) * w) * (1 / (1 + expf(-z))) (IEEE division), to fp16 (RN).
    One row per program and one warp; element 4j+i lives in lane j, so tl.sum
    over the 32 lanes is Triton's butterfly 16..1, which leaves every lane with
    lane 0's association of ATen's shfl_down sequence. Launched with
    enable_fp_fusion=False so no product is contracted into an FMA."""
    row = tl.program_id(0)
    t = row // num_heads
    h = row % num_heads
    lane = tl.arange(0, 32)
    xb = x_ptr + row * 128 + lane * 4
    zb = z_ptr + t * stride_zt + h * 128 + lane * 4
    wb = w_ptr + lane * 4
    x0 = tl.load(xb + 0).to(tl.float32)
    x1 = tl.load(xb + 1).to(tl.float32)
    x2 = tl.load(xb + 2).to(tl.float32)
    x3 = tl.load(xb + 3).to(tl.float32)
    part = ((x0 * x0 + x1 * x1) + x2 * x2) + x3 * x3
    var = tl.sum(part, axis=0) * 0.0078125
    r = libdevice.rsqrt(var + eps)
    one = tl.full([32], 1.0, tl.float32)
    for i in tl.static_range(4):
        if i == 0:
            xi = x0
        elif i == 1:
            xi = x1
        elif i == 2:
            xi = x2
        else:
            xi = x3
        wi = tl.load(wb + i).to(tl.float32)
        zi = tl.load(zb + i).to(tl.float32)
        sig = libdevice.div_rn(one, one + libdevice.exp(-zi))
        o = ((xi * r) * wi) * sig
        tl.store(out_ptr + row * 128 + lane * 4 + i, o.to(tl.float16))


def gdn_rmsnorm_sigmoid_gate_exact(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """x: [T, H, 128] or [T * H, 128] fp16 contiguous; z: [T, H * 128] fp16
    with unit inner stride (any row stride); weight: [128] fp16.
    Returns [T * H, 128] fp16."""
    assert x.dtype == torch.float16 and z.dtype == torch.float16
    assert weight.dtype == torch.float16 and weight.is_contiguous()
    assert x.is_contiguous() and weight.numel() == HEAD_DIM
    assert z.dim() == 2 and z.stride(1) == 1 and z.shape[1] % HEAD_DIM == 0
    num_heads = z.shape[1] // HEAD_DIM
    rows = x.numel() // HEAD_DIM
    assert rows == z.shape[0] * num_heads
    out = torch.empty((rows, HEAD_DIM), dtype=torch.float16, device=x.device)
    if rows:
        _gdn_rmsnorm_sigmoid_gate_exact_kernel[(rows,)](
            x,
            z,
            weight,
            out,
            num_heads,
            z.stride(0),
            eps,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return out
