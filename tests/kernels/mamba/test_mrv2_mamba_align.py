# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.mamba_align import (
    preprocess_mamba_align_fused_kernel,
    run_mamba_align_postprocess,
    run_mamba_align_precopy,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="MRV2 Mamba align needs CUDA/Triton"
)


def _make_context(
    conv: torch.Tensor,
    temporal: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_size: int = 4,
    max_num_reqs: int | None = None,
) -> SimpleNamespace:
    states = (conv, temporal)
    device = block_table.device
    return SimpleNamespace(
        is_initialized=True,
        num_layers=1,
        num_state_types=2,
        block_table_ptrs=torch.tensor(
            [block_table.data_ptr()], dtype=torch.int64, device=device
        ),
        block_table_stride_req=block_table.stride(0),
        state_base_addrs=torch.tensor(
            [state.data_ptr() for state in states],
            dtype=torch.int64,
            device=device,
        ),
        state_block_strides=torch.tensor(
            [state.stride(0) * state.element_size() for state in states],
            dtype=torch.int64,
            device=device,
        ),
        state_elem_sizes=torch.tensor(
            [state.element_size() for state in states],
            dtype=torch.int32,
            device=device,
        ),
        state_inner_sizes=torch.tensor(
            [conv.stride(1), temporal[0].numel()],
            dtype=torch.int64,
            device=device,
        ),
        state_conv_widths=torch.tensor(
            [conv.size(1), 0], dtype=torch.int32, device=device
        ),
        state_group_indices=torch.zeros(2, dtype=torch.int32, device=device),
        block_size=block_size,
        num_accepted_tokens_out=torch.empty(
            max_num_reqs or block_table.shape[0],
            dtype=torch.int32,
            device=device,
        ),
    )


def test_mrv2_align_prefix_seed_uses_resolved_mamba_block_size() -> None:
    device = torch.device("cuda")
    # Batch order is intentionally different from persistent request-slot order.
    idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device=device)
    state_idx = torch.full((2,), -2, dtype=torch.int32, device=device)
    num_computed = torch.tensor([8, 0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    num_accepted = torch.ones(2, dtype=torch.int32, device=device)
    src_col = torch.empty(2, dtype=torch.int32, device=device)
    token_bias = torch.empty(2, dtype=torch.int32, device=device)

    preprocess_mamba_align_fused_kernel[(1,)](
        idx_mapping,
        state_idx,
        num_computed,
        query_start_loc,
        num_accepted,
        src_col,
        token_bias,
        2,
        BLOCK_SIZE=triton.next_power_of_2(2),
        MAMBA_BLOCK_SIZE=8,
    )
    torch.accelerator.synchronize()

    # req0 resumes after one complete Mamba block and crosses into column 1;
    # req1 is fresh and has no source state before entering column 0.
    assert src_col.cpu().tolist() == [0, -1]
    assert state_idx.cpu().tolist() == [1, 0]
    assert token_bias.cpu().tolist() == [0, 0]


@pytest.mark.parametrize("token_bias", [0, 1, 2])
def test_mrv2_align_precopy_matches_v1_sd_semantics(token_bias: int) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    num_reqs, num_cols = 4, 6
    conv_width, conv_dim = 6, 128
    num_blocks = num_reqs * num_cols + 1
    block_table = torch.arange(1, num_blocks, dtype=torch.int32, device=device).reshape(
        num_reqs, num_cols
    )
    conv = torch.randn(
        num_blocks, conv_width, conv_dim, dtype=torch.float16, device=device
    )
    temporal = torch.randn(num_blocks, 4, 16, 16, dtype=torch.float32, device=device)
    conv_before = conv.clone()
    temporal_before = temporal.clone()

    # State arrays use persistent request-slot order; block-table rows use
    # current batch order. The non-identity mapping proves the two are not mixed.
    src_col = torch.tensor([-1, 1, 1, 2], dtype=torch.int32, device=device)
    dst_col = torch.tensor([0, 1, 2, 0], dtype=torch.int32, device=device)
    bias = torch.tensor(
        [0, 0, token_bias, token_bias], dtype=torch.int32, device=device
    )
    idx_mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)

    run_mamba_align_precopy(
        _make_context(conv, temporal, block_table),
        num_reqs,
        dst_col,
        src_col,
        bias,
        idx_mapping,
    )
    torch.accelerator.synchronize()

    conv_ref = conv_before.clone()
    temporal_ref = temporal_before.clone()
    for batch_idx, req_idx in enumerate(idx_mapping.cpu().tolist()):
        src = int(src_col[req_idx])
        dst = int(dst_col[req_idx])
        accepted_bias = int(bias[req_idx])
        if src < 0 or src == dst:
            continue
        src_block = int(block_table[batch_idx, src])
        dst_block = int(block_table[batch_idx, dst])
        temporal_block = int(block_table[batch_idx, src + accepted_bias])
        conv_ref[dst_block, : conv_width - accepted_bias] = conv_before[
            src_block, accepted_bias:
        ]
        temporal_ref[dst_block] = temporal_before[temporal_block]

    torch.testing.assert_close(conv, conv_ref, rtol=0, atol=0)
    torch.testing.assert_close(temporal, temporal_ref, rtol=0, atol=0)


def test_mrv2_align_postprocess_same_block_shift_is_exact() -> None:
    torch.manual_seed(43)
    device = torch.device("cuda")
    conv_width, conv_dim = 6, 128
    block_table = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32, device=device)
    conv = torch.randn(6, conv_width, conv_dim, dtype=torch.float16, device=device)
    temporal = torch.randn(6, 4, 16, 16, dtype=torch.float32, device=device)
    conv_before = conv.clone()
    temporal_before = temporal.clone()

    # accepted=3 and new_computed=8 means running=6, aligned=8, bias=2.
    # src_col == dst_col == 1 exercises the overlapping conv memmove.
    accepted = torch.tensor([3, 1, 1, 1], dtype=torch.int32, device=device)
    state_idx = torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=device)
    new_computed = torch.tensor([8, 0, 0, 0], dtype=torch.int32, device=device)
    idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    run_mamba_align_postprocess(
        _make_context(
            conv, temporal, block_table, block_size=4, max_num_reqs=accepted.numel()
        ),
        1,
        accepted,
        state_idx,
        new_computed,
        idx_mapping,
    )
    torch.accelerator.synchronize()

    dst_block = int(block_table[0, 1])
    temporal_src_block = int(block_table[0, 3])
    conv_ref = conv_before.clone()
    temporal_ref = temporal_before.clone()
    conv_ref[dst_block, : conv_width - 2] = conv_before[dst_block, 2:]
    temporal_ref[dst_block] = temporal_before[temporal_src_block]
    torch.testing.assert_close(conv, conv_ref, rtol=0, atol=0)
    torch.testing.assert_close(temporal, temporal_ref, rtol=0, atol=0)
    assert int(accepted[0]) == 1
