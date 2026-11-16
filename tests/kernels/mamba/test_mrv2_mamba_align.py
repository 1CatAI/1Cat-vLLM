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
    temporal_replay: bool = False,
) -> SimpleNamespace:
    states = (conv, temporal)
    device = block_table.device
    return SimpleNamespace(
        is_initialized=True,
        num_layers=1,
        num_state_types=2,
        num_states=len(states),
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
        temporal_replay=temporal_replay,
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
        torch.arange(num_reqs + 1, dtype=torch.int32, device=device) * 2,
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


@pytest.mark.parametrize("accepted_count", [2, 6, 8])
@pytest.mark.parametrize("query_len", [1, 8])
def test_speculative_to_single_token_tail_materializes_accepted_state(
    accepted_count: int, query_len: int
) -> None:
    torch.manual_seed(44)
    device = torch.device("cuda")
    # Physical IDs are deliberately unrelated to the speculative slot offsets.
    table = torch.tensor([[4, 7, 2, 9, 1, 8, 3, 6]], device=device, dtype=torch.int32)
    conv = torch.randn(10, 10, 128, device=device, dtype=torch.float16)
    temporal = torch.randn(10, 4, 16, 16, device=device, dtype=torch.float32)
    original_conv, original_temporal = conv.clone(), temporal.clone()
    ctx = _make_context(conv, temporal, table, block_size=8192)
    mapping = torch.zeros(1, device=device, dtype=torch.int32)
    state_idx = torch.zeros_like(mapping)
    computed = torch.full_like(mapping, 100)
    qsl = torch.tensor([0, query_len], device=device, dtype=torch.int32)
    accepted = torch.full_like(mapping, accepted_count)
    src, bias = torch.empty_like(mapping), torch.empty_like(mapping)

    def run():
        preprocess_mamba_align_fused_kernel[(1,)](
            mapping,
            state_idx,
            computed,
            qsl,
            accepted,
            src,
            bias,
            1,
            BLOCK_SIZE=1,
            MAMBA_BLOCK_SIZE=8192,
        )
        run_mamba_align_precopy(ctx, 1, state_idx, src, bias, mapping, qsl)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        conv.copy_(original_conv)
        temporal.copy_(original_temporal)
        accepted.fill_(accepted_count)
        state_idx.zero_()
        graph.replay()
        torch.accelerator.synchronize()
        expected_conv, expected_temporal = (
            original_conv.clone(),
            original_temporal.clone(),
        )
        if query_len == 1:
            offset = accepted_count - 1
            expected_conv[4, : 10 - offset] = original_conv[4, offset:]
            expected_temporal[4] = original_temporal[int(table[0, offset])]
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(temporal, expected_temporal, rtol=0, atol=0)
        assert int(accepted[0]) == (1 if query_len == 1 else accepted_count)


def test_mrv2_decode_ckpt_64_boundary_spec_replay_and_plain_decode() -> None:
    import numpy as np

    from vllm.model_executor.layers.mamba.gdn.spec_replay import SpecReplayState

    torch.manual_seed(45)
    device = torch.device("cuda")
    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
    conv = torch.randn(8, 8, 128, dtype=torch.float16, device=device)
    temporal = torch.randn(8, 4, 16, 16, dtype=torch.float32, device=device)

    state = SpecReplayState(max_reqs=4, max_rows=4, num_groups=1, device=device)
    # Mount even=5, odd=6 for req_idx=2.
    decode_ckpt = torch.full((1, 4, 2), -1, dtype=torch.int32, device=device)
    decode_ckpt[0, 2, 0] = 5
    decode_ckpt[0, 2, 1] = 6

    idx_mapping = torch.tensor([2], dtype=torch.int32, device=device)
    qsl = torch.tensor([0, 5], dtype=torch.int32, device=device)
    num_computed = torch.tensor([0, 0, 125, 0], dtype=torch.int32, device=device)
    ctx = _make_context(
        conv,
        temporal,
        block_table,
        block_size=2048,
        max_num_reqs=4,
        temporal_replay=True,
    )

    # 1. Speculative round crossing B64=128 (parity=0 -> slot 5, ckpt_token=2).
    state.prepare(
        idx_mapping,
        qsl,
        num_computed,
        np.array([4], dtype=np.int32),
        np.array([False]),
        1,
        ctx.block_table_ptrs,
        ctx.block_table_stride_req,
        2048,
        decode_ckpt_slots=decode_ckpt,
    )
    torch.accelerator.synchronize()
    assert int(state.spec_ckpt_token[0]) == 2
    assert int(state.spec_ckpt_slot[0, 0]) == 5
    assert int(state.verified[2]) == 1

    # Simulate verifier having written temporal[5] directly; postprocess must
    # preserve temporal[5] and shift conv from src_block=1 (bias=2) into conv[5].
    verifier_sentinel = torch.full_like(temporal[5], 77.5)
    temporal[5].copy_(verifier_sentinel)
    conv_before = conv.clone()
    accepted = torch.tensor([1, 1, 4, 1], dtype=torch.int32, device=device)
    state_idx = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=device)
    new_computed = torch.tensor([0, 0, 129, 0], dtype=torch.int32, device=device)
    run_mamba_align_postprocess(
        ctx,
        1,
        accepted,
        state_idx,
        new_computed,
        idx_mapping,
        decode_ckpt_slots=decode_ckpt,
        spec_verified=state.verified,
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(temporal[5], verifier_sentinel, rtol=0, atol=0)
    torch.testing.assert_close(conv[5, :6], conv_before[1, 2:], rtol=0, atol=0)

    # 2. Plain 1-token decode landing on B64=192 (parity=1 -> slot 6):
    # verified[2]==0, so BOTH temporal and conv copy from src_block=1 (bias=0).
    qsl_plain = torch.tensor([0, 1], dtype=torch.int32, device=device)
    num_computed[2] = 191
    state.prepare(
        idx_mapping,
        qsl_plain,
        num_computed,
        np.array([0], dtype=np.int32),
        np.array([False]),
        1,
        ctx.block_table_ptrs,
        ctx.block_table_stride_req,
        2048,
        decode_ckpt_slots=decode_ckpt,
    )
    accepted[2] = 1
    new_computed[2] = 192
    conv_before2 = conv.clone()
    temporal_before2 = temporal.clone()
    run_mamba_align_postprocess(
        ctx,
        1,
        accepted,
        state_idx,
        new_computed,
        idx_mapping,
        decode_ckpt_slots=decode_ckpt,
        spec_verified=state.verified,
    )
    torch.accelerator.synchronize()
    assert int(state.verified[2]) == 0
    torch.testing.assert_close(temporal[6], temporal_before2[1], rtol=0, atol=0)
    torch.testing.assert_close(conv[6], conv_before2[1], rtol=0, atol=0)


def test_mrv2_decode_ckpt_multi_request_staging_ring_no_async_overwrite() -> None:
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    device = torch.device("cuda")
    dummy_state = object.__new__(MambaHybridModelState)
    dummy_state._align_mode = True
    dummy_state.max_num_reqs = 8
    dummy_state.device = device
    dummy_state._decode_ckpt_slots_gpu = torch.full(
        (8, 8, 2), -1, dtype=torch.int32, device=device
    )
    dummy_state._decode_ckpt_cpu = torch.full((8, 8, 2), -1, dtype=torch.int32)
    dummy_state._decode_ckpt_stage_ring = [
        torch.empty((8, 8, 2), dtype=torch.int32, pin_memory=True) for _ in range(4)
    ]
    dummy_state._decode_ckpt_ring_idx = 0
    dummy_state._decode_ckpt_dirty = False

    # Mount 4 concurrent requests in the same step without synchronization.
    for r in range(4):
        dummy_state.mount_decode_ckpt_blocks(
            r,
            [100 + r * 10 + g for g in range(4)],
            [200 + r * 10 + g for g in range(4)],
        )
    dummy_state.flush_decode_ckpt_blocks()
    # Immediately mutate slot 0 across two subsequent async steps before sync.
    dummy_state.remove_request(0)
    dummy_state.flush_decode_ckpt_blocks()
    dummy_state.mount_decode_ckpt_blocks(0, [901, 902, 903, 904], [911, 912, 913, 914])
    dummy_state.flush_decode_ckpt_blocks()
    torch.accelerator.synchronize()

    got = dummy_state._decode_ckpt_slots_gpu[:4, :4, :].cpu()
    assert got[:, 0, 0].tolist() == [901, 902, 903, 904]
    assert got[:, 0, 1].tolist() == [911, 912, 913, 914]
    for r in (1, 2, 3):
        assert got[:, r, 0].tolist() == [100 + r * 10 + g for g in range(4)]
        assert got[:, r, 1].tolist() == [200 + r * 10 + g for g in range(4)]
