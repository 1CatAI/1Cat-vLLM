# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.llm_base_proposer import (
    SpecDecodeBaseProposer,
    _clone_drafter_mutable_metadata,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    eagle_prepare_next_token_padded_kernel,
    next_power_of_2,
)
from vllm.v1.worker.gpu_model_runner import (
    _async_spec_decode_participating_prev_positions,
    _count_contiguous_spec_tokens,
)

DEVICE_TYPE = current_platform.device_type


pytestmark = pytest.mark.skipif(
    DEVICE_TYPE != "cuda",
    reason="SM70 MTP safety tests exercise Triton CUDA helper kernels.",
)


def test_prepare_next_token_padded_counts_only_contiguous_prefix():
    device = torch.device(DEVICE_TYPE)
    sampled_token_ids = torch.tensor(
        [
            [11, 12, -1, 88, 89],
            [-1, 13, 14, 15, 16],
            [21, -2, 22, -1, 23],
            [31, 32, 33, 34, 35],
            [41, 42, 43, 44, 45],
        ],
        dtype=torch.int32,
        device=device,
    )
    discard_request_mask = torch.tensor(
        [False, False, False, False, True],
        dtype=torch.bool,
        device=device,
    )
    backup_next_token_ids = torch.tensor(
        [1000, 1001, 1002, 1003, 1004],
        dtype=torch.int32,
        device=device,
    )
    next_token_ids = torch.empty(5, dtype=torch.int32, device=device)
    valid_sampled_tokens_count = torch.empty(5, dtype=torch.int32, device=device)

    eagle_prepare_next_token_padded_kernel[(5,)](
        sampled_token_ids,
        discard_request_mask,
        backup_next_token_ids,
        next_token_ids,
        valid_sampled_tokens_count,
        100,
        sampled_token_ids.shape[1],
        5,
        sampled_token_ids.stride(0),
        BLOCK_SIZE_TOKENS=next_power_of_2(sampled_token_ids.shape[1]),
    )
    torch.accelerator.synchronize()

    assert torch.equal(
        next_token_ids.cpu(),
        torch.tensor([12, 1001, 21, 35, 1004], dtype=torch.int32, device="cpu"),
    )
    assert torch.equal(
        valid_sampled_tokens_count.cpu(),
        torch.tensor([2, 0, 1, 5, 0], dtype=torch.int32, device="cpu"),
    )


def test_prepare_inputs_padded_does_not_alias_target_seq_lens():
    device = torch.device(DEVICE_TYPE)
    query_start_loc = torch.tensor([0, 3, 6, 9], dtype=torch.int32, device=device)
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=torch.tensor([0, 3, 6, 9], dtype=torch.int32),
        seq_lens=torch.tensor([3, 3, 3], dtype=torch.int32, device=device),
        num_reqs=3,
        num_actual_tokens=9,
        max_query_len=3,
        max_seq_len=3,
        block_table_tensor=torch.zeros((3, 1), dtype=torch.int32, device=device),
        slot_mapping=torch.arange(9, dtype=torch.int64, device=device),
        seq_lens_cpu_upper_bound=torch.tensor([3, 3, 3], dtype=torch.int32),
        dcp_local_seq_lens=torch.tensor([3, 3, 3], dtype=torch.int32, device=device),
    )
    spec_decode_metadata = SpecDecodeMetadata.make_dummy(
        draft_token_ids=[[0, 0], [0, 0], [0, 0]],
        device=device,
    )
    valid_sampled_tokens_count = torch.tensor(
        [2, 3, 1], dtype=torch.int32, device=device
    )

    original_seq_lens = common_attn_metadata.seq_lens.clone()
    original_dcp_lens = common_attn_metadata.dcp_local_seq_lens.clone()
    original_upper_bound = common_attn_metadata.seq_lens_cpu_upper_bound.clone()

    output_metadata, _, num_rejected_tokens_gpu = (
        SpecDecodeBaseProposer.prepare_inputs_padded(
            None,
            common_attn_metadata,
            spec_decode_metadata,
            valid_sampled_tokens_count,
        )
    )

    assert (
        output_metadata.seq_lens.data_ptr() != common_attn_metadata.seq_lens.data_ptr()
    )
    assert (
        output_metadata.dcp_local_seq_lens.data_ptr()
        != common_attn_metadata.dcp_local_seq_lens.data_ptr()
    )
    assert (
        output_metadata.seq_lens_cpu_upper_bound.data_ptr()
        != common_attn_metadata.seq_lens_cpu_upper_bound.data_ptr()
    )

    output_metadata.seq_lens -= num_rejected_tokens_gpu
    output_metadata.dcp_local_seq_lens -= num_rejected_tokens_gpu
    output_metadata.seq_lens_cpu_upper_bound += 1
    assert torch.equal(common_attn_metadata.seq_lens, original_seq_lens)
    assert torch.equal(common_attn_metadata.dcp_local_seq_lens, original_dcp_lens)
    assert torch.equal(
        common_attn_metadata.seq_lens_cpu_upper_bound,
        original_upper_bound,
    )


def test_drafter_entry_metadata_clone_isolation():
    device = torch.device(DEVICE_TYPE)
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=device),
        _seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
        _num_computed_tokens_cpu=torch.tensor([7], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([8], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=8,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(1, dtype=torch.int64, device=device),
        dcp_local_seq_lens=torch.tensor([8], dtype=torch.int32, device=device),
    )

    cloned = _clone_drafter_mutable_metadata(common_attn_metadata)
    cloned.seq_lens += 1
    cloned.dcp_local_seq_lens += 1
    cloned._seq_lens_cpu += 1
    cloned._num_computed_tokens_cpu += 1
    cloned.seq_lens_cpu_upper_bound += 1

    assert common_attn_metadata.seq_lens.item() == 8
    assert common_attn_metadata.dcp_local_seq_lens.item() == 8
    assert common_attn_metadata._seq_lens_cpu.item() == 8
    assert common_attn_metadata._num_computed_tokens_cpu.item() == 7
    assert common_attn_metadata.seq_lens_cpu_upper_bound.item() == 8


def test_accepted_count_ignores_stale_tokens_after_first_rejection():
    device = torch.device(DEVICE_TYPE)
    output_token_ids = torch.tensor(
        [
            [101, 102, -1, 999, 1000],
            [-1, 201, 202, 203, 204],
            [301, -1, -1, 302, 303],
            [401, 402, 403, 404, 405],
        ],
        dtype=torch.int32,
        device=device,
    )

    accepted_counts = _count_contiguous_spec_tokens(output_token_ids)

    assert torch.equal(
        accepted_counts.cpu(),
        torch.tensor([2, 0, 1, 5], dtype=torch.int32, device="cpu"),
    )


def test_async_spec_decode_skips_stale_counts_without_previous_drafts():
    participating = _async_spec_decode_participating_prev_positions(
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
    )

    assert participating.size == 0


def test_async_spec_decode_selects_only_previous_rows_with_drafts():
    participating = _async_spec_decode_participating_prev_positions(
        np.array([2, -1, 0, 1], dtype=np.int32),
        np.array([0, 4, 2], dtype=np.int32),
    )

    assert participating.tolist() == [2, 1]


def test_dynamic_vocab_sampler_handles_all_nonfinite_candidates():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 fused dynamic-vocabulary ops require a V100 GPU.")

    from vllm import _sm70_ops

    device = torch.device(DEVICE_TYPE)
    hidden_states = torch.full((1, 64), torch.nan, dtype=torch.float16, device=device)
    weight = torch.randn((256, 64), dtype=torch.float16, device=device)
    prepared_weight, prepared_meta = _sm70_ops.sm70_f16_prepare(weight)
    base_values = torch.empty((1, 20), dtype=torch.float32, device=device)
    base_indices = torch.empty((1, 20), dtype=torch.int64, device=device)
    _sm70_ops.sm70_f16_lm_head_top20_tc_out(
        base_values,
        base_indices,
        hidden_states,
        prepared_weight,
        int(prepared_meta[0].item()),
        0,
        0,
    )
    base_token_ids = torch.arange(256, dtype=torch.int64, device=device)
    tail_logits = torch.full((32,), torch.inf, dtype=torch.float16, device=device)
    tail_token_ids = torch.arange(300, 332, dtype=torch.int64, device=device)
    pairs = torch.empty((20, 3), dtype=torch.float32, device=device)

    _sm70_ops.sm70_merge_tail_top20_pack_out(
        pairs,
        base_values[0],
        base_indices[0],
        base_token_ids,
        tail_logits,
        tail_token_ids,
        20,
    )
    sampled_token = torch.empty(1, dtype=torch.int64, device=device)
    sparse_ids = torch.empty(20, dtype=torch.int64, device=device)
    sparse_probs = torch.empty(20, dtype=torch.float32, device=device)
    exponential = torch.ones(64, dtype=torch.float32, device=device)
    _sm70_ops.sm70_sample_packed_top20_out(
        sampled_token,
        sparse_ids,
        sparse_probs,
        pairs,
        exponential,
        0.95,
    )
    torch.accelerator.synchronize()

    assert bool(((base_indices >= 0) & (base_indices < 256)).all())
    assert bool(torch.isfinite(pairs[:, 0]).all())
    assert bool(((sparse_ids >= 0) & (sparse_ids < 332)).all())
    assert bool(torch.isfinite(sparse_probs).all())
    assert float(sparse_probs.sum()) == pytest.approx(1.0, abs=1e-6)
    assert int(sampled_token.item()) in sparse_ids.tolist()


def test_dynamic_vocab_merge_preserves_finite_top20_order():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 fused dynamic-vocabulary ops require a V100 GPU.")

    from vllm import _sm70_ops

    device = torch.device(DEVICE_TYPE)
    base_values = torch.arange(40, 20, -1, dtype=torch.float32, device=device)
    base_indices = torch.arange(20, dtype=torch.int64, device=device)
    base_token_ids = torch.arange(100, 120, dtype=torch.int64, device=device)
    tail_logits = torch.arange(32, dtype=torch.float16, device=device)
    tail_token_ids = torch.arange(200, 232, dtype=torch.int64, device=device)
    pairs = torch.empty((20, 3), dtype=torch.float32, device=device)

    _sm70_ops.sm70_merge_tail_top20_pack_out(
        pairs,
        base_values,
        base_indices,
        base_token_ids,
        tail_logits,
        tail_token_ids,
        20,
    )
    torch.accelerator.synchronize()

    all_values = torch.cat((base_values, tail_logits.float()))
    all_ids = torch.cat((base_token_ids, tail_token_ids))
    expected_positions = torch.argsort(all_values, descending=True, stable=True)[:20]
    assert torch.equal(pairs[:, 0].cpu(), all_values[expected_positions].cpu())
    assert torch.equal(
        pairs[:, 1].to(torch.int64).cpu(),
        all_ids[expected_positions].cpu(),
    )
