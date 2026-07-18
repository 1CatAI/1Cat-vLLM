# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def _require_sm70_extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("split-KV BM32 entry is SM70-only")
    return pytest.importorskip("flash_attn_v100_cuda")


def _run_case(
    *, query_len: int, sequence_len: int, page_ids: list[int], seed: int
) -> dict[str, torch.Tensor]:
    extension = _require_sm70_extension()
    torch.manual_seed(seed)

    batch_size = 1
    num_q_heads = 6
    num_kv_heads = 1
    head_dim = 256
    split_parts = 3
    page_size = 784

    query = torch.randn(
        (batch_size, num_q_heads, query_len, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    key_cache = torch.randn(
        (max(page_ids) + 1, page_size, num_kv_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.tensor(
        [page_ids], device="cuda", dtype=torch.int32
    )
    sequence_lengths = torch.full(
        (batch_size,), sequence_len, device="cuda", dtype=torch.int32
    )
    softmax_scale = head_dim**-0.5

    unsplit_out = torch.empty_like(query)
    unsplit_lse = torch.empty(
        query.shape[:-1], device="cuda", dtype=torch.float32
    )
    split_out = torch.empty_like(query)
    split_lse = torch.empty_like(unsplit_lse)
    split_tmp_out = torch.empty(
        (batch_size, num_q_heads, split_parts, query_len, head_dim),
        device="cuda",
        dtype=torch.float32,
    )
    split_tmp_row_max = torch.empty(
        (batch_size, num_q_heads, split_parts, query_len),
        device="cuda",
        dtype=torch.float32,
    )
    split_tmp_row_sum = torch.empty_like(split_tmp_row_max)

    extension.prefill_paged_d256_bm32_allp_pair_scratch_fwd(
        query,
        key_cache,
        value_cache,
        unsplit_out,
        unsplit_lse,
        block_table,
        sequence_lengths,
        softmax_scale,
    )
    returned_out, returned_lse = (
        extension.prefill_paged_d256_bm32_allp_pair_scratch_splitkv3_fwd(
            query,
            key_cache,
            value_cache,
            split_out,
            split_lse,
            split_tmp_out,
            split_tmp_row_max,
            split_tmp_row_sum,
            block_table,
            sequence_len,
            softmax_scale,
        )
    )
    torch.cuda.synchronize()

    assert returned_out.data_ptr() == split_out.data_ptr()
    assert returned_lse.data_ptr() == split_lse.data_ptr()
    assert torch.isfinite(split_out).all()
    assert torch.isfinite(split_lse).all()

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "unsplit_out": unsplit_out,
        "unsplit_lse": unsplit_lse,
        "split_out": split_out,
        "split_lse": split_lse,
        "split_tmp_out": split_tmp_out,
        "split_tmp_row_max": split_tmp_row_max,
        "split_tmp_row_sum": split_tmp_row_sum,
    }


@torch.inference_mode()
def test_splitkv3_causal_fully_masked_partitions_are_zero() -> None:
    result = _run_case(
        query_len=512,
        sequence_len=513,
        page_ids=[0],
        seed=20260716,
    )

    # For query row zero the causal cutoff is token one. Splits one and two
    # start at token 128 and 384, so both must contribute exactly zero.
    assert torch.count_nonzero(result["split_tmp_row_sum"][0, :, 1:, 0]) == 0
    assert torch.count_nonzero(result["split_tmp_out"][0, :, 1:, 0]) == 0
    assert torch.all(result["split_tmp_row_max"][0, :, 1:, 0] == -1e30)

    torch.testing.assert_close(
        result["split_out"],
        result["unsplit_out"],
        rtol=0.0,
        atol=0.001953125,
    )
    torch.testing.assert_close(
        result["split_lse"], result["unsplit_lse"], rtol=0.0, atol=0.001
    )


@pytest.mark.parametrize("page_ids", ([0, 1], [1, 0]))
@torch.inference_mode()
def test_splitkv3_fast_visible_equality_boundary_and_page_crossing(
    page_ids: list[int],
) -> None:
    # ceil(785 / 128) = 7 and floor(2 * 7 / 3) * 128 = 512 = 785 - 273.
    # This is the fast-visible equality boundary, with a 17-token final tile
    # and a query tail crossing the 784-token page boundary.
    result = _run_case(
        query_len=273,
        sequence_len=785,
        page_ids=page_ids,
        seed=20260717,
    )

    torch.testing.assert_close(
        result["split_out"],
        result["unsplit_out"],
        rtol=0.0,
        atol=0.001953125,
    )
    torch.testing.assert_close(
        result["split_lse"], result["unsplit_lse"], rtol=0.0, atol=0.001
    )


@torch.inference_mode()
def test_splitkv3_python_wrapper_reuses_stream_workspace() -> None:
    interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    result = _run_case(
        query_len=273,
        sequence_len=785,
        page_ids=[1, 0],
        seed=20260718,
    )
    interface._prefill_splitkv3_workspace_cache.clear()

    first_out, first_lse = (
        interface.flash_attn_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3(
            result["query"],
            result["key_cache"],
            result["value_cache"],
            result["block_table"],
            785,
        )
    )
    second_out, second_lse = (
        interface.flash_attn_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3(
            result["query"],
            result["key_cache"],
            result["value_cache"],
            result["block_table"],
            785,
        )
    )
    torch.cuda.synchronize()

    assert first_out.data_ptr() == second_out.data_ptr()
    assert first_lse.data_ptr() == second_lse.data_ptr()
    assert len(interface._prefill_splitkv3_workspace_cache) == 1
    torch.testing.assert_close(
        second_out,
        result["unsplit_out"],
        rtol=0.0,
        atol=0.001953125,
    )
    torch.testing.assert_close(
        second_lse, result["unsplit_lse"], rtol=0.0, atol=0.001
    )
