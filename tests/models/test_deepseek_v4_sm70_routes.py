# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.platforms.interface import DeviceCapability


def test_sm70_sparse_backend_contract():
    from vllm.models.deepseek_v4.sm70.sparse import (
        DeepseekV4SM70SparseBackend,
        DeepseekV4SM70SparseImpl,
    )

    assert DeepseekV4SM70SparseBackend.get_name() == "V4_SM70_TRITON_SPARSE"
    assert DeepseekV4SM70SparseBackend.supported_dtypes == [torch.float16]
    assert DeepseekV4SM70SparseBackend.supports_compute_capability(
        DeviceCapability(7, 0)
    )
    assert not DeepseekV4SM70SparseBackend.supports_compute_capability(
        DeviceCapability(7, 5)
    )
    assert not DeepseekV4SM70SparseBackend.supports_compute_capability(
        DeviceCapability(8, 0)
    )
    assert DeepseekV4SM70SparseImpl.PREFILL_CHUNK_SIZE == 8


def test_sm70_sparse_backend_uses_v4_packed_kv_layout():
    from vllm.models.deepseek_v4.sm70.sparse import DeepseekV4SM70SparseBackend

    assert DeepseekV4SM70SparseBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        cache_dtype_str="fp8_ds_mla",
    ) == (3, 256, 584)


def test_sm70_selects_triton_sparse_impl():
    from vllm.models.deepseek_v4 import attention
    from vllm.models.deepseek_v4.sm70.sparse import DeepseekV4SM70SparseImpl

    platform = MagicMock()
    platform.is_rocm.return_value = False
    platform.is_cuda.return_value = True
    platform.is_device_capability.side_effect = lambda capability: capability == (
        7,
        0,
    )
    with patch.object(attention, "current_platform", platform):
        assert attention._select_v4_sparse_impl() is DeepseekV4SM70SparseImpl


def test_sm70_sparse_qk_dsplit_uses_graph_workspace():
    from vllm.models.deepseek_v4.sm70 import sparse

    q = torch.empty((1, 8, 512), dtype=torch.float16)
    output = torch.empty_like(q)
    layer = MagicMock()
    layer.compress_ratio = 1
    layer.swa_cache_layer.kv_cache = torch.empty((1, 256, 584), dtype=torch.uint8)
    layer.scale = 512**-0.5
    layer.attn_sink = torch.zeros(8, dtype=torch.float32)

    metadata = MagicMock()
    metadata.num_decode_tokens = 1
    metadata.decode_swa_indices = torch.zeros((1, 1, 128), dtype=torch.int32)
    metadata.decode_swa_lens = torch.full((1,), 128, dtype=torch.int32)

    workspace_manager = MagicMock()
    workspace_manager.get_simultaneous.side_effect = lambda *specs: tuple(
        torch.empty(shape, dtype=dtype) for shape, dtype in specs
    )
    with (
        patch.object(sparse.envs, "VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_SWA", True),
        patch.object(sparse.envs, "VLLM_SM70_DSV4_SPARSE_MLA_QK_DSPLIT", True),
        patch.object(
            sparse, "current_workspace_manager", return_value=workspace_manager
        ),
        patch.object(
            sparse, "sm70_sparse_attention_paged_fp8_splitk_qk_dsplit"
        ) as qk_dsplit,
        patch.object(sparse, "sm70_sparse_attention_paged_fp8_splitk") as splitk,
    ):
        sparse.DeepseekV4SM70SparseImpl._forward_decode(
            layer=layer,
            q=q,
            compressed_cache=None,
            output=output,
            sparse_metadata=None,
            swa_metadata=metadata,
            swa_only=True,
        )

    splitk.assert_not_called()
    qk_dsplit.assert_called_once()
    kwargs = qk_dsplit.call_args.kwargs
    assert kwargs["partial_qk"].shape == (1, 8, 8, 8, 16)
    assert kwargs["partial_probs"].shape == (1, 8, 8, 16)


def test_sm75_does_not_select_sm70_impl():
    from vllm.models.deepseek_v4 import attention
    from vllm.models.deepseek_v4.nvidia.flashmla import (
        DeepseekV4FlashMLASparseImpl,
    )

    platform = MagicMock()
    platform.is_rocm.return_value = False
    platform.is_cuda.return_value = True
    platform.is_device_capability.return_value = False
    with patch.object(attention, "current_platform", platform):
        assert attention._select_v4_sparse_impl() is DeepseekV4FlashMLASparseImpl


def test_v4_metadata_reuses_token_to_request_mapping():
    from vllm.v1.attention import backend
    from vllm.v1.attention.backend import CommonAttentionMetadata

    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor([2, 3], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=3,
        max_seq_len=3,
        block_table_tensor=torch.empty((2, 0), dtype=torch.int32),
        slot_mapping=torch.empty(8, dtype=torch.int64),
    )
    first_buffer = torch.full((8,), -1, dtype=torch.int32)
    second_buffer = torch.full((8,), -1, dtype=torch.int32)

    with patch.object(
        backend,
        "np_to_pinned_tensor",
        side_effect=lambda array: torch.from_numpy(array),
    ):
        first = metadata.token_to_req_indices(first_buffer)
        second = metadata.token_to_req_indices(second_buffer)

    torch.testing.assert_close(
        first, torch.tensor([0, 0, 1, 1, 1, 0, 0, 0], dtype=torch.int32)
    )
    assert second.data_ptr() == first.data_ptr()
    torch.testing.assert_close(second_buffer, torch.full_like(second_buffer, -1))


def test_v4_prefill_chunk_plan_uses_actual_sequence_widths():
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.empty((4, 0), dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int64),
        block_size=256,
        num_prefills=4,
        prefill_seq_lens_cpu=torch.tensor([100, 200, 300, 400]),
        prefill_query_lens_cpu=torch.tensor([10, 10, 10, 10]),
        prefill_window_size=128,
        prefill_max_model_len=512,
        prefill_max_num_batched_tokens=64,
    )

    assert metadata.get_prefill_chunk_plan(4, 2) == [
        (0, 3, 75, 212),
        (3, 4, 100, 237),
    ]


def test_v4_c128_boundary_detection():
    from vllm.models.deepseek_v4.compressor import _get_c128_boundary
    from vllm.v1.attention.backend import CommonAttentionMetadata

    def make_metadata(starts: list[int]) -> CommonAttentionMetadata:
        query_start_loc = torch.arange(len(starts) + 1, dtype=torch.int32)
        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc,
            seq_lens=torch.tensor(starts, dtype=torch.int32) + 1,
            _num_computed_tokens_cpu=torch.tensor(starts, dtype=torch.int32),
            num_reqs=len(starts),
            num_actual_tokens=len(starts),
            max_query_len=1,
            max_seq_len=max(starts) + 1,
            block_table_tensor=torch.empty((len(starts), 0), dtype=torch.int32),
            slot_mapping=torch.empty(len(starts), dtype=torch.int64),
        )

    assert _get_c128_boundary(make_metadata([1, 50])) is False
    assert _get_c128_boundary(make_metadata([127, 10])) is True


def test_v4_c128_metadata_keeps_graph_stable_row_stride():
    from vllm.v1.attention.backends.mla import flashmla_sparse

    launch_args = None

    class FakeKernel:
        def __getitem__(self, grid):
            assert grid == (4,)

            def launch(*args, **kwargs):
                nonlocal launch_args
                launch_args = (args, kwargs)

            return launch

    with patch.object(
        flashmla_sparse, "_build_c128a_topk_metadata_kernel", FakeKernel()
    ):
        global_decode, decode_lens, prefill_local = (
            flashmla_sparse.build_c128a_topk_metadata(
                positions=torch.arange(4, dtype=torch.int64),
                compress_ratio=128,
                num_decode_tokens=2,
                token_to_req_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
                block_table=torch.zeros((2, 1), dtype=torch.int32),
                block_size=2,
                slot_mapping=torch.arange(4, dtype=torch.int64),
                global_decode_buffer=torch.empty((4, 8192), dtype=torch.int32),
                decode_lens_buffer=torch.empty(4, dtype=torch.int32),
                prefill_buffer=torch.empty((4, 8192), dtype=torch.int32),
                max_compressed_tokens=128,
                fixed_row_stride=True,
            )
        )

    assert global_decode.shape == (2, 128)
    assert global_decode.stride() == (8192, 1)
    assert decode_lens.shape == (2,)
    assert prefill_local.shape == (2, 128)
    assert prefill_local.stride() == (8192, 1)
    assert launch_args is not None
    args, _ = launch_args
    assert args[1] == 8192
    assert args[4] == 8192


def test_v4_c128_metadata_keeps_upstream_packed_layout_by_default():
    from vllm.v1.attention.backends.mla import flashmla_sparse

    class FakeKernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: None

    with patch.object(
        flashmla_sparse, "_build_c128a_topk_metadata_kernel", FakeKernel()
    ):
        global_decode, _, prefill_local = flashmla_sparse.build_c128a_topk_metadata(
            positions=torch.arange(4, dtype=torch.int64),
            compress_ratio=128,
            num_decode_tokens=2,
            token_to_req_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
            block_table=torch.zeros((2, 1), dtype=torch.int32),
            block_size=2,
            slot_mapping=torch.arange(4, dtype=torch.int64),
            global_decode_buffer=torch.empty((4, 8192), dtype=torch.int32),
            decode_lens_buffer=torch.empty(4, dtype=torch.int32),
            prefill_buffer=torch.empty((4, 8192), dtype=torch.int32),
            max_compressed_tokens=128,
        )

    assert global_decode.stride() == (128, 1)
    assert prefill_local.stride() == (128, 1)
