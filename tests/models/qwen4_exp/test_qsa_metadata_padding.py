# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA metadata bounds when query offsets include graph-padding requests."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAMetadataBuilder
from vllm.triton_utils import HAS_TRITON
from vllm.utils import torch_utils
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MLAAttentionSpec

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("num_actual_tokens", [1, 4], ids=["decode", "verify"])
@pytest.mark.parametrize("padding", [0, 3], ids=["unpadded", "graph-padded"])
@pytest.mark.parametrize("cache_kind", ["plain", "compressed", "circular"])
def test_qsa_metadata_query_offset_bounds(
    backend: str,
    num_actual_tokens: int,
    padding: int,
    cache_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if backend == "triton" and (not HAS_TRITON or not torch.cuda.is_available()):
        pytest.skip("Triton metadata requires CUDA")
    device = torch.device("cuda" if backend == "triton" else "cpu")
    if backend == "torch":
        # Keep the real mapping helper, but avoid CUDA-pinned allocation in
        # this CPU-only reference path.
        monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    # One real request, followed by graph-padding requests of one token each.
    query_start_loc_cpu = torch.tensor(
        [0, *range(num_actual_tokens, num_actual_tokens + padding + 1)],
        dtype=torch.int32,
    )
    common = CommonAttentionMetadata(
        num_actual_tokens=num_actual_tokens,
        num_reqs=1 + padding,
        max_query_len=num_actual_tokens,
        max_seq_len=7 + num_actual_tokens,
        query_start_loc=query_start_loc_cpu.to(device),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=torch.tensor(
            [7 + num_actual_tokens] + [0] * padding,
            dtype=torch.int32,
            device=device,
        ),
        # Deliberately allocate only real slots, including for the Triton read.
        slot_mapping=torch.arange(
            100, 100 + num_actual_tokens, dtype=torch.int64, device=device
        ),
        block_table_tensor=torch.tensor(
            [[5, 6]] + [[0, 0]] * padding, dtype=torch.int32, device=device
        ),
    )
    sentinel = -12345
    # The real CommonAttentionMetadata mapping helper needs backing capacity
    # for every query offset, as in the scheduler's preallocated mapping buffer.
    mapping_capacity = num_actual_tokens + padding
    token_buffer = torch.full(
        (mapping_capacity + 1,), sentinel, dtype=torch.int32, device=device
    )
    position_buffer = torch.full(
        (num_actual_tokens + 1,), sentinel, dtype=torch.int64, device=device
    )
    slot_buffer = torch.full_like(position_buffer, sentinel)
    builder = (
        qsa_cache._build_qsa_metadata_torch
        if backend == "torch"
        else qsa_cache.build_qsa_metadata_triton
    )

    def build_metadata():
        return builder(
            common,
            token_buffer[:mapping_capacity],
            position_buffer[:num_actual_tokens],
            slot_buffer[:num_actual_tokens],
            storage_block_size=4,
            compress_ratio=2 if cache_kind == "compressed" else 1,
            circular_buffer_size=4 if cache_kind == "circular" else 0,
        )

    token_to_req, positions, slots = build_metadata()

    assert token_to_req.tolist() == [0] * num_actual_tokens
    assert positions.tolist() == list(range(7, 7 + num_actual_tokens))
    expected_slots = {
        "plain": [100, 101, 102, 103],
        "compressed": [23, -1, 24, -1],
        "circular": [23, 20, 21, 22],
    }
    assert slots.tolist() == expected_slots[cache_kind][:num_actual_tokens]
    assert token_buffer[-1].item() == sentinel
    assert position_buffer[-1].item() == sentinel
    assert slot_buffer[-1].item() == sentinel

    if backend == "triton":
        eager = tuple(tensor.clone() for tensor in (token_to_req, positions, slots))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = build_metadata()
        for _ in range(3):
            graph.replay()
            for actual, expected in zip(replayed, eager):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        # Replay must consume updated request data, not capture-time positions.
        common.seq_lens[0].add_(4)
        graph.replay()
        assert replayed[0].tolist() == [0] * num_actual_tokens
        assert replayed[1].tolist() == list(range(11, 11 + num_actual_tokens))
        updated_slots = (
            [25, -1, 26, -1]
            if cache_kind == "compressed"
            else expected_slots[cache_kind]
        )
        assert replayed[2].tolist() == updated_slots[:num_actual_tokens]
        assert token_buffer[-1].item() == sentinel
        assert position_buffer[-1].item() == sentinel
        assert slot_buffer[-1].item() == sentinel


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_dcp_replicated_selector_ignores_sharded_main_slot_mask(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if backend == "triton" and (not HAS_TRITON or not torch.cuda.is_available()):
        pytest.skip("Triton metadata requires CUDA")
    device = torch.device("cuda" if backend == "triton" else "cpu")
    if backend == "torch":
        monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    starts = torch.tensor([0, 4], dtype=torch.int32)
    common = CommonAttentionMetadata(
        num_actual_tokens=4,
        num_reqs=1,
        max_query_len=4,
        max_seq_len=11,
        query_start_loc=starts.to(device),
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([11], dtype=torch.int32, device=device),
        # All four target main K/V writes belong to the other DCP rank.
        slot_mapping=torch.full((4,), -1, dtype=torch.int64, device=device),
        block_table_tensor=torch.tensor([[5, 6]], dtype=torch.int32, device=device),
    )
    builder = (
        qsa_cache._build_qsa_metadata_torch
        if backend == "torch"
        else qsa_cache.build_qsa_metadata_triton
    )
    kwargs = dict(
        storage_block_size=4,
        compress_ratio=2,
    )

    def build(ignore_common_slot_mask: bool):
        return builder(
            common,
            torch.empty(4, dtype=torch.int32, device=device),
            torch.empty(4, dtype=torch.int64, device=device),
            torch.empty(4, dtype=torch.int64, device=device),
            ignore_common_slot_mask=ignore_common_slot_mask,
            **kwargs,
        )[2]

    assert build(False).tolist() == [-1, -1, -1, -1]
    assert build(True).tolist() == [23, -1, 24, -1]


def test_qsa_dummy_batch_suppresses_replicated_selector_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    monkeypatch.setattr(
        qsa_cache, "build_qsa_metadata", qsa_cache._build_qsa_metadata_torch
    )
    spec = MLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=2,
        dcp_sharded=False,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4, max_num_seqs=1),
    )
    builder = qsa_cache.QSAMetadataBuilder(
        spec,
        ["model.layers.3.self_attn.indexer.compressed_key_cache"],
        config,
        torch.device("cpu"),
        block_table_width=2,
    )
    starts = torch.tensor([0, 4], dtype=torch.int32)
    common = CommonAttentionMetadata(
        num_actual_tokens=4,
        num_reqs=1,
        max_query_len=4,
        max_seq_len=11,
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([11], dtype=torch.int32),
        slot_mapping=torch.arange(4, dtype=torch.int64),
        block_table_tensor=torch.tensor([[5, 6]], dtype=torch.int32),
    )
    normal_slots = builder.build(0, common).slot_mapping.clone()
    dummy = builder.build(0, replace(common, is_dummy_batch=True))
    assert normal_slots.tolist() == [23, -1, 24, -1]
    assert dummy.slot_mapping.tolist() == [-1, -1, -1, -1]


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_dcp_replicated_draft_main_uses_full_page_not_sharded_slot_map(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if backend == "triton" and (not HAS_TRITON or not torch.cuda.is_available()):
        pytest.skip("Triton metadata requires CUDA")
    device = torch.device("cuda" if backend == "triton" else "cpu")
    if backend == "torch":
        monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    starts = torch.tensor([0, 4], dtype=torch.int32)
    common = CommonAttentionMetadata(
        num_actual_tokens=4,
        num_reqs=1,
        max_query_len=4,
        max_seq_len=10,
        query_start_loc=starts.to(device),
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([10], dtype=torch.int32, device=device),
        slot_mapping=torch.full((4,), -1, dtype=torch.int64, device=device),
        block_table_tensor=torch.tensor([[5, 6]], dtype=torch.int32, device=device),
    )
    builder = (
        qsa_cache._build_qsa_metadata_torch
        if backend == "torch"
        else qsa_cache.build_qsa_metadata_triton
    )
    _, positions, slots = builder(
        common,
        torch.empty(4, dtype=torch.int32, device=device),
        torch.empty(4, dtype=torch.int64, device=device),
        torch.empty(4, dtype=torch.int64, device=device),
        storage_block_size=8,
        compress_ratio=1,
        map_plain_slot=True,
    )
    assert positions.tolist() == [6, 7, 8, 9]
    assert slots.tolist() == [46, 47, 48, 49]


@pytest.mark.parametrize(
    ("kernel_block_size", "block_table"),
    [(16, [5, 6]), (8, [5, 6]), (2, list(range(20, 29)))],
)
def test_dcp_draft_main_builder_suppresses_dummy_writes(
    monkeypatch: pytest.MonkeyPatch,
    kernel_block_size: int,
    block_table: list[int],
) -> None:
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.qsa.build_qsa_metadata",
        qsa_cache._build_qsa_metadata_torch,
    )
    monkeypatch.setattr(
        FlashAttentionMetadataBuilder,
        "build",
        lambda self, *args: SimpleNamespace(slot_mapping=None),
    )
    builder = object.__new__(Qwen4ExpQSAMetadataBuilder)
    builder.replicated_draft = True
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=8),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    builder.layer_names = ["mtp.layers.48.self_attn.attn"]
    builder.block_size = kernel_block_size
    builder.draft_token_to_req = torch.empty(4, dtype=torch.int32)
    builder.draft_logical_positions = torch.empty(4, dtype=torch.int64)
    builder.draft_slot_mapping = torch.empty(4, dtype=torch.int64)
    builder.draft_block_table_buffer = None
    starts = torch.tensor([0, 4], dtype=torch.int32)
    common = CommonAttentionMetadata(
        num_actual_tokens=4,
        num_reqs=1,
        max_query_len=4,
        max_seq_len=18,
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([18], dtype=torch.int32),
        slot_mapping=torch.full((4,), -1, dtype=torch.int64),
        block_table_tensor=torch.tensor([block_table], dtype=torch.int32),
    )
    assert builder.build(0, common).slot_mapping.tolist() == [94, 95, 96, 97]
    if kernel_block_size in (8, 2):
        assert builder.draft_block_table_buffer[0].tolist() == [
            virtual_block
            for group_block in block_table
            for virtual_block in (group_block * 2, group_block * 2 + 1)
        ]
    assert builder.build(
        0, replace(common, is_dummy_batch=True)
    ).slot_mapping.tolist() == [
        -1,
        -1,
        -1,
        -1,
    ]
    builder.block_size = 6
    with pytest.raises(RuntimeError, match="kernel block must divide"):
        builder.build(0, common)


def test_qsa_canonical_block_table_accepts_partial_virtual_page() -> None:
    builder = object.__new__(qsa_cache.QSAMetadataBuilder)
    builder.block_table_buffer = torch.empty((1, 2), dtype=torch.int32)
    builder.kv_cache_spec = SimpleNamespace(block_size=32, dcp_sharded=False)
    builder.kernel_block_size = 16
    builder.has_sharded_main_owner = False
    builder.dcp_world_size = 1
    table = torch.tensor([[14, 15, 16]], dtype=torch.int32)
    canonical = builder._canonical_block_table(table)
    assert canonical.tolist() == [[7, 8]]


def test_qsa_canonical_block_table_keeps_physical_pages() -> None:
    builder = object.__new__(qsa_cache.QSAMetadataBuilder)
    builder.block_table_buffer = torch.empty((1, 3), dtype=torch.int32)
    builder.kv_cache_spec = SimpleNamespace(block_size=32, dcp_sharded=False)
    builder.kernel_block_size = 16
    builder.has_sharded_main_owner = False
    builder.dcp_world_size = 1
    table = torch.tensor([[5, 6, 7]], dtype=torch.int32)
    assert builder._canonical_block_table(table) is table


def _dcp2_selector_builder(buffer_width: int) -> qsa_cache.QSAMetadataBuilder:
    """A target QSA selector builder with the measured TP4/DCP2 geometry.

    ``qsa_dcp_block_geometry`` gives the replicated selector a page covering
    ``1568 * 2 = 3136`` global tokens, while the shared common block table
    enumerates the 32-token kernel blocks this rank owns. Measured on the
    real model at layer 3 (2026-09-25): ``in_width=539``, ``buf_width=12``,
    ``spec_block=3136``, ``kernel_block=32``, ``storage_block=784``.
    """

    builder = object.__new__(qsa_cache.QSAMetadataBuilder)
    builder.block_table_buffer = torch.empty((1, buffer_width), dtype=torch.int32)
    builder.kv_cache_spec = SimpleNamespace(block_size=3136, dcp_sharded=False)
    builder.kernel_block_size = 32
    builder.has_sharded_main_owner = True
    builder.dcp_world_size = 2
    return builder


def test_qsa_replicated_side_table_uses_local_kernel_block_units() -> None:
    """A replicated selector still reads its sharded owner's local table.

    The selector page is declared as a global span, so the virtual expansion
    must be derived from the local span ``3136 // 2 = 1568``, i.e. 49 kernel
    blocks, not from the global 3136 (98 blocks). Using the global span
    halved the column count and divided every page ID by twice the correct
    factor, which mapped the whole table onto page 0.
    """

    builder = _dcp2_selector_builder(buffer_width=2)
    table = torch.arange(49, 49 + 98, dtype=torch.int32).unsqueeze(0)
    assert builder._canonical_block_table(table).tolist() == [[1, 2]]


def test_qsa_dcp2_selector_table_addresses_full_long_context() -> None:
    """Regression for the 30K-token selector slot-mapping failure.

    With the global span the 539-entry table collapsed to 6 columns, so the
    QSA slot kernel found no column for compressed group index 6 and wrote
    PAD_SLOT_ID for every group boundary in the chunk: the selector cache
    never received the current keys. A 30,030-token prompt needs
    ``ceil(ceil(30030 / 4) / 784) = 10`` columns.
    """

    builder = _dcp2_selector_builder(buffer_width=12)
    table = torch.arange(49, 49 + 539, dtype=torch.int32).unsqueeze(0)
    canonical = builder._canonical_block_table(table)
    # The slot kernel indexes with
    # ``(logical_position // compress_ratio) // storage_block_size``, so the
    # last token of a 30,030-token prompt needs column index 9.
    last_column_index = ((30030 - 1) // 4) // 784
    required_columns = last_column_index + 1
    assert required_columns == 10
    assert canonical.shape[1] == 11
    assert canonical.shape[1] >= required_columns
    # The first source entry is kernel block 49, i.e. selector page 1, not the
    # degenerate page 0 the doubled expansion produced.
    assert canonical[0, 0].item() == 1


def test_qsa_replicated_draft_side_table_keeps_global_page_geometry() -> None:
    """A standalone draft replicates every QSA cache, so its table is global.

    ``has_sharded_main_owner`` is false there, so no DCP division applies and
    the expansion stays ``3136 // 32 = 98``.
    """

    builder = _dcp2_selector_builder(buffer_width=2)
    builder.has_sharded_main_owner = False
    table = torch.arange(98, 98 + 196, dtype=torch.int32).unsqueeze(0)
    assert builder._canonical_block_table(table).tolist() == [[1, 2]]


def test_qsa_dcp_local_span_must_divide_kernel_block() -> None:
    """An indivisible local span is a real geometry error, not something to
    silently round: masking it would read the wrong compressed page."""

    builder = _dcp2_selector_builder(buffer_width=4)
    builder.kv_cache_spec = SimpleNamespace(block_size=3200, dcp_sharded=False)
    builder.kernel_block_size = 128
    table = torch.arange(675, 700, dtype=torch.int32).unsqueeze(0)
    with pytest.raises(RuntimeError, match="must be divisible"):
        builder._canonical_block_table(table)
