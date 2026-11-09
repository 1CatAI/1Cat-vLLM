# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused QSA pre-indexer clamps RoPE positions read from the ring tail.

The int64 MRoPE coordinates stored after each raw key in the compressor ring
are the only cache content that becomes an address (a cos/sin row). A stale or
foreign page there used to index out of bounds; now it selects a clamped row.
"""

import pytest
import torch

from vllm.models.qwen4_exp.common.qsa_cache import (
    circular_qsa_slot_mapping,
    compressed_qsa_slot_mapping,
)
from vllm.models.qwen4_exp.nvidia.ops import qsa_pre_indexer as pre_indexer_mod
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)

HQ, D = 4, 128
CR = 4
MROPE_SECTION = (11, 11, 10)
EPS = 1e-6
STATE_SIZE = 4
COMP_PAGE = 16 // CR
ROPE_POS_OFFSET = D
MAX_POSITION = 32768
# Requests whose first compressed group starts in the ring (history), so the
# kernel reads cached positions for it.
SEQ_LENS, QUERY_LENS, HISTORY_LENS = [260, 259, 138], [1, 1, 37], [8, 8, 8]


def _junk_positions(case: dict) -> tuple[int, int, int]:
    """(t, h, w) a foreign page can hold: each axis is out of range."""
    return (2**40, -(2**40), case["rope"].cos_sin_cache.shape[0])


def _clamped_positions(case: dict) -> tuple[int, int, int]:
    rows = case["rope"].cos_sin_cache.shape[0]
    return tuple(min(max(p, 0), rows - 1) for p in _junk_positions(case))


def _build_case(mrope: bool, seed: int = 0) -> dict:
    """Inputs for one pre-indexer launch with history in the ring."""
    from vllm.model_executor.layers.rotary_embedding import get_rope

    torch.manual_seed(seed)
    device = "cuda"
    rope_params = {
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    }
    if mrope:
        rope_params["mrope_interleaved"] = True
        rope_params["mrope_section"] = list(MROPE_SECTION)
    with torch.device(device):
        rope = get_rope(
            head_size=256,
            max_position=MAX_POSITION,
            rope_parameters=rope_params,
            dtype=torch.bfloat16,
        )

    token_to_req = torch.cat(
        [
            torch.full((length,), request, dtype=torch.int32)
            for request, length in enumerate(QUERY_LENS)
        ]
    ).to(device)
    logical_positions = torch.cat(
        [
            torch.arange(seq_len - query_len, seq_len, dtype=torch.int64)
            for seq_len, query_len in zip(SEQ_LENS, QUERY_LENS)
        ]
    ).to(device)
    num_tokens = logical_positions.numel()
    query_start_loc = torch.tensor(
        [0, *torch.tensor(QUERY_LENS).cumsum(0).tolist()], dtype=torch.int32
    ).to(device)
    positions = (
        torch.stack(
            [logical_positions, logical_positions // 7 + 3, logical_positions // 13]
        )
        if mrope
        else logical_positions
    )

    num_requests = len(SEQ_LENS)
    raw_block_table = torch.randperm(num_requests, dtype=torch.int32)[:, None]
    raw_block_table = raw_block_table.to(device)
    compressed_counts = [
        (seq_len // CR + COMP_PAGE - 1) // COMP_PAGE for seq_len in SEQ_LENS
    ]
    num_compressed_blocks = sum(compressed_counts)
    compressed_block_table = torch.full(
        (num_requests, max(compressed_counts)), -1, dtype=torch.int32
    )
    physical = torch.randperm(num_compressed_blocks, dtype=torch.int32)
    offset = 0
    for request, count in enumerate(compressed_counts):
        compressed_block_table[request, :count] = physical[offset : offset + count]
        offset += count
    raw_slots = circular_qsa_slot_mapping(
        raw_block_table, token_to_req, logical_positions, STATE_SIZE, query_start_loc
    )
    compressed_slots = compressed_qsa_slot_mapping(
        compressed_block_table.to(device),
        token_to_req,
        logical_positions,
        COMP_PAGE,
        CR,
    )

    group_counts = torch.tensor(
        [
            seq_len // CR - (seq_len - query_len) // CR
            for seq_len, query_len in zip(SEQ_LENS, QUERY_LENS)
        ],
        dtype=torch.int32,
        device=device,
    )
    k_work_counts = torch.maximum(group_counts, torch.ones_like(group_counts))
    k_start_loc = torch.cat([k_work_counts.new_zeros(1), k_work_counts.cumsum(0)])
    work_requests = torch.repeat_interleave(
        torch.arange(num_requests, dtype=torch.int32, device=device), k_work_counts
    )
    local_work = torch.arange(
        int(k_start_loc[-1]), dtype=torch.int32, device=device
    ) - torch.repeat_interleave(k_start_loc[:-1], k_work_counts)
    max_k_work = (num_tokens + (CR - 1) * num_requests) // CR
    k_work_metadata = torch.full((max_k_work, 2), -1, dtype=torch.int32, device=device)
    k_work_metadata[: work_requests.numel()] = torch.stack(
        (work_requests, local_work), dim=1
    )

    raw_width = D + 12
    raw_page_elements = STATE_SIZE * raw_width
    raw_storage = torch.zeros(
        num_requests, raw_page_elements + 16, dtype=torch.bfloat16, device=device
    )
    raw = torch.as_strided(
        raw_storage,
        (num_requests, STATE_SIZE, 1, raw_width),
        (raw_page_elements + 16, raw_width, raw_width, 1),
    )
    history_rows = []
    for request, history_len in enumerate(HISTORY_LENS):
        history_end = SEQ_LENS[request] - QUERY_LENS[request]
        block = int(raw_block_table[request, 0])
        for position in range(history_end - history_len, history_end):
            row = raw[block, position % STATE_SIZE, 0]
            row[:D] = torch.randn(D, dtype=torch.bfloat16, device=device)
            row[ROPE_POS_OFFSET:].view(torch.int64).copy_(
                torch.tensor(
                    [position, position // 7 + 3, position // 13],
                    dtype=torch.int64,
                    device=device,
                )
            )
            history_rows.append((block, position % STATE_SIZE))

    compressed_page_elements = COMP_PAGE * D
    compressed_storage = torch.zeros(
        num_compressed_blocks,
        compressed_page_elements + 16,
        dtype=torch.bfloat16,
        device=device,
    )
    compressed = torch.as_strided(
        compressed_storage,
        (num_compressed_blocks, COMP_PAGE, 1, D),
        (compressed_page_elements + 16, D, D, 1),
    )
    return {
        "rope": rope,
        "mrope": mrope,
        "positions": positions,
        "query_start_loc": query_start_loc,
        "logical_positions": logical_positions,
        "raw_block_table": raw_block_table,
        "raw_slots": raw_slots,
        "compressed_slots": compressed_slots,
        "k_work_metadata": k_work_metadata,
        "raw": raw,
        "history_rows": history_rows,
        "compressed": compressed,
        "projected_qk": torch.randn(
            num_tokens, (HQ + 1) * D, dtype=torch.bfloat16, device=device
        ),
        "q_weight": torch.randn(D, dtype=torch.bfloat16, device=device) * 0.2,
        "k_weight": torch.randn(D, dtype=torch.bfloat16, device=device) * 0.2,
    }


def _set_history_positions(case: dict, raw: torch.Tensor, values) -> None:
    tail = raw[..., ROPE_POS_OFFSET:].view(torch.int64)
    for block, row in case["history_rows"]:
        tail[block, row, 0, :3] = torch.tensor(values, dtype=torch.int64)


def _run(case: dict, tail_positions=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch on copies of the caches; optionally overwrite the cached
    positions of every history row first. Returns (query, compressed)."""
    raw = case["raw"].clone()
    if tail_positions is not None:
        _set_history_positions(case, raw, tail_positions)
    compressed = case["compressed"].clone()
    projected_qk = case["projected_qk"]
    num_tokens = projected_qk.shape[0]
    query = torch.empty(num_tokens, HQ, D, dtype=torch.bfloat16, device="cuda")
    pre_indexer_mod.qsa_pre_indexer(
        projected_qk[:, : HQ * D],
        projected_qk[:, HQ * D :],
        case["positions"],
        case["rope"].cos_sin_cache,
        case["q_weight"],
        case["k_weight"],
        EPS,
        query,
        raw,
        case["raw_slots"],
        case["raw_block_table"],
        case["query_start_loc"],
        case["logical_positions"],
        compressed,
        case["compressed_slots"],
        case["k_work_metadata"],
        compress_ratio=CR,
        mrope_section=MROPE_SECTION if case["mrope"] else None,
        rope_pos_offset=ROPE_POS_OFFSET,
    )
    torch.accelerator.synchronize()
    return query, compressed


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16)


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("mrope", [True, False], ids=["mrope", "text"])
def test_out_of_range_cached_positions_are_clamped(mrope: bool) -> None:
    case = _build_case(mrope)
    valid_query, valid_compressed = _run(case)

    junk_query, junk_compressed = _run(case, _junk_positions(case))
    clamped_query, clamped_compressed = _run(case, _clamped_positions(case))

    # No fault, and junk behaves exactly like the clamped rows.
    assert torch.equal(_bits(junk_query), _bits(clamped_query))
    assert torch.equal(_bits(junk_compressed), _bits(clamped_compressed))
    assert torch.isfinite(junk_compressed.float()).all()
    # Queries never read the ring; the compressed groups that start in the
    # ring did consume the (clamped) junk.
    assert torch.equal(_bits(junk_query), _bits(valid_query))
    assert not torch.equal(_bits(junk_compressed), _bits(valid_compressed))
