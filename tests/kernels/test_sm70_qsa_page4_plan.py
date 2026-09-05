# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation-invariance contract for the SM70 grouped QSA planner.

Keep logical selections, query grouping, and physical-page aliasing fixed.
Relocating KV pages must preserve masks, category padding, and the logical
reduction order. This is not a cross-batch-shape invariance contract.
"""

import pytest
import torch

WIDTH = 2051
OUTPUT_WIDTH = 4160


@pytest.fixture
def extension():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires a V100 / SM70 GPU")
    return pytest.importorskip("flash_attn_v100_cuda")


def make_case(kind="mixed", page_size=16, permute_selection=False):
    generator = torch.Generator().manual_seed(387466)
    if kind == "wide":
        requests = torch.arange(8, dtype=torch.int32)
        lengths = torch.full((8,), 8191, dtype=torch.int32)
        positions = lengths.to(torch.int64) - 1
    else:
        requests = torch.tensor([0, 0, 1, 1, 2, 3, 3, 3], dtype=torch.int32)
        lengths = torch.tensor([61, 57, 59, 63], dtype=torch.int32)
        positions = torch.tensor([16, 38, 19, 40, 36, 25, 32, 62])
    pages = (int(lengths.max()) + page_size - 1) // page_size
    table = torch.arange(len(lengths) * pages, dtype=torch.int32).view(-1, pages)
    if kind == "shared":
        table[:, 0] = table[0, 0]
    indices = torch.full((8, WIDTH), -1, dtype=torch.int32)
    for row, request in enumerate(requests.tolist()):
        visible = min(int(positions[row]) + 1, int(lengths[request]))
        blocks = torch.randperm(visible // 4, generator=generator)[:512].sort().values
        if permute_selection:
            blocks = blocks.flip(0)
        full = (blocks[:, None] * 4 + torch.arange(4)).flatten()
        indices[row, : len(full)] = full.to(torch.int32)
        tail = torch.arange(visible // 4 * 4, visible, dtype=torch.int32)
        indices[row, len(full) : len(full) + len(tail)] = tail
    if kind == "invalid":
        requests[1], requests[4] = -1, len(lengths)
        indices[6] = -1
    if kind == "empty":
        indices.fill_(-1)
    return indices, table, requests, positions, lengths


def relocate(case, layout):
    indices, table, requests, positions, lengths = case
    count = int(table.max()) + 1
    if layout == "collision":
        # PAGE16, interleaved K/V: stride=8, so 1024 cache pages collide
        # in the 8192-entry hash. No K/V allocation is needed for plan tests.
        mapping = torch.arange(count, dtype=torch.int32) * 2048 + 7
    elif layout == "shuffled":
        mapping = torch.randperm(count, generator=torch.Generator().manual_seed(466))
        mapping = mapping.to(torch.int32) * 3 + 11
    else:
        mapping = torch.arange(count, dtype=torch.int32)
    return (indices, mapping[table.long()], requests, positions, lengths), mapping


def category(mask):
    result = 0
    for query in range(8):
        if mask & (15 << (query * 4)):
            result |= 1 << (query * 6 // 16)
            result |= 1 << ((query * 6 + 5) // 16)
    return result


def reference(case, page_size, physical_stride):
    indices, table, requests, positions, lengths = case
    entries: dict[int, tuple[int, int]] = {}
    for query, request in enumerate(requests.tolist()):
        if not 0 <= request < len(lengths):
            continue
        visible = min(max(int(positions[query]) + 1, 0), int(lengths[request]))
        count = min(visible // 4, 512) * 4 + visible % 4
        for token in indices[query, :count].tolist():
            if not 0 <= token < visible:
                continue
            physical = int(table[request, token // page_size])
            physical = physical * physical_stride + token % page_size // 4
            owner = (query << 29) | (token // 4)
            old_mask, old_owner = entries.get(physical, (0, (1 << 32) - 1))
            entries[physical] = (
                old_mask | (1 << (query * 4 + token % 4)),
                min(old_owner, owner),
            )
    pages: list[int] = []
    masks: list[int] = []
    for group_category in range(1, 8):
        bucket = sorted(
            (owner, physical, mask)
            for physical, (mask, owner) in entries.items()
            if category(mask) == group_category
        )
        pages.extend(physical for _, physical, _ in bucket)
        masks.extend(mask for _, _, mask in bucket)
        padding = -len(bucket) % 8
        pages.extend([0] * padding)
        masks.extend([0] * padding)
    return pages, masks


def run_plan(extension, case, page_size, physical_stride, num_cache_blocks=None):
    device_case = tuple(t.cuda() for t in case)
    if num_cache_blocks is None:
        num_cache_blocks = int(case[1].max()) + 1
    pages = torch.full((1, OUTPUT_WIDTH), -17, dtype=torch.int32, device="cuda")
    masks = torch.zeros_like(pages, dtype=torch.uint32)
    lengths = torch.empty(1, dtype=torch.int32, device="cuda")

    def launch():
        extension.grouped_sparse_page4_plan_fwd(
            *device_case,
            pages,
            masks,
            lengths,
            page_size,
            physical_stride,
            num_cache_blocks,
        )

    launch()
    return pages, masks, lengths, launch, device_case


@pytest.mark.parametrize("kind", ["mixed", "shared", "invalid", "empty", "wide"])
@pytest.mark.parametrize("layout", ["compact", "shuffled", "collision"])
def test_plan_matches_logical_reference(extension, kind, layout):
    case, _ = relocate(make_case(kind), layout)
    expected_pages, expected_masks = reference(case, 16, 8)
    pages, masks, lengths, launch, _ = run_plan(extension, case, 16, 8)
    assert int(lengths[0]) == len(expected_pages) * 4
    count = len(expected_pages)
    assert pages[0, :count].cpu().tolist() == expected_pages
    assert masks[0, :count].cpu().tolist() == expected_masks
    first = (pages.clone(), masks.clone(), lengths.clone())
    for _ in range(3):
        launch()
        for before, after in zip(first, (pages, masks, lengths)):
            assert torch.equal(before, after)


@pytest.mark.parametrize("page_size", [4, 16, 32])
def test_plan_selection_order_and_graph_relocation(extension, page_size):
    case = make_case("shared", page_size)
    changed_case = make_case("shared", page_size, permute_selection=True)
    changed_case, _ = relocate(changed_case, "shuffled")
    stride = page_size // 4
    pages, masks, lengths, launch, device_case = run_plan(
        extension, case, page_size, stride, int(changed_case[1].max()) + 1
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for source, target in zip(changed_case, device_case):
        target.copy_(source)
    expected_pages, expected_masks = reference(changed_case, page_size, stride)
    for _ in range(3):
        graph.replay()
        count = len(expected_pages)
        assert int(lengths[0]) == count * 4
        assert pages[0, :count].cpu().tolist() == expected_pages
        assert masks[0, :count].cpu().tolist() == expected_masks


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("kv_dtype", ["auto", "fp8_e4m3"])
def test_attention_is_bitwise_invariant_to_physical_relocation(
    extension, interleaved, kv_dtype
):
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    case = make_case("shared")
    generator = torch.Generator(device="cuda").manual_seed(387)
    count = int(case[1].max()) + 1
    kv = torch.randn(
        count, 2, 16, 1, 256, generator=generator, dtype=torch.float16, device="cuda"
    )
    if kv_dtype == "fp8_e4m3":
        kv = kv.to(torch.float8_e4m3fn).view(torch.uint8)
    query = torch.randn(
        8, 6, 256, generator=generator, dtype=torch.float16, device="cuda"
    )
    outputs = []
    for layout in ("compact", "shuffled"):
        remapped, mapping = relocate(case, layout)
        cache = torch.zeros(
            int(mapping.max()) + 1, 2, 16, 1, 256, dtype=kv.dtype, device="cuda"
        )
        cache[mapping.long().cuda()] = kv
        key, value = cache[:, 0], cache[:, 1]
        if not interleaved:
            key, value = key.contiguous(), value.contiguous()
        stride = key.stride(0) // (4 * 256)
        pages, masks, lengths, _, _ = run_plan(extension, remapped, 16, stride)
        physical_k, physical_v = qsa._qsa_xqa_page4_physical_kv(query, key, value)
        out = torch.empty_like(query)
        lse = torch.empty((8, 6), dtype=torch.float32, device="cuda")
        extension.grouped_sparse_page4_fwd(
            query,
            physical_k,
            physical_v,
            out,
            pages,
            masks,
            lengths,
            lse,
            256**-0.5,
            kv_dtype,
            0.125,
            0.25,
        )
        outputs.append(out.clone())
    assert torch.equal(*outputs)


@pytest.mark.parametrize("permute_selection", [False, True])
def test_single_row_page4_table_uses_logical_order(extension, permute_selection):
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    case, _ = relocate(make_case(permute_selection=permute_selection), "shuffled")
    indices, table, requests, positions, lengths = case
    pages, seq_lens = qsa._qsa_xqa_page4_block_table(
        *(tensor.cuda() for tensor in case), int(table.max()) + 1, 16, 8
    )
    for row, request in enumerate(requests.tolist()):
        visible = min(int(positions[row]) + 1, int(lengths[request]))
        tokens = sorted(indices[row, :visible:4].tolist())
        expected = [int(table[request, t // 16]) * 8 + t % 16 // 4 for t in tokens]
        assert pages[row, : len(expected)].cpu().tolist() == expected
        assert int(seq_lens[row]) == visible


@pytest.mark.parametrize("kv_dtype", ["auto", "fp8_e4m3"])
def test_mixed_grouped_and_xqa_tail_is_allocation_invariant(
    extension, monkeypatch, kv_dtype
):
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    monkeypatch.setattr(qsa, "_SM70_QSA_XQA_PAGE4", True)
    monkeypatch.setattr(qsa, "_SM70_QSA_XQA_PAGE4_MIN_ROWS", 64)
    monkeypatch.setattr(qsa, "_SM70_QSA_GROUPED_PAGE4", True)
    calls = []
    grouped = qsa._qsa_sparse_paged_attention_sm70_grouped_page4
    tail = qsa._qsa_sparse_paged_attention_sm70_xqa_page4_batch

    def record_grouped(query, *args):
        calls.append(("grouped", query.shape[0]))
        return grouped(query, *args)

    def record_tail(query, *args):
        calls.append(("tail", query.shape[0]))
        return tail(query, *args)

    monkeypatch.setattr(
        qsa, "_qsa_sparse_paged_attention_sm70_grouped_page4", record_grouped
    )
    monkeypatch.setattr(
        qsa, "_qsa_sparse_paged_attention_sm70_xqa_page4_batch", record_tail
    )
    case = make_case("shared")
    generator = torch.Generator(device="cuda").manual_seed(185)
    count = int(case[1].max()) + 1
    kv = torch.randn(
        count, 2, 16, 1, 256, generator=generator, dtype=torch.float16, device="cuda"
    )
    if kv_dtype == "fp8_e4m3":
        kv = kv.to(torch.float8_e4m3fn).view(torch.uint8)
    query = torch.randn(
        185, 6, 256, generator=generator, dtype=torch.float16, device="cuda"
    )
    outputs = []
    for layout in ("compact", "shuffled"):
        remapped, mapping = relocate(case, layout)
        indices, table, requests, positions, lengths = [t.cuda() for t in remapped]
        indices = indices.repeat(24, 1)[:185].contiguous()
        requests = requests.repeat(24)[:185].contiguous()
        positions = positions.repeat(24)[:185].contiguous()
        cache = torch.zeros(
            int(mapping.max()) + 1, 2, 16, 1, 256, dtype=kv.dtype, device="cuda"
        )
        cache[mapping.long().cuda()] = kv
        outputs.append(
            qsa.qsa_sparse_paged_attention(
                query,
                cache[:, 0],
                cache[:, 1],
                indices,
                table,
                requests,
                query_positions=positions,
                sequence_lengths=lengths,
                kv_cache_dtype=kv_dtype,
                k_scale=0.125,
                v_scale=0.25,
            ).clone()
        )
    assert torch.equal(*outputs)
    assert calls == [("grouped", 184), ("tail", 1)] * 2
