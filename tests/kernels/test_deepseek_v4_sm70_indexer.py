# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.sm70_sparse import sparse_policy
from vllm.models.deepseek_v4.sm70 import indexer as sm70_indexer
from vllm.models.deepseek_v4.sm70.indexer import (
    sm70_indexer_decode_logits,
    sm70_indexer_prefill_logits,
)
from vllm.utils.torch_utils import current_stream


@pytest.fixture(autouse=True)
def indexer_engine_config():
    with set_current_vllm_config(VllmConfig()):
        yield


requires_sm70 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7,
    reason="requires a Volta or Turing GPU (SM70/SM75)",
)

_HEAD_DIM = 128
_BLOCK_SIZE = 64

# E4M3FN bit patterns with exact values. Building keys from a palette keeps the
# reference bit-exact and needs no native FP8 conversion, which SM70 lacks.
_FP8_BITS = torch.tensor([0x00, 0x30, 0x38, 0x3C, 0x40, 0xB0, 0xB8, 0xC0])
_FP8_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -2.0])


def _random_fp8_keys(num_keys: int, generator: torch.Generator):
    """Return (uint8 bit patterns, exact fp32 values) for `num_keys` keys."""
    picks = torch.randint(len(_FP8_BITS), (num_keys, _HEAD_DIM), generator=generator)
    return _FP8_BITS[picks].to(torch.uint8).cuda(), _FP8_VALUES[picks].cuda()


def _reference_index_logits(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """I[t, s] = sum_h weights[t, h] * relu(q[t, h] . k[s]).

    The model's actual indexer scoring function (reference `Indexer.forward`,
    inference/model.py:427; the in-repo `fp8_mqa_logits` reference at
    v1/attention/ops/rocm_aiter_mla_sparse.py:510 applies the same relu). It
    does NOT factor to (sum_h weights[t, h] * q[t, h]) . k[s] - the relu sits
    between the weighting and the head sum.
    """
    per_head = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    return torch.einsum("mhn,mh->mn", torch.relu(per_head), weights.float())


def _reference_factored_logits(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """The relu-free form, kept reachable via VLLM_SM70_INDEXER_RELU=0."""
    weighted_q = torch.einsum("thd,th->td", q.float(), weights.float())
    return weighted_q @ k.float().T


def _build_paged_index_cache(
    value_bits: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Pack keys into the block-major layout the C++ cache kernels write.

    `indexer_k_quant_and_cache` / `cp_gather_indexer_k_quant_cache`
    (csrc/libtorch_stable/cache_kernels.cu) store, per block, all `block_size`
    rows of 128 FP8 values first and only then that block's FP32 scales:

        value: block_base + pos_in_block * 128 + d
        scale: block_base + block_size * 128 + pos_in_block * 4

    It is NOT a per-token [128 values][4 scale] record, which is why only
    token 0 of each block used to read back correctly.
    """
    num_blocks, block_size, head_dim = value_bits.shape
    assert scales.shape == (num_blocks, block_size)
    value_bytes = value_bits.reshape(num_blocks, block_size * head_dim)
    scale_bytes = scales.contiguous().view(torch.uint8)
    return (
        torch.cat([value_bytes, scale_bytes], dim=1)
        .reshape(num_blocks, block_size, head_dim + 4)
        .contiguous()
    )


def _make_queries(rows: int, num_heads: int):
    q = (
        torch.randn((rows, num_heads, _HEAD_DIM), device="cuda", dtype=torch.float16)
        * 0.25
    )
    weights = torch.randn((rows, num_heads), device="cuda", dtype=torch.float32)
    return q, weights


@requires_sm70
@pytest.mark.parametrize("use_cublas", [True, False])
def test_prefill_matches_the_reference_scoring_function(monkeypatch, use_cublas):
    torch.manual_seed(20260819)
    generator = torch.Generator().manual_seed(20260819)
    num_queries, num_heads, num_keys = 5, 8, 37
    monkeypatch.setattr(sparse_policy(), "indexer_prefill_cublas", use_cublas)

    q, weights = _make_queries(num_queries, num_heads)
    bits, values = _random_fp8_keys(num_keys, generator)
    scales = torch.linspace(0.5, 2.0, num_keys, device="cuda", dtype=torch.float32)
    k = values * scales[:, None]

    actual = sm70_indexer_prefill_logits(
        q, bits.view(torch.float8_e4m3fn), scales, weights
    )
    expected = _reference_index_logits(q, weights, k)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    # The relu is not a rounding detail: without it the scoring is a different
    # function, which is what broke long-context top-k selection.
    factored = _reference_factored_logits(q, weights, k)
    assert not torch.allclose(expected, factored, rtol=1e-2, atol=1e-2)


@requires_sm70
def test_prefill_relu_off_restores_the_factored_form(monkeypatch):
    """VLLM_SM70_INDEXER_RELU=0 keeps the old scoring reachable for A/B."""
    torch.manual_seed(20260819)
    generator = torch.Generator().manual_seed(20260819)
    num_queries, num_heads, num_keys = 4, 8, 23
    monkeypatch.setattr(sparse_policy(), "indexer_relu", False)

    q, weights = _make_queries(num_queries, num_heads)
    bits, values = _random_fp8_keys(num_keys, generator)
    scales = torch.linspace(0.5, 2.0, num_keys, device="cuda", dtype=torch.float32)

    actual = sm70_indexer_prefill_logits(
        q, bits.view(torch.float8_e4m3fn), scales, weights
    )
    expected = _reference_factored_logits(q, weights, values * scales[:, None])
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@requires_sm70
@pytest.mark.parametrize(
    "relu,fused",
    [
        pytest.param(True, True, id="relu"),
        pytest.param(False, True, id="factored-fused"),
        pytest.param(False, False, id="factored-gather"),
    ],
)
def test_decode_reads_the_block_major_paged_cache(monkeypatch, relu, fused):
    """Every decode path must address values and scales the way the cache is
    written. Reading a token as a [128 values][4 scale] record lands on the
    wrong bytes for every token but the first of each block, and reinterprets
    four FP8 value bytes as the FP32 scale.
    """
    torch.manual_seed(20260819)
    generator = torch.Generator().manual_seed(20260819)
    monkeypatch.setattr(sparse_policy(), "indexer_relu", relu)
    monkeypatch.setattr(sparse_policy(), "indexer_fused_logits", fused)

    num_heads = 8
    seq_lens_list = [1, _BLOCK_SIZE + 1, 3 * _BLOCK_SIZE - 5]
    rows = len(seq_lens_list)
    max_seq_len = max(seq_lens_list)
    blocks_per_row = (max_seq_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    num_blocks = rows * blocks_per_row + 2  # +2 so no row can start at block 0

    bits, values = _random_fp8_keys(num_blocks * _BLOCK_SIZE, generator)
    value_bits = bits.reshape(num_blocks, _BLOCK_SIZE, _HEAD_DIM)
    block_values = values.reshape(num_blocks, _BLOCK_SIZE, _HEAD_DIM)
    # Distinct per-token scales inside every block: with one scale per block
    # the layout bug is invisible.
    scales = torch.rand((num_blocks, _BLOCK_SIZE), generator=generator).cuda() + 0.5
    cache = _build_paged_index_cache(value_bits, scales)

    # Shuffled, non-identity mapping so a wrong block stride cannot pass.
    perm = torch.randperm(num_blocks, generator=generator)[: rows * blocks_per_row]
    block_table = perm.to(torch.int32).cuda().reshape(rows, blocks_per_row)
    seq_lens = torch.tensor(seq_lens_list, device="cuda", dtype=torch.int32)
    q, weights = _make_queries(rows, num_heads)

    actual = sm70_indexer_decode_logits(
        q,
        cache,
        weights,
        seq_lens,
        block_table,
        max_seq_len,
        indexer_decode_cublas=True,
    )

    reference = _reference_index_logits if relu else _reference_factored_logits
    for row, seq_len in enumerate(seq_lens_list):
        positions = torch.arange(seq_len, device="cuda")
        physical = block_table[row][positions // _BLOCK_SIZE].long()
        pos_in_block = positions % _BLOCK_SIZE
        k = (
            block_values[physical, pos_in_block]
            * scales[physical, pos_in_block][:, None]
        )
        expected = reference(q[row : row + 1], weights[row : row + 1], k)
        torch.testing.assert_close(
            actual[row, :seq_len].unsqueeze(0), expected, rtol=1e-2, atol=1e-2
        )


# The serving cache pads its blocks (stride(0) exceeds a block's bytes), so it
# is not contiguous; the route has to take it, and read the right rows.
@pytest.mark.parametrize("padded_blocks", [False, True])
# native: one block-table row and [1, rows] lengths. flattened: what a uniform
# speculative decode with more than two verifier tokens arrives as, one
# block-table row and one length per token. two_requests: the same, twice.
@pytest.mark.parametrize("layout", ["native", "flattened", "two_requests"])
@pytest.mark.parametrize("num_rows", [3, 8, 13])
@requires_sm70
def test_decode_cublas_keeps_the_fp32_topk_set_and_masks_graph_tail(
    monkeypatch, padded_blocks, layout, num_rows
):
    torch.manual_seed(20260824)
    generator = torch.Generator().manual_seed(20260824)
    num_heads = 64
    live_seq_len = 1017
    graph_width = 2048
    blocks = (live_seq_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    bits, _ = _random_fp8_keys(blocks * _BLOCK_SIZE, generator)
    value_bits = bits.reshape(blocks, _BLOCK_SIZE, _HEAD_DIM)
    scales = torch.rand((blocks, _BLOCK_SIZE), generator=generator).cuda() + 0.5
    cache = _build_paged_index_cache(value_bits, scales)
    if padded_blocks:
        block_bytes = cache.shape[1] * cache.shape[2]
        storage = torch.zeros(
            (blocks, block_bytes + 40), dtype=torch.uint8, device="cuda"
        )
        storage[:, :block_bytes] = cache.reshape(blocks, block_bytes)
        cache = storage[:, :block_bytes].view(cache.shape)
        assert not cache.is_contiguous()
    block_table = torch.full(
        (1, graph_width // _BLOCK_SIZE),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    block_table[0, :blocks] = torch.randperm(blocks, generator=generator).to(
        device="cuda", dtype=torch.int32
    )
    seq_lens = torch.arange(
        live_seq_len - num_rows + 1,
        live_seq_len + 1,
        dtype=torch.int32,
        device="cuda",
    ).view(1, num_rows)
    table_rows_per_request = 1
    num_requests = 2 if layout == "two_requests" else 1
    if layout != "native":
        table_rows_per_request = num_rows
        second_table = block_table.clone()
        second_table[0, :blocks] = block_table[0, :blocks].flip(0)
        tables = [block_table, second_table][:num_requests]
        block_table = torch.cat(
            [table.repeat(num_rows, 1) for table in tables]
        ).contiguous()
        seq_lens = torch.cat(
            [seq_lens.view(num_rows, 1) - 300 * i for i in range(num_requests)]
        ).contiguous()
    q, weights = _make_queries(num_rows * num_requests, num_heads)

    monkeypatch.setattr(sparse_policy(), "indexer_decode_cublas_enabled", False)
    baseline = sm70_indexer_decode_logits(
        q,
        cache,
        weights,
        seq_lens,
        block_table,
        graph_width,
        table_rows_per_request,
        indexer_decode_cublas=True,
    )

    static_workspace = (
        torch.empty((graph_width, _HEAD_DIM), dtype=torch.float16, device="cuda"),
        torch.empty((graph_width,), dtype=torch.float32, device="cuda"),
        torch.empty(
            (num_rows * num_heads, graph_width),
            dtype=torch.float32,
            device="cuda",
        ),
    )

    workspace_requests = []

    class StaticWorkspace:
        @staticmethod
        def get_simultaneous(*specs):
            assert len(specs) == len(static_workspace)
            workspace_requests.append(specs)
            return static_workspace

    monkeypatch.setattr(sparse_policy(), "indexer_decode_cublas_enabled", True)
    monkeypatch.setattr(sparse_policy(), "indexer_decode_min_keys", 1)
    monkeypatch.setattr(
        sm70_indexer, "current_workspace_manager", lambda: StaticWorkspace()
    )

    def candidate_call():
        return sm70_indexer_decode_logits(
            q,
            cache,
            weights,
            seq_lens,
            block_table,
            graph_width,
            table_rows_per_request,
            indexer_decode_cublas=True,
        )

    capture_stream = torch.cuda.Stream()
    # vLLM's stream, so that leaving the capture contexts restores it instead
    # of recording torch's default stream as vLLM's current stream, which
    # breaks later graph captures in the same process.
    capture_stream.wait_stream(current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            candidate_call()
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        candidate = candidate_call()
    graph.replay()
    torch.accelerator.synchronize()
    # Only the cuBLAS route asks for a workspace: this card took it.
    assert workspace_requests

    for row, row_len in enumerate(seq_lens.reshape(-1).tolist()):
        expected_topk = torch.topk(baseline[row, :row_len], 512).indices.sort().values
        actual_topk = torch.topk(candidate[row, :row_len], 512).indices.sort().values
        assert torch.equal(actual_topk, expected_topk)

    # The graph bucket has no allocated blocks after the live request tail.
    # Reaching this assertion proves the gather masked those -1 entries.
    assert torch.isfinite(candidate[:, :live_seq_len]).all()

    # Replay the same graph with a shorter live request. This exercises the
    # device-derived bound instead of merely validating the capture-time row.
    seq_lens.sub_(127)
    monkeypatch.setattr(sparse_policy(), "indexer_decode_cublas_enabled", False)
    replay_baseline = sm70_indexer_decode_logits(
        q,
        cache,
        weights,
        seq_lens,
        block_table,
        graph_width,
        table_rows_per_request,
        indexer_decode_cublas=True,
    )
    graph.replay()
    torch.accelerator.synchronize()
    for row, row_len in enumerate(seq_lens.reshape(-1).tolist()):
        expected_topk = (
            torch.topk(replay_baseline[row, :row_len], 512).indices.sort().values
        )
        actual_topk = torch.topk(candidate[row, :row_len], 512).indices.sort().values
        assert torch.equal(actual_topk, expected_topk)


@requires_sm70
def test_sm70_indexer_decode_valid_logits_do_not_depend_on_workspace_width() -> None:
    torch.manual_seed(20260804)
    num_queries = 8
    num_heads = 4
    num_keys = 7
    head_dim = 128
    cache_block_size = 64

    q = torch.randn(
        (num_queries, num_heads, head_dim), device="cuda", dtype=torch.float16
    )
    weights = torch.randn((num_queries, num_heads), device="cuda", dtype=torch.float16)
    cache = torch.zeros(
        (1, cache_block_size, head_dim + 4), device="cuda", dtype=torch.uint8
    )
    cache[..., :head_dim].fill_(0x38)
    cache[..., head_dim:].view(torch.float32).fill_(1.0)
    seq_lens = torch.full((1, num_queries), num_keys, device="cuda", dtype=torch.int32)
    block_table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)

    narrow = sm70_indexer_decode_logits(
        q,
        cache,
        weights,
        seq_lens,
        block_table,
        max_seq_len=16,
        indexer_decode_cublas=True,
    )
    wide = sm70_indexer_decode_logits(
        q,
        cache,
        weights,
        seq_lens,
        block_table,
        max_seq_len=64,
        indexer_decode_cublas=True,
    )

    assert narrow.shape == (num_queries, 16)
    assert wide.shape == (num_queries, 64)
    torch.testing.assert_close(narrow[:, :num_keys], wide[:, :num_keys], rtol=0, atol=0)
