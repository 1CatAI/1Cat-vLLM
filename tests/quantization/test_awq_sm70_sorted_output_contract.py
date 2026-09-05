# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    _use_qwen38_chunked_w2,
)

pytestmark = pytest.mark.skip_global_cleanup


def _reference_unpermute(
    sorted_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
) -> torch.Tensor:
    num_tokens, top_k = topk_weights.shape
    output = torch.zeros((num_tokens, sorted_output.shape[1]), dtype=torch.float32)
    for token in range(num_tokens):
        for route in range(top_k):
            sorted_row = int(inv_permuted_idx[token, route])
            output[token] += (
                topk_weights[token, route] * sorted_output[sorted_row].float()
            )
    return output.to(sorted_output.dtype)


def _route_order_direct_reduce(
    route_output: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    num_tokens, top_k, hidden = route_output.shape
    output = torch.zeros((num_tokens, hidden), dtype=torch.float32)
    for route in range(top_k):
        output += topk_weights[:, route, None] * route_output[:, route].float()
    return output.to(route_output.dtype)


def _chunked_reference_unpermute(
    sorted_output: torch.Tensor,
    topk_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    expert_offsets: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    num_tokens, top_k = topk_weights.shape
    chunks = []
    for token_begin in range(0, num_tokens, chunk_tokens):
        current_tokens = min(chunk_tokens, num_tokens - token_begin)
        source_begin = token_begin * top_k
        source_end = source_begin + current_tokens * top_k
        chunk_rows: list[int] = []
        for expert in range(expert_offsets.numel() - 1):
            expert_begin = int(expert_offsets[expert])
            expert_end = int(expert_offsets[expert + 1])
            expert_sources = permuted_idx[expert_begin:expert_end]
            local_begin = int(torch.searchsorted(expert_sources, source_begin))
            local_end = int(torch.searchsorted(expert_sources, source_end))
            chunk_rows.extend(
                range(expert_begin + local_begin, expert_begin + local_end)
            )

        chunk_rows_tensor = torch.tensor(chunk_rows, dtype=torch.int64)
        chunk_sources = permuted_idx[chunk_rows_tensor] - source_begin
        chunk_inverse = torch.empty(current_tokens * top_k, dtype=torch.int64)
        chunk_inverse[chunk_sources] = torch.arange(current_tokens * top_k)
        chunks.append(
            _reference_unpermute(
                sorted_output[chunk_rows_tensor],
                topk_weights[token_begin : token_begin + current_tokens],
                chunk_inverse.view(current_tokens, top_k),
            )
        )
    return torch.cat(chunks)


@pytest.mark.parametrize(
    ("num_tokens", "top_k", "hidden"), [(1, 10, 64), (8, 10, 256), (16, 10, 64)]
)
def test_route_order_direct_reduce_is_bitwise_reference(
    num_tokens: int,
    top_k: int,
    hidden: int,
) -> None:
    generator = torch.Generator().manual_seed(num_tokens * 1000 + hidden)
    route_output = torch.randn(
        (num_tokens, top_k, hidden), generator=generator, dtype=torch.float16
    )
    topk_weights = torch.softmax(
        torch.randn((num_tokens, top_k), generator=generator), dim=-1
    )

    # Model the stable expert sort: permuted_idx maps each sorted row back to
    # the flattened source route, and inv_permuted_idx is its inverse.
    expert_ids = torch.randint(
        0, 512, (num_tokens, top_k), generator=generator, dtype=torch.int64
    )
    flat_experts = expert_ids.flatten()
    permuted_idx = torch.argsort(flat_experts, stable=True)
    inv_permuted_idx = torch.empty_like(permuted_idx)
    inv_permuted_idx[permuted_idx] = torch.arange(num_tokens * top_k)
    inv_permuted_idx = inv_permuted_idx.view(num_tokens, top_k)
    sorted_output = route_output.view(-1, hidden)[permuted_idx]

    reference = _reference_unpermute(sorted_output, topk_weights, inv_permuted_idx)
    direct = _route_order_direct_reduce(route_output, topk_weights)
    assert torch.equal(direct, reference)


def test_fp16_atomic_style_accumulation_is_not_bitwise_reference() -> None:
    generator = torch.Generator().manual_seed(0)
    route_output = (
        torch.randn((1, 10, 256), generator=generator, dtype=torch.float16) * 4
    )
    topk_weights = torch.softmax(torch.randn((1, 10), generator=generator), dim=-1)

    reference = _route_order_direct_reduce(route_output, topk_weights)
    atomic_style = torch.zeros_like(reference)
    for route in range(10):
        contribution = (
            topk_weights[:, route, None] * route_output[:, route].float()
        ).half()
        atomic_style = (atomic_style + contribution).half()

    assert not torch.equal(atomic_style, reference)
    assert int((atomic_style != reference).sum()) == 168
    assert float(
        (atomic_style.float() - reference.float()).abs().max()
    ) == pytest.approx(0.00390625)


@pytest.mark.parametrize("chunk_tokens", [1, 3, 8])
def test_token_chunks_preserve_stable_expert_order_and_final_output(
    chunk_tokens: int,
) -> None:
    num_tokens = 17
    top_k = 10
    num_experts = 32
    hidden = 64
    generator = torch.Generator().manual_seed(20260903 + chunk_tokens)
    route_output = torch.randn(
        (num_tokens, top_k, hidden), generator=generator, dtype=torch.float16
    )
    topk_weights = torch.softmax(
        torch.randn((num_tokens, top_k), generator=generator), dim=-1
    )
    expert_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        generator=generator,
        dtype=torch.int64,
    )
    flat_experts = expert_ids.flatten()
    permuted_idx = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[permuted_idx]
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    expert_offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    sorted_output = route_output.view(-1, hidden)[permuted_idx]

    full_inverse = torch.empty_like(permuted_idx)
    full_inverse[permuted_idx] = torch.arange(num_tokens * top_k)
    expected = _reference_unpermute(
        sorted_output,
        topk_weights,
        full_inverse.view(num_tokens, top_k),
    )
    actual = _chunked_reference_unpermute(
        sorted_output,
        topk_weights,
        permuted_idx,
        expert_offsets,
        chunk_tokens,
    )
    assert torch.equal(actual, expected)


def test_qwen38_sorted_output_memory_ledger() -> None:
    layers = 48
    hidden = 2560
    top_k = 10
    element_bytes = 2

    max_batched_tokens_bytes = 8192 * top_k * hidden * element_bytes
    persistent_cap8_bytes = layers * 8 * top_k * hidden * element_bytes
    persistent_cap32_bytes = layers * 32 * top_k * hidden * element_bytes

    def chunk_scratch_bytes(chunk_tokens: int) -> int:
        chunk_slots = chunk_tokens * top_k
        chunk_output = chunk_slots * hidden * element_bytes
        chunk_metadata = (513 + 512 + 512 + chunk_slots + chunk_slots) * 4
        return chunk_output + chunk_metadata

    assert max_batched_tokens_bytes == 400 * 1024**2
    assert chunk_scratch_bytes(4096) == 200 * 1024**2 + 333_828
    assert max_batched_tokens_bytes - chunk_scratch_bytes(4096) == 209_381_372
    assert chunk_scratch_bytes(6144) == 300 * 1024**2 + 497_668
    assert max_batched_tokens_bytes - chunk_scratch_bytes(6144) == 104_359_932
    assert persistent_cap8_bytes == pytest.approx(18.75 * 1024**2)
    assert persistent_cap32_bytes == 75 * 1024**2


@pytest.mark.parametrize(
    ("num_tokens", "indexed_w13", "expected"),
    [
        (6144, True, False),
        (6537, True, False),
        (6538, True, False),
        (8192, True, True),
        (8193, True, True),
        (12288, True, True),
        (12289, True, False),
        (16384, True, True),
        (8192, False, False),
    ],
)
def test_chunked_w2_admission_is_profitable_and_tail_safe(
    num_tokens: int,
    indexed_w13: bool,
    expected: bool,
) -> None:
    layer = SimpleNamespace(
        sm70_awq_qwen38_w2_chunk_tokens=6144,
        _awq_moe_buf_top_k=10,
        sm70_w2_n_dim=2560,
        sm70_num_experts=512,
    )
    assert _use_qwen38_chunked_w2(layer, num_tokens, indexed_w13) is expected


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not hasattr(torch.ops._C, "awq_moe_chunked_w2_sm70_out"),
    reason="requires the CUDA chunked W2 extension",
)
def test_chunked_w2_cuda_op_rejects_small_tail() -> None:
    device = torch.device("cuda")
    out = torch.empty(1, dtype=torch.float16, device=device)
    chunk_output = torch.empty_like(out)
    input = torch.empty_like(out)
    int_scratch = [torch.empty(1, dtype=torch.int32, device=device) for _ in range(7)]
    topk_weights = torch.empty(1, dtype=torch.float32, device=device)
    ptrs_w = torch.empty(1, dtype=torch.uint8, device=device)
    ptrs_s = torch.empty_like(ptrs_w)

    with pytest.raises(
        RuntimeError,
        match="tail chunk must contain at least 2048 tokens",
    ):
        torch.ops._C.awq_moe_chunked_w2_sm70_out(
            out,
            chunk_output,
            input,
            int_scratch[0],
            int_scratch[1],
            topk_weights,
            int_scratch[2],
            int_scratch[3],
            int_scratch[4],
            int_scratch[5],
            int_scratch[6],
            ptrs_w,
            ptrs_s,
            12289,
            10,
            512,
            160,
            2560,
            2560,
            32,
            6144,
        )
