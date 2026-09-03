# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

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


def test_qwen38_sorted_output_memory_ledger() -> None:
    layers = 48
    hidden = 2560
    top_k = 10
    element_bytes = 2

    max_batched_tokens_bytes = 8192 * top_k * hidden * element_bytes
    persistent_cap8_bytes = layers * 8 * top_k * hidden * element_bytes
    persistent_cap32_bytes = layers * 32 * top_k * hidden * element_bytes

    assert max_batched_tokens_bytes == 400 * 1024**2
    assert persistent_cap8_bytes == pytest.approx(18.75 * 1024**2)
    assert persistent_cap32_bytes == 75 * 1024**2
