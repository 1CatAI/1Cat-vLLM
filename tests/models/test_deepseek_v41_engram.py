# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise hash semantics, speculative prefix commit and graph replay."""

import copy

import pytest
import torch

from benchmarks.deepseek_v41.oracle_primitives import EngramHashState
from vllm.model_executor.layers.deepseek_v41_engram import (
    V41EngramHashState,
    apply_engram_gate,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _auxiliaries() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    token_map = torch.randint(99092, (129280,), dtype=torch.int32, generator=generator)
    primes = torch.tensor(
        [
            [16000003 + 2 * i for i in range(24)],
            [16000117 + 2 * i for i in range(24)],
        ],
        dtype=torch.int64,
    )
    offsets = torch.cat(
        (torch.zeros(2, 1, dtype=torch.int64), primes.cumsum(1)[:, :-1]), 1
    )
    # Products exercise high int64 bits before XOR and modulus.
    multipliers = torch.tensor(
        [
            [75940234776931, 51874328537443, 64173895873611, 28176371828291],
            [13984732984811, 81658718364891, 71637276818521, 51483762739211],
        ],
        dtype=torch.int64,
    )
    return token_map, primes, offsets, multipliers


def _states(max_tokens=2048):
    values = _auxiliaries()
    cpu = EngramHashState(*values, pad_token_id=2)
    gpu = V41EngramHashState(*(t.cuda() for t in values), max_tokens=max_tokens)
    return cpu, gpu


@pytest.mark.parametrize(
    "length,chunk", [(64, 64), (2048, 2048), (2048, 127), (257, 63), (16384, 2048)]
)
def test_v41_hash_matches_verified_reference_across_chunks_masks_and_int64(
    length, chunk
):
    cpu, gpu = _states()
    generator = torch.Generator().manual_seed(41)
    tokens = torch.randint(129280, (1, length), generator=generator)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 61:64] = False
    mask[:, -2] = False
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        expected = cpu(tokens[:, start:end], start, mask[:, start:end])
        actual = gpu(
            tokens[0, start:end].cuda(),
            torch.arange(start, end, device="cuda"),
            mask[0, start:end].cuda(),
        )
        torch.testing.assert_close(actual.cpu(), expected[0], rtol=0, atol=0)
        gpu.commit(torch.tensor([end - start], device="cuda", dtype=torch.int32))
        torch.testing.assert_close(
            gpu.history[:3].cpu(), cpu.history[0], rtol=0, atol=0
        )
        assert gpu.history[3].item() == cpu.next_pos


@pytest.mark.parametrize("accepted_drafts", range(6))
def test_v41_hash_commits_only_accepted_inputs_and_can_replace_rejected_suffix(
    accepted_drafts,
):
    cpu, gpu = _states()
    tokens = torch.arange(80) * 37
    prompt, candidate, following = tokens[:63], tokens[63:69], torch.tensor([888, 999])
    cpu(prompt[None], 0)
    gpu(prompt.cuda(), torch.arange(63, device="cuda"))
    gpu.commit(torch.tensor([63], device="cuda", dtype=torch.int32))
    # No history is committed by verification itself.
    expected_history = gpu.history.clone()
    actual = gpu(candidate.cuda(), torch.arange(63, 69, device="cuda"))
    torch.testing.assert_close(gpu.history, expected_history, rtol=0, atol=0)
    before = EngramHashState(*_auxiliaries(), pad_token_id=2)
    before(prompt[None], 0)
    expected = before(candidate[None], 63)
    torch.testing.assert_close(actual.cpu(), expected[0], rtol=0, atol=0)
    # Zero accepted drafts still commits the anchor input. A bonus token is
    # not yet an input and must not be inserted into the three-token history.
    accepted_inputs = accepted_drafts + 1
    gpu.commit(torch.tensor([accepted_inputs], device="cuda", dtype=torch.int32))
    cpu(candidate[None, :accepted_inputs], 63)
    next_pos = 63 + accepted_inputs
    actual = gpu(following.cuda(), torch.arange(next_pos, next_pos + 2, device="cuda"))
    expected = cpu(following[None], next_pos)
    torch.testing.assert_close(actual.cpu(), expected[0], rtol=0, atol=0)
    gpu.commit(torch.tensor([2], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(gpu.history[:3].cpu(), cpu.history[0], rtol=0, atol=0)


def test_v41_hash_graph_replay_with_changing_acceptance_and_new_sequence():
    cpu, gpu = _states()
    tokens = torch.arange(6, device="cuda", dtype=torch.int64) + 100
    positions = torch.arange(6, device="cuda")
    selector = torch.tensor([1], device="cuda", dtype=torch.int32)
    gpu(tokens, positions)
    gpu.commit(selector)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gpu(tokens, positions)
        gpu.commit(selector)
    # start_pos=0 explicitly starts a new sequence, even after a different
    # accepted prefix in the previous sequence.
    for start, count in ((0, 6), (6, 1), (7, 4), (0, 2)):
        positions.copy_(torch.arange(start, start + 6, device="cuda"))
        tokens.add_(11)
        selector.fill_(count)
        verification = copy.deepcopy(cpu)
        expected = verification(tokens.cpu()[None], start)
        cpu(tokens.cpu()[None, :count], start)
        graph.replay()
        torch.testing.assert_close(output.cpu(), expected[0], rtol=0, atol=0)
        torch.testing.assert_close(
            gpu.history[:3].cpu(), cpu.history[0], rtol=0, atol=0
        )
        assert gpu.history[3].item() == start + count


@pytest.mark.parametrize("tokens", [1, 6, 64])
@pytest.mark.parametrize("inplace", [False, True])
def test_v41_engram_gate_per_stream_fp32_and_bounded_fp16_store(tokens, inplace):
    generator = torch.Generator().manual_seed(41)
    residual = torch.randn(tokens, 4, 5120, generator=generator).half()
    projection = torch.randn(tokens, 25600, generator=generator).half()
    q_weight = torch.rand(4, 5120, generator=generator)
    k_weight = torch.rand(4, 5120, generator=generator)
    # Include zero-norm streams and masked tokens.
    residual[0].zero_()
    projection[0, : 4 * 5120].zero_()
    mask = torch.ones(tokens, dtype=torch.bool)
    if tokens > 1:
        mask[1] = False
    h, key = residual.float(), projection[:, : 4 * 5120].float().view(tokens, 4, 5120)
    rstd = torch.rsqrt(h.square().mean(-1) + 1e-20) * torch.rsqrt(
        key.square().mean(-1) + 1e-20
    )
    dot = (h * (q_weight * k_weight) * key).sum(-1) * rstd * 5120**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    gate.masked_fill_(~mask[:, None], 0)
    expected = (h + gate[..., None] * projection[:, 4 * 5120 :].float()[:, None]).half()
    x = residual.cuda()
    actual = apply_engram_gate(
        x,
        projection.cuda(),
        q_weight.cuda(),
        k_weight.cuda(),
        mask.cuda(),
        inplace=inplace,
    )
    assert (actual.data_ptr() == x.data_ptr()) == inplace
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.001, atol=2e-6)
    if tokens > 1:
        torch.testing.assert_close(actual[1].cpu(), residual[1], rtol=0, atol=0)


def test_engram_preserves_wide_fp32_hc_storage_inplace():
    generator = torch.Generator(device="cuda").manual_seed(712)
    residual = torch.randn(6, 4, 5120, device="cuda", generator=generator) * 2.0**19
    projection = torch.randn(6, 25600, device="cuda", generator=generator).half()
    q = torch.rand(4, 5120, device="cuda", generator=generator)
    k = torch.rand(4, 5120, device="cuda", generator=generator)
    key = projection[:, :20480].float().view(6, 4, 5120)
    dot = ((residual * (q * k)) * key).sum(-1)
    dot *= torch.rsqrt(residual.square().mean(-1) + 1e-20)
    dot *= torch.rsqrt(key.square().mean(-1) + 1e-20) * 5120**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    expected = residual + gate[..., None] * projection[:, 20480:].float()[:, None]
    actual = apply_engram_gate(residual, projection, q, k, inplace=True)
    assert actual.data_ptr() == residual.data_ptr()
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-6)
