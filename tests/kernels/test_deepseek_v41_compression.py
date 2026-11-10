# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest
import torch

from vllm.model_executor.kernels.deepseek_v41_compression import V41C2State

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Reference:
    def __init__(self, norm):
        self.norm = norm
        self.partial = torch.zeros(2, 512)
        self.position = 0

    def feed(self, kv, score, start):
        if start == 0:
            self.partial.zero_()
            self.position = 0
        assert self.position == start
        result = torch.zeros_like(kv, dtype=torch.float16)
        complete = torch.zeros(kv.shape[0], dtype=torch.bool)
        for i in range(kv.shape[0]):
            if (start + i) % 2 == 0:
                self.partial[0] = kv[i]
                self.partial[1] = score[i]
            else:
                values = torch.stack((self.partial[0], kv[i]))
                scores = torch.stack((self.partial[1], score[i]))
                latent = (values * scores.softmax(0)).sum(0).half().float()
                result[i] = (
                    latent * torch.rsqrt(latent.square().mean() + 1e-20) * self.norm
                ).half()
                complete[i] = True
                self.partial.zero_()
        self.position += kv.shape[0]
        return result, complete


def _inputs(n, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, 512, generator=generator), torch.randn(
        n, 512, generator=generator
    ) * 3


def _states():
    norm = torch.linspace(0.5, 1.5, 512)
    return _Reference(norm), V41C2State(norm.cuda())


@pytest.mark.parametrize(
    "length,chunk", [(64, 64), (257, 63), (2048, 127), (2048, 2048)]
)
def test_c2_pooling_matches_fp32_pool_then_fp16_norm_across_odd_chunks(length, chunk):
    cpu, gpu = _states()
    kv, score = _inputs(length, 41)
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        expected, mask = cpu.feed(kv[start:end], score[start:end], start)
        actual, complete = gpu(
            kv[start:end].cuda(),
            score[start:end].cuda(),
            torch.arange(start, end, device="cuda"),
        )
        torch.testing.assert_close(complete.cpu(), mask, rtol=0, atol=0)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.001, atol=2e-6)
        gpu.commit(torch.tensor([end - start], dtype=torch.int32, device="cuda"))
        torch.testing.assert_close(gpu.partial.cpu(), cpu.partial, rtol=0, atol=0)
        assert gpu.position.item() == end


@pytest.mark.parametrize("start", [63, 64])
@pytest.mark.parametrize("accepted_drafts", range(6))
def test_c2_commit_all_draft_acceptance_lengths_and_both_group_parities(
    start, accepted_drafts
):
    cpu, gpu = _states()
    kv, score = _inputs(start + 9, 42)
    cpu.feed(kv[:start], score[:start], 0)
    gpu(kv[:start].cuda(), score[:start].cuda(), torch.arange(start, device="cuda"))
    gpu.commit(torch.tensor([start], dtype=torch.int32, device="cuda"))
    before = gpu.partial.clone()
    actual, complete = gpu(
        kv[start : start + 6].cuda(),
        score[start : start + 6].cuda(),
        torch.arange(start, start + 6, device="cuda"),
    )
    expected, mask = copy.deepcopy(cpu).feed(
        kv[start : start + 6], score[start : start + 6], start
    )
    torch.testing.assert_close(gpu.partial, before, rtol=0, atol=0)
    torch.testing.assert_close(complete.cpu(), mask, rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.001, atol=2e-6)
    accepted = accepted_drafts + 1
    gpu.commit(torch.tensor([accepted], dtype=torch.int32, device="cuda"))
    cpu.feed(kv[start : start + accepted], score[start : start + accepted], start)
    torch.testing.assert_close(gpu.partial.cpu(), cpu.partial, rtol=0, atol=0)
    # Replacement input after a rejection must pool with the last accepted
    # partial group, never a rejected candidate's projection or score.
    following_kv, following_score = _inputs(2, 100 + accepted)
    next_pos = start + accepted
    actual, complete = gpu(
        following_kv.cuda(),
        following_score.cuda(),
        torch.arange(next_pos, next_pos + 2, device="cuda"),
    )
    expected, mask = cpu.feed(following_kv, following_score, next_pos)
    torch.testing.assert_close(complete.cpu(), mask, rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.001, atol=2e-6)


def test_c2_graph_replays_different_prefix_selectors():
    cpu, gpu = _states()
    kv, score = _inputs(6, 43)
    a, b = kv.cuda(), score.cuda()
    positions = torch.arange(6, device="cuda")
    selector = torch.tensor([1], dtype=torch.int32, device="cuda")
    gpu(a, b, positions)
    gpu.commit(selector)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, complete = gpu(a, b, positions)
        gpu.commit(selector)
    for start, accepted in ((0, 5), (5, 1), (6, 6), (0, 2)):
        positions.copy_(torch.arange(start, start + 6, device="cuda"))
        a.mul_(-0.75)
        b.add_(0.01)
        selector.fill_(accepted)
        expected, mask = copy.deepcopy(cpu).feed(a.cpu(), b.cpu(), start)
        cpu.feed(a[:accepted].cpu(), b[:accepted].cpu(), start)
        graph.replay()
        torch.testing.assert_close(complete.cpu(), mask, rtol=0, atol=0)
        torch.testing.assert_close(output.cpu(), expected, rtol=0.001, atol=2e-6)
        torch.testing.assert_close(gpu.partial.cpu(), cpu.partial, rtol=0, atol=0)
