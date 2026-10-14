# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped shared-key scoring preserves paged request identity and live state."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_mqa_paged, qsa_select_paged_tokens
from vllm.models.qwen4_exp.nvidia.ops.qsa_shared_key import (
    maybe_shared_key_batch_scores,
)
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SM70 CUDA required"
)


@pytest.mark.parametrize("requests", [2, 4, 6, 8])
@pytest.mark.parametrize("host", [False, True])
def test_shared_key_batch_scores_and_replay_match_retained_paged_reader(requests, host):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    rows, page = requests * 5, 204
    torch.manual_seed(20261015 + requests)
    data = torch.randn(requests * 3, page, 1, 128, dtype=torch.float16) * 0.125
    if host:
        data = data.pin_memory()
        cache = get_accelerator_view_from_cpu_tensor(data)
    else:
        cache = data.cuda()
    q = torch.randn(rows, 4, 128, dtype=torch.float16, device="cuda") * 0.125
    table = torch.full((requests, 4), -1, dtype=torch.int32, device="cuda")
    table[:, :3] = torch.arange(requests * 3, dtype=torch.int32, device="cuda").view(
        requests, 3
    )
    table[-1, 1] = -1
    req = torch.arange(requests, dtype=torch.int32, device="cuda").repeat_interleave(5)
    positions = torch.full((rows,), 2399, dtype=torch.int64, device="cuda")
    lengths = (
        torch.arange(1, requests + 1, dtype=torch.int32, device="cuda") * 128 + 1024
    )
    starts = torch.arange(0, rows + 1, 5, dtype=torch.int32)

    def run(enabled):
        return qsa_mqa_paged(
            q,
            cache,
            table,
            req,
            positions,
            lengths,
            4,
            shared_key_scoring=enabled,
            shared_key_query_start_loc_cpu=starts if enabled else None,
        )

    def check(actual, visible):
        expected, expected_visible = run(False)
        torch.testing.assert_close(visible, expected_visible, rtol=0, atol=0)
        for row, n in enumerate(visible.tolist()):
            a, b = actual[row, :n], expected[row, :n]
            assert torch.equal(torch.isfinite(a), torch.isfinite(b))
            finite = torch.isfinite(b)
            torch.testing.assert_close(a[finite], b[finite], rtol=0.003, atol=0.0002)

    output = run(True)
    check(*output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(True)
    for shift in (1, 2):
        q.mul_(0.75)
        table[:, :3].copy_(table[:, :3].roll(shift, 0))
        positions.sub_(47)
        lengths[0] = 0
        req[-1] = -1
        graph.replay()
        check(*output)


@pytest.mark.parametrize("requests", [2, 8])
def test_grouped_selection_matches_independent_native_requests(requests):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    rows = requests * 5
    torch.manual_seed(20261016)
    q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.float16) * 0.125
    cache = (
        torch.randn(requests * 16, 64, 1, 128, device="cuda", dtype=torch.float16)
        * 0.125
    )
    table = torch.arange(requests * 16, dtype=torch.int32, device="cuda").view(
        requests, 16
    )
    mapping = torch.arange(
        requests, dtype=torch.int32, device="cuda"
    ).repeat_interleave(5)
    positions = torch.full((rows,), 4095, device="cuda", dtype=torch.int64)
    lengths = torch.full((requests,), 4096, device="cuda", dtype=torch.int32)
    starts = torch.arange(0, rows + 1, 5, dtype=torch.int32)

    def run():
        return qsa_select_paged_tokens(
            q,
            cache,
            table,
            mapping,
            positions,
            lengths,
            2048,
            4,
            query_start_loc_cpu=starts,
            shared_key_scoring=True,
            shared_key_batch_scoring=True,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for shift in (1, 2):
        table.copy_(table.roll(shift, 0))
        q.mul_(0.75)
        lengths[0] = 0
        graph.replay()
        expected = torch.cat(
            [
                qsa_select_paged_tokens(
                    q[5 * i : 5 * i + 5],
                    cache,
                    table[i : i + 1],
                    mapping[5 * i : 5 * i + 5] - i,
                    positions[5 * i : 5 * i + 5],
                    lengths[i : i + 1],
                    2048,
                    4,
                    shared_key_scoring=True,
                )
                for i in range(requests)
            ]
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("starts", [[0, 1, 10], [0, 9, 10], [0, 5, 9], [0, 7, 6, 10]])
def test_shared_key_batch_declines_unqualified_host_layouts(starts):
    q = torch.zeros(10, 4, 128, dtype=torch.float16, device="cuda")
    table = torch.zeros(len(starts) - 1, 2, dtype=torch.int32, device="cuda")
    assert (
        maybe_shared_key_batch_scores(
            q,
            q.new_zeros(2, 64, 1, 128),
            table,
            torch.zeros(10, dtype=torch.int32, device="cuda"),
            torch.zeros(10, dtype=torch.int64, device="cuda"),
            torch.zeros(len(starts) - 1, dtype=torch.int32, device="cuda"),
            4,
            128,
            128**0.5,
            torch.tensor(starts, dtype=torch.int32),
        )
        is None
    )
