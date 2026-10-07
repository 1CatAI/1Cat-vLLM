# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_device_capability(70),
    reason="requires SM70",
)


@pytest.mark.parametrize(
    "groups,rows,context",
    [
        (1, 2, 1091),
        (1, 8, 640),
        (1, 8, 1024),
        (1, 8, 1091),
        (1, 8, 1536),
        (1, 8, 2048),
        (1, 8, 2049),
        (1, 8, 4096),
        (4, 8, 8192),
        (1, 8, 131072),
        (1, 8, 262144),
    ],
)
def test_grouped_fp16_reference_and_graph_metadata(groups, rows, context):
    from vllm.vllm_flash_attn.flash_attn_interface import ensure_fa2_library_loaded

    ensure_fa2_library_loaded()
    operator = torch.ops._vllm_fa2_C.sm70_grouped_fp16_fwd
    page = 832
    pages = math.ceil(context / page)
    generator = torch.Generator(device="cuda").manual_seed(123)
    storage = torch.randn(
        (groups * pages, 2, page, 1, 264),
        generator=generator,
        dtype=torch.float16,
        device="cuda",
    )
    k, v = storage[:, 0, :, :, :256], storage[:, 1, :, :, :256]
    q = torch.randn(
        (groups * rows, 6, 256),
        generator=generator,
        dtype=torch.float16,
        device="cuda",
    )
    table = torch.stack(
        [
            torch.randperm(pages, generator=generator, device="cuda") + g * pages
            for g in range(groups)
        ]
    ).int()
    lengths = torch.tensor(
        [context - rows + i + 1 for g in range(groups) for i in range(rows)],
        dtype=torch.int32,
        device="cuda",
    )
    if rows == 8:
        lengths.reshape(groups, rows)[:, 2] = 0
    out = torch.empty_like(q)
    partial = torch.full((groups, 80, 8, 6, 256), float("nan"), device="cuda")
    lse = torch.full((groups, 80, 8, 6, 2), float("nan"), device="cuda")
    if groups == 1:
        partial, lse = partial[0], lse[0]

    def run():
        return operator(q, k, v, out, table, lengths, partial, lse, 1 / 16)

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = torch.empty_like(q, dtype=torch.float32)
        for g in range(groups):
            keys = k.index_select(0, table[g].long()).reshape(-1, 256).float()
            values = v.index_select(0, table[g].long()).reshape(-1, 256).float()
            panel = slice(g * rows, (g + 1) * rows)
            scores = q[panel].float() @ keys.T / 16
            scores.masked_fill_(
                torch.arange(keys.shape[0], device=q.device)[None, None, :]
                >= lengths[panel, None, None],
                -torch.inf,
            )
            reference[panel] = torch.softmax(scores, dim=-1).nan_to_num() @ values
        actual = run().clone()
        torch.testing.assert_close(actual.float(), reference, atol=0.0005, rtol=0.01)
        for _ in range(2):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, actual, atol=0, rtol=0)
        lengths.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(out) == 0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_short_plan_boundary_replays_match_legacy():
    from vllm.vllm_flash_attn.flash_attn_interface import ensure_fa2_library_loaded

    ensure_fa2_library_loaded()
    operator = torch.ops._vllm_fa2_C.sm70_grouped_fp16_fwd
    storage = torch.randn((3, 2, 832, 1, 264), device="cuda", dtype=torch.float16)
    k, v = storage[:, 0, :, :, :256], storage[:, 1, :, :, :256]
    q = torch.randn((8, 6, 256), device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device="cuda")
    lengths = torch.arange(2041, 2049, dtype=torch.int32, device="cuda")
    partial = torch.empty((80, 8, 6, 256), device="cuda")
    lse = torch.empty((80, 8, 6, 2), device="cuda")

    def run(enabled):
        operator(q, k, v, out, table, lengths, partial, lse, 1 / 16, enabled)

    run(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(True)
    for context in (2049, 2048, 2496, 128):
        lengths.copy_(torch.arange(context - 7, context + 1, device="cuda"))
        graph.replay()
        actual = out.clone()
        run(False)
        if context == 2048:
            torch.testing.assert_close(actual, out, atol=0.0005, rtol=0.01)
        else:
            assert torch.equal(actual.view(torch.int16), out.view(torch.int16))
