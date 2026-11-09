# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Balanced W2 scratch boundaries against full-output and dense oracles."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.fixture(scope="module", params=[False, True], ids=["standard", "compact"])
def w2_bank(request):
    from benchmarks.benchmark_sm70_turbomind_exactness import _awq_reference_weight
    from vllm import _sm70_ops as sm70

    compact = request.param

    if hasattr(torch.ops.five_w2_audit, "chunked"):
        native = torch.ops.five_w2_audit
        ops = SimpleNamespace(
            prepare=native.prepare_compact if compact else native.prepare,
            ptrs=native.ptrs,
            baseline=native.baseline,
            chunked=native.chunked,
        )
    elif hasattr(torch.ops._C, "awq_moe_chunked_w2_sm70_out"):
        ops = SimpleNamespace(
            prepare=(
                sm70.awq_sm70_prepare_compact if compact else sm70.awq_sm70_prepare
            ),
            ptrs=sm70.awq_moe_build_strided_ptrs,
            baseline=sm70.awq_moe_gemm_sm70_per_expert_dispatch_out,
            chunked=sm70.awq_moe_chunked_w2_sm70_out,
        )
    else:
        pytest.skip("requires a freshly built chunked W2 extension")
    torch.manual_seed(470)
    experts, k, n, group = 512, 160, 2560, 32
    active = (0, 1, 7, 31, 64, 127, 255, 257, 500, 511)
    weights, metadata, decoded = [], [], {}
    for expert in range(experts):
        raw = torch.randint(0, 2**31 - 1, (k, n // 8), dtype=torch.int32, device="cuda")
        zero = torch.full(
            (k // group, n // 8), 0x44444444, dtype=torch.int32, device="cuda"
        )
        scale = torch.full(
            (k // group, n),
            0.003 + expert * 0.00001,
            dtype=torch.float16,
            device="cuda",
        )
        weight, meta, strides = ops.prepare(raw, scale, zero, group, False)
        weights.append(weight)
        metadata.append(meta)
        decoded[expert] = _awq_reference_weight(raw, scale, zero, group)
    weights, metadata = torch.stack(weights), torch.stack(metadata)
    ptr_w, ptr_s = ops.ptrs(
        weights, metadata, int(strides[0]), int(strides[1]), experts
    )
    return ops, weights, metadata, ptr_w, ptr_s, active, decoded


@pytest.mark.parametrize(
    "tokens,cap",
    [
        (4097, 4096),
        (6145, 6144),
        (8192, 4096),
        (8192, 6144),
        (8193, 4096),
        (12289, 6144),
    ],
)
def test_balanced_w2_values_and_changing_route_graph(w2_bank, tokens, cap):
    ops, weights, metadata, ptr_w, ptr_s, active, decoded = w2_bank
    top_k, experts, k, n = 10, 512, 160, 2560
    slots = tokens * top_k
    source = torch.randn(slots, k, dtype=torch.float16, device="cuda") * 0.1
    sorted_input = torch.empty_like(source)
    offsets = torch.empty(experts + 1, dtype=torch.int32, device="cuda")
    permutation = torch.empty(slots, dtype=torch.int32, device="cuda")
    inverse = torch.empty(tokens, top_k, dtype=torch.int32, device="cuda")
    topk_weights = torch.empty(tokens, top_k, device="cuda")
    full = torch.empty(slots, n, dtype=torch.float16, device="cuda")
    reference = torch.empty(tokens, n, dtype=torch.float16, device="cuda")
    actual = torch.empty_like(reference)
    scratch = torch.empty(cap * top_k, n, dtype=torch.float16, device="cuda")
    chunk_offsets = torch.empty_like(offsets)
    begin = torch.empty(experts, dtype=torch.int32, device="cuda")
    end = torch.empty_like(begin)
    indices = torch.empty(cap * top_k, dtype=torch.int32, device="cuda")
    chunk_inverse = torch.empty_like(indices)
    ids = torch.empty(tokens, top_k, dtype=torch.int64, device="cuda")

    def configure(phase):
        table = torch.tensor(active, device="cuda")
        token = torch.arange(tokens, device="cuda")[:, None]
        route = torch.arange(top_k, device="cuda")[None, :]
        # Rotate the ten active experts, leaving 502 empty segments. Distinct
        # routes per token match production, and phase changes the stable sort.
        ids.copy_(
            (token * 17 + route * 53 + phase) % experts
            if phase == 7
            else table[(token + route + phase) % top_k]
        )
        order = torch.argsort(ids.flatten(), stable=True)
        permutation.copy_(order)
        inverse.view(-1)[order] = torch.arange(slots, dtype=torch.int32, device="cuda")
        offsets[0] = 0
        offsets[1:].copy_(torch.bincount(ids.flatten(), minlength=experts).cumsum(0))
        source.normal_(std=0.1)
        sorted_input.copy_(source[order])
        topk_weights.copy_(torch.softmax(torch.randn_like(topk_weights), -1))

    def baseline():
        ops.baseline(
            full, sorted_input, offsets, ptr_w, ptr_s, experts, k, n, 32, False
        )
        torch.ops._moe_C.moe_unpermute(
            full, topk_weights, inverse, None, top_k, reference
        )

    def chunked():
        ops.chunked(
            actual,
            scratch,
            sorted_input,
            offsets,
            permutation,
            topk_weights,
            chunk_offsets,
            begin,
            end,
            indices,
            chunk_inverse,
            ptr_w,
            ptr_s,
            tokens,
            top_k,
            experts,
            k,
            n,
            n,
            32,
            cap,
        )

    configure(0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        baseline()
        chunked()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        chunked()
    for phase in (0, 1, 7):
        configure(phase)
        baseline()
        graph.replay()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, atol=0.002, rtol=0.01)
        # Independent decoded-weight oracle at both ends and across a chunk
        # boundary; tolerate FP16 materialization, not wrong/missing routes.
        for token in (0, tokens // 2, tokens - 1):
            oracle = torch.zeros(n, dtype=torch.float32, device="cuda")
            for route, expert in enumerate(ids[token].tolist()):
                value = (
                    source[token * top_k + route].float() @ decoded[expert].float()
                ).half()
                oracle += value.float() * topk_weights[token, route]
            error = (actual[token].float() - oracle).abs()
            assert error.max() <= oracle.abs().max() * 0.005 + 0.0001
            assert error.norm() <= oracle.norm() * 0.005 + 0.001
