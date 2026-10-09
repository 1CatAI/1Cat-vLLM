# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from test_gguf_lattice_transcode import source

from vllm import _custom_ops  # noqa: F401
from vllm.model_executor.kernels.gguf import dp4a_expert_capabilities
from vllm.model_executor.layers.quantization import gguf_turbomind_moe  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_lattice_staging import (
    GGUFLatticeStaging,
)
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_lut_transcode import transcode_lut4
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def prepared_bank(kind, n, k, seed):
    weights, stats, raw = [], [], []
    # Eight distinct experts, repeated to fill the real 512-expert geometry.
    for i in range(8):
        rows = source(kind, 0.0009765625, n=n, k=k, seed=seed + i)
        if kind == 20:
            projection = transcode_lut4(rows, kind)
            w, s, ld = torch.ops._C.gguf_lut4_sm70_prepare(
                torch.from_numpy(projection.codes).cuda(),
                torch.from_numpy(projection.scales).cuda(),
                0,
                32,
            )
        else:
            projection = transcode_lattice(rows, kind)
            codes, metadata = projection.mma884_storage()
            metadata = metadata.view(np.int32 if kind == 22 else np.int64)
            w, s, ld = torch.ops._C.gguf_lattice_sm70_prepare(
                torch.from_numpy(codes).cuda(),
                torch.from_numpy(metadata).cuda(),
                kind,
                projection.group_size,
            )
            raw.append(RawGGUFProjection.from_rows(rows, kind).data)
        weights.append(w)
        stats.append(s)
    w = torch.stack(weights).repeat(64, 1, 1)
    s = torch.stack(stats).repeat(64, 1, 1)
    wp, sp = torch.ops._C.awq_moe_build_strided_ptrs(w, s, *ld.tolist(), 512)
    original = torch.from_numpy(np.stack(raw)).cuda().repeat(64, 1, 1) if raw else wp
    return w, s, wp, sp, original


def operands(pool, kind, gate, up, down, staged, dp4a):
    if staged:
        pool.bind(gate[4])
        gc, gs = pool.slot("w1", kind)
        uc, us = pool.slot("w3", kind)
        gw, gst = torch.ops._C.awq_moe_build_strided_ptrs(gc, gs, 2560 * 32, 160, 512)
        uw, ust = torch.ops._C.awq_moe_build_strided_ptrs(uc, us, 2560 * 32, 160, 512)
    else:
        gw, gst, uw, ust = gate[2], gate[3], up[2], up[3]
    return (
        gate[4],
        up[4],
        gw,
        gst,
        uw,
        ust,
        down[2],
        down[3],
        kind,
        20,
        0,
        512,
        16 if kind == 22 else 32,
        160,
        [],
        [],
        [],
        dp4a,
        [],
        staged,
    )


@pytest.mark.parametrize("kind", [18, 21, 22])
def test_real_shape_staging_compile_graph_and_small_m_bypass(kind):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    pool = GGUFLatticeStaging(512, 160, 2560, "cuda")
    next_kind = {18: 21, 21: 22, 22: 18}[kind]
    down = prepared_bank(20, 2560, 160, 300)
    prepared = [
        (prepared_bank(t, 160, 2560, 110 + t), prepared_bank(t, 160, 2560, 210 + t))
        for t in (kind, next_kind)
    ]
    control = [
        operands(pool, t, *b, down, False, [])
        for t, b in zip((kind, next_kind), prepared)
    ]
    candidate = [
        operands(pool, t, *b, down, True, [])
        for t, b in zip((kind, next_kind), prepared)
    ]

    def chain(x, ids, probabilities, args):
        first = torch.ops.vllm.gguf_expert_dp4a(x, ids, probabilities, *args[0])
        return torch.ops.vllm.gguf_expert_dp4a(x + first, ids, probabilities, *args[1])

    compiled = torch.compile(
        lambda x, ids, p: chain(x, ids, p, candidate), fullgraph=True
    )
    torch.manual_seed(951 + kind)
    for m in (5, 20, 512):
        x = (torch.randn(m, 2560, device="cuda") * 0.03125).half()
        ids = (
            (
                torch.arange(m, device="cuda")[:, None] * 17
                + torch.arange(10, device="cuda")[None, :] * 31
            )
            % 512
        ).int()
        probabilities = torch.softmax(torch.randn(m, 10, device="cuda"), -1)
        expected = chain(x, ids, probabilities, control)
        actual = compiled(x, ids, probabilities)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # A compiler-induced canonical clone would cost at least 50 MiB for
        # one operand; all ordinary outputs/scratch at these M fit below this.
        torch.accelerator.synchronize()
        before = torch.accelerator.memory_allocated()
        torch.accelerator.reset_peak_memory_stats()
        actual = compiled(x, ids, probabilities)
        torch.accelerator.synchronize()
        assert torch.accelerator.max_memory_allocated() - before < 48 * 1024**2
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if m <= 20:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = compiled(x, ids, probabilities)
            x.mul_(0.5)
            ids.copy_(ids.flip(0))
            probabilities.copy_(probabilities.flip(1))
            expected = chain(x, ids, probabilities, control)
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            del graph, captured
        del actual, expected, x, ids, probabilities

    batches = [
        c.min_m
        for c in dp4a_expert_capabilities(
            kind,
            20,
            2560,
            160,
            512,
            torch.float16,
            is_sm70=True,
            enabled=True,
            original_storage_available=True,
        )
        if c.reason is None
    ]
    assert 5 in batches
    gate, up = prepared[0]
    canonical_args = operands(pool, kind, gate, up, down, False, batches)
    staged_args = operands(pool, kind, gate, up, down, True, batches)
    for m in batches:
        x = (torch.randn(m, 2560, device="cuda") * 0.03125).half()
        ids = torch.randint(512, (m, 10), device="cuda", dtype=torch.int32)
        probabilities = torch.softmax(torch.randn(m, 10, device="cuda"), -1)
        pool.codes.fill_(0x5A5A5A5A)
        pool.metadata.fill_(17)
        expected = torch.ops.vllm.gguf_expert_dp4a(
            x, ids, probabilities, *canonical_args
        )
        actual = torch.ops.vllm.gguf_expert_dp4a(x, ids, probabilities, *staged_args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.all(pool.codes.flatten()[:256] == 0x5A5A5A5A)
        assert torch.all(pool.metadata.flatten()[:256] == 17)
