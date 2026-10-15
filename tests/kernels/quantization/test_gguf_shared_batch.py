# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared expert batch policy, native execution and changed-input replay."""

from types import SimpleNamespace

import gguf
import pytest
import torch

from tests.kernels.quantization.test_gguf_dense_segments import raw_weights
from tools.sm70.native_trace import NativeDispatchTrace
from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.quantization.gguf_dense_hmma import (
    _shared_expert,
    maybe_apply_shared_expert,
)
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    prepare_gguf_projections,
    prepared_projection_arguments,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "sources,down_source",
    [
        ((12, 23), 8),
        ((23, 12), 8),
        ((23, 23), 8),
        ((12, 14), 8),
        ((13, 23), 8),
        ((23, 13), 8),
        ((8, 23), 8),
        ((23, 8), 8),
        ((20, 23), 8),
        pytest.param((8, 8), 8, id="draft-q8"),
        ((12, 23), 20),
    ],
)
def test_shared_batch_native_and_official_oracle_with_graph(sources, down_source):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    raw = [raw_weights(s, 160, 2560) for s in sources] + [
        raw_weights(down_source, 2560, 160)
    ]
    decoded = [
        torch.from_numpy(gguf.quants.dequantize(r, gguf.GGMLQuantizationType(s))).cuda()
        for r, s in zip(raw, (*sources, down_source))
    ]
    cfg = SimpleNamespace(
        kernel_config=KernelConfig(sm70_gguf={"shared_expert_batch": True}),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with set_current_vllm_config(cfg):
        gu = prepare_gguf_projections(
            [(torch.from_numpy(r).cuda(), s) for r, s in zip(raw[:2], sources)],
            torch.float16,
            True,
            8,
        )
        down = prepare_gguf_projections(
            [(torch.from_numpy(raw[2]).cuda(), down_source)], torch.float16, True, 8
        )
    assert all(p.shared_expert_batch for p in (*gu, *down))
    pairs = []
    for p in gu:
        tile = 0
        for n in p.source_output_sizes:
            assert n == 160
            cs = 20 * 4 * 512 * (2 if p.segment_format == 4 else 1)
            hs = (
                20
                * 512
                * (1 if p.segment_format == 1 else 2 if p.segment_format == 2 else 0)
            )
            ss = 20 * 512
            pairs.append(
                (
                    p.codes[tile * cs : (tile + 5) * cs],
                    p.segment_high[tile * hs : (tile + 5) * hs],
                    p.stats[tile * ss : (tile + 5) * ss],
                    p.segment_format,
                )
            )
            tile += 5
    p = down[0]
    gate = torch.randn(2560, device="cuda", dtype=torch.float16) * 0.02
    layer = SimpleNamespace(
        expert_gate=SimpleNamespace(weight=gate),
        gate_up_proj=SimpleNamespace(gguf_tm_projections=gu),
        down_proj=SimpleNamespace(gguf_tm_projections=down, reduce_results=False),
    )
    arguments = (
        *pairs[0][:3],
        *pairs[1][:3],
        p.codes,
        p.segment_high,
        p.stats,
        gate,
        [pairs[0][3], pairs[1][3], p.segment_format],
        *prepared_projection_arguments(gu),
    )
    for m in (5, 8, 9, 10, 16, 17, 20, 21, 24, 25, 30, 32, 33, 39, 40, 41):
        x = torch.randn(m, 2560, device="cuda", dtype=torch.float16) * 0.125
        with NativeDispatchTrace() as trace:
            actual = _shared_expert(x, *arguments, [p.gguf_tm_k_ld, p.gguf_tm_q_ld, 1])
        native = [
            r
            for r in trace.records
            if r["operator"] == "_C::gguf_shared_gate_up_sm70_out"
        ]
        accelerated = down_source == 8 and m <= 40
        assert len(native) == (
            (m + 19) // 20 if accelerated else int(m <= 8 or m == 20)
        )
        if accelerated:
            assert all("restore" not in r["operator"] for r in trace.records)
        legacy = _shared_expert(x, *arguments, [p.gguf_tm_k_ld, p.gguf_tm_q_ld])
        if m in (5, 8, 20):
            assert torch.equal(actual, legacy)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = maybe_apply_shared_expert(layer, x)
        for _ in range(2):
            x.normal_(std=0.125)
            graph.replay()
            legacy = _shared_expert(x, *arguments, [p.gguf_tm_k_ld, p.gguf_tm_q_ld])
            assert torch.isfinite(output).all()
            paired = (output.float() - legacy.float()).norm() / (
                legacy.float().norm() + 1e-9
            )
            assert float(paired) < 0.003
            g, u = [(x.float() @ w.float().T).half() for w in decoded[:2]]
            h = (torch.nn.functional.silu(g.float()) * u.float()).half()
            d = (h.float() @ decoded[2].float().T).half()
            sg = (x.float() @ gate.float()).half()
            expected = (d.float() * torch.sigmoid(sg.float())[:, None]).half()
            error = (output.float() - expected.float()).norm() / (
                expected.float().norm() + 1e-9
            )
            assert float(error) < 0.003
