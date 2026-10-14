# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow indexed GGUF prefill: padding, empty groups and graph replay."""

import gguf
import numpy as np
import pytest
import torch
from test_gguf_lattice_transcode import source as lattice_source
from test_gguf_lut_transcode import source as lut_source

from vllm import _custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_expert_cache import _canonical

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("weight_type", [18, 21, 22, 23])
@pytest.mark.parametrize(
    "m,n,k,gated,indexed",
    [
        (511, 320, 256, True, True),  # below the new tile admission
        (512, 64, 512, False, False),
        (513, 320, 768, True, True),
        (2049, 512, 2560, False, True),
        (769, 352, 512, True, True),  # N32 padding: existing fallback
        (513, 576, 256, False, False),  # wide projection: existing fallback
    ],
)
def test_narrow_grouped_prefill(weight_type, m, n, k, gated, indexed):
    experts = 5
    rows = n // 2 if gated else n
    parts = []
    dense = []
    for part in range(2 if gated else 1):
        source = lut_source if weight_type == 23 else lattice_source
        data = np.stack(
            [
                source(
                    weight_type,
                    scale=0.0009765625,
                    n=rows,
                    k=k,
                    seed=9683 + part * experts + expert,
                )
                for expert in range(experts)
            ]
        )
        parts.append(torch.from_numpy(data).cuda())
        dense.append(
            torch.from_numpy(
                gguf.quants.dequantize(data, gguf.GGMLQuantizationType(weight_type))
            ).cuda()
        )
    family, decoder, group, weight_bytes, stats_bytes = _canonical(weight_type, n, k)
    weights = torch.empty(experts * weight_bytes, dtype=torch.uint8, device="cuda")
    stats = torch.empty(experts * stats_bytes, dtype=torch.uint8, device="cuda")
    sequence = torch.arange(experts, device="cuda", dtype=torch.int64)
    pointers = [p.data_ptr() + sequence * p[0].numel() for p in parts]
    error = torch.zeros(1, dtype=torch.int32, device="cuda")
    torch.ops._C.gguf_prefill_transcode_sm70(
        weights,
        stats,
        weight_bytes,
        stats_bytes,
        pointers[0],
        pointers[1] if gated else None,
        sequence.int(),
        torch.tensor([experts], dtype=torch.int32, device="cuda"),
        128,
        weight_type,
        n,
        k,
        0,
        parts[0].shape[-1],
        error,
    )
    assert error.item() == 0
    tables = []
    for bank, stride, ld in ((weights, weight_bytes, 32 * k), (stats, stats_bytes, n)):
        table = torch.empty(experts, 2, dtype=torch.int64, device="cuda")
        table[:, 0] = bank.data_ptr() + sequence * stride
        table[:, 1] = ld
        tables.append(table.view(torch.uint8).flatten())
    bounds = [0, 0, 17, m - 1, m, m]
    offsets = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(9683 + m)
    x = (
        torch.randn(m + 7 if indexed else m, k, device="cuda", generator=gen) * 0.125
    ).half()
    indices = (
        torch.randint(x.shape[0], (m,), device="cuda", dtype=torch.int32, generator=gen)
        if indexed
        else None
    )
    storage = torch.full((m * rows + 32,), 37, dtype=torch.half, device="cuda")
    out = storage[16:-16].view(m, rows)

    def run(tune=False):
        torch.ops._C.gguf_grouped_prefill_gemm_sm70_out(
            out,
            x,
            offsets,
            *tables,
            family,
            decoder,
            n,
            experts,
            group,
            gated,
            tune,
            indices,
        )

    def check():
        actual_x = x if indices is None else x[indices.long()]
        expected = torch.empty_like(out, dtype=torch.float32)
        for expert, (begin, end) in enumerate(zip(bounds, bounds[1:])):
            a = actual_x[begin:end].float()
            projected = a @ dense[0][expert].float().T
            if gated:
                projected = torch.nn.functional.silu(projected) * (
                    a @ dense[1][expert].float().T
                )
            expected[begin:end] = projected
        assert torch.isfinite(expected.half()).all(), "reference exceeds FP16 range"
        finite = torch.isfinite(out)
        assert finite.all(), {
            "bad_indices": (~finite).nonzero()[:8].tolist(),
            "expected_abs_max": expected.abs().max().item(),
        }
        relative = (out.float() - expected).norm(dim=1) / expected.norm(
            dim=1
        ).clamp_min(1e-6)
        assert relative.max().item() < 0.005
        assert (storage[:16] == 37).all() and (storage[-16:] == 37).all()

    run()  # no pre-existing native tuning entry
    check()
    run(tune=True)
    run()
    check()
    if m == 513:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        # Address-stable replay with different activations and gather indices.
        x.mul_(-0.75)
        if indices is not None:
            indices.copy_(torch.flip(indices, (0,)))
        graph.replay()
        check()
