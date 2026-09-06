# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wheel-native GDN geometry isolation and replay/state ownership checks.

TP1/2/4 head partitions of the same layer must produce the same result. This
is an operator test, not a model-score or distributed-communication gate.
"""

import importlib.util
import os

import pytest
import torch


@pytest.fixture(scope="module")
def native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    path = os.environ.get("SM70_FLASHINFER_GDN_TEST_LIBRARY")
    if not path:
        spec = importlib.util.find_spec("vllm._sm70_flashinfer_gdn_C")
        path = spec.origin if spec else None
    if not path:
        pytest.skip("Install the native GDN wheel fragment or set its test path")
    torch.ops.load_library(path)


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32, 64])
def test_head_partition_graph_state_equivalence(native, rows):
    torch.manual_seed(20260906)
    device = "cuda"
    pool, hq, hv, dim, hidden = rows + 3, 16, 48, 128, 2560
    width = (2 * hq + hv) * dim
    x = torch.randn(rows, hidden, device=device, dtype=torch.float16) * 0.1
    qkv = torch.randn(rows, width, device=device, dtype=torch.float16) * 0.1
    weights = torch.randn(hidden, 2 * hv, device=device, dtype=torch.float16) * 0.01
    cw = torch.randn(width, 4, device=device, dtype=torch.float16) * 0.1
    bias = torch.randn(width, device=device, dtype=torch.float16) * 0.01
    a = torch.randn(hv, device=device, dtype=torch.float32) * 0.1
    dt = torch.randn(hv, device=device, dtype=torch.float16) * 0.1
    conv = torch.randn(pool, width, 3, device=device, dtype=torch.float16) * 0.1
    state = torch.randn(pool, hv, dim, dim, device=device) * 0.01
    indices = torch.arange(rows, device=device, dtype=torch.int32)
    groups = []
    for tp in (1, 2, 4):
        parts = []
        q, v = hq // tp, hv // tp
        for rank in range(tp):
            qs = torch.arange(rank * q * dim, (rank + 1) * q * dim, device=device)
            vs = torch.arange(rank * v * dim, (rank + 1) * v * dim, device=device)
            channels = torch.cat((qs, qs + hq * dim, vs + 2 * hq * dim))
            heads = torch.arange(rank * v, (rank + 1) * v, device=device)
            columns = torch.cat((heads, heads + hv))
            # Use production's SD storage represented as a [pool, C, 3] view.
            c = conv[:, channels].transpose(1, 2).contiguous().transpose(1, 2)
            # Preserve a padded pool stride rather than requiring dense states.
            s = torch.empty(pool * 2, v, dim, dim, device=device)[::2]
            s.copy_(state[:, heads])
            raw = qkv[:, channels].contiguous()
            out = torch.empty(rows, v, dim, device=device, dtype=torch.float16)
            conv_out = torch.empty_like(raw)
            partial = torch.empty(rows * 2 * v * 160, device=device)
            run = getattr(torch.ops, f"_C_flashinfer_gdn_sm70_h2560_q{q}_v{v}").run
            args = (
                x,
                weights[:, columns].contiguous(),
                raw,
                cw[channels],
                bias[channels],
                c,
                a[heads],
                dt[heads],
                s,
                indices,
                out,
                conv_out,
                partial,
            )
            parts.append(
                dict(
                    run=run,
                    args=args,
                    raw=raw,
                    channels=channels,
                    heads=heads,
                    state=s,
                    conv=c,
                    out=out,
                )
            )
        groups.append(parts)

    def launch():
        for parts in groups:
            for p in parts:
                p["run"](*p["args"])

    for _ in range(3):
        launch()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for parts in groups:
        for p in parts:
            p["conv"].copy_(conv[:, p["channels"]])
            p["state"].copy_(state[:, p["heads"]])
    for step in range(8):
        x.normal_().mul_(0.1)
        qkv.normal_().mul_(0.1)
        indices.copy_(torch.randperm(pool, device=device)[:rows])
        if step % 3 == 1:
            indices[-1] = -1
        for parts in groups:
            for p in parts:
                p["raw"].copy_(qkv[:, p["channels"]])
                p["out"].fill_(torch.nan)
        graph.replay()
        full = groups[0][0]
        for parts in groups[1:]:
            for p in parts:
                torch.testing.assert_close(
                    p["conv"], full["conv"][:, p["channels"]], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    p["state"], full["state"][:, p["heads"]], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    p["out"], full["out"][:, p["heads"]], atol=0, rtol=0
                )
                assert torch.isfinite(p["out"]).all()
