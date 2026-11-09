# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-process canonical/one-bank graph timings with actual GGUF TP shards."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig, Sm70GgufConfig
from vllm.model_executor.layers.quantization.gguf_dense_hmma import (
    maybe_apply_shared_expert,
)
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
)


def projections(record, enabled):
    cfg = cast(
        VllmConfig,
        SimpleNamespace(
            kernel_config=KernelConfig(sm70_gguf=Sm70GgufConfig(small_m_hmma=enabled))
        ),
    )
    with set_current_vllm_config(cfg):
        return prepare_gguf_projections(
            [(s["raw"].cuda(), s["gtype"]) for s in record["segs"]],
            torch.float16,
            True,
            8,
        )


def time_graph(fn):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):
            fn()
    samples = []
    for _ in range(3):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        for _ in range(50):
            graph.replay()
        b.record()
        torch.accelerator.synchronize()
        samples.append(a.elapsed_time(b) * 1000 / 50 / 16)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("gates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--all-attention", action="store_true")
    args = parser.parse_args()
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    gates = torch.load(args.gates, map_location="cpu", weights_only=True)
    examples = {}
    for record in data:
        examples.setdefault(record["kind"], record)
    cases = []
    dense_records = (
        [r for r in data if r["kind"] == "attn_in"]
        if args.all_attention
        else [examples[kind] for kind in ("gdn_in", "gdn_out", "attn_in", "attn_out")]
    )
    for record in dense_records:
        kind = record["kind"]
        a, b = projections(record, False), projections(record, True)
        assert all(hasattr(p, "segment_format") for p in b)
        assert all(p.codes.dtype == torch.uint8 for p in b)
        for m in (5, 20):
            x = record["x"].repeat(3, 1)[:m].cuda()
            control = apply_prepared_gguf_projections(x, a)
            candidate = apply_prepared_gguf_projections(x, b)
            error = float(
                (candidate.float() - control.float()).norm() / control.float().norm()
            )
            assert error < 0.005
            rows = []
            for tag in ("A", "B", "B", "A"):
                timings = time_graph(
                    lambda x=x, bank=a if tag == "A" else b: (
                        apply_prepared_gguf_projections(x, bank)
                    )
                )
                rows.append(dict(tag=tag, graph_us=timings))
            cases.append(
                dict(
                    kind=kind,
                    layer=record["layer"],
                    shapes=[list(p.kernel.config.partition_weight_shape) for p in b],
                    m=m,
                    relative_l2=error,
                    abba=rows,
                    resident_bytes=sum(
                        p.numel() * p.element_size()
                        for layer in b
                        for p in layer.parameters()
                    ),
                )
            )
            print(cases[-1], flush=True)
    gu = examples["shexp_in"]
    down = next(
        d for d in data if d["kind"] == "shexp_down" and d["layer"] == gu["layer"]
    )
    ga, gb = projections(gu, False), projections(gu, True)
    da, db = projections(down, False), projections(down, True)
    layer = SimpleNamespace(
        expert_gate=SimpleNamespace(weight=gates[gu["layer"]].half().cuda()),
        gate_up_proj=SimpleNamespace(gguf_tm_projections=gb),
        down_proj=SimpleNamespace(gguf_tm_projections=db),
    )
    for m in (5, 20):
        x = gu["x"].repeat(3, 1)[:m].cuda()

        def control(x=x, m=m):
            gate_up = apply_prepared_gguf_projections(x, ga)
            h = torch.empty((m, 160), device=x.device, dtype=x.dtype)
            torch.ops._C.silu_and_mul(h, gate_up)
            out = apply_prepared_gguf_projections(h, da)
            return out * torch.sigmoid(torch.mm(x, layer.expert_gate.weight[:, None]))

        reference = control()
        candidate = maybe_apply_shared_expert(layer, x)
        assert candidate is not None
        error = float(
            (candidate.float() - reference.float()).norm() / reference.float().norm()
        )
        assert error < 0.005
        cases.append(
            dict(
                kind="shared",
                m=m,
                relative_l2=error,
                control=time_graph(control),
                candidate=time_graph(lambda x=x: maybe_apply_shared_expert(layer, x)),
            )
        )
        print(cases[-1], flush=True)
    args.output.write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
