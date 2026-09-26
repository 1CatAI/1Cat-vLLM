# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired whole-input-chain graph benchmark, real weights, synthetic activations.

Measures all selected GDN layers consecutively, including QKVZ, b/a and output
layout. It is not a model benchmark. The installed source-built native op is
required; this benchmark never loads a private extension.
"""

import argparse
import hashlib
import json
import statistics
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
    _pack_gdn_input_weight,
    _qwen38_sm70_fp16_gdn_input,
)


def checkpoint_weights(model: Path, layers: list[int], rank: int):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    result = []
    with ExitStack() as stack:
        handles = {}
        for layer in layers:

            def read(projection, layer=layer):
                name = (
                    f"model.language_model.layers.{layer}.linear_attn."
                    f"{projection}.weight"
                )
                shard = index[name]
                if shard not in handles:
                    handles[shard] = stack.enter_context(
                        safe_open(model / shard, framework="pt", device="cpu")
                    )
                return handles[shard].get_tensor(name).half()

            q, k, v = read("in_proj_qkv").split((2048, 2048, 6144))
            z = read("in_proj_z")
            qkvz = torch.cat(
                (
                    q[rank * 512 : (rank + 1) * 512],
                    k[rank * 512 : (rank + 1) * 512],
                    v[rank * 1536 : (rank + 1) * 1536],
                    z[rank * 1536 : (rank + 1) * 1536],
                )
            ).cuda()
            ba = torch.cat(
                tuple(
                    read(p)[rank * 12 : (rank + 1) * 12]
                    for p in ("in_proj_b", "in_proj_a")
                )
            ).cuda()
            result.append((qkvz, ba))
    return result


def capture(xs, weights, packed):
    outputs = []

    def run():
        outputs.clear()
        for x, (q, b), (pq, pb) in zip(xs, weights, packed):
            outputs.append(_qwen38_sm70_fp16_gdn_input(x, q, b, pq, pb))

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph, outputs


def time_graph(graph, replays):
    for _ in range(3):
        graph.replay()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / replays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--layers", default="all")
    p.add_argument("--rank", type=int, choices=range(4), default=0)
    p.add_argument("--rows", default="2,4,8,16")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not envs.VLLM_SM70_QWEN38_GDN_INPUT_BATCH:
        p.error("Set VLLM_SM70_QWEN38_GDN_INPUT_BATCH=1 before startup")
    if not hasattr(torch.ops._C, "qwen38_gdn_input_batch_sm70_out"):
        p.error("Build/install this worktree's ordinary SM70 extension first")
    torch.set_num_threads(1)
    torch.manual_seed(20260926 + args.rank)
    layers = (
        [i for i in range(48) if i % 4 != 3]
        if args.layers == "all"
        else list(map(int, args.layers.split(",")))
    )
    weights = checkpoint_weights(args.model, layers, args.rank)
    packed = [tuple(_pack_gdn_input_weight(w) for w in pair) for pair in weights]
    rows = []
    report = dict(
        model=str(args.model),
        layers=layers,
        tp4_weight_rank=args.rank,
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        synthetic_activations=True,
        packed_bytes=sum(w.numel() * w.element_size() for pair in packed for w in pair),
        config_sha256=hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        results=rows,
    )
    for m in map(int, args.rows.split(",")):
        if not 2 <= m <= 16:
            p.error("This component benchmark covers M2..16")
        xs = [
            torch.randn(m, 2560, device="cuda", dtype=torch.float16) * 0.1
            for _ in weights
        ]
        base, expected = capture(xs, weights, [(None, None)] * len(weights))
        candidate, actual = capture(xs, weights, packed)
        checks = []
        for scale in (0.0, 0.001, 0.03, 0.1, 1.0, 3.0):
            for x in xs:
                x.normal_(0, scale)
            for outputs in actual:
                for output in outputs:
                    output.fill_(float("nan"))
            base.replay()
            candidate.replay()
            checks.append(
                [
                    [
                        int(
                            (a.view(torch.int16) != e.view(torch.int16)).count_nonzero()
                        )
                        for a, e in zip(aa, ee)
                    ]
                    for aa, ee in zip(actual, expected)
                ]
            )
        exact = not any(v for scale in checks for layer in scale for v in layer)
        samples = [[], []]
        if exact:
            for trial in range(7):
                for arm in (0, 1) if trial % 2 == 0 else (1, 0):
                    samples[arm].append(time_graph((base, candidate)[arm], 16))
        medians = [statistics.median(s) if s else None for s in samples]
        row = dict(
            rows=m,
            bit_exact=exact,
            checks=checks,
            medians_ms=medians,
            samples_ms=samples,
        )
        rows.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "checks"}), flush=True)
        if not exact:
            raise AssertionError(
                "GDN chain bitwise comparison failed; retained all mismatches"
            )
        del base, candidate, xs, expected, actual


if __name__ == "__main__":
    main()
