# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate HC projection association using a retained failed chain input.

Replays actual weights and all 96 HC modules without IPC or collectives.
Attention/MoE outputs remain fixed external inputs. This is a numerical
diagnostic, not distributed performance, model quality, or a runtime policy.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from benchmarks.kernels.benchmark_sm70_flashinfer_gdn_conv import check_exclusive
from benchmarks.kernels.benchmark_sm70_hc_tp4 import load_weights
from vllm.models.qwen4_exp.nvidia.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)


def difference(actual, expected):
    delta = actual.float() - expected.float()
    limit = 3e-3 + 3e-3 * expected.float().abs()
    return {
        "changed": int((actual != expected).sum()),
        "outside_envelope": int((delta.abs() > limit).sum()),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-30)),
        "finite": bool(torch.isfinite(actual).all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality-inputs", type=int, default=16)
    args = parser.parse_args()
    if args.out.exists() or args.quality_inputs < 1:
        parser.error("Require a fresh output and positive quality-input count")
    check_exclusive()
    torch.set_num_threads(4)
    retained = torch.load(args.failure, weights_only=True, map_location="cpu")
    initial = retained["initial"].cuda()
    cores = retained["external_core_outputs"].cuda()
    weights = load_weights(args.model)
    assert len(weights) == len(cores) == 96
    mapping = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    norms = []
    for i in range(96):
        role = ("attn", "mlp")[i % 2]
        prefix = f"model.language_model.layers.{i // 2}.{role}_hyper_connection."
        name = prefix + "hc_norm.weight"
        with safe_open(args.model / mapping[name], framework="pt", device="cpu") as f:
            norms.append(f.get_tensor(name).half().cuda())
    up_shards = [
        [
            up.view(4, 2560, 320)[:, r * 640 : (r + 1) * 640]
            .reshape(2560, 320)
            .contiguous()
            for r in range(4)
        ]
        for _, up in weights
    ]

    def project(x, i, split_down, split_up):
        down, up = weights[i]
        if split_down:
            pieces = [
                torch.nn.functional.linear(x, down[r * 80 : r * 80 + 88])
                for r in range(4)
            ]
            raw = torch.cat([p[:, :80] for p in pieces], dim=-1)
            injection = pieces[3][:, 80:84]
        else:
            packed = torch.nn.functional.linear(x, down)
            raw, injection = packed[:, :320], packed[:, 320:324]
        lora = hc_silu(raw, 4)
        if split_up:
            pieces = [
                torch.nn.functional.linear(lora, w).view(len(x), 4, 640)
                for w in up_shards[i]
            ]
            gate = torch.cat(pieces, dim=-1).reshape(len(x), 10240)
        else:
            gate = torch.nn.functional.linear(lora, up)
        return hc_gate_mix(x, gate, 4), injection

    def chain(split_down, split_up):
        collected = []
        state, injection = initial, None
        for i in range(96):
            if i == 2:
                state = hc_combine(state, cores[i - 1], injection, 4)
            if i in (0, 2):
                xn = grouped_gemma_rmsnorm(state, norms[i], 1e-6, 4)
            else:
                state, xn = hc_combine_norm(
                    state, cores[i - 1], injection, norms[i], 1e-6, 4
                )
            block, injection = project(xn, i, split_down, split_up)
            collected.append((state, xn, block, injection))
        return collected

    modes = {
        "reference": (False, False),
        "both_sharded": (True, True),
        "only_up_sharded": (False, True),
        "only_down_sharded": (True, False),
    }
    graphs, outputs = {}, {}
    for name, split in modes.items():
        for _ in range(2):
            chain(*split)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = chain(*split)
        graphs[name] = graph

    results = []
    generator = torch.Generator(device="cuda").manual_seed(20260905)
    for sample in range(args.quality_inputs):
        check_exclusive()
        if sample:
            initial.normal_(generator=generator)
            cores.normal_(generator=generator)
        for graph in graphs.values():
            graph.replay()
        torch.accelerator.synchronize()
        if sample == 0:
            index = retained["hc_module"]
            # A new ablation must first reproduce the retained negative result.
            assert torch.equal(
                outputs["reference"][index][1].cpu(), retained["reference"]
            )
            assert torch.equal(
                outputs["both_sharded"][index][1].cpu(), retained["candidate"]
            )
        record = {"input": sample, "arms": {}}
        for name in modes:
            if name == "reference":
                continue
            first_failure = None
            changed = 0
            max_error = 0.0
            for i, (a, b) in enumerate(zip(outputs[name], outputs["reference"])):
                for field, actual, expected in zip(
                    ("state", "normalized", "block", "injection"), a, b
                ):
                    stats = difference(actual, expected)
                    changed += stats["changed"]
                    max_error = max(max_error, stats["max_abs"])
                    if first_failure is None and (
                        not stats["finite"] or stats["outside_envelope"]
                    ):
                        first_failure = {"module": i, "field": field, **stats}
            record["arms"][name] = {
                "first_failure": first_failure,
                "total_changed": changed,
                "max_abs": max_error,
            }
        results.append(record)
        print(json.dumps(record), flush=True)
    args.out.write_text(
        json.dumps(
            {
                "scope": "96-module arithmetic isolation, no IPC/performance admission",
                "rows": len(initial),
                "saved_four_rank_failure_reproduced": True,
                "envelope": {"atol": 3e-3, "rtol": 3e-3},
                "inputs": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
