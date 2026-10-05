# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All 48 checkpoint shared-expert chains, five launches vs one launch.

Includes gate/up/SiLU/down/output gate, excludes identical TP transport.
The candidate uses a residency-checked cooperative barrier and fixed-order
down partial sums. This screen is not full-model numerical admission.
"""

import argparse
import hashlib
import json
import statistics
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from benchmarks.kernels.benchmark_sm70_mtp_fp32_dense import capture, elapsed


def load_chain(model, rank):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = sorted(
        n
        for n in index
        if "mtp" not in n and n.endswith(".mlp.shared_expert.gate_proj.weight")
    )
    assert len(names) == 48
    chains = []
    with ExitStack() as stack:
        files = {}

        def get(name):
            file = index[name]
            if file not in files:
                files[file] = stack.enter_context(
                    safe_open(model / file, framework="pt")
                )
            return files[file].get_tensor(name)

        for name in names:
            prefix = name.removesuffix("shared_expert.gate_proj.weight")
            lo, hi = rank * 160, (rank + 1) * 160
            up = (
                torch.cat(
                    (
                        get(name)[lo:hi],
                        get(prefix + "shared_expert.up_proj.weight")[lo:hi],
                    )
                )
                .cuda()
                .half()
                .contiguous()
            )
            down = (
                get(prefix + "shared_expert.down_proj.weight")[:, lo:hi]
                .cuda()
                .half()
                .contiguous()
            )
            gate = get(prefix + "shared_expert_gate.weight").cuda().half().contiguous()
            packed_up = (
                up.reshape(10, 32, 160, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
            )
            packed_down = (
                down.reshape(80, 32, 10, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
            )
            chains.append((up, down, gate, packed_up, packed_down))
    return chains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    chains = load_chain(args.model, args.rank)
    xs = torch.randn(48, 5, 2560, device="cuda", dtype=torch.float16)
    up_outputs = [xs.new_empty(5, 160) for _ in chains]
    partials = [xs.new_empty(8, 5, 320, dtype=torch.float32) for _ in chains]
    gate_logits = [xs.new_empty(5, dtype=torch.float32) for _ in chains]
    outputs = {
        arm: [xs.new_empty(5, 2560) for _ in chains] for arm in ("control", "fused")
    }

    def control():
        for i, (_, down, gate, packed_up, _) in enumerate(chains):
            torch.ops._C.qwen38_shared_up_batch_fp32_sm70_out(
                up_outputs[i], partials[i], xs[i], packed_up
            )
            g = torch.nn.functional.linear(xs[i], gate)
            y = torch.nn.functional.linear(up_outputs[i], down)
            torch.ops._C.qwen38_shared_gate_mul_sm70_out(outputs["control"][i], g, y)

    def fused():
        for i, (_, _, gate, packed_up, packed_down) in enumerate(chains):
            torch.ops._C.qwen38_shared_chain_sm70_out(
                outputs["fused"][i],
                partials[i],
                gate_logits[i],
                xs[i],
                packed_up,
                packed_down,
                gate,
            )

    graphs = {"control": capture(control), "fused": capture(fused)}
    checks = []
    for scale in (0.0, 0.03, 1.0, 3.0):
        xs.normal_(0, scale)
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        errors = []
        for i, (a, b) in enumerate(zip(outputs["control"], outputs["fused"])):
            assert torch.isfinite(b).all(), (scale, i)
            errors.append((a - b).abs().max().item())
        checks.append(dict(scale=scale, max_error=max(errors)))
    trials = {arm: [] for arm in graphs}
    for trial in range(7):
        order = ("control", "fused") if trial % 2 else ("fused", "control")
        for arm in order:
            trials[arm].append(elapsed(graphs[arm]))
    result = dict(
        model_admission=False,
        layers=48,
        rows=5,
        ctas=80,
        warps_per_cta=8,
        cooperative_residency_checked=True,
        floating_point_atomics=False,
        weight_bytes=48 * (320 * 2560 + 2560 * 160 + 2560) * 2,
        checks=checks,
        samples_ms=trials,
        median_ms={arm: statistics.median(s) for arm, s in trials.items()},
        native_sha256=hashlib.sha256(Path("vllm/_C.abi3.so").read_bytes()).hexdigest(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
