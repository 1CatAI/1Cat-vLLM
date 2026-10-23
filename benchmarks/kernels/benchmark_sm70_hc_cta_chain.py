# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 M5 HC graph: combine/norm, down/gather/SiLU, up/mix/gather.

Compare production with 8/16-warp CTA reductions on all 96 checkpoint pairs.
Attention/MoE outputs are fixed external inputs. This is not model-round speed.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from benchmarks.kernels.benchmark_sm70_hc_tp4 import load_weights
from benchmarks.kernels.benchmark_sm70_mtp_fp32_dense import capture, elapsed
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.models.qwen4_exp.nvidia.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
)
from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import _pack_hc_batch_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fused-down-finish", action="store_true")
    args = parser.parse_args()
    arms = (0, -8, -16) if args.fused_down_finish else (0, 8, 16)
    rank = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    assert torch.cuda.get_device_capability() == (7, 0)
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    raw = load_weights(args.model)
    packed = [
        (_pack_hc_batch_weight(d, "down", rank), _pack_hc_batch_weight(u, "up", rank))
        for d, u in raw
    ]
    mapping = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    norms = []
    for i in range(48):
        for role in ("attn", "mlp"):
            prefix = f"model.language_model.layers.{i}.{role}_hyper_connection."
            name = prefix + "hc_norm.weight"
            with safe_open(args.model / mapping[name], framework="pt") as f:
                norms.append(f.get_tensor(name).half().cuda())
    initial = torch.randn(5, 10240, device="cuda", dtype=torch.half)
    cores = torch.randn(96, 5, 2560, device="cuda", dtype=torch.half).mul_(0.1)
    scratch = [
        (
            torch.empty(20, 5, 96, device="cuda", dtype=torch.float32),
            initial.new_empty(5, 320),
            initial.new_empty(5, 640),
            initial.new_empty(5, 2560),
            initial.new_empty(5, 4),
        )
        for _ in packed
    ]
    comm = CustomAllreduce(dist.group.WORLD, device=rank)
    assert comm.can_sm70_qwen38_hc_batch(initial)
    torch.accelerator.synchronize()
    dist.barrier()
    native = ops._custom_ar_owner_namespace()

    def pair(xn, d, u, work, warps):
        native.sm70_qwen38_hc_batch(
            comm._ptr, xn, d, u, *work, False, False, False, True, warps
        )

    def chain(warps, saved):
        state, injection = initial, None
        for i, ((d, u), norm, work) in enumerate(zip(packed, norms, scratch)):
            if i == 2:
                state = hc_combine(state, cores[i - 1], injection, 4)
            if i in (0, 2):
                xn = grouped_gemma_rmsnorm(state, norm, 1e-6, 4)
            else:
                state, xn = hc_combine_norm(
                    state, cores[i - 1], injection, norm, 1e-6, 4
                )
            pair(xn, d, u, work, warps)
            injection = work[-1]
            saved.extend((state, xn, work[-2], injection))

    outputs, graphs = {}, {}
    for warps in arms:
        saved = []
        dist.barrier()
        graphs[warps] = capture(partial(chain, warps, saved))
        # capture() warms three times then records once; retain only one chain.
        outputs[warps] = saved[-384:]
    checks = []
    for scale in (0.0, 0.03, 1.0, 3.0):
        initial.normal_(0, scale)
        graphs[0].replay()
        reference = [t.clone() for t in outputs[0]]
        for warps in arms[1:]:
            graphs[warps].replay()
            torch.accelerator.synchronize()
            assert all(torch.isfinite(t).all() for t in outputs[warps])
            delta = max(
                (a.float() - b.float()).abs().max().item()
                for a, b in zip(outputs[warps], reference)
            )
            checks.append(dict(warps=warps, scale=scale, max_abs=delta))
    trials = {w: [] for w in graphs}
    for trial in range(7):
        order = arms if trial % 2 else arms[::-1]
        for warps in order:
            dist.barrier()
            times = [None] * 4
            dist.all_gather_object(times, elapsed(graphs[warps]))
            trials[warps].append(times)
    # Change widths and flip packet epochs with a single HC pair. This checks
    # that extra CTA warps cannot publish duplicate packets or stale row tags.
    transitions = []
    for rows in (2, 5, 8, 16, 5, 2):
        x = initial.new_empty(rows, 10240).normal_(0, 0.1)
        work = (
            torch.empty(20, rows, 96, device="cuda", dtype=torch.float32),
            x.new_empty(rows, 320),
            x.new_empty(rows, 640),
            x.new_empty(rows, 2560),
            x.new_empty(rows, 4),
        )
        for warps in arms:
            dist.barrier()
            g = capture(partial(pair, x, *packed[0], work, warps))
            g.replay()
            if warps == 0:
                reference = [t.clone() for t in work[1:]]
            else:
                torch.accelerator.synchronize()
                assert all(torch.isfinite(t).all() for t in work[1:])
                transitions.append(
                    dict(
                        rows=rows,
                        warps=warps,
                        max_abs=max(
                            (a.float() - b.float()).abs().max().item()
                            for a, b in zip(work[1:], reference)
                        ),
                    )
                )
    all_checks = [None] * 4
    dist.all_gather_object(all_checks, dict(chains=checks, transitions=transitions))
    root = Path(__file__).resolve().parents[2]
    report = dict(
        complete=True,
        model_admission=False,
        source=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        native_sha256=hashlib.sha256(
            (root / "vllm/_C.abi3.so").read_bytes()
        ).hexdigest(),
        rows=5,
        pairs=96,
        exclusions="attention/MoE/PLE computation and final mixer",
        samples_ms=trials,
        rank_max_medians_ms={
            w: statistics.median(max(t) for t in s) for w, s in trials.items()
        },
        operator_checks=all_checks,
    )
    dist.barrier()
    comm.close()
    dist.destroy_process_group()
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ("operator_checks", "samples_ms")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
