# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired gate-projection/conv/recurrent component screen, no engine launch.

Uses real checkpoint weights with synthetic changing hidden states. Both arms
include an identical input-refresh copy because production conv mutates QKV.
This is not a model-quality test or complete GDN layer timing.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

from benchmarks.kernels.flashinfer_sm70_gdn_conv import FusedGDN, build


def check_exclusive():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Reserve exactly one GPU")
    info = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            visible,
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    for row in info.splitlines():
        pid, _, name = row.partition(",")
        if pid.strip().isdigit() and int(pid) != os.getpid() and "snapd" not in name:
            raise RuntimeError(f"Foreign GPU owner: {row}")


def load_weights(model, layer=0, rank=0, tp=4):
    mapping = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    config = json.loads((model / "config.json").read_text())["text_config"]
    if (
        tp < 1
        or not 0 <= rank < tp
        or config["linear_num_key_heads"] % tp
        or config["linear_num_value_heads"] % tp
        or config["linear_key_head_dim"] != 128
        or config["linear_value_head_dim"] != 128
    ):
        raise ValueError("Require D128 and a valid evenly sharded TP geometry")
    prefix = f"model.language_model.layers.{layer}.linear_attn."

    def get(name, dtype=torch.float16):
        name = prefix + name
        with safe_open(model / mapping[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name).to(dtype)

    qfull = config["linear_num_key_heads"] * 128
    vfull = config["linear_num_value_heads"] * 128
    hq, hv = (
        config["linear_num_key_heads"] // tp,
        config["linear_num_value_heads"] // tp,
    )
    selection = torch.cat(
        [
            torch.arange(rank * hq * 128, (rank + 1) * hq * 128),
            torch.arange(qfull + rank * hq * 128, qfull + (rank + 1) * hq * 128),
            torch.arange(
                2 * qfull + rank * hv * 128, 2 * qfull + (rank + 1) * hv * 128
            ),
        ]
    )
    assert qfull * 2 + vfull == get("conv1d.weight").shape[0]
    wqkv = get("in_proj_qkv.weight")[selection].contiguous().cuda()
    ba = (
        torch.cat(
            [
                get("in_proj_b.weight")[rank * hv : (rank + 1) * hv],
                get("in_proj_a.weight")[rank * hv : (rank + 1) * hv],
            ]
        )
        .contiguous()
        .cuda()
    )
    conv = get("conv1d.weight").reshape(-1, 4)[selection].contiguous().cuda()
    A = get("A_log", torch.float32)[rank * hv : (rank + 1) * hv].contiguous().cuda()
    dt = get("dt_bias")[rank * hv : (rank + 1) * hv].contiguous().cuda()
    return config["hidden_size"], hq, hv, wqkv, ba, conv, A, dt


def capture(fn, repeats=1):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            fn()
    return graph


def error(actual, reference):
    a, b = actual.float(), reference.float()
    if not a.numel():
        return {"max_abs": 0.0, "relative_l2": 0.0, "finite": True, "exact": True}
    return {
        "max_abs": (a - b).abs().max().item(),
        "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-20)).item(),
        "finite": bool(torch.isfinite(a).all()),
        "exact": bool(torch.equal(actual, reference)),
    }


@torch.inference_mode()
def screen(rows, weights, args):
    from flash_qla.ops.gated_delta_rule.chunk.sm70 import fused_fwd as qla
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    hidden, hq, hv, wqkv, ba_weight, cw, A, dt = weights
    pool = rows + 3
    width = (2 * hq + hv) * 128
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
    raw = torch.empty(rows, width, device="cuda", dtype=torch.float16)
    raw.copy_(x @ wqkv.t())
    base_in, candidate_in = torch.empty_like(raw), torch.empty_like(raw)
    c0 = torch.randn(pool, 3, width, device="cuda", dtype=torch.float16).transpose(1, 2)
    s0 = torch.randn(pool, hv, 128, 128, device="cuda", dtype=torch.float32) * 0.01
    cb, cc, sb, sc = c0.clone(), c0.clone(), s0.clone(), s0.clone()
    indices = torch.arange(rows, device="cuda", dtype=torch.int32)
    bias = torch.empty(0, device="cuda", dtype=torch.float16)
    packed = ba_weight.t().contiguous()
    ba = torch.empty(rows, 2 * hv, device="cuda", dtype=torch.float16)
    b, a = (
        torch.empty(rows, hv, device="cuda", dtype=torch.float16),
        torch.empty(rows, hv, device="cuda", dtype=torch.float16),
    )
    out = torch.empty(rows, hv, 128, device="cuda", dtype=torch.float16)
    candidate = FusedGDN(rows, hq, hv, hidden=hidden)

    def baseline():
        base_in.copy_(raw)
        torch.mm(x, ba_weight.t(), out=ba)
        b.copy_(ba[:, :hv])
        a.copy_(ba[:, hv:])
        causal_conv1d_update(
            base_in,
            cb,
            cw,
            None,
            "silu",
            conv_state_indices=indices,
            validate_data=False,
        )
        return qla.gdn_decode_mixed_qkv_global_state_sm70(
            base_in, a, b, A, dt, sb, indices, out
        )

    def fused():
        candidate_in.copy_(raw)
        return candidate(x, packed, candidate_in, cw, bias, cc, A, dt, sc, indices)

    baseline()
    candidate_graph = capture(fused)
    checks = []
    for cycle in range(args.steps):
        x.normal_().mul_((0.25, 1.0, 3.0)[cycle % 3])
        raw.copy_(x @ wqkv.t())
        indices.copy_(torch.randperm(pool, device="cuda")[:rows])
        if cycle % 8 == 7:
            indices[-1] = -1
        cb.copy_(c0)
        cc.copy_(c0)
        sb.copy_(s0)
        sc.copy_(s0)
        baseline()
        fused()
        live = indices >= 0
        checks.append(
            {
                "cycle": cycle,
                "out": error(candidate.output[live], out[live]),
                "state": error(sc, sb),
                "conv_state": error(cc, cb),
                "conv_out": error(candidate.conv_out[live], base_in[live]),
            }
        )
        eager_out, eager_state, eager_conv = (
            candidate.output.clone(),
            sc.clone(),
            cc.clone(),
        )
        cc.copy_(c0)
        sc.copy_(s0)
        candidate.output.fill_(float("nan"))
        candidate.partial.fill_(float("nan"))
        candidate_graph.replay()
        torch.cuda.synchronize()
        for actual, ref in (
            (candidate.output, eager_out),
            (sc, eager_state),
            (cc, eager_conv),
        ):
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)
        # Advance a real recurrent history rather than always testing zero state.
        c0.copy_(cb)
        s0.copy_(sb)
    maxima = {
        part: {
            metric: max(c[part][metric] for c in checks)
            for metric in ("max_abs", "relative_l2")
        }
        for part in ("out", "state", "conv_out")
    }
    gate = (
        all(c[part]["finite"] for c in checks for part in ("out", "state", "conv_out"))
        and all(c["conv_state"]["exact"] for c in checks)
        and maxima["out"]["relative_l2"] < 5e-3
        and maxima["state"]["relative_l2"] < 5e-3
    )
    # Unlike the local comparisons above, neither arm is reset from the other
    # arm in this phase. This screens accumulation through independent FP32
    # recurrent histories, including graph replays and recycled request slots.
    cb.copy_(c0)
    cc.copy_(c0)
    sb.copy_(s0)
    sc.copy_(s0)
    history_maxima = {
        part: {"max_abs": 0.0, "relative_l2": 0.0}
        for part in ("out", "state", "conv_out")
    }
    history_gate, history_first_failure = True, None
    for cycle in range(args.history_steps):
        x.normal_().mul_((0.25, 1.0, 3.0)[cycle % 3])
        raw.copy_(x @ wqkv.t())
        indices.copy_(torch.randperm(pool, device="cuda")[:rows])
        if cycle % 8 == 7:
            indices[-1] = -1
        if cycle % 31 == 30:
            # A retired slot is cleared identically before being reused. Do
            # not synchronize either arm's other, independently evolved slots.
            slot = (cycle // 31) % pool
            cb[slot].zero_()
            cc[slot].zero_()
            sb[slot].zero_()
            sc[slot].zero_()
        baseline()
        candidate.output.fill_(float("nan"))
        candidate.partial.fill_(float("nan"))
        candidate_graph.replay()
        live = indices >= 0
        current = {
            "out": error(candidate.output[live], out[live]),
            "state": error(sc, sb),
            "conv_out": error(candidate.conv_out[live], base_in[live]),
        }
        for part, metrics in history_maxima.items():
            for metric in metrics:
                metrics[metric] = max(metrics[metric], current[part][metric])
        current_gate = (
            all(check["finite"] for check in current.values())
            and torch.equal(cc, cb)
            and bool((candidate.output[~live] == 0).all())
            and current["out"]["relative_l2"] < 5e-3
            and current["state"]["relative_l2"] < 5e-3
        )
        if not current_gate and history_first_failure is None:
            history_first_failure = {"cycle": cycle, "errors": current}
        history_gate = history_gate and current_gate
    gate = gate and history_gate
    record = {
        "rows": rows,
        "q_heads": hq,
        "v_heads": hv,
        "checks": checks,
        "maxima": maxima,
        "independent_history": {
            "steps": args.history_steps,
            "maxima": history_maxima,
            "gate": history_gate,
            "first_failure": history_first_failure,
        },
        "operator_gate": gate,
    }
    if gate:
        indices.copy_(torch.arange(rows, device="cuda"))
        graphs = [capture(baseline, args.calls), capture(fused, args.calls)]
        for _ in range(20):
            for graph in graphs:
                graph.replay()
        torch.cuda.synchronize()
        samples = [[], []]
        check_exclusive()
        for repeat in range(args.samples):
            for i in [0, 1] if repeat % 2 == 0 else [1, 0]:
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                graphs[i].replay()
                end.record()
                end.synchronize()
                samples[i].append(start.elapsed_time(end) * 1000 / args.calls)
        check_exclusive()
        record.update(
            samples_us=samples, median_us=[statistics.median(s) for s in samples]
        )
    record["reference_qla_binary_sha256"] = hashlib.sha256(
        Path(qla._load_ext().__file__).read_bytes()
    ).hexdigest()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--history-steps", type=int, default=256)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--calls", type=int, default=30)
    args = parser.parse_args()
    if min(args.steps, args.history_steps, args.samples, args.calls) <= 0:
        parser.error("Step, sample and call counts must be positive")
    torch.manual_seed(20260906)
    check_exclusive()
    weights = load_weights(args.model, args.layer, args.rank, args.tp)
    build(*weights[:3])
    for rows in args.rows:
        print(json.dumps(screen(rows, weights, args)), flush=True)


if __name__ == "__main__":
    main()
