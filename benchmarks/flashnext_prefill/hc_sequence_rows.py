# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen row-owned HC state with ReduceScatter and transient weight gathers.

This is a complete HC stage screen, not a serving path. The control includes
FP16 AllReduce, combine/norm, both projections and both HC AllGathers. The
candidate includes ReduceScatter, local combine/norm, both weight gathers,
both projections, all layout copies and the final block AllGather. Injection
and residual state remain local. Before execution, the finite paired maximum
row-relative L2 limit is 0.001 separately for hidden, block and injection.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from collectives import measure
from hc_pipeline import sequential
from moe_tiles import relative_l2

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.models.qwen4_exp.nvidia.ops.hc import (
    hc_combine_norm,
    hc_gate_mix_gathered,
    hc_packet_silu,
)

ORDER = (0, 1, 2, 3)


def fp32_reference(nccl, hidden, injection, partial, weight, down, up):
    """Independent FP32 arithmetic with explicit original FP16 boundaries."""
    rows = hidden.shape[0]
    pieces = partial.new_empty((4, *partial.shape))
    nccl.all_gather(pieces, partial)
    reduced = pieces.float().sum(dim=0).half()
    gate = 2 * torch.sigmoid(injection.float() / 4)
    updated = (
        hidden.float().reshape(rows, 4, 2560)
        + reduced.float()[:, None] * gate[..., None]
    ).half()
    h = updated.float()
    norm = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + 1e-6)
    norm = (norm + norm * weight.float().reshape(4, 2560)).half().flatten(1)
    full_down = down.new_empty((4 * down.shape[0], down.shape[1]))
    full_up = up.new_empty((4 * up.shape[0], up.shape[1]))
    nccl.all_gather(full_down, down)
    nccl.all_gather(full_up, up)
    projected = norm.float() @ full_down.float().T
    logits = projected[:, 3 * 88 + 80 : 3 * 88 + 84]
    lora = (
        projected.half().reshape(rows, 4, 88)[:, :, :80].reshape(rows, 320).float() / 4
    )
    lora = (lora * torch.sigmoid(lora)).half()
    gates = (lora.float() @ full_up.float().T).half()
    gates = gates.reshape(rows, 4, 4, 640).permute(0, 2, 1, 3).reshape(rows, 4, 2560)
    product = torch.sigmoid(gates.float()) * norm.float().reshape(rows, 4, 2560)
    block = (product[:, 0] + product[:, 1] + product[:, 2] + product[:, 3]) / 4
    return updated.flatten(1), block, logits


def reference_errors(actual, reference):
    hidden, block, logits = actual
    ref_hidden, ref_block, ref_logits = reference
    scale = ref_logits.float().square().mean(dim=0).sqrt().clamp_min(1e-12)
    errors = {
        "hidden_row_l2": relative_l2(hidden, ref_hidden),
        "block_row_l2": relative_l2(block, ref_block),
        "logit_aggregate_l2": float(
            (logits.float() - ref_logits.float()).norm()
            / ref_logits.float().norm().clamp_min(1e-12)
        ),
        "logit_max_column_rms_error": float(
            ((logits.float() - ref_logits.float()).abs() / scale).max()
        ),
        "finite": all(bool(torch.isfinite(v).all()) for v in actual),
        "injection_factor_max_abs_error": float(
            (
                2 * torch.sigmoid(logits.float() / 4)
                - 2 * torch.sigmoid(ref_logits.float() / 4)
            )
            .abs()
            .max()
        ),
    }
    errors["passed"] = (
        errors["finite"]
        and max(
            errors["hidden_row_l2"],
            errors["block_row_l2"],
            errors["logit_aggregate_l2"],
        )
        <= 0.001
        and errors["logit_max_column_rms_error"] <= 4 * torch.finfo(torch.float16).eps
        and errors["injection_factor_max_abs_error"] <= 0.0005
    )
    return errors


def local_project(nccl, x, down, up):
    full_down = down.new_empty((4 * down.shape[0], down.shape[1]))
    nccl.all_gather(full_down, down)
    local = torch.mm(x, full_down.T)
    del full_down
    packet = local.reshape(-1, 4, 88).transpose(0, 1).contiguous().flatten(0, 1)
    del local
    lora, injection = hc_packet_silu(packet, ORDER, 4)
    del packet
    full_up = up.new_empty((4 * up.shape[0], up.shape[1]))
    nccl.all_gather(full_up, up)
    gate = torch.mm(lora, full_up.T)
    del lora, full_up
    gate = gate.reshape(-1, 4, 2560).transpose(0, 1).contiguous().flatten(0, 1)
    mixed = hc_gate_mix_gathered(x, gate, ORDER)
    del gate
    gathered = mixed.new_empty((4 * x.shape[0], 2560))
    nccl.all_gather(gathered, mixed)
    return gathered, injection


def peak(fn):
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    before = torch.accelerator.memory_allocated()
    result = fn()
    torch.accelerator.synchronize()
    growth = torch.accelerator.max_memory_allocated() - before
    del result
    return growth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Screen FP32 ReduceScatter against independent arithmetic",
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[7344, 8976, 11424])
    parser.add_argument("--drift-steps", type=int, default=0)
    parser.add_argument("--reduced-precision", action="store_true")
    parser.add_argument(
        "--drift-communication", choices=("fp16", "fp32"), default="fp16"
    )
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    nccl = PyNcclCommunicator(dist.group.WORLD, rank)
    assert not nccl.disabled
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        args.reduced_precision
    )
    report = {"cases": [], "paired_max_row_l2_limit": 0.001}
    passed = True
    try:
        for rows in args.tokens:
            step = rows // 4
            span = slice(rank * step, (rank + 1) * step)
            torch.manual_seed(rows)
            hidden = torch.randn(rows, 10240, device="cuda", dtype=torch.half) * 0.3
            injection = torch.randn(rows, 4, device="cuda", dtype=torch.half) * 0.1
            weight = torch.randn(10240, device="cuda", dtype=torch.half) * 0.01
            local_hidden, local_injection = (
                hidden[span].clone(),
                injection[span].clone(),
            )
            torch.manual_seed(100 + rank)
            partial = torch.randn(rows, 2560, device="cuda", dtype=torch.half) * 0.03
            down = torch.randn(88, 10240, device="cuda", dtype=torch.half) * 0.01
            up = torch.randn(2560, 320, device="cuda", dtype=torch.half) * 0.01

            def control(
                hidden=hidden,
                injection=injection,
                partial=partial,
                weight=weight,
                down=down,
                up=up,
            ):
                reduced = torch.empty_like(partial)
                nccl.all_reduce(partial, reduced)
                updated, normalized = hc_combine_norm(
                    hidden, reduced, injection, weight, 1e-6, 4
                )
                block, logits = sequential(nccl, normalized, down, up, rank)
                return updated, block, logits

            def candidate(
                hidden=local_hidden,
                injection=local_injection,
                partial=partial,
                weight=weight,
                down=down,
                up=up,
                step=step,
            ):
                reduced = partial.new_empty((step, 2560))
                nccl.reduce_scatter(reduced, partial)
                updated, normalized = hc_combine_norm(
                    hidden, reduced, injection, weight, 1e-6, 4
                )
                block, logits = local_project(nccl, normalized, down, up)
                return updated, block, logits

            expected, actual = control(), candidate()
            targets = (expected[0][span], expected[1], expected[2][span])
            errors = [relative_l2(a, b) for a, b in zip(actual, targets)]
            checks = [None] * 4
            finite = all(bool(torch.isfinite(a).all()) for a in actual)
            dist.all_gather_object(checks, (errors, finite))
            row = {
                "tokens": rows,
                "max_rank_errors": [max(e[i] for e, _ in checks) for i in range(3)],
                "numerical_pass": all(max(e) <= 0.001 and f for e, f in checks),
                "control": measure(control),
                "sequence_rows": measure(candidate),
                "control_repeat": measure(control),
                "control_temporary_peak_bytes": peak(control),
                "candidate_temporary_peak_bytes": peak(candidate),
                "hidden_state_bytes_control": hidden.numel() * hidden.element_size(),
                "hidden_state_bytes_candidate": local_hidden.numel()
                * local_hidden.element_size(),
            }
            if args.reference:
                # This changes the reduction arithmetic rather than relaxing
                # the earlier failed paired near-zero row-ratio screen.
                def fp32_candidate(
                    hidden=local_hidden,
                    injection=local_injection,
                    partial=partial,
                    weight=weight,
                    down=down,
                    up=up,
                    step=step,
                ):
                    source = partial.float()
                    reduced = source.new_empty((step, 2560))
                    nccl.reduce_scatter(reduced, source)
                    reduced = reduced.half()
                    del source
                    updated, normalized = hc_combine_norm(
                        hidden, reduced, injection, weight, 1e-6, 4
                    )
                    block, logits = local_project(nccl, normalized, down, up)
                    return updated, block, logits

                reference = fp32_reference(
                    nccl, hidden, injection, partial, weight, down, up
                )
                control_check = reference_errors(expected, reference)
                ref_local = (reference[0][span], reference[1], reference[2][span])
                fp16_check = reference_errors(actual, ref_local)
                fp32_check = reference_errors(fp32_candidate(), ref_local)
                all_checks = [None] * 4
                dist.all_gather_object(
                    all_checks, (control_check, fp16_check, fp32_check)
                )
                row["independent_reference_checks"] = all_checks
                row["fp32_reduce_scatter"] = measure(fp32_candidate)
                row["fp32_temporary_peak_bytes"] = peak(fp32_candidate)
                row["reference_gate"] = {
                    "hidden_and_block_max_row_relative_l2": 0.001,
                    "logit_aggregate_relative_l2": 0.001,
                    "logit_max_column_rms_normalized_error": 4
                    * torch.finfo(torch.float16).eps,
                    "injection_factor_max_abs_error": 0.0005,
                }
                row["fp32_reference_pass"] = all(
                    c[0]["passed"] and c[2]["passed"] for c in all_checks
                )
                passed &= row["fp32_reference_pass"]
                row["fp16_reference_pass"] = all(
                    c[0]["passed"] and c[1]["passed"] for c in all_checks
                )
                passed &= row["fp16_reference_pass"]
                if args.drift_steps:
                    ch, cb, ci = hidden, torch.zeros_like(partial), injection
                    sh, sb, si = (
                        local_hidden,
                        torch.zeros_like(partial),
                        local_injection,
                    )
                    rh, rb, ri = hidden, torch.zeros_like(partial), injection
                    snapshots = []
                    for iteration in range(1, args.drift_steps + 1):
                        torch.manual_seed(9700 + rank + 4 * iteration)
                        noise = torch.randn_like(partial) * 0.03
                        ch, cb, ci = control(
                            hidden=ch, injection=ci, partial=(noise + cb * 0.025).half()
                        )
                        drift_candidate = (
                            fp32_candidate
                            if args.drift_communication == "fp32"
                            else candidate
                        )
                        sh, sb, si = drift_candidate(
                            hidden=sh, injection=si, partial=(noise + sb * 0.025).half()
                        )
                        rh, rb, ri = fp32_reference(
                            nccl,
                            rh,
                            ri.half(),
                            (noise + rb * 0.025).half(),
                            weight,
                            down,
                            up,
                        )
                        if iteration in (1, 8, 16, 32, args.drift_steps):
                            c = reference_errors((ch, cb, ci), (rh, rb, ri))
                            s = reference_errors((sh, sb, si), (rh[span], rb, ri[span]))
                            checks = [None] * 4
                            dist.all_gather_object(checks, (c, s))
                            drift_pass = all(
                                e["finite"]
                                and max(
                                    e["hidden_row_l2"],
                                    e["block_row_l2"],
                                    e["logit_aggregate_l2"],
                                )
                                <= 0.003
                                and e["injection_factor_max_abs_error"] <= 0.0005
                                for pair in checks
                                for e in pair
                            )
                            snapshots.append(
                                {
                                    "iteration": iteration,
                                    "all_rank_checks": checks,
                                    "passed": drift_pass,
                                }
                            )
                            passed &= drift_pass
                    row["drift"] = {
                        "steps": args.drift_steps,
                        "state_block_logit_l2_limit": 0.003,
                        "factor_abs_limit": 0.0005,
                        "snapshots": snapshots,
                    }
            else:
                passed &= row["numerical_pass"]
            report["cases"].append(row)
            if rank == 0:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    finally:
        torch.accelerator.synchronize()
        dist.destroy_process_group()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
