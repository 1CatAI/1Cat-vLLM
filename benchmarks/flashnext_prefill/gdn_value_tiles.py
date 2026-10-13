# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen value-column tiling of the existing SM70 TileLang prefill core.

Synthetic normalized operands use Flash-Next TP4 dimensions. The benchmark
captures the ordinary wrapper's native launch, changing only block_DV in its
existing factory. No serving route is installed. Timings include restoring
the indexed FP32 state before each call, but exclude operand preparation.
"""

import argparse
import importlib
import json
from pathlib import Path

import torch
from moe_tiles import measure, relative_l2

from flash_qla.ops.gated_delta_rule.chunk.hopper.kkt_solve import kkt_solve
from flash_qla.ops.utils import chunk_local_cumsum
from flash_qla.utils import prepare_chunk_indices, prepare_chunk_offsets


def prepare(rows):
    torch.manual_seed(rows)
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(1, rows, 4, 128, device="cuda"), dim=-1
        ).half()
        for _ in range(2)
    ]
    v = torch.randn(1, rows, 12, 128, device="cuda", dtype=torch.half) * 0.1
    beta = torch.rand(1, rows, 12, device="cuda")
    gate = -torch.rand_like(beta) * 0.08
    cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    indices = prepare_chunk_indices(cu, 64)
    offsets, _ = prepare_chunk_offsets(cu, 64)
    g = chunk_local_cumsum(gate, 64, cu_seqlens=cu, chunk_indices=indices)
    a = kkt_solve(k, beta, cu_seqlens=cu, chunk_indices=indices)
    seed = torch.randn(4, 12, 128, 128, device="cuda") * 0.01
    return {
        "q": q,
        "k": k,
        "v": v,
        "a": a,
        "g": g,
        "b": beta,
        "cu_seqlens": cu,
        "chunk_offsets": offsets,
        "state_indices": torch.tensor([2], device="cuda", dtype=torch.int32),
        "has_initial_state": torch.tensor([True], device="cuda"),
        "state_layout_vlk": True,
        "sm70_original": True,
        "inplace_final_state": True,
    }, seed


def capture_launch(operands, seed, width):
    module = importlib.import_module(
        "flash_qla.ops.gated_delta_rule.chunk.hopper.fused_fwd"
    )
    original = module.tilelang_fused_chunk_gdr_fwd_sm70
    captured = []

    def factory(*args, **kwargs):
        kwargs["block_DV"] = width
        kernel = original(*args, **kwargs)

        def launch(*tensors):
            captured.append((kernel, tensors))
            return kernel(*tensors)

        return launch

    state = seed.clone()
    try:
        module.tilelang_fused_chunk_gdr_fwd_sm70 = factory
        output, _, _ = module.fused_gdr_fwd(**operands, initial_state=state)
    finally:
        module.tilelang_fused_chunk_gdr_fwd_sm70 = original
    assert len(captured) == 1
    kernel, tensors = captured[0]

    def run():
        state.copy_(seed)
        kernel(*tensors)

    return run, output, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[7344, 5347])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "synthetic_screen_only": True,
        "maximum_row_relative_l2_limit": 0.003,
        "state_reset_included": True,
        "cases": [],
    }
    for rows in args.tokens:
        operands, seed = prepare(rows)
        base, base_out, base_state = capture_launch(operands, seed, 32)
        candidate, candidate_out, candidate_state = capture_launch(operands, seed, 16)
        base()
        candidate()
        errors = {
            "output_rel_l2": relative_l2(candidate_out, base_out),
            "state_rel_l2": relative_l2(candidate_state[2], base_state[2]),
            "finite": bool(torch.isfinite(candidate_out).all())
            and bool(torch.isfinite(candidate_state).all()),
            "untouched_state_exact": torch.equal(candidate_state[:2], seed[:2])
            and torch.equal(candidate_state[3:], seed[3:]),
        }
        assert errors["finite"] and errors["untouched_state_exact"], errors
        assert max(errors["output_rel_l2"], errors["state_rel_l2"]) <= 0.003, errors
        row = {
            "tokens": rows,
            "baseline_dv32": measure(base, 10),
            "candidate_dv16": measure(candidate, 10),
            "baseline_repeat": measure(base, 10),
            "errors": errors,
        }
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
