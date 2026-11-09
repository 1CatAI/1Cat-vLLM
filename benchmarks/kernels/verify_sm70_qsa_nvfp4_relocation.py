# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay actual Qwen3.8 Layer3 tensors across old/fixed page4 planners.

Inputs are the task's saved per-rank first-prefill observer dictionaries, not
model weights. Verify native replay against the captured model output, hold
attention arithmetic fixed while changing only the planner, then reverse the
intervention. This is an operator causal test, not whole-model acceptance.
"""

# Each closure is consumed synchronously within its current layout iteration.
# ruff: noqa: B023

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from flash_attn_v100 import flash_attn_interface as interface

from vllm.models.qwen4_exp.nvidia.ops import qsa


def load_frozen(path: Path):
    name = "frozen_qsa.flash_attn_v100_cuda"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen extension: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def difference(a: torch.Tensor, b: torch.Tensor) -> dict:
    assert a.shape == b.shape and a.dtype == b.dtype == torch.float16
    return {
        "elements": a.numel(),
        "bit_mismatches": int((a.view(torch.int16) != b.view(torch.int16)).sum()),
        "max_abs": float((a.float() - b.float()).abs().max()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def fingerprint(path: Path) -> dict:
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


@torch.inference_mode()
def replay(args, rank, frozen, fixed):
    records = [
        torch.load(
            args.capture_dir / f"{case}_rank{rank}_step00.pt", weights_only=True
        )["qsa"]
        for case in args.cases
    ]
    prefix = "layer03_prefill_"
    records = [
        {
            name.removeprefix(prefix): value
            for name, value in record.items()
            if name.startswith(prefix)
        }
        for record in records
    ]
    a, b = records
    inputs_equal = {
        name: torch.equal(
            a[name].contiguous().view(torch.uint8),
            b[name].contiguous().view(torch.uint8),
        )
        for name in ("hidden", "query", "key", "value", "gate", "selected", "positions")
    }
    assert all(inputs_equal.values()), inputs_equal
    positions = a["positions"].long()
    context = positions.numel()
    assert torch.equal(positions, torch.arange(context))
    assert context % 8 == 0, "This replay is for a complete grouped-prefill chunk"
    assert ((a["selected"] < 0) | (a["selected"] <= positions[:, None])).all()
    page_size = 400
    logical_blocks = (context + page_size - 1) // page_size
    outputs = {"native": [], "planner_only": [], "fixed": []}
    native_checks, arithmetic_checks, reversals = [], [], []

    for record in records:
        table_cpu = record["block_table"].long()
        used = table_cpu[0, :logical_blocks]
        assert used.unique().numel() == logical_blocks
        blocks = int(used.max()) + 1
        cache = torch.zeros(blocks, 2, page_size, 1, 256, dtype=torch.float16)
        physical_ids = table_cpu[0, positions // page_size]
        cache[physical_ids, 0, positions % page_size] = record["key"]
        cache[physical_ids, 1, positions % page_size] = record["value"]
        cache = cache.cuda()
        key, value = cache[:, 0], cache[:, 1]
        query, gate = record["query"].cuda(), record["gate"].cuda()
        selected, table = record["selected"].cuda(), record["block_table"].cuda()
        requests = torch.zeros(context, dtype=torch.int32, device="cuda")
        lens = torch.tensor([context], dtype=torch.int32, device="cuda")
        query_positions = positions.cuda()
        pp, mm, lengths, lse = qsa._qsa_grouped_page4_workspace(query)
        physical_k, physical_v = qsa._qsa_xqa_page4_physical_kv(query, key, value)
        output = torch.empty_like(query)

        def plan(extension):
            extension.grouped_sparse_page4_plan_fwd(
                selected,
                table,
                requests,
                query_positions,
                lens,
                pp,
                mm,
                lengths,
                page_size,
                key.stride(0) // (4 * 256),
                blocks,
            )

        def forward(extension):
            qsa._qsa_grouped_page4_forward(
                extension,
                query,
                physical_k,
                physical_v,
                output,
                pp,
                mm,
                lengths,
                lse,
                1 / 16,
                "auto",
                1.0,
                1.0,
            )
            qsa._qsa_output_gate(output, gate.view_as(query))
            return output.cpu()

        plan(frozen)
        native = forward(frozen)
        check = difference(native, record["output"])
        assert check["bit_mismatches"] == 0, (rank, "native vs capture", check)
        native_checks.append(check)
        outputs["native"].append(native)

        # The freshly rebuilt attention must preserve the frozen arithmetic
        # when it consumes exactly the same old physical plan.
        check = difference(forward(fixed), native)
        assert check["bit_mismatches"] == 0, (rank, "attention rebuild", check)
        arithmetic_checks.append(check)

        plan(fixed)
        planner_only = forward(frozen)
        fixed_output = forward(fixed)
        assert difference(planner_only, fixed_output)["bit_mismatches"] == 0
        outputs["planner_only"].append(planner_only)
        outputs["fixed"].append(fixed_output)

        plan(frozen)
        reversal = difference(forward(frozen), native)
        assert reversal["bit_mismatches"] == 0
        reversals.append(reversal)

    deltas = {name: difference(*pair) for name, pair in outputs.items()}
    assert deltas["native"]["bit_mismatches"] > 0, "Missing negative control"
    assert deltas["planner_only"]["bit_mismatches"] == 0, deltas
    assert deltas["fixed"]["bit_mismatches"] == 0, deltas
    return {
        "rank": rank,
        "context": context,
        "inputs_bitwise_equal": inputs_equal,
        "native_matches_captured": native_checks,
        "attention_rebuild_unchanged": arithmetic_checks,
        "allocation_differences": deltas,
        "native_reversal": reversals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--frozen-dso", type=Path, required=True)
    parser.add_argument("--cases", nargs=2, default=["repeat0", "repeat1"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--import-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    frozen = load_frozen(args.frozen_dso)
    fixed = interface.flash_attn_v100_cuda
    assert Path(frozen.__file__).resolve() != Path(fixed.__file__).resolve()
    report = {
        "scope": "actual NVFP4 Layer3 causal replay, not whole-model acceptance",
        "frozen": fingerprint(args.frozen_dso),
        "fixed": fingerprint(Path(fixed.__file__)),
        "ranks": [],
    }
    if not args.import_only:
        assert torch.cuda.get_device_capability() == (7, 0)
        for rank in range(4):
            result = replay(args, rank, frozen, fixed)
            report["ranks"].append(result)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    else:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print("Frozen and fixed extension imports passed", flush=True)


if __name__ == "__main__":
    main()
