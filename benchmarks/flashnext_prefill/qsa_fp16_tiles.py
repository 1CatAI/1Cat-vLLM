# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen the existing FP16 staged-QSA tile loop using saved real selections.

An FP8 main-model dump is decoded to FP16 for this algorithm screen. It is
not a draft-model workload qualification. Timings include the complete tiled
reader and merge, with the same preallocated workspace in every arm.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch
from moe_tiles import measure, relative_l2


def reference_row(g, row):
    req = int(g["requests"][row])
    logical = g["indices"][row].long()
    page = g["history"].shape[2]
    table = g["table"]
    if req < 0 or req >= table.shape[0]:
        return torch.zeros_like(g["query"][row])
    valid = (
        (logical >= 0)
        & (logical <= g["positions"][row])
        & (logical < g["lengths"][req])
        & (logical // page < table.shape[1])
    )
    physical = table[req, (logical // page).clamp(0, table.shape[1] - 1)].long()
    valid &= (physical >= 0) & (physical < g["history"].shape[0])
    physical, offset = physical[valid], logical[valid] % page
    if not physical.numel():
        return torch.zeros_like(g["query"][row])
    keys = g["history"][physical, 0, offset, 0].float()
    values = g["history"][physical, 1, offset, 0].float()
    scores = g["query"][row].float() @ keys.T / 16
    out = (scores.softmax(dim=-1) @ values).half()
    if g["gate"] is not None:
        out = (out.float() * g["gate"][row].float().sigmoid()).half()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--tiles", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = importlib.import_module("vllm._sm70_qsa_device_C")
    native = Path(module.__file__)
    saved = torch.load(args.dump, map_location="cpu", weights_only=True)
    source_dtype = str(saved["history"].dtype)
    if saved["history"].dtype == torch.uint8:
        blocks, _, page, _, _ = saved["history"].shape
        scales = saved["scales"].view(blocks, page, 2).permute(0, 2, 1)
        values = saved["history"].view(torch.float8_e4m3fn).float()
        saved["history"] = (values * scales[..., None, None]).half()
        saved["scales"] = torch.ones(2, dtype=torch.float32)
    assert saved["history"].dtype == torch.float16
    g = {k: v.cuda() if v is not None else None for k, v in saved.items()}
    if g["gate"] is not None:
        g["gate"] = g["gate"].reshape_as(g["query"])
    m = g["query"].shape[0]
    splits = (g["indices"].shape[1] + 63) // 64
    partial = torch.empty(64 * 6 * splits * 256, device="cuda")
    state = torch.empty(64 * 6 * splits * 2, device="cuda")
    out = torch.empty_like(g["query"])

    def run(tiles):
        torch.ops.vllm_sm70_qsa_device.run_tiled(
            g["query"],
            g["history"],
            g["scales"],
            g["indices"],
            g["table"],
            g["requests"],
            g["positions"],
            g["lengths"],
            out,
            g["gate"],
            partial,
            state,
            64,
            tiles,
        )

    run(1)
    control = out.clone()
    rows = sorted({0, 1, m // 2, m - 1})
    expected = torch.stack([reference_row(g, row) for row in rows])
    report = {
        "native": str(native),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "source_history_dtype": source_dtype,
        "query_shape": list(g["query"].shape),
        "workspace_bytes": (partial.numel() + state.numel()) * 4,
        "control_fp32_max_rel_l2": relative_l2(control[rows], expected),
        "cases": [],
    }
    assert report["control_fp32_max_rel_l2"] <= 0.003
    for index, tiles in enumerate([1, *args.tiles, 1]):
        run(tiles)
        errors = {
            "paired_max_rel_l2": relative_l2(out, control),
            "fp32_max_rel_l2": relative_l2(out[rows], expected),
        }
        row = {
            "tiles": tiles,
            "repeat_control": index == len(args.tiles) + 1,
            **measure(lambda tiles=tiles: run(tiles), args.iterations),
            **errors,
            "passed": bool(torch.isfinite(out).all()) and max(errors.values()) <= 0.003,
        }
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    raise SystemExit(0 if all(r["passed"] for r in report["cases"]) else 1)


if __name__ == "__main__":
    main()
