# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen bounded expert-batch FP16 GEMMs against the quantized MoE chain.

Real TP4 weights and routes; both arms exclude routing and host DMA. The BLAS
arm includes dequantization, padded input gather, scatter and weighted unroute.
Its contiguous raw GPU banks are an optimistic source layout: production's
resident/cold slots would need additional gathering. This is not a new backend.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from moe_tiles import (
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    TOPK,
    artifact_identity,
    load_layer,
    measure,
    relative_l2,
)

from vllm.model_executor.layers.quantization.gguf import dequantize_gguf_rows
from vllm.model_executor.layers.quantization.gguf_expert_cache import (
    TurboMindPrefill,
    _unroute,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, quant_size
from vllm.triton_utils import tl, triton


@triton.jit
def _pad_gather(
    x,
    rows,
    offsets,
    padded,
    FIRST: tl.constexpr,
    PAD: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    pos = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    r, col = pos // K, pos % K
    begin = tl.load(offsets + FIRST + expert)
    end = tl.load(offsets + FIRST + expert + 1)
    token = tl.load(rows + begin + r, (r < PAD) & (begin + r < end), 0)
    value = tl.load(x + token * K + col, (r < PAD) & (begin + r < end), 0)
    tl.store(padded + expert * PAD * K + pos, value, r < PAD)


@triton.jit
def _scatter(
    padded,
    offsets,
    out,
    FIRST: tl.constexpr,
    PAD: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    pos = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    r, col = pos // K, pos % K
    begin = tl.load(offsets + FIRST + expert)
    end = tl.load(offsets + FIRST + expert + 1)
    valid = (r < PAD) & (begin + r < end)
    value = tl.load(padded + expert * PAD * K + pos, valid, 0)
    tl.store(out + (begin + r) * K + col, value, valid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=7344)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info, tables, banks, raw, left = load_layer(GGUFReader(args.model), args.layer)
    runner = TurboMindPrefill.__new__(TurboMindPrefill)
    runner.cache = SimpleNamespace(experts=EXPERTS, prefill_tokens=4096)
    runner.expert_ids = torch.arange(EXPERTS + 1, dtype=torch.int32, device="cuda")
    runner.tuned = set()
    runner._tune(
        args.layer,
        info,
        tables,
        torch.zeros(8192, HIDDEN, device="cuda", dtype=torch.half),
    )
    torch.manual_seed(args.tokens)
    x = (torch.randn(args.tokens, HIDDEN, device="cuda") * 0.5).half()
    scores, ids = torch.randn(args.tokens, EXPERTS, device="cuda").topk(TOPK, dim=1)
    weights = scores.softmax(dim=1)
    route = runner._route(ids, 0, EXPERTS)
    offsets, _, rows, _, inverse, _ = route
    pad = triton.cdiv(int((offsets[1:] - offsets[:-1]).max()), 32) * 32
    subroutes = [
        (
            t,
            min(t + 4096, args.tokens),
            runner._subroute(route, t, min(t + 4096, args.tokens)),
        )
        for t in range(0, args.tokens, 4096)
    ]
    control = torch.empty_like(x)

    def baseline():
        hidden = runner._gate_up(info, tables, x, route, False)
        for first, last, sub in subroutes:
            runner._down(
                info,
                tables,
                hidden,
                sub,
                weights[first:last],
                control[first:last],
                False,
            )

    baseline()
    raw_gu = torch.cat([torch.from_numpy(raw[p]).cuda() for p in ("gate", "up")], 1)
    raw_down = torch.from_numpy(raw["down"]).cuda()
    block, size = quant_size(info["down"][0])
    down_k = raw_down.shape[-1] // size * block
    routed = torch.empty(args.tokens * TOPK, HIDDEN, device="cuda", dtype=torch.half)
    output = torch.empty_like(x)
    report = {
        "artifact": artifact_identity(),
        "layer": args.layer,
        "tokens": args.tokens,
        "padded_rows_per_expert": pad,
        "control": measure(baseline, 3),
        "cases": [],
    }
    for count in (32, 64, 128):
        padded = torch.empty(count, pad, HIDDEN, device="cuda", dtype=torch.half)

        def candidate(count=count, padded=padded):
            for first in range(0, EXPERTS, count):
                gu = dequantize_gguf_rows(
                    raw_gu[first : first + count].flatten(0, 1),
                    info["gate"][0],
                    HIDDEN,
                    torch.half,
                ).view(count, 2 * INTERMEDIATE, HIDDEN)
                dn = dequantize_gguf_rows(
                    raw_down[first : first + count].flatten(0, 1),
                    info["down"][0],
                    down_k,
                    torch.half,
                ).view(count, HIDDEN, down_k)
                dn = dn[:, :, left : left + INTERMEDIATE]
                grid = (count, triton.cdiv(pad * HIDDEN, 2048))
                _pad_gather[grid](x, rows, offsets, padded, first, pad, HIDDEN, 2048)
                projected = torch.bmm(
                    padded, gu.transpose(1, 2), out_dtype=torch.float32
                )
                hidden = (
                    torch.nn.functional.silu(projected[:, :, :INTERMEDIATE])
                    * projected[:, :, INTERMEDIATE:]
                ).half()
                down = torch.bmm(hidden, dn.transpose(1, 2))
                _scatter[grid](down, offsets, routed, first, pad, HIDDEN, 2048)
            _unroute[(args.tokens, triton.cdiv(HIDDEN, 1024))](
                routed, weights, inverse, output, HIDDEN, TOPK=TOPK, BLOCK=1024
            )

        candidate()
        error = relative_l2(output, control)
        finite = bool(torch.isfinite(output).all())
        row = {
            "experts_per_batch": count,
            **measure(candidate, 3),
            "control_max_rel_l2": error,
            "finite": finite,
            "passed": finite and error <= 0.003,
        }
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["control_repeat"] = measure(baseline, 3)
    # Localize a rejected whole-chain screen before considering native work.
    count = 64
    packed_gu = raw_gu[:count].flatten(0, 1)
    packed_dn = raw_down[:count].flatten(0, 1)

    def dequant_gu():
        return dequantize_gguf_rows(
            packed_gu, info["gate"][0], HIDDEN, torch.half
        ).view(count, 2 * INTERMEDIATE, HIDDEN)

    def dequant_dn():
        return dequantize_gguf_rows(
            packed_dn, info["down"][0], down_k, torch.half
        ).view(count, HIDDEN, down_k)[:, :, left : left + INTERMEDIATE]

    gu, dn = dequant_gu(), dequant_dn()
    padded = torch.empty(count, pad, HIDDEN, device="cuda", dtype=torch.half)
    grid = (count, triton.cdiv(pad * HIDDEN, 2048))

    def gather():
        _pad_gather[grid](x, rows, offsets, padded, 0, pad, HIDDEN, 2048)

    gather()
    projected = torch.bmm(padded, gu.transpose(1, 2), out_dtype=torch.float32)
    hidden = (
        torch.nn.functional.silu(projected[:, :, :INTERMEDIATE])
        * projected[:, :, INTERMEDIATE:]
    ).half()
    down = torch.bmm(hidden, dn.transpose(1, 2))
    parts = {
        "dequant_gate_up": dequant_gu,
        "dequant_down": dequant_dn,
        "gather": gather,
        "gate_up_bmm_fp32_output": lambda: torch.bmm(
            padded, gu.transpose(1, 2), out_dtype=torch.float32
        ),
        "gate_up_bmm_fp16_output": lambda: torch.bmm(padded, gu.transpose(1, 2)),
        "down_bmm": lambda: torch.bmm(hidden, dn.transpose(1, 2)),
        "scatter": lambda: _scatter[grid](down, offsets, routed, 0, pad, HIDDEN, 2048),
    }
    report["first_64_expert_components"] = {
        name: measure(fn, 20) for name, fn in parts.items()
    }
    print(json.dumps(report["first_64_expert_components"]), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if all(row["passed"] for row in report["cases"]) else 1)


if __name__ == "__main__":
    main()
