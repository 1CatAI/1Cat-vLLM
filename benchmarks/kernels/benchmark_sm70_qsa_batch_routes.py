# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen QSA attention for independent no-MTP requests, not verifier rows.

Compare forced Triton, direct XQA, and padded grouped Page4 with the same
metadata. This is attention only, not the indexer, model round or quality score.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda as native

from vllm.models.qwen4_exp.nvidia.ops import qsa as ops


def check_exclusive():
    report = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    for line in report.splitlines():
        pid, _, name = line.partition(",")
        if (
            pid.strip().isdigit()
            and int(pid) != os.getpid()
            and "snapd-desktop-integration" not in name
        ):
            raise RuntimeError(f"Foreign GPU process; discard timing: {line}")


@torch.inference_mode()
def screen(rows, args):
    device = "cuda"
    page = 784
    blocks = (args.seq_len + page - 1) // page
    q = torch.randn((rows, 6, 256), device=device, dtype=torch.float16)
    k = torch.randn((rows * blocks, page, 1, 256), device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    generator = torch.Generator().manual_seed(7)
    selected_blocks = torch.stack(
        [
            torch.randperm(args.seq_len // 4 - 1, generator=generator)[:512]
            for _ in range(rows)
        ]
    ).to(device=device, dtype=torch.int32)
    # Distinct requests have distinct physical pages, even at equal positions.
    table = torch.randperm(rows * blocks, device=device, dtype=torch.int32)
    table = table.view(rows, blocks)
    req = torch.arange(rows, device=device, dtype=torch.int32)
    pos = torch.full((rows,), args.seq_len - 1, device=device, dtype=torch.int64)
    lengths = torch.full((rows,), args.seq_len, device=device, dtype=torch.int32)
    indices = ops.expand_qsa_block_indices_cuda(
        selected_blocks,
        pos,
        lengths,
        req,
        4,
        2048,
    )
    outputs = [torch.empty_like(q) for _ in range(3)]
    padded = ((rows + 7) // 8) * 8
    pq = q.new_zeros((padded, 6, 256))
    pi = indices.new_full((padded, indices.shape[1]), -1)
    pr = req.new_full((padded,), -1)
    pp = pos.new_zeros(padded)
    po = torch.empty_like(pq)

    def triton_call():
        ops.qsa_sparse_paged_attention(
            q,
            k,
            v,
            indices,
            table,
            req,
            outputs[0],
            pos,
            lengths,
        )

    def xqa_call():
        ops._qsa_sparse_paged_attention_sm70_xqa_page4_batch(
            q,
            k,
            v,
            indices,
            table,
            req,
            pos,
            lengths,
            outputs[1],
            "auto",
            1.0,
            1.0,
            native,
        )

    def grouped_call():
        pq[:rows].copy_(q)
        pi[:rows].copy_(indices)
        pr[:rows].copy_(req)
        pp[:rows].copy_(pos)
        ops._qsa_sparse_paged_attention_sm70_grouped_page4(
            pq,
            k,
            v,
            pi,
            table,
            pr,
            pp,
            lengths,
            po,
            "auto",
            1.0,
            1.0,
            native,
        )
        outputs[2].copy_(po[:rows])

    calls = [triton_call, xqa_call, grouped_call]
    graphs = []
    stream = torch.cuda.Stream()
    for call in calls:
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.calls_per_graph):
                call()
        graphs.append(graph)

    checks = []
    passed = [True] * 3
    for cycle in range(6):
        q.normal_().mul_((0.25, 1.0, 3.0)[cycle % 3])
        # A canonical QSA tail has (visible_length % 4) entries, not always
        # three. Sweep all residues while keeping captured pointers fixed.
        lengths.fill_(args.seq_len - cycle % 4)
        pos.copy_(lengths.long() - 1)
        ops.expand_qsa_block_indices_cuda(
            selected_blocks,
            pos,
            lengths,
            req,
            4,
            2048,
            out=indices,
        )
        complete = (
            selected_blocks[:, :, None] * 4 + torch.arange(4, device=device)
        ).reshape(rows, 2048)
        tail_offset = torch.arange(3, device=device)
        tail = (lengths[:, None] // 4) * 4 + tail_offset
        tail = torch.where(tail_offset < lengths[:, None] % 4, tail, -1)
        torch.testing.assert_close(
            indices.long(), torch.cat((complete, tail), 1).long()
        )
        # Exercise mutable slot mappings while graph pointers remain fixed.
        table.copy_(
            torch.randperm(rows * blocks, device=device, dtype=torch.int32).view_as(
                table
            )
        )
        safe = indices.clamp_min(0)
        physical = table.gather(1, (safe // page).long()).long()
        selected_k = k[physical, (safe % page).long(), 0].float()
        selected_v = v[physical, (safe % page).long(), 0].float()
        scores = q.float() @ selected_k.transpose(1, 2) / 16
        scores.masked_fill_((indices < 0)[:, None, :], float("-inf"))
        expected = torch.softmax(scores, dim=-1) @ selected_v
        for call in calls:
            call()
        eager = [out.clone() for out in outputs]
        for out in outputs:
            out.fill_(float("nan"))
        po.fill_(float("nan"))
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        errors = []
        for route, (out, ref) in enumerate(zip(outputs, eager)):
            exact = torch.equal(out, ref)
            close = torch.allclose(out.float(), expected, atol=2e-3, rtol=1e-2)
            err = out.float() - expected
            rel = (err.norm() / expected.norm()).item()
            passed[route] &= exact and close and rel <= 5e-3
            errors.append(
                {
                    "max_abs": err.abs().max().item(),
                    "relative_l2": rel,
                    "graph_equals_eager": exact,
                    "fp32_oracle_close": close,
                }
            )
        if torch.count_nonzero(po[rows:]).item():
            passed[2] = False
        checks.append(errors)

    print(
        json.dumps(
            {
                "rows": rows,
                "route_order": ["triton", "direct_xqa", "padded_grouped"],
                "micro_gate_passed": passed,
                "checks": checks,
            }
        ),
        flush=True,
    )

    # Timing retains the declared length, not the final residue-check length.
    lengths.fill_(args.seq_len)
    pos.copy_(lengths.long() - 1)
    ops.expand_qsa_block_indices_cuda(
        selected_blocks,
        pos,
        lengths,
        req,
        4,
        2048,
        out=indices,
    )
    samples = [[] for _ in calls]
    for sample in range(5):
        check_exclusive()
        for i in range(3) if sample % 2 == 0 else reversed(range(3)):
            # Retain diagnostics, but never admit timing for a failed oracle.
            if not passed[0] or not passed[i]:
                continue
            graph = graphs[i]
            for _ in range(5):
                graph.replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples[i].append(
                start.elapsed_time(end) * 1000 / args.replays / args.calls_per_graph
            )
        check_exclusive()
    return {
        "rows": rows,
        "independent_requests": True,
        "query_heads": 6,
        "kv_heads": 1,
        "head_dim": 256,
        "kv_dtype": "float16",
        "selection_width": indices.shape[1],
        "padded_rows": padded,
        "samples_us": samples,
        "median_us": [statistics.median(s) if s else None for s in samples],
        "route_order": ["triton", "direct_xqa", "padded_grouped"],
        "changed_input_slot_map_cycles": checks,
        "micro_gate_passed": passed,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", default="1,4,8,16")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--replays", type=int, default=40)
    p.add_argument("--calls-per-graph", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.seq_len < 4096 or args.seq_len % 4:
        p.error("seq-len must be a multiple of four and at least 4096")
    rows = [int(m) for m in args.rows.split(",")]
    if not rows or min(rows) < 1 or min(args.replays, args.calls_per_graph) < 1:
        p.error("rows and replay counts must be positive")
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("SM70 required")
    if not ops._qsa_grouped_page4_supported(native, "auto"):
        raise RuntimeError("grouped Page4 native capability unavailable")
    original_gate = ops._SM70_QSA_XQA_PAGE4
    try:
        # Force the reference; candidate calls bypass the gate explicitly.
        ops._SM70_QSA_XQA_PAGE4 = False
        torch.manual_seed(7)
        results = [screen(m, args) for m in rows]
    finally:
        ops._SM70_QSA_XQA_PAGE4 = original_gate
    payload = {
        "qsa_source": ops.__file__,
        "native": native.__file__,
        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "grouped_abi": ops._qsa_grouped_page4_abi_version(native),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "seq_len": args.seq_len,
        "calls_per_graph": args.calls_per_graph,
        "replays": args.replays,
        "results": results,
        "scope": "Single-GPU attention micro, not TP4/model quality or throughput",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not all(all(r["micro_gate_passed"]) for r in results):
        raise SystemExit(
            "A QSA micro quality gate failed; inspect retained diagnostics"
        )


if __name__ == "__main__":
    main()
