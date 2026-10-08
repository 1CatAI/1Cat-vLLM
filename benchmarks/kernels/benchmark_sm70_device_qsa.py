# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research screen: direct device E4M3 history versus protected-cache QSA."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
from vllm.models.qwen4_exp.nvidia.ops.host_kv_attention import host_qsa_attention


def make_inputs(rows, context, owners, dtype=torch.uint8):
    device = torch.device("cuda:0")
    requests_count, page, width = max(1, rows // 5), 832, 2051
    pages = (context + page - 1) // page
    blocks = pages * requests_count
    table = torch.arange(blocks, device=device, dtype=torch.int32).view(
        requests_count, pages
    )
    requests = torch.arange(rows, device=device, dtype=torch.int32) // 5
    positions = context - 5 + torch.arange(rows, device=device, dtype=torch.int64) % 5
    lengths = torch.full((requests_count,), context, dtype=torch.int32, device=device)
    selected = torch.full((rows, width), -1, dtype=torch.int32)
    generator = torch.Generator().manual_seed(83)
    common_pages = {}
    for row in range(rows):
        pos = context - 5 + row % 5
        complete = (pos + 1) // 4
        if complete > 512:
            # Correlated five-token selections: ~350 common pages, 162 private.
            common = common_pages.setdefault(
                row // 5,
                torch.randperm((context - 5) // 4 - 1, generator=generator)[:350],
            )
            private = torch.randperm(complete - 1, generator=generator)
            private = private[~torch.isin(private, common)][:162]
            compressed = torch.cat((common, private)).sort().values
        else:
            compressed = torch.arange(complete)
        tokens = (compressed[:, None] * 4 + torch.arange(4)[None, :]).flatten()
        tokens = torch.cat((tokens, torch.arange(complete * 4, pos + 1)))
        selected[row, : tokens.numel()] = tokens.int()
    selected = selected.to(device)
    states = []
    for _ in range(owners):
        state = HostQSAKV(
            blocks,
            page,
            256,
            device,
            hot_tokens=8192,
            rows=rows,
            width=width,
            device_reference=True,
            dtype=dtype,
        )
        key = torch.randn(blocks * page, 1, 256, device=device, dtype=torch.float16)
        value = torch.randn_like(key)
        state.write(key, value, torch.arange(blocks * page, device=device))
        q = torch.randn(rows, 6, 256, device=device, dtype=torch.float16)
        gate = torch.randn_like(q)
        splits = (width + 63) // 64
        states.append(
            dict(
                state=state,
                q=q,
                gate=gate,
                candidate=torch.empty_like(q),
                control=torch.empty_like(q),
                partial=torch.empty(rows * 6 * splits * 256, device=device),
                sums=torch.empty(rows * 6 * splits * 2, device=device),
            )
        )
    return states, selected, table, requests, positions, lengths


def call(record, metadata, candidate):
    indices, table, requests, positions, lengths = record.get("metadata", metadata)
    if candidate:
        state = record["state"]
        torch.ops.round15_device_qsa.run(
            record["q"],
            state.history,
            state.scales,
            indices,
            table,
            requests,
            positions,
            lengths,
            record["candidate"],
            record["gate"],
            record["partial"],
            record["sums"],
        )
    else:
        host_qsa_attention(
            record["q"],
            record["state"],
            indices,
            table,
            requests,
            positions,
            lengths,
            record["control"],
            record["gate"],
        )


def oracle(record, metadata):
    indices, table, requests, positions, lengths = record.get("metadata", metadata)
    state = record["state"]
    if state.fp8:
        decoded = state.history.view(torch.float8_e4m3fn).float().squeeze(3)
        scale = state.scales.view(state.blocks, state.page_size, 2).transpose(1, 2)
        decoded = (decoded * scale[..., None]).half()
    else:
        decoded = state.history.squeeze(3)
    expected = torch.zeros_like(record["q"])
    for row in range(indices.shape[0]):
        req, pos = int(requests[row]), int(positions[row])
        if req < 0 or req >= table.shape[0]:
            continue
        logical = indices[row]
        logical = logical[(logical >= 0) & (logical <= pos) & (logical < lengths[req])]
        logical = logical[logical // state.page_size < table.shape[1]]
        physical = table[req, logical // state.page_size].long()
        valid = (physical >= 0) & (physical < state.blocks)
        physical, logical = physical[valid], logical[valid]
        if logical.numel() == 0:
            continue
        offset = logical % state.page_size
        key = decoded[physical, 0, offset].float()
        value = decoded[physical, 1, offset].float()
        scores = record["q"][row].float() @ key.T / 16
        attention = (scores.softmax(-1) @ value).half()
        expected[row] = (
            attention.float() * record["gate"][row].float().sigmoid()
        ).half()
    return expected


def error(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    return dict(
        max_abs=diff.max().item(),
        relative_l2=(diff.norm() / expected.float().norm().clamp_min(1e-30)).item(),
        max_scaled=(diff.max() / expected.abs().max().clamp_min(1e-30)).item(),
    )


def graph_for(records, metadata, candidate):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for record in records:
            call(record, metadata, candidate)
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def graph_time(graph, repeats):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5, choices=[1, 5, 20])
    parser.add_argument("--context", type=int, default=8448)
    parser.add_argument("--owners", type=int, default=12)
    parser.add_argument("--fp16-history", action="store_true")
    parser.add_argument(
        "--include-draft",
        action="store_true",
        help="Append M5/M1/M1/M1 on one shared FP16 history",
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    library_sha256 = hashlib.sha256(args.library.read_bytes()).hexdigest()
    torch.ops.load_library(str(args.library))
    records, *metadata = make_inputs(
        args.rows,
        args.context,
        args.owners,
        torch.float16 if args.fp16_history else torch.uint8,
    )
    if args.include_draft:
        assert args.rows in (5, 20) and not args.fp16_history
        drafts, *draft_metadata = make_inputs(args.rows, args.context, 1, torch.float16)
        drafts[0]["metadata"] = draft_metadata
        one_metadata = [
            draft_metadata[0][4::5].clone(),
            draft_metadata[1],
            draft_metadata[2][4::5].clone(),
            draft_metadata[3][4::5].clone(),
            draft_metadata[4],
        ]
        for step in range(3):
            first = drafts[0]
            r = dict(
                state=first["state"],
                q=first["q"][4::5].clone(),
                gate=first["gate"][4::5].clone(),
                candidate=first["candidate"][4::5].clone(),
                control=first["control"][4::5].clone(),
                partial=first["partial"],
                sums=first["sums"],
                metadata=one_metadata,
            )
            drafts.append(r)
        records.extend(drafts)
    # Warm the same cache ownership route before graph capture; both arms read
    # identical authoritative E4M3 bytes and per-vector FP32 scales.
    for _ in range(16):
        for record in records:
            call(record, metadata, False)
    for record in records:
        call(record, metadata, True)
    torch.cuda.synchronize()
    checks = []
    for record in records:
        ref = oracle(record, metadata)
        checks.append(
            dict(
                candidate_official=error(record["candidate"], ref),
                control_official=error(record["control"], ref),
                candidate_control=error(record["candidate"], record["control"]),
            )
        )
        torch.testing.assert_close(record["candidate"], ref, rtol=0.005, atol=0.0003)
    print(json.dumps({"numerical": checks}), flush=True)
    edge = records[0]
    original_q = edge["q"].clone()
    edge["q"].mul_(8)
    call(edge, metadata, True)
    torch.testing.assert_close(
        edge["candidate"], oracle(edge, metadata), rtol=0.005, atol=0.0005
    )
    edge["q"].copy_(original_q)
    original_requests = metadata[2].clone()
    metadata[2][0] = -1
    call(edge, metadata, True)
    torch.testing.assert_close(
        edge["candidate"], oracle(edge, metadata), rtol=0.005, atol=0.0003
    )
    metadata[2].copy_(original_requests)
    original_positions = metadata[3].clone()
    metadata[3][0] = -1
    call(edge, metadata, True)
    torch.testing.assert_close(
        edge["candidate"], oracle(edge, metadata), rtol=0.005, atol=0.0003
    )
    metadata[3].copy_(original_positions)
    original_queries = [record["q"].clone() for record in records]
    control = graph_for(records, metadata, False)
    candidate = graph_for(records, metadata, True)
    for mutation in [0.5, -1.0]:
        for record in records:
            record["q"].mul_(mutation)
        control.replay()
        candidate.replay()
        torch.cuda.synchronize()
        for record in records:
            torch.testing.assert_close(
                record["candidate"], oracle(record, metadata), rtol=0.005, atol=0.0003
            )
    for record, original in zip(records, original_queries):
        record["q"].copy_(original)
    control.replay()
    candidate.replay()
    torch.cuda.synchronize()
    timings = {"control": [], "candidate": []}
    for _ in range(args.samples):
        for arm in ["control", "candidate", "candidate", "control"]:
            graph = control if arm == "control" else candidate
            timings[arm].append(graph_time(graph, args.repeats))
    result = dict(
        rows=args.rows,
        context=args.context,
        owners=args.owners,
        calls=len(records),
        query_rows=[r["q"].shape[0] for r in records],
        include_draft=args.include_draft,
        history_dtype="fp16" if args.fp16_history else "e4m3fn",
        library_sha256=library_sha256,
        checks=checks,
        graph_changed_inputs=2,
        edge_checks=["q_std8", "invalid_request", "position_minus1"],
        timing_query_std=1,
        samples_us=timings,
        medians_us={k: statistics.median(v) for k, v in timings.items()},
        scope="research-only; synthetic correlated selectors; not model endpoint",
        candidate_kernels=2 * len(records),
        control_kernels=4 * len(records),
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["medians_us"]), flush=True)


if __name__ == "__main__":
    main()
