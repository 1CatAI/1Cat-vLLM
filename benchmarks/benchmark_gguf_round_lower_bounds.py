# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconcile payload lower bounds without mixing profiled and endpoint timing.

Unknown traffic/instruction counters stay null; peak-bandwidth floors are
optimistic hardware bounds, never estimates of a deliverable speedup.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--resident", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-clock-mhz", type=float, default=877)
    parser.add_argument("--sm-clock-mhz", type=float, required=True)
    args = parser.parse_args()
    resident = json.loads(args.resident.read_text())
    trace = json.loads(args.trace.read_text())["ranks"][0]
    peak_gbps = args.memory_clock_mhz * 1e6 * 2 * 4096 / 8 / 1e9
    rows, missing = [], []
    detail_by_module = defaultdict(list)
    for detail in trace["projection_details"]:
        detail_by_module[detail["module"]].append(detail)
    totals = defaultdict(int)
    logical_weights = defaultdict(int)
    for bank in resident["resident_micro"][0]["banks"]:
        size = bank["bytes"]
        if bank["route"] != "dmv":
            parts = detail_by_module[bank["name"]]
            assert parts and all(part.get("bytes") for part in parts), bank
            size = sum(part["bytes"] for part in parts)
            missing.append(
                dict(
                    module=bank["name"],
                    bytes=size,
                    provenance=(
                        "prior loaded-bank inventory; "
                        "verify current storage at model A/B"
                    ),
                )
            )
            weights = sum(part["n"] * part["k"] for part in parts)
        else:
            weights = bank["k"] * sum(bank["n"])
        totals[bank["role"]] += size
        logical_weights[bank["role"]] += weights
    for role, size in totals.items():
        measured = resident["resident_micro"][0]["results"][role]
        rows.append(
            dict(
                role="target." + role,
                payload_read_bytes=size,
                peak_hbm_floor_ms=size / (peak_gbps * 1e6),
                logical_weight_values=logical_weights[role],
                mma_flops=16 * logical_weights[role],
                measured_resident_chain_ms=measured["mean_ms"],
                measured_scope=resident["resident_micro"][0]["measurement"],
                decode_warp_instructions=None,
            )
        )
    target = gguf.GGUFReader(str(args.target))
    head = next(t for t in target.tensors if t.name == "output.weight")
    hk, hn = map(int, head.shape)
    assert hn % 4 == 0 and hk % 32 == 0
    head_bytes = hn // 4 * hk // 2 + hn // 4 * (hk // 32) * 4
    for role in ("target_head", "draft_shared_head"):
        rows.append(
            dict(
                role=role,
                payload_read_bytes=head_bytes,
                peak_hbm_floor_ms=head_bytes / (peak_gbps * 1e6),
                profiled_reference_service_ms=trace["tail_categories"][role][
                    "service_ms_per_round"
                ],
                measured_scope="old diagnostic trace; not 54633 A/B",
                decode_warp_instructions=None,
            )
        )
    draft = gguf.GGUFReader(str(args.draft))
    draft_roles = defaultdict(int)
    for tensor in draft.tensors:
        k, n = map(int, tensor.shape) if len(tensor.shape) == 2 else (0, 0)
        name = tensor.name
        if name == "fc.weight":
            draft_roles["context_fc.u8"] += n // 4 * k * 9 // 8
        elif any(
            name.endswith(s)
            for s in (
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            )
        ):
            draft_roles["query_projections.u8"] += n * k // 4 * 9 // 8
        elif name.endswith("conv_proj.weight"):
            draft_roles["replicated_convolution_projection.fp16"] += n * k * 2
        elif name == "selector_hidden.weight":
            draft_roles["selector_hidden.fp16"] += n * k * 2
    # The draft creates a single dense context KV matrix from all five layers.
    draft_roles["fused_context_kv.fp16"] = 5 * 2 * 2 * 128 * 5120 * 2
    for role, size in draft_roles.items():
        rows.append(
            dict(
                role="draft." + role,
                payload_read_bytes=size,
                peak_hbm_floor_ms=size / (peak_gbps * 1e6),
                provenance=(
                    "source shapes and current u8/fp16 loader layout; "
                    "not measured DRAM traffic"
                ),
                decode_warp_instructions=None,
            )
        )
    state_bytes = 48 * 12 * 128 * 128 * 4
    rows.append(
        dict(
            role="target.GDN_state",
            mandatory_state_read_bytes=state_bytes,
            speculative_state_write_bytes=state_bytes * 8,
            total_state_bytes=state_bytes * 9,
            peak_hbm_floor_ms=state_bytes * 9 / (peak_gbps * 1e6),
            provenance=(
                "fused_recurrent.py: one FP32 initial state and a full FP32 "
                "snapshot per live token; M8/C1"
            ),
            profiled_reference_service_ms=trace["target_aux_categories"][
                "GDN_state_or_gating"
            ]["service_ms_per_round"],
            decode_warp_instructions=None,
        )
    )
    # At 1K: target 16 layers, local 1 KV head of dimension 256, E4M3;
    # draft 5 layers, local 2 KV heads of dimension 128, FP16. These are unique
    # KV bytes; partition rereads and paged-layout traffic need counters.
    for role, size in (
        ("target_attention.unique_kv_1k", 16 * 2 * 1 * 256 * 1024),
        ("draft_attention.unique_kv_1k", 5 * 2 * 2 * 128 * 2 * 1024),
    ):
        rows.append(
            dict(
                role=role,
                unique_read_bytes=size,
                peak_hbm_floor_ms=size / (peak_gbps * 1e6),
                actual_dram_bytes=None,
                decode_warp_instructions=None,
            )
        )
    message_bytes = 8 * 5120 * 2
    communication = dict(
        target_layer_collectives=128,
        bytes_per_collective=message_bytes,
        remote_bytes_sent_per_card_per_collective=3 * message_bytes,
        pair_nv2_one_way_peak_gbps=50,
        aggregate_three_peer_peak_gbps=150,
        serialization_floor_ms=128 * 3 * message_bytes / (150 * 1e6),
        synchronization_latency_floor_ms=None,
        profiled_reference_category_calls=130,
        profiled_reference_service_ms=trace["target_aux_categories"][
            "communication_or_reduction"
        ]["service_ms_per_round"],
        note=(
            "128 layer reductions; trace category also includes other reductions; "
            "peak link floor excludes protocol and dependencies"
        ),
    )
    report = dict(
        base_source="c4f6245f841466782752a8c3283e4727565cf17a",
        memory_clock_mhz=args.memory_clock_mhz,
        sm_clock_mhz=args.sm_clock_mhz,
        peak_hbm_gbps=peak_gbps,
        warp_issue_slots_per_second=80 * 4 * args.sm_clock_mhz * 1e6,
        timing_policy=(
            "No new baseline run. Old resident/trace data is reference only; "
            "54633 gains require same-wheel A/B."
        ),
        rows=rows,
        communication=communication,
        recovered_missing_banks=missing,
        target_projection_payload_bytes=sum(totals.values()),
        target_projection_peak_floor_ms=sum(totals.values()) / (peak_gbps * 1e6),
        known_instruction_counter=dict(
            role="IQ3_S GDN out M8/N5120/K1536",
            executed_warp_instructions=1513280,
            ideal_issue_floor_us=1513280 / (80 * 4 * args.sm_clock_mhz),
            note=(
                "Prior NCU observation; do not extrapolate to other readers or kernels"
            ),
        ),
        unresolved=[
            "Dynamic instruction counts for other target/draft/head kernels",
            "Sampling/handoff actual reads and synchronization latency",
            "Current 54633 installed-model bank footprints and graph ledger",
            "Actual DRAM traffic, including activation and table cache hits",
        ],
        input_evidence={
            str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.resident, args.trace)
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                target_bytes=report["target_projection_payload_bytes"],
                target_floor_ms=report["target_projection_peak_floor_ms"],
                restored_missing_bytes=sum(m["bytes"] for m in missing),
                draft_bytes=sum(draft_roles.values()),
                head_bytes_per_call=head_bytes,
            )
        )
    )


if __name__ == "__main__":
    main()
