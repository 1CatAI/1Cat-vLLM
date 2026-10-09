# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Account for selected KV operands and cache counters in untimed TP4 records.

Operand counts are not measured HBM transfers. Cache counters include a
prefill prefix in the first snapshot, which is excluded. Target unions are
single-request unions; C4 cannot combine identical logical IDs from requests.
"""

import argparse
import json
import statistics
from pathlib import Path

from analyze_flashnext_round_cost import draft_groups


def selected_reads(records, vector_bytes, encoded_bytes):
    if not records or any(r["logical_union_single_request"] is None for r in records):
        raise ValueError("A single-request KV ledger requires explicit request unions")
    samples = []
    for r in records:
        issued = sum(r["valid_per_row"])
        hits, misses, contention = r["cache_hit_miss_contention_delta"]
        if hits + misses != issued:
            raise ValueError(
                "Cache delta includes unsampled work; do not attribute it to this call"
            )
        samples.append(
            {
                "ordinal": r["ordinal"],
                "rows": r["rows"],
                "unique_tokens": r["logical_union_single_request"],
                "issued_tokens": issued,
                "fp16_operand_union_bytes": r["logical_union_single_request"]
                * vector_bytes,
                "fp16_query_list_bytes": issued * vector_bytes,
                "encoded_operand_union_bytes": r["logical_union_single_request"]
                * encoded_bytes,
                "encoded_miss_read_bytes": misses * encoded_bytes,
                "hits": hits,
                "misses": misses,
                "contended_pages": contention,
            }
        )
    means = {
        f"mean_{key}": statistics.mean(r[key] for r in samples)
        for key in samples[0]
        if key not in ("ordinal", "rows")
    }
    return {**means, "samples": samples}


def analyze(report, trim=8):
    workers = sorted(report["round_cost"], key=lambda r: r["rank"])
    if [r["rank"] for r in workers] != [0, 1, 2, 3]:
        raise ValueError("TP4 records required")
    config = report["config"]["kernel_config"]
    if not config["qsa_host_kv_device_reference"]:
        raise ValueError("Host costs must be kept separate from device-history costs")
    # These are captured Flash-Next D256 caches: two vectors per token,
    # per-vector FP32 scales for E4M3; no scales for FP16 draft history.
    if (
        config["qsa_host_kv_dtype"] != "fp8_e4m3"
        or config["qsa_host_kv_draft_dtype"] != "float16"
    ):
        raise ValueError("This ledger expects E4M3 target and FP16 draft history")
    out = {
        "scope": "untimed D256 selected operands, not measured DRAM bytes",
        "per_rank": [],
    }
    reference = None
    for worker in workers:
        targets, drafts, signature = [], [], {}
        for name, history in sorted(worker["selections"].items()):
            records = history["records"]
            signature[name] = [
                {
                    key: r[key]
                    for key in (
                        "ordinal",
                        "rows",
                        "valid_per_row",
                        "unique_per_row",
                        "logical_union_single_request",
                    )
                }
                for r in records
            ]
            if name.startswith("mtp."):
                groups = draft_groups(records, 5)
                # The bounded selection ring keeps only the last sixteen rounds.
                groups = groups[2:-2]
                if not groups:
                    raise ValueError("No complete central draft rounds")
                drafts.append(
                    {
                        "layer": name,
                        "dropped": history["dropped"],
                        "rounds": len(groups),
                        "steps": [
                            selected_reads([g[i] for g in groups], 1024, 1024)
                            for i in range(4)
                        ],
                    }
                )
            else:
                if history["dropped"] or any(r["rows"] != 5 for r in records):
                    raise ValueError("Target capture must contain complete M5 calls")
                targets.append(
                    {"layer": name, **selected_reads(records[trim:-trim], 1024, 520)}
                )
        if len(targets) != 12 or len(drafts) != 1:
            raise ValueError("Expected twelve target QSA layers and one draft layer")
        if reference is not None and signature != reference:
            raise ValueError("Ranks disagree on causal selections")
        reference = signature
        totals = {
            key: sum(layer[key] for layer in targets)
            for key in targets[0]
            if key.startswith("mean_")
        }
        out["per_rank"].append(
            {
                "rank": worker["rank"],
                "target": targets,
                "target_total": totals,
                "draft": drafts[0],
            }
        )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.result.read_text()))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
