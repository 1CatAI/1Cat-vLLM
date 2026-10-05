# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inventory GGUF source bytes and mixed FFN pairs without a CUDA runtime.

Examples:
    python benchmarks/benchmark_gguf_quantization_inventory.py model.gguf \
        --output-dir inventory --tensor-parallel-size 4
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import regex as re

_LAYER = re.compile(r"blk\.(\d+)\.(.+)")
_PROJECTIONS = {
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_qkv.weight",
    "attn_output.weight",
    "attn_gate.weight",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_out.weight",
    "output.weight",
}


def family(name):
    if name in {"F32", "F16", "BF16", "F64"}:
        return "floating"
    if name.startswith("TQ"):
        return "D_ternary"
    if name.startswith("IQ4") or name in {"MXFP4", "NVFP4"}:
        return "B_lut4"
    if name.startswith("IQ"):
        return "C_lattice"
    if name.startswith("Q"):
        return "A_affine"
    return "other"


def aggregate(records, key, payload_bytes):
    groups = defaultdict(list)
    for record in records:
        groups[record[key]].append(record)
    rows = []
    for value, items in groups.items():
        size = sum(item["source_bytes"] for item in items)
        rows.append(
            {
                key: value,
                "source_bytes": size,
                "payload_percent": size * 100 / payload_bytes,
                "tensor_count": len(items),
                "layers": sorted({r["layer"] for r in items if r["layer"] is not None}),
                "roles": sorted({r["role"] for r in items}),
            }
        )
    return sorted(rows, key=lambda row: (-row["source_bytes"], str(row[key])))


def build_inventory(reader, model_path, tp_size):
    records = []
    for tensor in reader.tensors:
        match = _LAYER.fullmatch(tensor.name)
        layer, role = (int(match[1]), match[2]) if match else (None, tensor.name)
        shape = list(map(int, tensor.shape[::-1]))
        type_name = tensor.tensor_type.name
        records.append(
            {
                "name": tensor.name,
                "layer": layer,
                "role": role,
                "gguf_type_id": int(tensor.tensor_type),
                "gguf_type": type_name,
                "family": family(type_name),
                "shape": shape,
                "source_bytes": int(tensor.n_bytes),
                "data_offset": int(tensor.data_offset),
                "is_projection": role in _PROJECTIONS and len(shape) == 2,
            }
        )
    assert len({r["name"] for r in records}) == len(records)
    payload = sum(record["source_bytes"] for record in records)
    assert 0 < payload <= model_path.stat().st_size
    layers = sorted({r["layer"] for r in records if r["layer"] is not None})
    assert layers == list(range(len(layers))), "Noncontiguous GGUF layer indices"
    by_name = {r["name"]: r for r in records}
    pairs = []
    for layer in layers:
        gate = by_name.get(f"blk.{layer}.ffn_gate.weight")
        up = by_name.get(f"blk.{layer}.ffn_up.weight")
        if gate is None and up is None:
            continue
        assert gate is not None and up is not None, "Incomplete gate/up pair"
        assert gate["shape"] == up["shape"] and len(gate["shape"]) == 2
        size = gate["source_bytes"] + up["source_bytes"]
        assert size % tp_size == 0
        n, k = gate["shape"]
        assert n % tp_size == 0, "Output rows cannot divide evenly across TP"
        pairs.append(
            {
                "layer": layer,
                "gate_type_id": gate["gguf_type_id"],
                "gate_type": gate["gguf_type"],
                "up_type_id": up["gguf_type_id"],
                "up_type": up["gguf_type"],
                "gate_family": gate["family"],
                "up_family": up["family"],
                "mixed_type": gate["gguf_type_id"] != up["gguf_type_id"],
                "N": n,
                "K": k,
                "tp_N": n // tp_size,
                "gate_source_bytes": gate["source_bytes"],
                "up_source_bytes": up["source_bytes"],
                "source_bytes": size,
                "tp_source_bytes": size // tp_size,
            }
        )
    role_groups = defaultdict(list)
    for record in records:
        role_groups[record["gguf_type"], record["role"]].append(record)
    type_roles = []
    for (type_name, role), items in role_groups.items():
        size = sum(item["source_bytes"] for item in items)
        type_roles.append(
            {
                "gguf_type": type_name,
                "role": role,
                "tensor_count": len(items),
                "source_bytes": size,
                "payload_percent": size * 100 / payload,
                "layers": sorted({r["layer"] for r in items if r["layer"] is not None}),
            }
        )
    type_roles.sort(
        key=lambda row: (-row["source_bytes"], row["gguf_type"], row["role"])
    )
    pair_bytes = sum(p["source_bytes"] for p in pairs)
    mixed = [p for p in pairs if p["mixed_type"]]
    mixed_bytes = sum(p["source_bytes"] for p in mixed)
    pair_groups = defaultdict(list)
    for pair in pairs:
        pair_groups[pair["gate_type"], pair["up_type"]].append(pair)
    priorities = []
    for (gate, up), items in pair_groups.items():
        if gate == up:
            continue
        size = sum(item["source_bytes"] for item in items)
        priorities.append(
            {
                "gate_type": gate,
                "up_type": up,
                "gate_family": items[0]["gate_family"],
                "up_family": items[0]["up_family"],
                "layers": [item["layer"] for item in items],
                "layer_count": len(items),
                "source_bytes": size,
                "tp_source_bytes": size // tp_size,
                "mixed_pair_percent": size * 100 / mixed_bytes,
                "all_pair_percent": size * 100 / pair_bytes,
                "model_payload_percent": size * 100 / payload,
            }
        )
    priorities.sort(
        key=lambda row: (-row["source_bytes"], row["gate_type"], row["up_type"])
    )
    cumulative_bytes = cumulative_layers = 0
    for rank, row in enumerate(priorities, start=1):
        cumulative_bytes += row["source_bytes"]
        cumulative_layers += row["layer_count"]
        row.update(
            priority=rank,
            cumulative_layer_count=cumulative_layers,
            cumulative_mixed_percent=cumulative_bytes * 100 / mixed_bytes,
        )
    sets = defaultdict(list)
    for pair in mixed:
        sets[tuple(sorted((pair["gate_type"], pair["up_type"])))].append(pair)
    set_priorities = []
    for types, items in sets.items():
        size = sum(item["source_bytes"] for item in items)
        orientations = defaultdict(list)
        for item in items:
            orientations[f"{item['gate_type']}/{item['up_type']}"].append(item["layer"])
        set_priorities.append(
            {
                "types": list(types),
                "layer_count": len(items),
                "layers": sorted(item["layer"] for item in items),
                "orientations": dict(orientations),
                "source_bytes": size,
                "tp_source_bytes": size // tp_size,
                "mixed_pair_percent": size * 100 / mixed_bytes,
            }
        )
    set_priorities.sort(key=lambda row: (-row["source_bytes"], row["types"]))
    cumulative_bytes = cumulative_layers = 0
    for rank, row in enumerate(set_priorities, start=1):
        cumulative_bytes += row["source_bytes"]
        cumulative_layers += row["layer_count"]
        row.update(
            priority=rank,
            cumulative_layer_count=cumulative_layers,
            cumulative_mixed_percent=cumulative_bytes * 100 / mixed_bytes,
        )
    assert sum(row["source_bytes"] for row in priorities) == mixed_bytes
    assert sum(row["layer_count"] for row in priorities) == len(mixed)
    return {
        "model_filename": model_path.name,
        "file_bytes": model_path.stat().st_size,
        "tensor_payload_bytes": payload,
        "non_payload_bytes": model_path.stat().st_size - payload,
        "layer_count": len(layers),
        "tensor_parallel_size": tp_size,
        "tensor_count": len(records),
        "by_type": aggregate(records, "gguf_type", payload),
        "by_family": aggregate(records, "family", payload),
        "by_role": aggregate(records, "role", payload),
        "by_type_role": type_roles,
        "gate_up_summary": {
            "pair_count": len(pairs),
            "source_bytes": pair_bytes,
            "mixed_pair_count": len(mixed),
            "mixed_source_bytes": mixed_bytes,
            "mixed_pair_source_percent": mixed_bytes * 100 / pair_bytes
            if pair_bytes
            else 0,
        },
        "mixed_pair_priority": priorities,
        "mixed_type_set_priority": set_priorities,
        "gate_up_pairs": pairs,
        "tensors": records,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )


def main():
    import gguf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    args = parser.parse_args()
    if args.tensor_parallel_size < 1:
        parser.error("tensor-parallel-size must be positive")
    inventory = build_inventory(
        gguf.GGUFReader(str(args.model)), args.model, args.tensor_parallel_size
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    for key in (
        "by_type",
        "by_family",
        "by_role",
        "by_type_role",
        "gate_up_pairs",
        "mixed_pair_priority",
        "mixed_type_set_priority",
        "tensors",
    ):
        write_csv(args.output_dir / f"{key}.csv", inventory[key])
    print(
        json.dumps(
            {
                key: value
                for key, value in inventory.items()
                if key not in {"tensors", "gate_up_pairs", "by_role", "by_type_role"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
