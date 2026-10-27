# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only checkpoint and original-block TP storage audit for V4.1 Flash.

No tensor payload is materialized. Bank sizes describe the current packed TP
representation, not measured runtime memory or a recommended cache fraction.
"""

import argparse
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gguf
import regex as re

EXPERT = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    kind: int
    nbytes: int


def packed_bank_bytes(tensor: Tensor, tp_size: int) -> list[int]:
    """Account for shared boundary blocks and the existing final-row guard."""
    match = EXPERT.fullmatch(tensor.name)
    if match is None or len(tensor.shape) != 3 or tp_size <= 0:
        raise ValueError("Expected a routed expert tensor and positive TP size")
    k, rows, experts = tensor.shape
    block, unit = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(tensor.kind)]
    if min(k, rows, experts) <= 0 or k % block:
        raise ValueError("Expert tensor must contain complete source blocks")
    if tensor.nbytes != math.prod(tensor.shape) // block * unit:
        raise ValueError("Expert byte count does not match its quantization type")
    result = []
    for rank in range(tp_size):
        if match[2] == "down":
            if k % tp_size:
                raise ValueError("Expert input width is not divisible by TP")
            local = k // tp_size
            first = rank * local // block
            last = ((rank + 1) * local + block - 1) // block
            physical_k, local_rows = (last - first) * block, rows
        else:
            if rows % tp_size:
                raise ValueError("Expert output width is not divisible by TP")
            physical_k, local_rows = k, rows // tp_size
        tail = ((-physical_k) % 512) // block * unit
        result.append(experts * local_rows * (physical_k // block) * unit + tail)
    return result


def verified_paths(manifest: str, log: str) -> set[str]:
    expected = set()
    for line in manifest.splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip(" *")
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError("Invalid SHA256 manifest")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Manifest path escapes checkpoint directory")
        expected.add(name)
    exits = re.findall(r"^EXIT=(\d+)$", log, re.MULTILINE)
    actual = {
        match[1] for match in re.finditer(r"^(.+): (?:OK|成功)$", log, re.MULTILINE)
    }
    if not expected or not exits or exits[-1] != "0" or actual != expected:
        raise ValueError("Checkpoint SHA256 verification is incomplete or failed")
    return expected


def safetensors_names(path: Path) -> dict[str, dict]:
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if size > 16 * 1024**2 or 8 + size > path.stat().st_size:
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(source.read(size))
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    ranges = []
    for value in tensors.values():
        start, end = value["data_offsets"]
        if not 0 <= start <= end <= path.stat().st_size - 8 - size:
            raise ValueError(f"Safetensors payload is out of bounds: {path}")
        ranges.append((start, end))
    ordered = sorted(ranges)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"Overlapping safetensors payloads: {path}")
    return tensors


def shard_header(reader: gguf.GGUFReader) -> dict:
    def value(name):
        field = reader.get_field(name)
        return None if field is None else field.contents()

    architecture = value("general.architecture")
    index, count = value("split.no"), value("split.count")
    if index is None and count is None:
        if architecture != "deepseek41-engram":
            raise ValueError("Expected the standalone Engram GGUF")
    else:
        if index is None or count is None or not 0 <= index < count:
            raise ValueError("Invalid GGUF split metadata")
        if (index == 0 and architecture != "deepseek41") or (
            index > 0 and architecture not in (None, "deepseek41")
        ):
            raise ValueError("Unexpected backbone GGUF architecture")
    return {
        "architecture": architecture,
        "split_index": index,
        "split_count": count,
        "split_tensor_count": value("split.tensors.count"),
    }


def audit(root: Path, tp_size: int) -> dict:
    verified = verified_paths(
        (root / "SHA256SUMS").read_text(), (root / "verify.log").read_text()
    )
    if len(verified) != 8:
        raise ValueError("Expected four backbone, one Engram and three DSpark files")
    config = json.loads((root / "base/config.json").read_text())
    if config["model_type"] != "deepseek_v41":
        raise ValueError("Expected a DeepSeek V4.1 checkpoint")
    text = config["text_config"]
    layers = text["num_hidden_layers"]
    tensors: dict[str, Tensor] = {}
    file_records = []
    padding_ids = set()
    for path in sorted((root / "gguf").glob("*.gguf")):
        reader = gguf.GGUFReader(path, mode="r")
        header = shard_header(reader)
        padding = reader.get_field("tokenizer.ggml.padding_token_id")
        if padding is not None:
            padding_ids.add(int(padding.contents()))
        intervals = []
        for item in reader.tensors:
            if item.name in tensors:
                raise ValueError(f"Duplicate tensor across shards: {item.name}")
            if not reader.data_offset <= item.data_offset:
                raise ValueError(f"Tensor overlaps GGUF header: {item.name}")
            end = item.data_offset + item.n_bytes
            if end > path.stat().st_size:
                raise ValueError(f"Tensor is outside GGUF file: {item.name}")
            intervals.append((item.data_offset, end))
            tensors[item.name] = Tensor(
                item.name,
                tuple(map(int, item.shape)),
                int(item.tensor_type),
                int(item.n_bytes),
            )
        ordered = sorted(intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"Overlapping GGUF tensor payloads: {path}")
        file_records.append({"name": path.name, "bytes": path.stat().st_size, **header})
    splits = [item for item in file_records if item["split_index"] is not None]
    if (
        len(splits) != 4
        or {item["split_count"] for item in splits} != {4}
        or {item["split_index"] for item in splits} != set(range(4))
    ):
        raise ValueError("Expected the complete four-part backbone GGUF")

    expert_bytes = 0
    rank_bytes = [0] * tp_size
    layer_records = []
    for layer in range(layers):
        types = {}
        for role in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            tensor = tensors[name]
            if tensor.kind not in (16, 17, 29):
                raise ValueError(f"Unexpected routed expert quantization: {name}")
            expected = (text["hidden_size"], text["moe_intermediate_size"])
            if role == "down":
                expected = expected[::-1]
            if tensor.shape != (*expected, text["n_routed_experts"]):
                raise ValueError(f"Expert geometry does not match HF config: {name}")
            expert_bytes += tensor.nbytes
            types[role] = gguf.GGMLQuantizationType(tensor.kind).name
            rank_bytes = [
                a + b for a, b in zip(rank_bytes, packed_bank_bytes(tensor, tp_size))
            ]
        layer_records.append({"layer": layer, "types": types})
    if sum(EXPERT.fullmatch(name) is not None for name in tensors) != layers * 3:
        raise ValueError("Unexpected routed expert tensor count")

    engram_bytes = 0
    for layer, rows in zip(
        text["engram_layer_ids"], text["engram_num_embeddings"], strict=True
    ):
        tensor = tensors[f"blk.{layer}.engram_embd.weight"]
        if tensor.shape != (text["engram_head_dim"], rows) or tensor.kind != 20:
            raise ValueError("Engram table does not match HF config or IQ4_NL")
        engram_bytes += tensor.nbytes

    expected_mtp = {
        name: file
        for name, file in json.loads(
            (root / "base/model.safetensors.index.json").read_text()
        )["weight_map"].items()
        if name.startswith("mtp.")
    }
    seen_mtp = {}
    dtypes: Counter[str] = Counter()
    for path in sorted((root / "base/dspark").glob("*.safetensors")):
        for name, value in safetensors_names(path).items():
            if name in seen_mtp:
                raise ValueError(f"Duplicate DSpark tensor: {name}")
            seen_mtp[name] = path.name
            dtypes[value["dtype"]] += 1
    if seen_mtp != expected_mtp:
        raise ValueError("DSpark shards do not exactly cover the HF MTP index")
    return {
        "sha256_verified_files": sorted(verified),
        "files": file_records,
        "tensor_count": len(tensors),
        "tp_size": tp_size,
        "expert_source_bytes": expert_bytes,
        "expert_packed_bank_bytes_per_rank": rank_bytes,
        "expert_tp_boundary_overhead_bytes": sum(rank_bytes) - expert_bytes,
        "cold_expert_storage_estimates": {
            str(fraction): int(sum(rank_bytes) * (1 - fraction))
            for fraction in (0.65, 0.70, 0.75, 0.80)
        },
        "cold_estimate_assumptions": (
            "Ideal fraction only; excludes spare slots, host slack, arena "
            "alignment, load staging and prefill workspace"
        ),
        "expert_layers": layer_records,
        "engram_table_bytes": engram_bytes,
        "hf_pad_token_id": config["pad_token_id"],
        "gguf_tokenizer_padding_ids": sorted(padding_ids),
        "dspark_tensor_count": len(seen_mtp),
        "dspark_dtypes": dict(dtypes),
        "dspark_block_size": text["dspark_block_size"],
        "dspark_layers": text["num_nextn_predict_layers"],
        "memory_note": (
            "Packed layout estimate; excludes runtime buffers and measured allocation."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.model_root, args.tp_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("expert_layers", "files")
            }
        )
    )


if __name__ == "__main__":
    main()
