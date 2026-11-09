# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only byte and operand oracle for real mixed IQ4_XS/IQ3_S projections."""

import argparse
import json
import runpy
from pathlib import Path

import gguf
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=256)
    args = parser.parse_args()
    if args.chunk_rows <= 0 or args.chunk_rows % 32:
        parser.error("--chunk-rows must be a positive multiple of 32")
    source = Path(__file__).resolve().parents[2]
    api = runpy.run_path(
        str(source / "vllm/model_executor/layers/quantization/gguf_iq4_native.py")
    )
    reader = gguf.GGUFReader(str(args.model))
    tensors = {t.name: t for t in reader.tensors}
    layers = []
    for layer in range(64):
        pair = [tensors[f"blk.{layer}.ffn_{p}.weight"] for p in ("gate", "up")]
        if {int(t.tensor_type) for t in pair} == {21, 23}:
            layers.append(layer)
    assert len(layers) == 11
    rows = []
    for layer in layers:
        for role in ("gate", "up"):
            tensor = tensors[f"blk.{layer}.ffn_{role}.weight"]
            typ = int(tensor.tensor_type)
            if typ != 23:
                continue
            n, k = (int(v) for v in reversed(tensor.shape))
            record = {
                "layer": layer,
                "role": role,
                "gguf_type_id": typ,
                "N": n,
                "K": k,
                "source_bytes": int(tensor.n_bytes),
                "record_bytes": 0,
                "inverse_byte_mismatches": 0,
            }
            if typ == 23:
                record.update(
                    float_bit_mismatches=0,
                    half_bit_mismatches=0,
                    max_abs_error=0.0,
                    max_relative_error=0.0,
                    max_abs_weight=0.0,
                    half_nonfinite=0,
                )
            for start in range(0, n, args.chunk_rows):
                original = tensor.data[start : start + args.chunk_rows].copy()
                count = original.shape[0]
                packed = api["pack_iq4_xs_records"](original)
                restored = api["unpack_iq4_xs_records"](packed, count, k)
                record["inverse_byte_mismatches"] += int(
                    np.count_nonzero(restored != original)
                )
                official = gguf.quants.dequantize(
                    original, gguf.GGMLQuantizationType.IQ4_XS
                )
                decoded = api["dequantize_iq4_xs_records"](packed, count, k)
                half = api["dequantize_iq4_xs_records"](
                    packed, count, k, dtype=np.float16
                )
                record["float_bit_mismatches"] += int(
                    np.count_nonzero(
                        decoded.view(np.uint32) != official.view(np.uint32)
                    )
                )
                record["half_bit_mismatches"] += int(
                    np.count_nonzero(
                        half.view(np.uint16)
                        != official.astype(np.float16).view(np.uint16)
                    )
                )
                delta = np.abs(decoded - official)
                record["max_abs_error"] = max(
                    record["max_abs_error"], float(delta.max())
                )
                relative = delta / np.maximum(
                    np.abs(official), np.finfo(np.float32).tiny
                )
                record["max_relative_error"] = max(
                    record["max_relative_error"], float(relative.max())
                )
                record["max_abs_weight"] = max(
                    record["max_abs_weight"], float(np.abs(official).max())
                )
                record["half_nonfinite"] += int(np.count_nonzero(~np.isfinite(half)))
                assert packed.nbytes == original.nbytes
                record["record_bytes"] += packed.nbytes
            assert record["record_bytes"] == record["source_bytes"]
            assert record["inverse_byte_mismatches"] == 0
            if typ == 23:
                assert (
                    record["float_bit_mismatches"] == record["half_bit_mismatches"] == 0
                )
                assert record["half_nonfinite"] == 0
            rows.append(record)
            print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"complete": True, "cpu_only": True, "layers": layers, "tensors": rows},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
