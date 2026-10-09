# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare CPU expert conversion on RAM-resident GGUF payloads, in ABBA order."""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np

from vllm.model_executor.layers.quantization import gguf_lattice_transcode as lattice
from vllm.model_executor.layers.quantization import gguf_lut_transcode as lut
from vllm.model_executor.layers.quantization import gguf_transcode as affine
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, quant_type_name


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def convert(data, kind, affine_module, lattice_module):
    if kind in lattice.LATTICE_TYPES:
        return lattice_module.transcode_lattice(data, kind)
    if kind in lut.LUT4_TYPES:
        return lut.transcode_lut4(data, kind)
    return affine_module.transcode_affine(data, kind)


parser = argparse.ArgumentParser()
parser.add_argument("--candidate-dir", required=True)
parser.add_argument("--control-dir")
parser.add_argument("--output", required=True)
parser.add_argument("models", nargs="+")
args = parser.parse_args()
candidate_dir = Path(args.candidate_dir)
new_affine = load("candidate_affine", candidate_dir / "gguf_transcode.py")
new_lattice = load("candidate_lattice", candidate_dir / "gguf_lattice_transcode.py")
if args.control_dir:
    control_dir = Path(args.control_dir)
    affine = load("control_affine", control_dir / "gguf_transcode.py")
    lattice = load("control_lattice", control_dir / "gguf_lattice_transcode.py")
fixtures = {}
counts = Counter()
readers = []
for path in args.models:
    reader = GGUFReader(path, mode="r")
    readers.append(reader)
    for tensor in reader.tensors:
        if not tensor.name.endswith("_exps.weight") or tensor.data.ndim != 3:
            continue
        kind = int(tensor.tensor_type)
        axis = 1 if ".ffn_down_exps." in tensor.name else 0
        key = (kind, axis, tuple(int(x) for x in tensor.shape[:2]))
        counts[key] += int(tensor.data.shape[0])
        fixtures.setdefault(key, (tensor.name, tensor.data[0]))


def prepare(data, kind, axis, rank, candidate):
    if candidate:
        part = new_affine.tp_slice_packed(data, kind, rank, 4, axis=axis)
        result = convert(data if part is None else part, kind, new_affine, new_lattice)
        if part is None:
            result = result.tp_slice(rank, 4, axis=axis)
    else:
        result = convert(data, kind, affine, lattice).tp_slice(rank, 4, axis=axis)
    packets = result.mma884_storage() if kind in lattice.LATTICE_TYPES else ()
    return result, packets


results = []
for key, (name, data) in fixtures.items():
    kind, axis, _shape = key
    source = np.array(data, copy=True, order="C")
    for rank in range(4):
        original, expected_packets = prepare(source, kind, axis, rank, False)
        candidate, actual_packets = prepare(source, kind, axis, rank, True)
        for field in fields(original):
            expected, actual = (
                getattr(original, field.name),
                getattr(candidate, field.name),
            )
            if isinstance(expected, np.ndarray):
                assert actual.dtype == expected.dtype
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected
        for expected, actual in zip(expected_packets, actual_packets):
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
    timings = {False: [], True: []}
    for _ in range(7):
        for candidate in (False, True, True, False):
            start = time.perf_counter()
            prepared = prepare(source, kind, axis, 0, candidate)
            elapsed = (time.perf_counter() - start) * 1000
            timings[candidate].append(elapsed)
            del prepared
    control_ms = statistics.median(timings[False])
    candidate_ms = statistics.median(timings[True])
    record = {
        "type": quant_type_name(kind),
        "axis": axis,
        "example": name,
        "expert_projections_per_rank": counts[key],
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "speedup": control_ms / candidate_ms,
        "all_four_ranks_bitwise_equal": True,
    }
    results.append(record)
    print(json.dumps(record), flush=True)
report = {
    "scope": (
        "RAM-resident CPU codec ABBA; excludes model startup, GPU preparation "
        "and disk I/O"
    ),
    "projections": results,
    "control_expert_conversion_estimate_s": sum(
        r["control_ms"] * r["expert_projections_per_rank"] / 1000 for r in results
    ),
    "candidate_expert_conversion_estimate_s": sum(
        r["candidate_ms"] * r["expert_projections_per_rank"] / 1000 for r in results
    ),
}
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "projections"}), flush=True)
