# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare TP2 scalar attention over distinct KV layer working sets.

This measures attention operators, not model verification rounds. Run the
bitwise kernel tests separately before considering a model experiment.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch

FLAG = "VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[270, 1100])
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("requires an owned SM70 GPU")
    from flash_attn_v100 import flash_attn_interface as interface

    native = interface.flash_attn_v100_cuda
    if getattr(native, "tp2_e4m3_scalar_fast_version", lambda: 0)() < 2:
        raise RuntimeError("rebuild Flash-V100 with TP2 scalar fast revision 2")
    if args.layers < 1 or any(not 8 <= n <= 262144 for n in args.context_lengths):
        raise ValueError("positive layer count and context lengths 8..262144 required")
    torch.manual_seed(20260908)
    report = {
        "measurement": "distinct-KV attention working set, not complete model rounds",
        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "layers": args.layers,
        "rows": [],
    }
    previous = os.environ.get(FLAG)
    try:
        for length in args.context_lengths:
            report["rows"].append(measure(native, length, args.layers))
    finally:
        if previous is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = previous
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def measure(native, length, layers):
    page, parts = 3296, 256
    pages = (length + page - 1) // page
    operands = []
    for _ in range(layers):
        kv = torch.randn((pages, 2, page, 2, 256), device="cuda", dtype=torch.float16)
        k, v = kv.to(torch.float8_e4m3fn).view(torch.uint8).unbind(1)
        q = torch.randn((8, 12, 256), device="cuda", dtype=torch.float16)
        table = torch.randperm(pages, device="cuda").int()[None].repeat(8, 1)
        seq = torch.arange(length - 7, length + 1, device="cuda").int()
        operands.append((q, k, v, table, seq))
    out = torch.empty_like(operands[0][0])
    tmp = torch.empty((8, 12, parts, 256), device="cuda")
    maxima = torch.empty((8, 12, parts), device="cuda")
    sums = torch.empty_like(maxima)
    active = torch.full((1,), parts, device="cuda", dtype=torch.int32)

    def call(operand):
        q, k, v, table, seq = operand
        native.decode_paged_fwd(
            q,
            k,
            v,
            out,
            table,
            seq,
            tmp,
            maxima,
            sums,
            active,
            0.0625,
            1024,
            parts,
            "fp8_e4m3",
            0.5,
            1.25,
            -1,
            -1,
            None,
            0,
        )

    # Compare every layer before capturing a shared-workspace timing graph.
    for operand in operands:
        os.environ[FLAG] = "0"
        call(operand)
        expected = out.clone()
        os.environ[FLAG] = "1"
        call(operand)
        if not torch.equal(out.view(torch.int16), expected.view(torch.int16)):
            raise AssertionError("candidate output differs from control")
    graphs = {}
    for enabled in ("0", "1"):
        os.environ[FLAG] = enabled
        graph = torch.cuda.CUDAGraph()
        before = native.tp2_e4m3_scalar_fast_launch_count()
        with torch.cuda.graph(graph):
            for operand in operands:
                call(operand)
        count = native.tp2_e4m3_scalar_fast_launch_count() - before
        assert count == (layers if enabled == "1" else 0)
        graphs[enabled] = graph
    samples = {enabled: [] for enabled in graphs}
    for trial in range(7):
        for enabled in ("0", "1") if trial % 2 == 0 else ("1", "0"):
            graph = graphs[enabled]
            graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(4):
                graph.replay()
            end.record()
            end.synchronize()
            samples[enabled].append(start.elapsed_time(end) / 4)
    return {
        "context_length": length,
        "all_layer_outputs_bitwise_equal": True,
        "samples_ms": samples,
        "median_ms": {k: statistics.median(v) for k, v in samples.items()},
    }


if __name__ == "__main__":
    main()
