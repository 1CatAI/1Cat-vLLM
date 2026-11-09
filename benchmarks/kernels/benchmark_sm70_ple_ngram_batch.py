# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the existing GPU PLE n-gram pipeline with one fused launch."""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

import vllm
from vllm.models.qwen4_exp.nvidia import ple_layer


def graph_us(call):
    for _ in range(5):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            call()
    samples = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 800)
    return statistics.median(samples), samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    capability = ple_layer.SM70_PLE_NGRAM
    for m in (5, 10, 15, 20, 32):
        layer = object.__new__(ple_layer.Qwen4ExpNGramEmbedding)
        torch.nn.Module.__init__(layer)
        layer.ngram_size, layer.heads_per_ngram, layer.ngram_heads = 3, 8, 16
        layer.eos_token_id = 151645
        layer.positions_buffer = torch.arange(m, device="cuda")
        layer.padded_buffer = torch.empty((32, m), dtype=torch.long, device="cuda")
        layer.layer_multipliers = torch.tensor(
            [-7046029254386353131, -4658895280553007687, -7723592293110705685],
            device="cuda",
        )
        layer.ngram_heads_vocab_sizes = torch.arange(20000101, 20000117, device="cuda")
        layer.ngram_heads_offsets = torch.arange(16, device="cuda") * 20000117
        ids = torch.arange(m, device="cuda", dtype=torch.int32) * 7919 + 12
        starts = torch.tensor(
            list(range(0, m, 5)) + [m], device="cuda", dtype=torch.int32
        )
        context = torch.full(
            (starts.numel() - 1, 2), 17, device="cuda", dtype=torch.int32
        )
        call = partial(layer.compute_ngram_ids, ids, starts, context)
        ple_layer.SM70_PLE_NGRAM = SimpleNamespace(reason=lambda *a: "legacy_control")
        expected = call()
        old, old_samples = graph_us(call)
        ple_layer.SM70_PLE_NGRAM = capability
        actual = call()
        if not torch.equal(actual, expected):
            raise AssertionError(f"M={m}: fused IDs differ from the existing GPU path")
        new, new_samples = graph_us(call)
        row = dict(
            m=m,
            requests=starts.numel() - 1,
            legacy_us=old,
            fused_us=new,
            saved_us=old - new,
            exact=True,
            legacy_samples=old_samples,
            fused_samples=new_samples,
        )
        results.append(row)
        print(json.dumps(row), flush=True)
    core = Path(vllm.__file__).parent / "_C.abi3.so"
    result = dict(
        version=vllm.__version__,
        package=str(vllm.__file__),
        core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        results=results,
        scope="Synthetic integer inputs; model geometry; CUDA graph operator timing",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
