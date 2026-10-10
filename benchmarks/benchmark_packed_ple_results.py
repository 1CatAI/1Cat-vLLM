# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU producer A/B using real GGUF rows and recorded output-token prefixes.

This measures CPU hash, lookup and four-rank result publication. It excludes
IPC, page-fault latency after warmup, and GPU consumption. It is not a model
latency or acceptance benchmark.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import torch

from vllm.model_executor.kernels.ple.host_result import publish_host_flag
from vllm.model_executor.kernels.ple.packed_result import packed_result_capability
from vllm.model_executor.layers.ple_offload_layer import mark_as_offload_worker
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding
from vllm.transformers_utils.gguf_files import gguf_shard_paths
from vllm.transformers_utils.gguf_rows import PackedGGUFRowReader
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader


def summarize(samples):
    values = np.asarray(samples, dtype=float) / 1000
    return dict(
        mean_us=float(values.mean()),
        p50_us=float(np.median(values)),
        p90_us=float(np.quantile(values, 0.9)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("acceptance_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=64)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("cycles must be positive")
    torch.set_num_threads(1)
    mark_as_offload_worker()
    owners = [GGUFReader(path) for path in gguf_shard_paths(args.model)]
    tensor = next(
        t
        for owner in owners
        for t in owner.tensors
        if t.name == "per_layer_token_embd.weight"
    )
    if int(tensor.tensor_type) != 20 or int(tensor.shape[0]) != 160:
        parser.error("This diagnostic requires IQ4_NL PLE rows of width 160")
    metadata = {
        key: value.contents()
        for owner in owners
        for key, value in owner.fields.items()
        if key.startswith("qwen4exp.ple.")
    }
    reader = PackedGGUFRowReader(tensor.data, 20, 160)
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(module)
    module.layer_name = "producer_diagnostic"
    module.ngram_size, module.heads_per_ngram, module.ngram_heads = 3, 8, 16
    module.embedding_dim, module.head_dim = 2560, 160
    module.eos_token_id = int(metadata["qwen4exp.ple.eos_token_id"])
    for attribute, key in [
        ("layer_multipliers", "layer_multipliers"),
        ("ngram_heads_vocab_sizes", "head_vocab_sizes"),
        ("ngram_heads_offsets", "head_offsets"),
    ]:
        module.register_buffer(
            attribute, torch.tensor(metadata["qwen4exp.ple." + key], dtype=torch.int64)
        )
    module.positions_buffer = torch.arange(512)
    module.padded_buffer = torch.empty((4, 512), dtype=torch.int64)
    module._packed_gguf, module._cascade = True, False
    module.ngram_embedding = SimpleNamespace(
        _cpu_reader=reader,
        embedding_lookup=lambda ids: torch.from_numpy(reader.lookup(ids.numpy())),
    )
    layout = packed_result_capability(
        enabled=True,
        source_type=20,
        row_width=160,
        heads=16,
        fp16=True,
        sm70=True,
        offloaded=True,
        local_tables=False,
    )
    sequences = [
        row["output_token_ids"]
        for row in json.loads(args.acceptance_report.read_text())["rows"]
    ]
    if len(sequences) < 4 or any(len(seq) < 170 for seq in sequences[:4]):
        parser.error(
            "Four recorded prefixes with at least 170 output tokens are required"
        )
    results = []
    with torch.inference_mode():
        for m, requests in [(5, 1), (20, 4)]:
            starts = torch.arange(requests + 1, dtype=torch.int32) * 5
            batches = [
                (
                    torch.tensor(
                        [
                            token
                            for seq in sequences[:requests]
                            for token in seq[pos : pos + 5]
                        ],
                        dtype=torch.int32,
                    ),
                    torch.tensor(
                        [seq[pos - 2 : pos] for seq in sequences[:requests]],
                        dtype=torch.int32,
                    ),
                )
                for pos in range(2, 162, 5)
            ]
            buffers = [
                [torch.empty((m, width), dtype=dtype) for _ in range(4)]
                for width, dtype in [(2560, torch.float16), (1440, torch.uint8)]
            ]
            flags = [torch.zeros(16, dtype=torch.int32) for _ in range(4)]

            def produce(arm, batch, starts=starts, buffers=buffers, flags=flags):
                module._packed_result_layout = layout if arm else None
                ids, context = batch
                result = module.forward_impl(ids, ids, starts, context, buffers[arm][0])
                for rank in range(4):
                    if rank:
                        buffers[arm][rank].copy_(result)
                    publish_host_flag(flags[rank])
                return result

            for batch in batches:
                expected = produce(0, batch).numpy().copy()
                packed = produce(1, batch).numpy().reshape(m * 16, 90)
                actual = (
                    gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.IQ4_NL)
                    .astype(np.float16)
                    .reshape(m, 2560)
                )
                np.testing.assert_array_equal(
                    actual.view(np.uint8), expected.view(np.uint8)
                )
            times = [[], []]
            for cycle in range(args.cycles):
                batch = batches[cycle % len(batches)]
                order = [0, 1, 1, 0] if cycle % 2 == 0 else [1, 0, 0, 1]
                for arm in order:
                    begin = time.perf_counter_ns()
                    produce(arm, batch)
                    times[arm].append(time.perf_counter_ns() - begin)
            results.append(
                dict(
                    m=m,
                    requests=requests,
                    check="All warmup result bytes exact",
                    control=summarize(times[0]),
                    packed=summarize(times[1]),
                    saving_mean_us=float(
                        (np.mean(times[0]) - np.mean(times[1])) / 1000
                    ),
                )
            )
    report = dict(
        scope=__doc__,
        torch_threads=torch.get_num_threads(),
        source_type="IQ4_NL",
        k=160,
        heads=16,
        cycles=args.cycles,
        results=results,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
