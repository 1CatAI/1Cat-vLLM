# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy MTP verification graph screen; excludes head and TP communication."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_sm70_mtp_fp32_dense import capture, elapsed
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    greedy_rejection_from_top_tokens,
    rejection_sample,
)


def screen(num_reqs: int) -> dict:
    rows = num_reqs * 5
    logits = torch.randn(rows, 248320, dtype=torch.float16, device="cuda")
    indices = torch.arange(rows, dtype=torch.int64, device="cuda")
    proposals = logits.argmax(-1).to(torch.int32).roll(1)
    cu = torch.arange(num_reqs + 1, dtype=torch.int32, device="cuda") * 5
    mapping = torch.arange(num_reqs, dtype=torch.int32, device="cuda")
    expanded = mapping.repeat_interleave(5)
    positions = torch.arange(5, dtype=torch.int32, device="cuda").repeat(num_reqs)
    temperature = torch.zeros(num_reqs, device="cuda")
    seeds = mapping.to(torch.int64)
    top_ids = logits.argmax(-1)
    outputs = [None, None, None]

    def dense():
        # MRv2 apply_sampling_params materializes an FP32 copy even when no
        # transforms are requested. Projection and all-gather are excluded.
        processed = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
        outputs[0] = rejection_sample(
            processed,
            None,
            proposals,
            cu,
            positions,
            mapping,
            expanded,
            positions,
            temperature,
            seeds,
            4,
        )

    def compact_with_argmax():
        outputs[1] = greedy_rejection_from_top_tokens(
            logits.argmax(-1), proposals, indices, cu, 4
        )

    def compact_ids_only():
        outputs[2] = greedy_rejection_from_top_tokens(
            top_ids, proposals, indices, cu, 4
        )

    graphs = [capture(fn) for fn in (dense, compact_with_argmax, compact_ids_only)]
    for graph in graphs:
        graph.replay()
    torch.accelerator.synchronize()
    reference, counts = outputs[0]
    mask = torch.arange(5, device="cuda")[None, :] < counts[:, None]
    for candidate, candidate_counts in outputs[1:]:
        torch.testing.assert_close(candidate_counts, counts)
        torch.testing.assert_close(candidate[mask], reference[mask])
    samples = [[], [], []]
    for trial in range(7):
        for idx in (0, 1, 2) if trial % 2 else (2, 1, 0):
            samples[idx].append(elapsed(graphs[idx]))
    return {
        "num_reqs": num_reqs,
        "logit_rows": rows,
        "arms": ["dense_fp32_copy_rejection", "argmax_rejection", "ids_rejection"],
        "median_ms": [statistics.median(values) for values in samples],
        "samples_ms": samples,
        "outputs_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    report = {
        "complete_round_measurement": False,
        "includes_head_or_tp_communication": False,
        "rows": [screen(num_reqs) for num_reqs in (1, 4)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
