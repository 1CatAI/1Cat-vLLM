# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen PLE prefill history elimination, including state commit and SiLU.

Before candidate execution: finite outputs and maximum row relative-L2 <=
0.001 versus the production convolution; state must match exactly, including
unused/speculative slots. This is a microbenchmark, not model qualification.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from moe_tiles import measure, relative_l2

from vllm.model_executor.kernels.ple.short_conv import dilated_prefill_short_conv
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"cases": [], "maximum_row_relative_l2_limit": 0.001}
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(module)
    module.conv_state_len = 9
    module.short_conv_dilation = 3
    for lengths in ([7344], [8976], [1600] * 4, [4, 0, 13, 7]):
        n, rows, hidden = len(lengths), sum(lengths), 10240
        torch.manual_seed(9683 + rows)
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.half) * 0.5
        weight = torch.randn(hidden, 4, device="cuda", dtype=torch.half) * 0.125
        seed = torch.randn(n + 2, hidden, 13, device="cuda", dtype=torch.half) * 0.5
        control_state, candidate_state = seed.clone(), seed.clone()
        starts = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        slots = torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
        initial = torch.arange(n, device="cuda") % 2 == 0
        if lengths == [4, 0, 13, 7]:
            slots[-1] = NULL_BLOCK_ID
        metadata = SimpleNamespace(
            non_spec_query_start_loc=starts,
            has_initial_states_p=initial,
            max_prefill_query_len=max(lengths),
        )
        max_length = max(lengths)

        def control(
            inputs=(x, metadata, control_state, weight, slots, n, 0, rows), seed=seed
        ):
            control_state = inputs[2]
            control_state.copy_(seed)
            return module._short_conv_dilated_prefill_batched(*inputs)

        def candidate(
            tile,
            inputs=(
                x,
                candidate_state,
                weight,
                starts,
                slots,
                initial,
                max_length,
                0,
                3,
                NULL_BLOCK_ID,
            ),
            seed=seed,
        ):
            candidate_state = inputs[1]
            candidate_state.copy_(seed)
            return dilated_prefill_short_conv(
                *inputs,
                token_tile=tile,
            )

        def peak(fn):
            torch.accelerator.synchronize()
            torch.accelerator.reset_peak_memory_stats()
            before = torch.accelerator.memory_allocated()
            output = fn()
            torch.accelerator.synchronize()
            grown = torch.accelerator.max_memory_allocated() - before
            del output
            return grown

        expected = control()
        row = {
            "lengths": lengths,
            "control": measure(control, 5),
            "control_peak_bytes": peak(control),
            "candidates": [],
        }
        for tile in (4, 8, 16):
            actual = candidate(tile)
            error = relative_l2(actual, expected)
            exact_state = torch.equal(candidate_state, control_state)
            case = {
                "token_tile": tile,
                "paired_max_rel_l2": error,
                "output_exact": torch.equal(actual, expected),
                "state_exact": exact_state,
                "passed": error <= 0.001
                and exact_state
                and bool(torch.isfinite(actual).all()),
                **measure(lambda tile=tile: candidate(tile), 5),
                "peak_bytes": peak(lambda tile=tile: candidate(tile)),
            }
            row["candidates"].append(case)
        row["control_repeat"] = measure(control, 5)
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    raise SystemExit(
        0 if all(c["passed"] for r in report["cases"] for c in r["candidates"]) else 1
    )


if __name__ == "__main__":
    main()
