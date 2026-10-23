# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen copying only occupied cold slots, retaining their staging offsets.

Uses three real TP4 layer-zero shard sizes and a synthetic fragmented slot
map. Timing includes Python/native submission of every contiguous range.
This is a DMA screen; model transcode overlap must be measured separately.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from moe_tiles import artifact_identity, measure

from vllm.model_executor.layers.quantization.gguf_expert_cache import _native
from vllm.transformers_utils.gguf_tensor_reader import quant_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate_block, gate_size = quant_size(18)
    down_block, down_size = quant_size(20)
    slot_bytes = [160 * (2560 // gate_block) * gate_size] * 2 + [
        2560 * (160 // down_block) * down_size
    ]
    slots = 136
    host = [
        torch.empty(slots, size, dtype=torch.uint8).pin_memory() for size in slot_bytes
    ]
    for tensor in host:
        tensor.random_(0, 256)
    stage = [torch.empty_like(tensor, device="cuda") for tensor in host]
    copy = _native().copy_h2d
    stream = torch.cuda.current_stream().cuda_stream
    report = {"artifact": artifact_identity(), "slot_bytes": slot_bytes, "cases": []}
    for cold in (110, 104):
        selected = torch.randperm(slots, generator=torch.Generator().manual_seed(9683))[
            :cold
        ]
        occupied = set(selected.tolist())
        extent = max(occupied) + 1
        ranges = []
        for slot in sorted(occupied):
            if ranges and ranges[-1][1] == slot:
                ranges[-1][1] += 1
            else:
                ranges.append([slot, slot + 1])

        def submit(spans):
            for src, dst, size in zip(host, stage, slot_bytes):
                for begin, end in spans:
                    copy(
                        dst.data_ptr() + begin * size,
                        src.data_ptr() + begin * size,
                        (end - begin) * size,
                        stream,
                    )

        bulk = partial(submit, [(0, extent)])
        sparse = partial(submit, ranges)
        for dst in stage:
            dst.zero_()
        sparse()
        torch.accelerator.synchronize()
        for src, dst in zip(host, stage):
            actual = dst.cpu()
            torch.testing.assert_close(actual[selected], src[selected], rtol=0, atol=0)
            holes = torch.tensor(sorted(set(range(slots)) - occupied))
            assert (actual[holes] == 0).all()
        row = {
            "cold_slots": cold,
            "extent": extent,
            "ranges": len(ranges),
            "bulk_bytes": extent * sum(slot_bytes),
            "selected_bytes": cold * sum(slot_bytes),
            "bulk": measure(bulk, 10),
            "selected": measure(sparse, 10),
            "bulk_repeat": measure(bulk, 10),
            "cold_exact_holes_untouched": True,
        }
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
