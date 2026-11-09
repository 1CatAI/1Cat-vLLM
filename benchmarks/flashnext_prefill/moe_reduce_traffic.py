# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen the traffic cost of a routed FP32-atomic GEMM epilogue.

Both paths read the same synthetic FP16 GEMM result. Control materializes it
then uses the serving unroute; candidate zeroes a FP32 output, scatters weighted
values with atomics, and converts to FP16. This omits GEMM and is only a cheap
rejection screen, not evidence that a fused GEMM will have the same timing.
"""

import argparse
import json
from pathlib import Path

import torch
from moe_tiles import measure, relative_l2

from vllm.model_executor.layers.quantization.gguf_expert_cache import _unroute
from vllm.triton_utils import tl, triton


@triton.jit
def scatter(source, order, weights, output, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    route = tl.load(order + row).to(tl.int64)
    weight = tl.load(weights + route)
    value = tl.load(source + row.to(tl.int64) * H + col, col < H, other=0)
    tl.atomic_add(
        output + (route // 10) * H + col,
        value.to(tl.float32) * weight,
        mask=col < H,
        sem="relaxed",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[4096, 3248])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens) < 1 or args.iterations < 1:
        parser.error("tokens and iterations must be positive")
    cases = []
    for tokens in args.tokens:
        torch.manual_seed(tokens)
        scores = torch.randn(tokens, 512, device="cuda")
        top, ids = scores.topk(10, dim=1)
        weights = top.softmax(dim=1).float()
        _, order = ids.flatten().sort(stable=True)
        inverse = torch.empty_like(order, dtype=torch.int32)
        inverse[order] = torch.arange(order.numel(), device="cuda", dtype=torch.int32)
        source = torch.randn(tokens * 10, 2560, dtype=torch.half, device="cuda")
        down = torch.empty_like(source)
        output = torch.empty(tokens, 2560, dtype=torch.half, device="cuda")
        accumulator = torch.empty_like(output, dtype=torch.float32)

        def control(
            down=down,
            source=source,
            tokens=tokens,
            weights=weights,
            inverse=inverse,
            output=output,
        ):
            down.copy_(source)
            _unroute[(tokens, 3)](
                down, weights, inverse, output, 2560, TOPK=10, BLOCK=1024
            )
            return output

        def candidate(
            block,
            accumulator=accumulator,
            tokens=tokens,
            source=source,
            order=order,
            weights=weights,
            output=output,
        ):
            accumulator.zero_()
            scatter[(tokens * 10, triton.cdiv(2560, block))](
                source, order, weights, accumulator, 2560, block, num_warps=4
            )
            output.copy_(accumulator)
            return output

        reference = control().clone()
        row = {"tokens": tokens, "control": measure(control, args.iterations)}
        results = []
        for block in (256, 1024):
            actual = candidate(block)
            error = relative_l2(actual, reference)
            finite = bool(torch.isfinite(actual).all())
            # Only FP32 accumulation order changes; retain the FP16 source
            # rounding boundary. Predetermined paired row-relative L2 bound.
            passed = finite and error <= 0.001
            result = {
                "block": block,
                "max_row_relative_l2": error,
                "finite": finite,
                "passed": passed,
                **measure(lambda block=block: candidate(block), args.iterations),
            }
            results.append(result)
        row["candidates"] = results
        row["control_repeat"] = measure(control, args.iterations)
        cases.append(row)
        print(json.dumps(row), flush=True)
    report = {"gpu": torch.cuda.get_device_name(), "cases": cases}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(c["passed"] for row in cases for c in row["candidates"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
