# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test whether HC's narrow N=88 projection benefits from more K partitions.

Synthetic FP16 operands, serving dimensions; all packing/reduction costs are
included. This is a rejection/admission experiment, not model qualification.
"""

import argparse
import json
from functools import partial

import torch
from moe_tiles import measure, relative_l2


def partitioned(x, weight, splits):
    rows, k = x.shape
    n = weight.shape[0]
    partials = torch.bmm(
        x.view(rows, splits, k // splits).permute(1, 0, 2),
        weight.view(n, splits, k // splits).permute(1, 2, 0),
        out_dtype=torch.float32,
    )
    return partials.sum(dim=0).half()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[7344, 5347])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    for rows in args.tokens:
        gen = torch.Generator(device="cuda").manual_seed(rows)
        x = torch.randn(rows, 10240, device="cuda", dtype=torch.half, generator=gen)
        weight = (
            torch.randn(88, 10240, device="cuda", generator=gen) / 10240**0.5
        ).half()
        expected = x.float() @ weight.float().T
        baseline = torch.mm(x, weight.T)
        result = {"tokens": rows, "candidates": []}
        result["baseline"] = measure(partial(torch.mm, x, weight.T), 20)
        result["baseline_rel_l2"] = relative_l2(baseline, expected)
        for splits in (2, 4, 8):
            candidate = partitioned(x, weight, splits)
            result["candidates"].append(
                {
                    "splits": splits,
                    **measure(partial(partitioned, x, weight, splits), 20),
                    "fp32_rel_l2": relative_l2(candidate, expected),
                    "baseline_rel_l2": relative_l2(candidate, baseline),
                    "finite": bool(torch.isfinite(candidate).all()),
                }
            )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
