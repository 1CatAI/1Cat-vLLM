# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen existing HC combine/norm launch parameters at real prefill shapes.

Synthetic inputs have the checkpoint's HC4/D2560 layout. All arms use the
same preallocated outputs and preserve FP16 rounding before normalization.
This screen changes no runtime admission or launch defaults.
"""

import argparse
import json
from pathlib import Path

import torch
from moe_tiles import measure, relative_l2

from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_combine_norm_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[7344, 8976])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"cases": []}
    for m in args.tokens:
        torch.manual_seed(9683 + m)
        residual = torch.randn(m, 4 * 2560, device="cuda", dtype=torch.half) * 0.5
        block = torch.randn(m, 2560, device="cuda", dtype=torch.half) * 0.5
        injection = torch.randn(m, 4, device="cuda", dtype=torch.half)
        weight = torch.randn(4 * 2560, device="cuda", dtype=torch.half) * 0.1
        out, normed = torch.empty_like(residual), torch.empty_like(residual)

        def run(tile, warps, tensors=(block, residual, injection, weight, out, normed)):
            block, residual, injection, weight, out, normed = tensors
            return _hc_combine_norm_kernel[(block.shape[0], 4)](
                block,
                residual,
                injection,
                weight,
                out,
                normed,
                block.stride(0),
                residual.stride(0),
                injection.stride(0),
                out.stride(0),
                normed.stride(0),
                2560,
                4,
                W_SHARED=False,
                EPS=1e-6,
                BLOCK_SIZE=tile,
                launch_pdl=False,
                PREFETCH_WEIGHT=False,
                num_warps=warps,
            )

        run(512, 4)
        control = (out.clone(), normed.clone())
        r = residual[:16].float().view(16, 4, 2560)
        scale = 2 * (injection[:16].float() / 4).sigmoid()
        expected_out = (r + block[:16, None].float() * scale[..., None]).half()
        x = expected_out.float()
        normalized = x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        expected_norm = (normalized * (1 + weight.float().view(4, 2560))).half()
        expected = (expected_out.flatten(1), expected_norm.flatten(1))
        for index, (tile, warps) in enumerate(
            [(512, 4), (512, 8), (1024, 4), (1024, 8), (256, 8), (512, 4)]
        ):
            compiled = run(tile, warps)
            paired = max(relative_l2(a, b) for a, b in zip((out, normed), control))
            fp32 = max(relative_l2(a[:16], b) for a, b in zip((out, normed), expected))
            row = {
                "tokens": m,
                "tile": tile,
                "warps": warps,
                "repeat_control": index == 5,
                "registers": getattr(compiled, "n_regs", None),
                "spills": getattr(compiled, "n_spills", None),
                "paired_max_rel_l2": paired,
                "fp32_max_rel_l2": fp32,
                **measure(lambda tile=tile, warps=warps: run(tile, warps), 5),
                "passed": bool(torch.isfinite(out).all())
                and bool(torch.isfinite(normed).all())
                and max(paired, fp32) <= 0.001,
            }
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    raise SystemExit(0 if all(row["passed"] for row in report["cases"]) else 1)


if __name__ == "__main__":
    main()
