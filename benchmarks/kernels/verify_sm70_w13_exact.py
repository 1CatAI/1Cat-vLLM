# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W13 scheduling screen with frozen-binary oracle and changing graph inputs."""

import argparse
import json
from pathlib import Path

import torch
from verify_sm70_qsa_router_exact import capture, paired_timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--old-library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--mode", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    torch.accelerator.set_device_index(0)
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(20260906)
    torch.ops.load_library(str(args.old_library))
    torch.ops.load_library(str(args.library))
    old = torch.ops._C_qwen38.nvfp4_qwen38_w13_fused_swiglu_out
    op = torch.ops._C_w13_exact.run
    weights = torch.randint(
        -(2**31), 2**31 - 1, (512, 2560, 40), dtype=torch.int32, device="cuda"
    )
    scales = torch.empty(512, 160, 320, dtype=torch.float16, device="cuda")
    scales.uniform_(2**-10, 2**-6)
    x = torch.randn(48, 1, 2560, dtype=torch.float16, device="cuda")
    ids = torch.randint(512, (48, 10), dtype=torch.int32, device="cuda")
    a = torch.empty(48, 10, 160, dtype=torch.float16, device="cuda")
    b, c = torch.empty_like(a), torch.empty_like(a)
    scratch = torch.empty(48, 10, 10, 16, 32, dtype=torch.float32, device="cuda")

    def run(dst, mode):
        for i in range(48):
            if mode == "installed":
                old(dst[i], x[i], weights, scales, ids[i])
            else:
                op(
                    dst[i],
                    x[i],
                    weights,
                    scales,
                    ids[i],
                    scratch[i],
                    args.mode if mode == "static" else 0,
                )

    ga = capture(lambda: run(a, "installed"))
    gb = capture(lambda: run(b, "dynamic"))
    gc = capture(lambda: run(c, "static"))
    if args.profile_only:
        for _ in range(300):
            gb.replay()
            gc.replay()
        torch.accelerator.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        op(b[0], x[0], weights, scales, ids[0], scratch[0], 0)
        op(c[0], x[0], weights, scales, ids[0], scratch[0], args.mode)
        torch.accelerator.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    for replay in range(64):
        x.normal_()
        ids.random_(512)
        if replay % 8 == 0:
            ids[:, ::3] = -1
        elif replay % 8 == 1:
            ids[:, ::3] = 512
        elif replay % 8 == 2:
            ids.zero_()
        if replay % 4 == 0:
            x.mul_(1e-3)
        elif replay % 4 == 1:
            x.mul_(16)
        elif replay % 4 == 2:
            x[:, :, ::3] = -0.0
        b.fill_(float("nan"))
        c.fill_(float("nan"))
        scratch.fill_(float("nan"))
        ga.replay()
        gb.replay()
        gc.replay()
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (replay, "build")
        assert torch.equal(a.view(torch.int16), c.view(torch.int16)), (replay, "static")
    # Time varied valid experts, not an all-invalid or repeated-expert case.
    ids.copy_(
        (
            torch.arange(10, device="cuda")
            + 13 * torch.arange(48, device="cuda")[:, None]
        )
        % 512
    )
    x.normal_()
    result = {
        "scope": "synthetic packed NVFP4 payload/scale screen, not model quality",
        "calls": 48,
        "candidate_mode": args.mode,
        "split_k": 16,
        "m1_changed_input_graph_replays": 64,
        "exact_vs_installed_old_and_rebuilt_control": True,
        "times": paired_timing(gb, gc),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
