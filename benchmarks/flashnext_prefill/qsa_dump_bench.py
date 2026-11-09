# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time / check QSA prefill readers on a real dump: run_tiled (64-row steps)
vs optional new op. argv: dump [op_name]"""

import os
import sys

import torch

if os.environ.get("QSA_SO"):
    torch.ops.load_library(os.environ["QSA_SO"])
else:
    import vllm._sm70_qsa_device_C  # noqa: F401
d = torch.load(sys.argv[1])
g = {k: (v.cuda() if v is not None else None) for k, v in d.items()}
if g["gate"] is not None:
    g["gate"] = g["gate"].reshape(g["query"].shape)
g["output"] = g["output"].reshape(g["query"].shape)
rows = g["query"].shape[0]
width = g["indices"].shape[1]
splits = (width + 63) // 64
partial = torch.empty(64 * 6 * splits * 256, device="cuda")
state = torch.empty(64 * 6 * splits * 2, device="cuda")


def base(out):
    torch.ops.vllm_sm70_qsa_device.run_tiled(
        g["query"],
        g["history"],
        g["scales"],
        g["indices"],
        g["table"],
        g["requests"],
        g["positions"],
        g["lengths"],
        out,
        g["gate"],
        partial,
        state,
        64,
    )


def timeit(fn, out, it=10):
    fn(out)
    torch.accelerator.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn(out)
    e.record()
    torch.accelerator.synchronize()
    return s.elapsed_time(e) / it


out = torch.empty_like(g["query"])
ms = timeit(base, out)
ref = g["output"]
print(
    f"run_tiled: {ms:.2f} ms for {rows} rows; "
    f"matches dumped output: {torch.equal(out, ref)} "
    f"(max diff {(out.float() - ref.float()).abs().max().item():.3g})"
)
if len(sys.argv) > 2:
    op = getattr(torch.ops.vllm_sm70_qsa_device, sys.argv[2])
    extra = [int(x) for x in sys.argv[3:]]

    def new(o):
        op(
            g["query"],
            g["history"],
            g["scales"],
            g["indices"],
            g["table"],
            g["requests"],
            g["positions"],
            g["lengths"],
            o,
            g["gate"],
            *extra,
        )

    out2 = torch.empty_like(out)
    ms2 = timeit(new, out2)
    diff = (out2.float() - ref.float()).abs()
    rel = diff.norm(dim=-1) / ref.float().norm(dim=-1).clamp_min(1e-6)
    print(
        f"{sys.argv[2:]}: {ms2:.2f} ms ({ms / ms2:.2f}x); "
        f"max abs diff {diff.max().item():.3g}, "
        f"max rel-L2 per head {rel.max().item():.3g}, "
        f"mean {rel.mean().item():.3g}, nan {torch.isnan(out2).any().item()}"
    )
