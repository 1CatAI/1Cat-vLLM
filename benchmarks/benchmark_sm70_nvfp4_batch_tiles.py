# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen batched NVFP4 projections with real TP weights and cold L2.

Prototype extensions are for standalone measurements only. They never replace
operators in a vLLM service. Projection error against the installed operator
is diagnostic and does not substitute for a teacher-forcing quality gate.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _sm70_ops as ops
from vllm.model_executor.kernels.linear.nvfp4.sm70 import _qpn2_config

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--m", type=int, nargs="+", default=[8, 16, 32, 64])
p.add_argument("--projection", choices=["gate_up", "down", "both"], default="both")
p.add_argument("--ncu", action="store_true")
p.add_argument("--ncu-candidate", action="store_true")
p.add_argument("--candidate", type=Path)
p.add_argument("--candidate-so", type=Path)
a = p.parse_args()
if a.ncu_candidate and not (a.candidate or a.candidate_so):
    p.error("--ncu-candidate requires a candidate source or binary")
a.out.parent.mkdir(parents=True, exist_ok=True)
candidate_binary = a.candidate_so
torch.manual_seed(20261005)
torch.backends.cuda.matmul.allow_tf32 = False
if a.candidate_so:
    torch.ops.load_library(str(a.candidate_so))
elif a.candidate:
    from torch.utils.cpp_extension import load as load_extension

    extension_name = (
        "qwen_batch_screen_" + hashlib.sha256(a.candidate.read_bytes()).hexdigest()[:12]
    )
    candidate_binary = Path(
        load_extension(
            name=extension_name,
            sources=[str(a.candidate)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "--ptxas-options=-v",
                "--maxrregcount=96",
                "-gencode=arch=compute_70,code=sm_70",
            ],
            is_python_module=False,
            verbose=True,
        )
    )

index = json.loads((a.model / "model.safetensors.index.json").read_text())["weight_map"]
pre = "model.language_model.layers.55.mlp."


def load(proj, field):
    name = pre + proj + "." + field
    with safe_open(a.model / index[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def shard(proj, down=False):
    w = load(proj, "weight_packed")
    s = load(proj, "weight_scale")
    if down:
        w = w[:, : w.shape[1] // 4].contiguous()
        s = s[:, : s.shape[1] // 4].contiguous()
    else:
        w = w[: w.shape[0] // 4].contiguous()
        s = s[: s.shape[0] // 4].contiguous()
    return w, s, 1 / float(load(proj, "weight_global_scale").max())


result = {
    "candidate_source_sha256": hashlib.sha256(a.candidate.read_bytes()).hexdigest()
    if a.candidate
    else None,
    "candidate_binary_sha256": (
        hashlib.sha256(candidate_binary.read_bytes()).hexdigest()
        if candidate_binary
        else None
    ),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(),
    "cold_l2_bytes": 128 * 1024 * 1024,
    "activation_fixture": "seeded Gaussian FP16, standard deviation 0.1",
    "numerical_gate": "projection diagnostics only; no teacher-forcing claim",
    "route": (
        "installed batched TM shared-weight dispatch; "
        "prescaled FP16 TM and compact E4M3 QPN scales"
    ),
    "rows": [],
}
evict = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
for proj in ["gate_up", "down"] if a.projection == "both" else [a.projection]:
    if proj == "gate_up":
        w, s, g = shard("gate_proj")
        w2, s2, g2 = shard("up_proj")
        if g != g2:
            raise ValueError("unequal gate/up globals need model-loader rescaling")
        w, s = torch.cat([w, w2]), torch.cat([s, s2])
    else:
        w, s, g = shard("down_proj", True)
    w, s = w.cuda(), s.cuda()
    # Check against an FP32 dense multiply of the same FP16 dequantized
    # weights. This isolates accumulation from load-time weight rounding.
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float16,
        device="cuda",
    )
    raw_codes = torch.stack([w & 15, w >> 4], -1).flatten(1).long()
    effective_rows = (s.float() * g).half().repeat_interleave(16, dim=1)
    dense_weights = (lookup[raw_codes] * effective_rows).float()
    del raw_codes, effective_rows
    n, k = w.shape[0], w.shape[1] * 2
    codes = torch.stack([w & 15, w >> 4], -1).flatten(1).t().contiguous()
    effective = (s.float().t() * g).half().contiguous()
    tmw, tms, meta = ops.nvfp4_sm70_prepare(codes, effective, 16, False)
    kl, ql = [int(v) for v in meta[:2]]
    cs = ops.nvfp4_qpn2_prepare_scales_sm70(s)
    if float(tms.abs().max()) > 65504 / 16384:
        raise ValueError("batch prescale unsupported")
    tms.mul_(16384)
    split, acc = _qpn2_config(k, n, proj == "gate_up")
    for m in a.m:
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        out = torch.empty(
            m, n // 2 if proj == "gate_up" else n, device="cuda", dtype=torch.float16
        )

        def run(
            out=out,
            x=x,
            tmw=tmw,
            cs=cs,
            g=g,
            split=split,
            acc=acc,
            tms=tms,
            kl=kl,
            ql=ql,
            proj=proj,
        ):
            ops.nvfp4_qpn2_tm_dispatch_sm70_out(
                out,
                x,
                tmw,
                cs,
                g,
                split,
                acc,
                tms,
                16,
                kl,
                ql,
                proj == "gate_up",
                256,
                True,
            )

        for _ in range(3):
            run()
        torch.cuda.synchronize()
        dense_reference = torch.nn.functional.linear(x.float(), dense_weights)
        if proj == "gate_up":
            gate, up = dense_reference.half().chunk(2, dim=1)
            dense_reference = (
                torch.nn.functional.silu(gate.float()).half() * up
            ).float()
        else:
            dense_reference = dense_reference.half().float()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        times = []
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        for st, en in zip(starts, ends):
            evict.fill_(17)
            st.record()
            graph.replay()
            en.record()
        torch.cuda.synchronize()
        times = [st.elapsed_time(en) * 1000 for st, en in zip(starts, ends)]
        if a.ncu and not a.ncu_candidate:
            torch.cuda.cudart().cudaProfilerStart()
            evict.fill_(17)
            graph.replay()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        elif not a.ncu:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                evict.fill_(17)
                graph.replay()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(a.out.with_suffix(f".{proj}.m{m}.trace.json")))
        control_diff = out.float() - dense_reference
        row = {
            "control_vs_fp32_dense": {
                "max_abs": float(control_diff.abs().max()),
                "mean_abs": float(control_diff.abs().mean()),
            },
            "projection": proj,
            "m": m,
            "n": n,
            "k": k,
            "split": split,
            "nacc": acc,
            "median_us": statistics.median(times),
            "mean_us": statistics.mean(times),
            "weight_bytes": w.numel() + s.numel(),
            "weight_only_GBs": (w.numel() + s.numel())
            / statistics.median(times)
            / 1000,
        }
        if (a.candidate or a.candidate_so) and m > 8:
            candidate_out = torch.empty_like(out)
            packed = torch.empty_like(x)
            scratch = torch.empty(
                8 * (2 if proj == "gate_up" else 1) * candidate_out.numel(),
                dtype=torch.float32,
                device="cuda",
            )

            def candidate_run(
                candidate_out=candidate_out,
                x=x,
                tmw=tmw,
                cs=cs,
                packed=packed,
                scratch=scratch,
                g=g,
                proj=proj,
            ):
                torch.ops._qwen_batch_screen.run(
                    candidate_out, x, tmw, cs, packed, scratch, g, proj == "gate_up"
                )

            candidate_run()
            graph.replay()
            torch.cuda.synchronize()
            oracle_diff = candidate_out.float() - dense_reference
            row["candidate_vs_fp32_dense"] = {
                "max_abs": float(oracle_diff.abs().max()),
                "mean_abs": float(oracle_diff.abs().mean()),
            }
            diff = candidate_out.float() - out.float()
            row["candidate_vs_production"] = {
                "max_abs": float(diff.abs().max()),
                "mean_abs": float(diff.abs().mean()),
                "bitwise_fraction": float((candidate_out == out).float().mean()),
            }
            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg):
                candidate_run()
            for _ in range(20):
                cg.replay()
            torch.cuda.synchronize()
            ct = []
            for st, en in zip(starts, ends):
                evict.fill_(17)
                st.record()
                cg.replay()
                en.record()
            torch.cuda.synchronize()
            ct = [st.elapsed_time(en) * 1000 for st, en in zip(starts, ends)]
            row["candidate_median_us"] = statistics.median(ct)
            row["speedup"] = row["median_us"] / row["candidate_median_us"]
            if a.ncu_candidate:
                torch.cuda.cudart().cudaProfilerStart()
                evict.fill_(17)
                cg.replay()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            elif not a.ncu:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    evict.fill_(17)
                    cg.replay()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(
                    str(a.out.with_suffix(f".{proj}.m{m}.candidate.trace.json"))
                )
        result["rows"].append(row)
        print(json.dumps(row), flush=True)
    del tmw, tms, cs, w, s, codes, effective, dense_weights, dense_reference
    torch.cuda.empty_cache()
a.out.write_text(json.dumps(result, indent=2) + "\n")
