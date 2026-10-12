# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen shorter E4M3 verifier partitions using real projection weights.

The source's 64-token minimum uses only seventeen active CTAs at 1K. A 32-token
minimum uses thirty-three, retaining the original N32 QK/PV/compensation. Inputs
derive from real TP4 attention weights and synthetic normalized activations;
this is an operator screen, not model teacher-forcing or full-round admission.
"""

import argparse
import hashlib
import json
import math
import sys
from functools import partial
from pathlib import Path

import torch
from benchmark_sm70_qpn2_effective_scale import graph_pair
from safetensors import safe_open
from torch.utils.cpp_extension import load


def function_end(source, anchor):
    begin = source.index("{", source.index(anchor))
    cursor, depth = begin + 1, 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return cursor


def generate(root, compact=False):
    source = (root / "kernel/grouped-attention.cu").read_text()
    end = function_end(
        source, "void flash_attention_grouped_verify_e5m2_combine_kernel("
    )
    prefix = source[:end] + "\n} // namespace\n"
    begin = source.index("at::Tensor private_grouped_e4m3_fp32_paged(")
    end = function_end(source, "at::Tensor private_grouped_e4m3_fp32_paged(")
    text = "#include <torch/extension.h>\n" + prefix + source[begin:end]
    old = "constexpr int kGroupedVerifyMinTokensPerSplit = 64;"
    assert text.count(old) == 1
    if compact:
        # The retained compact builder expects the older single-request launch.
        # Restore the batch dimension after changing only the q8 CTA geometry.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from benchmarks.kernels.build_sm70_grouped_attention_compact_cta import (
            compact_cta,
        )

        batch_launch = (
            "kernel<<<dim3(1, 80, static_cast<unsigned>(batch_size)),\n"
            "           kGroupedVerifyThreads, kCompensatedSmemBytes, stream>>>"
        )
        flat_launch = (
            "kernel<<<dim3(1, 80), kGroupedVerifyThreads, "
            "kCompensatedSmemBytes, stream>>>"
        )
        assert text.count(batch_launch) == 1
        text = compact_cta(text.replace(batch_launch, flat_launch))
        text = text.replace(
            "dim3(q.size(0) == 8 ? 3 : 1, 80)",
            "dim3(query_len == 8 ? 3 : 1, 80, static_cast<unsigned>(batch_size))",
        )
        text = text.replace("q.size(0) == 8 ?", "query_len == 8 ?")
        # The current paired path preloads one K vector per thread for 512
        # vectors. Compact CTAs have only 256 threads. The general panel loader
        # covers the complete panel with a thread-strided loop instead.
        assert text.count("bool paired = true;") == 1
        text = text.replace("bool paired = true;", "bool paired = false;")
    else:
        text = text.replace(old, old.replace("64", "32"))
    return (
        text
        + r"""
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("run",&private_grouped_e4m3_fp32_paged);
}
"""
    )


def rotated(value, positions):
    frequencies = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, device="cuda") / 64))
    phase = positions[:, None] * frequencies[None]
    c, s = phase.cos()[:, None], phase.sin()[:, None]
    left, right = value[..., :32].float(), value[..., 32:64].float()
    out = value.clone()
    out[..., :32] = (left * c - right * s).half()
    out[..., 32:64] = (right * c + left * s).half()
    return out


def real_case(weights, layer, length, page):
    prefix = f"model.language_model.layers.{layer}.self_attn."

    def read(name):
        return weights.get_tensor(prefix + name)

    def dense(name, rows):
        w = read(name + ".weight")[rows].float().cuda()
        scales = read(name + ".weight_scale")[rows].half().float().cuda()
        return (w * scales.reshape(-1, 1)).half().float()

    q_rows = [h * 512 + d for h in range(6) for d in range(256)]
    q_weight = dense("q_proj", q_rows)
    k_weight = dense("k_proj", slice(0, 256))
    v_weight = dense("v_proj", slice(0, 256))
    x = torch.randn(length, 5120, device="cuda").half().float()
    q = (x[-8:] @ q_weight.T).half().view(8, 6, 256)
    k = (x @ k_weight.T).half().view(length, 1, 256)
    v = (x @ v_weight.T).half().view(length, 1, 256)

    def normalize(value, name):
        f = value.float()
        inverse = torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)
        return (f * inverse * (read(name).float().cuda() + 1)).half()

    q = normalize(q, "q_norm.weight")
    k = normalize(k, "k_norm.weight")
    positions = torch.arange(length, device="cuda")
    q = rotated(q, positions[-8:]).contiguous()
    k = rotated(k, positions)
    pages = math.ceil(length / page)
    cache = torch.zeros(pages, 2, page, 1, 256, device="cuda", dtype=torch.uint8)
    table = torch.randperm(pages, device="cuda").int()[None]
    k_scale, v_scale = float(read("k_scale")), float(read("v_scale"))
    assert k_scale > 0 and v_scale > 0
    logical = torch.stack((k.float() / k_scale, v.float() / v_scale), 1)
    logical = logical.to(torch.float8_e4m3fn).view(torch.uint8)
    padded = torch.zeros(pages * page, 2, 1, 256, device="cuda", dtype=torch.uint8)
    padded[:length].copy_(logical)
    cache.index_copy_(
        0,
        table[0].long(),
        padded.view(pages, page, 2, 1, 256).transpose(1, 2).contiguous(),
    )
    buffers = [
        (
            torch.empty_like(q),
            torch.empty(80, 8, 6, 256, device="cuda"),
            torch.empty(80, 8, 6, 2, device="cuda"),
        )
        for _ in range(2)
    ]
    return {
        "q": q,
        "k": cache[:, 0],
        "v": cache[:, 1],
        "table": table,
        "lengths": torch.arange(
            length - 7, length + 1, device="cuda", dtype=torch.int32
        ),
        "buffers": buffers,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }


def oracle(case):
    length = int(case["lengths"][-1])
    ids = case["table"][0].long()
    k = case["k"].index_select(0, ids).reshape(-1, 256)[:length]
    v = case["v"].index_select(0, ids).reshape(-1, 256)[:length]
    k = k.contiguous().view(torch.float8_e4m3fn).double()
    v = v.contiguous().view(torch.float8_e4m3fn).double()
    q = case["q"].view(-1, 256).double()
    scores = q @ k.T * (0.0625 * case["k_scale"])
    visible = (
        torch.arange(length, device="cuda")[None]
        < case["lengths"].repeat_interleave(6)[:, None]
    )
    scores.masked_fill_(~visible, -torch.inf)
    return (scores.softmax(-1) @ v * case["v_scale"]).view_as(case["q"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--compact-cta", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = args.source_root / "csrc/attention/sm70_grouped_long"
    text = generate(root, args.compact_cta)
    source = args.out / "short_splits.cu"
    if not source.exists() or source.read_text() != text:
        source.write_text(text)
    extension = load(
        name="e4m3_compact_short_nopair_screen"
        if args.compact_cta
        else "e4m3_short_splits_screen",
        sources=[str(source)],
        extra_include_paths=[str(root / "include"), str(root / "kernel")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-extended-lambda",
            "--ptxas-options=-v",
        ],
        verbose=True,
    )
    if args.compile_only:
        return
    from vllm.v1.attention.ops.sm70_e4m3_long import builtin_long_attention

    baseline = builtin_long_attention()
    assert baseline is not None
    operators = [baseline, extension.run]
    torch.manual_seed(123)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    result = {
        "scope": "Real TP4 attention weights, synthetic hidden states; "
        "sixteen attention calls only; no full-layer/round admission",
        "cuda_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "results": [],
    }
    with safe_open(args.model / "model.safetensors", framework="pt") as weights:
        for context in (1024, 8192):
            cases = [
                real_case(weights, layer, context + 8, 2048)
                for layer in range(3, 64, 4)
            ]

            def call(arm, cases=cases):
                for case in cases:
                    out, partial_out, lse = case["buffers"][arm]
                    operators[arm](
                        case["q"],
                        case["k"],
                        case["v"],
                        out,
                        case["table"],
                        case["lengths"],
                        partial_out,
                        lse,
                        0.0625,
                        case["k_scale"],
                        case["v_scale"],
                    )

            call(0)
            call(1)
            errors, bitwise = [], []
            for case in cases:
                ref = oracle(case)
                assert torch.isfinite(ref).all()
                assert all(torch.isfinite(buf[0]).all() for buf in case["buffers"])
                bitwise.append(
                    torch.equal(
                        case["buffers"][0][0].view(torch.int16),
                        case["buffers"][1][0].view(torch.int16),
                    )
                )
                errors.append(
                    [
                        {
                            "max_abs": (buffers[0].double() - ref).abs().max().item(),
                            "relative_l2": (
                                (buffers[0].double() - ref).norm()
                                / ref.norm().clamp_min(1e-30)
                            ).item(),
                        }
                        for buffers in case["buffers"]
                    ]
                )
            # An operator guard rejects geometry mistakes before any speed
            # result is admitted. This does not replace model teacher-forcing.
            assert max(e[1]["relative_l2"] for e in errors) < 0.001, errors
            timed = graph_pair(partial(call, 0), partial(call, 1), eviction, args.iters)
            result["results"].append(
                {
                    "context": context,
                    "attention_calls": 16,
                    "errors_vs_fp64": errors,
                    "bitwise_control_candidate": bitwise,
                    "compact_cta": args.compact_cta,
                    **timed,
                }
            )
            (args.out / "result.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        "context": context,
                        **{k: v for k, v in timed.items() if k != "samples_us"},
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
