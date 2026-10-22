# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight TP4 MoE microbenchmark, using the serving route/subroute code.

Run control/candidate in fresh processes with separate native tuning caches.
This isolates resident canonical-bank compute: it does not measure host DMA,
transcoding overlap, TP communication, or end-to-end model quality.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import vllm._C_gguf  # noqa: F401

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_expert_cache import (
    TurboMindPrefill,
    _canonical,
)
from vllm.model_executor.layers.quantization.gguf_native import packed_tp_span
from vllm.transformers_utils.gguf_tensor_reader import (
    GGUFReader,
    dequantize,
    quant_size,
)

EXPERTS, TOPK, HIDDEN, INTERMEDIATE = 512, 10, 2560, 160


def artifact_identity():
    import vllm._C

    import vllm

    native = Path(vllm._C.__file__).resolve()
    return {
        "package": str(Path(vllm.__file__).resolve()),
        "native": str(native),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }


def measure(fn, iterations):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(5):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / iterations)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def pointer_table(tensor, stride, ld):
    table = torch.empty(EXPERTS, 2, dtype=torch.int64, device="cuda")
    table[:, 0] = tensor.data_ptr() + torch.arange(EXPERTS, device="cuda") * stride
    table[:, 1] = ld
    return table.view(torch.uint8).flatten()


def relative_l2(actual, expected):
    actual, expected = actual.float(), expected.float()
    return (
        ((actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-6))
        .max()
        .item()
    )


def read_layer(reader, layer):
    tensors = {
        part: next(
            t for t in reader.tensors if t.name == f"blk.{layer}.ffn_{part}_exps.weight"
        )
        for part in ("gate", "up", "down")
    }
    gtype, dtype = (int(tensors[p].tensor_type) for p in ("gate", "down"))
    assert int(tensors["up"].tensor_type) == gtype
    raw = {
        p: np.ascontiguousarray(np.asarray(tensors[p].data)[:, :INTERMEDIATE])
        for p in ("gate", "up")
    }
    full = np.asarray(tensors["down"].data)
    block, size = quant_size(dtype)
    first, last, left, _ = packed_tp_span(full.shape[2] // size * block, dtype, 4, 0)
    raw["down"] = np.ascontiguousarray(full[:, :, first:last])
    return raw, gtype, dtype, left


def load_layer(reader, layer):
    raw, gtype, dtype, left = read_layer(reader, layer)
    source = {p: torch.from_numpy(a).cuda() for p, a in raw.items()}
    info = {
        "gate": (gtype,),
        "down": (dtype,),
        "rows": INTERMEDIATE,
        "hidden": HIDDEN,
        "gu": _canonical(gtype, 2 * INTERMEDIATE, HIDDEN),
        "dn": _canonical(dtype, HIDDEN, INTERMEDIATE),
    }
    banks, tables = {}, {}
    ids = torch.arange(EXPERTS, dtype=torch.int32, device="cuda")
    count = torch.full((1,), EXPERTS, dtype=torch.int32, device="cuda")
    error = torch.zeros(1, dtype=torch.int32, device="cuda")
    pointers = {
        p: t.data_ptr()
        + torch.arange(EXPERTS, device="cuda", dtype=torch.int64) * t[0].numel()
        for p, t in source.items()
    }
    for key, part, n, k, skip in (
        ("gu", "gate", 2 * INTERMEDIATE, HIDDEN, 0),
        ("dn", "down", HIDDEN, INTERMEDIATE, left),
    ):
        _, _, _, weight_bytes, stats_bytes = info[key]
        weight = torch.empty(EXPERTS * weight_bytes, dtype=torch.uint8, device="cuda")
        stats = torch.empty(EXPERTS * stats_bytes, dtype=torch.uint8, device="cuda")
        banks[key] = (weight, stats)
        tables[key] = (
            pointer_table(weight, weight_bytes, 32 * k),
            pointer_table(stats, stats_bytes, n),
        )
        torch.ops._C.gguf_prefill_transcode_sm70(
            weight,
            stats,
            weight_bytes,
            stats_bytes,
            pointers[part],
            pointers["up"] if key == "gu" else None,
            ids,
            count,
            1024,
            gtype if key == "gu" else dtype,
            n,
            k,
            skip,
            raw[part].shape[2],
            error,
        )
    assert error.item() == 0, "transcode reported an invalid expert"
    return info, tables, banks, raw, left


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 17, 47])
    parser.add_argument("--tokens", type=int, nargs="+", default=[7344, 5347])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tune-tokens", type=int, default=8192)
    parser.add_argument("--down-tokens", type=int, default=4096)
    parser.add_argument("--routing", choices=["uniform", "skewed"], default="uniform")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-reference", type=Path)
    parser.add_argument("--compare-reference", type=Path)
    args = parser.parse_args()
    assert args.iterations > 0 and min(args.tokens) > 0 and args.down_tokens > 0
    torch.backends.cuda.matmul.allow_tf32 = False
    reader = GGUFReader(args.model)
    references = (
        torch.load(args.compare_reference, map_location="cpu", weights_only=True)
        if args.compare_reference
        else {}
    )
    saved, results = {}, []
    report = {
        "artifact": artifact_identity(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "routing": args.routing,
        "tune_tokens": args.tune_tokens,
        "down_tokens": args.down_tokens,
        "cases": results,
    }
    for layer in args.layers:
        info, tables, banks, raw, left = load_layer(reader, layer)
        # Reuse the production sorting, sub-batching, fused GEMM, and unroute.
        runner = TurboMindPrefill.__new__(TurboMindPrefill)
        runner.cache = SimpleNamespace(experts=EXPERTS, prefill_tokens=args.down_tokens)
        runner.expert_ids = torch.arange(EXPERTS + 1, dtype=torch.int32, device="cuda")
        runner.tuned = set()
        probe = torch.zeros(args.tune_tokens, HIDDEN, dtype=torch.half, device="cuda")
        runner._tune(layer, info, tables, probe)
        del probe
        dequantized = {}

        def ref_weight(
            expert, part, dequantized=dequantized, info=info, raw=raw, left=left
        ):
            key = (expert, part)
            if key not in dequantized:
                ty = info["down" if part == "down" else "gate"][0]
                array = dequantize(raw[part][expert], ty)
                if part == "down":
                    array = array[:, left : left + INTERMEDIATE]
                dequantized[key] = (
                    torch.from_numpy(np.ascontiguousarray(array)).cuda().float()
                )
            return dequantized[key]

        for tokens in args.tokens:
            gen = torch.Generator(device="cuda").manual_seed(tokens)
            x = (torch.randn(tokens, HIDDEN, device="cuda", generator=gen) * 0.5).half()
            scores = torch.randn(tokens, EXPERTS, device="cuda", generator=gen)
            if args.routing == "skewed":
                scores[:, :32] += 2
            scores, ids = scores.topk(TOPK, dim=1)
            weights = scores.softmax(dim=1)
            route = runner._route(ids, 0, EXPERTS)
            subroutes = (
                [
                    (
                        start,
                        min(tokens, start + args.down_tokens),
                        runner._subroute(
                            route, start, min(tokens, start + args.down_tokens)
                        ),
                    )
                    for start in range(0, tokens, args.down_tokens)
                ]
                if tokens > args.down_tokens
                else [(0, tokens, route)]
            )
            out = torch.empty_like(x)

            def gu(runner=runner, info=info, tables=tables, x=x, route=route):
                return runner._gate_up(info, tables, x, route, False)

            hidden = gu()

            def down(
                h=hidden,
                runner=runner,
                info=info,
                tables=tables,
                subroutes=subroutes,
                weights=weights,
                out=out,
            ):
                for start, end, subroute in subroutes:
                    runner._down(
                        info,
                        tables,
                        h,
                        subroute,
                        weights[start:end],
                        out[start:end],
                        False,
                    )

            def chain(gu=gu, down=down):
                down(gu())

            result = {
                "layer": layer,
                "gate_type": info["gate"][0],
                "down_type": info["down"][0],
                "tokens": tokens,
                "gate_up": measure(gu, args.iterations),
                "down_unroute": measure(down, args.iterations),
                "chain": measure(chain, args.iterations),
            }
            down()
            # FP32 reference at expert boundaries catches ragged-row/layout errors.
            offsets = route[0].cpu().tolist()
            positions = sorted(
                {
                    p
                    for e in (0, 1, 31, 32, 127, 255, 510, 511)
                    if offsets[e] < offsets[e + 1]
                    for p in (offsets[e], offsets[e + 1] - 1)
                }
            )
            hidden_errors = []
            for position in positions:
                token = int(route[2][position])
                expert = int(ids.flatten()[route[1][position]])
                expected = torch.nn.functional.silu(
                    ref_weight(expert, "gate") @ x[token].float()
                )
                expected *= ref_weight(expert, "up") @ x[token].float()
                hidden_errors.append(relative_l2(hidden[position], expected))
            chosen_tokens = torch.tensor(
                sorted({0, 1, tokens // 2, tokens - 1}), device="cuda"
            ).clamp_max(tokens - 1)
            sample = {
                "hidden": hidden[positions].cpu(),
                "out": out[chosen_tokens].cpu(),
            }
            expected_rows = []
            for token in chosen_tokens.tolist():
                expected_row = torch.zeros(HIDDEN, device="cuda")
                for slot in range(TOPK):
                    expert = int(ids[token, slot])
                    expected_hidden = torch.nn.functional.silu(
                        ref_weight(expert, "gate") @ x[token].float()
                    ) * (ref_weight(expert, "up") @ x[token].float())
                    expected_down = (
                        (ref_weight(expert, "down") @ expected_hidden.half().float())
                        .half()
                        .float()
                    )
                    expected_row += weights[token, slot] * expected_down
                expected_rows.append(expected_row.half())
            output_error = relative_l2(out[chosen_tokens], torch.stack(expected_rows))
            key = f"{layer}/{tokens}/{args.routing}"
            result["hidden_fp32_max_rel_l2"] = max(hidden_errors)
            result["output_fp32_max_rel_l2"] = output_error
            result["finite"] = bool(
                torch.isfinite(hidden).all() & torch.isfinite(out).all()
            )
            passed = (
                result["finite"] and max(hidden_errors) <= 0.01 and output_error <= 0.01
            )
            if args.compare_reference:
                errors = {
                    name: relative_l2(value, references[key][name])
                    for name, value in sample.items()
                }
                result["control_max_rel_l2"] = errors
                passed &= max(errors.values()) <= 0.003
            result["passed"] = passed
            saved[key] = sample
            results.append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        del banks, tables, dequantized, hidden, out, runner
    if args.save_reference:
        torch.save(saved, args.save_reference)
    if not all(case["passed"] for case in results):
        raise SystemExit("numerical gate failed")


if __name__ == "__main__":
    main()
