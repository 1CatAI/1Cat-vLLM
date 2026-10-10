# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 QSA operator A/B; no model or serving-engine throughput claims.

Requires the pinned source trees, Torch/Triton/TileLang/tvm-ffi, CUDA 12.x,
and a Flash-V100 extension built from the supplied 1Cat tree.
"""

import argparse
import gc
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch
from qsa_sxm2_adapters import load_operators


def error_metrics(actual, reference):
    delta = actual.float() - reference.float()
    assert torch.isfinite(actual).all(), "nonfinite output"
    return {
        "max_abs": delta.abs().max().item(),
        "relative_rms": (
            delta.square().mean().sqrt()
            / reference.float().square().mean().sqrt().clamp_min(1e-8)
        ).item(),
    }


def repage(cache, table, tokens, page_size):
    """Repack identical logical values into an independent physical layout."""
    requests = table.shape[0]
    pages = math.ceil(tokens / page_size)
    if page_size == cache.shape[1]:
        return cache, table
    logical = cache[table.long()].reshape(requests, -1, *cache.shape[2:])
    padded = torch.zeros(
        (requests, pages * page_size, *cache.shape[2:]),
        dtype=cache.dtype,
        device=cache.device,
    )
    padded[:, :tokens] = logical[:, :tokens]
    destination = torch.randperm(requests * pages, device=cache.device).to(torch.int32)
    output = torch.empty(
        (requests * pages, page_size, *cache.shape[2:]),
        dtype=cache.dtype,
        device=cache.device,
    )
    output[destination.long()] = padded.reshape_as(output)
    return output, destination.view(requests, pages)


def make_case(context, requests, width, seed, index_page=4, kv_page=16):
    torch.manual_seed(seed)
    m = requests * width
    columns = context // 4
    index_table = (
        torch.randperm(requests * columns // 4, device="cuda")
        .to(torch.int32)
        .view(requests, columns // 4)
    )
    kv_table = (
        torch.randperm(requests * context // 16, device="cuda")
        .to(torch.int32)
        .view(requests, context // 16)
    )
    cache = torch.randn(
        (requests * columns // 4, 4, 1, 128), device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        (requests * context // 16, 16, 1, 256), device="cuda", dtype=torch.float16
    )
    v = torch.randn_like(k)
    scales = torch.ones(2, dtype=torch.float32, device="cuda")
    device_partial = torch.empty(20 * 6 * 33 * 256, dtype=torch.float32, device="cuda")
    device_state = torch.empty(20 * 6 * 33 * 2, dtype=torch.float32, device="cuda")
    qi = torch.randn((m, 4, 128), device="cuda", dtype=torch.float16)
    qa = torch.randn((m, 6, 256), device="cuda", dtype=torch.float16)
    req = torch.arange(requests, device="cuda", dtype=torch.int32).repeat_interleave(
        width
    )
    pos = torch.arange(context - width, context, device="cuda").repeat(requests)
    seq = torch.full((requests,), context, device="cuda", dtype=torch.int32)
    row_lengths = (pos + 1).to(torch.int32)
    visible = row_lengths // 4
    # Layout adapters built once, like persistent serving metadata.
    sgl_index_table = index_table[req.long()].contiguous()
    offsets = torch.arange(16, device="cuda", dtype=torch.int32)
    request_to_token = (kv_table[:, :, None] * 16 + offsets).view(requests, context)
    one_cache, one_index_table = repage(cache, index_table, columns, index_page)
    # Repack K and V together so they share the same physical block mapping.
    both, one_kv_table = repage(torch.stack((k, v), dim=2), kv_table, context, kv_page)
    one_k = both[:, :, 0].contiguous()
    one_v = both[:, :, 1].contiguous()
    history = torch.stack((one_k, one_v), dim=1).contiguous()
    return locals()


def score_reference(c):
    refs = []
    for row in range(c["m"]):
        request = row // c["width"]
        keys = c["cache"][c["index_table"][request].long()].reshape(-1, 128).float()
        dot = c["qi"][row].float() @ keys.T
        scores = dot.relu().sum(0) / math.sqrt(128)
        scores[c["visible"][row] :] = -float("inf")
        refs.append(scores)
    return torch.stack(refs)


def attention_reference(c, indices):
    refs = []
    for row in range(c["m"]):
        logical = indices[row].long()
        logical = logical[(logical >= 0) & (logical <= c["pos"][row])]
        slots = c["request_to_token"][row // c["width"], logical].long()
        keys = c["k"].view(-1, 256)[slots].float()
        values = c["v"].view(-1, 256)[slots].float()
        probs = torch.softmax((c["qa"][row].float() @ keys.T) / 16, dim=-1)
        refs.append(probs @ values)
    return torch.stack(refs)


def same_set(left, right):
    return bool(torch.equal(left.sort(dim=1).values, right.sort(dim=1).values))


def paired_timing(functions, groups, replays, unroll, cold_cache=False):
    """Interleaved ABBA CUDA graph replay, including all stage allocations.

    Repeated calls inside one graph amortize host graph-launch gaps. Host
    enqueue numbers are eager Python submission cost, not isolated CPU work.
    """
    graphs, outputs, allocated, device_events = {}, {}, {}, {}
    scrub = (
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        if cold_cache
        else None
    )
    for name, fn in functions.items():
        for _ in range(5):
            fn()
        torch.accelerator.synchronize()
        baseline = torch.accelerator.memory_allocated()
        torch.accelerator.reset_peak_memory_stats()
        out = fn()
        torch.accelerator.synchronize()
        allocated[name] = torch.accelerator.max_memory_allocated() - baseline
        del out
        graph = torch.cuda.CUDAGraph()
        if cold_cache:
            begin = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            device_events[name] = (begin, end)
            graph._benchmark_resources = (begin, end, scrub)
        with torch.cuda.graph(graph):
            if cold_cache:
                scrub.zero_()
                begin.record()
                outputs[name] = fn()
                end.record()
            else:
                for _ in range(unroll):
                    outputs[name] = fn()
        graphs[name] = graph
    torch.accelerator.synchronize()
    samples = {name: [] for name in functions}
    names = list(functions)
    for group in range(groups):
        order = names if group % 2 == 0 else names[::-1]
        for name in order:
            for _ in range(3):
                graphs[name].replay()
            if cold_cache:
                begin, end = device_events[name]
                measurements = []
                for _ in range(replays):
                    graphs[name].replay()
                    end.synchronize()
                    measurements.append(begin.elapsed_time(end) * 1000)
                samples[name].append(statistics.median(measurements))
                continue
            start, stop = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(replays):
                graphs[name].replay()
            stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) * 1000 / replays / unroll)
    result = {}
    for name, fn in functions.items():
        host = []
        for _ in range(3):
            torch.accelerator.synchronize()
            start = time.perf_counter_ns()
            for _ in range(20):
                fn()
            host.append((time.perf_counter_ns() - start) / 20 / 1000)
            torch.accelerator.synchronize()
        result[name] = dict(
            gpu_us=statistics.median(samples[name]),
            gpu_samples_us=samples[name],
            eager_submission_us=statistics.median(host),
            peak_allocated_increment_bytes=allocated[name],
        )
    result["onecat_speedup"] = result["sglang"]["gpu_us"] / result["onecat"]["gpu_us"]
    # Replay after changing the query tests that captures read dynamic tensors.
    return result, graphs, outputs


def run_case(ops, context, requests, width, args):
    c = make_case(
        context, requests, width, args.seed, args.onecat_index_page, args.onecat_kv_page
    )
    qsa = ops.onecat

    def one_score():
        return qsa.qsa_mqa_paged(
            c["qi"],
            c["one_cache"],
            c["one_index_table"],
            c["req"],
            c["pos"],
            c["seq"],
            4,
            num_columns=c["columns"],
            shared_key_scoring=True,
        )[0]

    def sgl_score():
        return ops.sgl_score(
            c["qi"], c["cache"], c["sgl_index_table"], c["visible"], c["columns"]
        )

    def one_topk(scores):
        selected = torch.empty((c["m"], 512), device="cuda", dtype=torch.int32)
        torch.ops._C.qsa_lexicographic_topk(
            scores, c["visible"], selected, 512, c["m"] in (5, 10)
        )
        return selected

    def one_expand(blocks):
        return qsa.expand_qsa_block_indices_cuda(
            blocks, c["pos"], c["seq"], c["req"], 4, 2048
        )

    def sgl_expand(blocks):
        return ops.sgl_expand(blocks, c["pos"], c["row_lengths"], 4, 2048)

    def one_select():
        return qsa.qsa_select_paged_tokens(
            c["qi"],
            c["one_cache"],
            c["one_index_table"],
            c["req"],
            c["pos"],
            c["seq"],
            2048,
            4,
            shared_key_scoring=True,
        )

    def sgl_select():
        return sgl_expand(ops.sgl_topk(sgl_score(), c["visible"]))

    def one_attention(indices):
        return qsa.qsa_sparse_paged_attention(
            c["qa"],
            c["one_k"],
            c["one_v"],
            indices,
            c["one_kv_table"],
            c["req"],
            query_positions=c["pos"],
            sequence_lengths=c["seq"],
        )

    def sgl_attention(indices):
        return ops.sgl_attention(
            c["qa"],
            c["k"].view(-1, 1, 256),
            c["v"].view(-1, 1, 256),
            c["request_to_token"],
            c["req"],
            indices,
            c["row_lengths"],
        )

    def device_attention(indices):
        out = torch.empty_like(c["qa"])
        torch.ops.vllm_sm70_qsa_device.run(
            c["qa"],
            c["history"],
            c["scales"],
            indices,
            c["one_kv_table"],
            c["req"],
            c["pos"],
            c["seq"],
            out,
            None,
            c["device_partial"],
            c["device_state"],
        )
        return out

    scores = {"onecat": one_score(), "sglang": sgl_score()}
    oracle = score_reference(c)
    valid = torch.arange(c["columns"], device="cuda")[None, :] < c["visible"][:, None]
    correctness = {
        "score": {
            name: error_metrics(value[valid], oracle[valid])
            for name, value in scores.items()
        }
    }
    for metrics in correctness["score"].values():
        assert metrics["max_abs"] < 1e-4, metrics
    common = scores["onecat"]
    blocks = {"onecat": one_topk(common), "sglang": ops.sgl_topk(common, c["visible"])}
    correctness["common_scores_topk_same_set"] = same_set(*blocks.values())
    assert correctness["common_scores_topk_same_set"], (
        "top-k selection differs on identical scores"
    )
    correctness["native_scores_topk_same_set"] = same_set(
        one_topk(scores["onecat"]), ops.sgl_topk(scores["sglang"], c["visible"])
    )
    selected = one_expand(blocks["onecat"])
    correctness["expansion_same_set"] = same_set(selected, sgl_expand(blocks["sglang"]))
    assert correctness["expansion_same_set"]
    ref = attention_reference(c, selected)
    correctness["attention"] = {
        "onecat": error_metrics(one_attention(selected), ref),
        "onecat_device": error_metrics(device_attention(selected), ref),
        "sglang": error_metrics(sgl_attention(selected), ref),
    }
    for metrics in correctness["attention"].values():
        assert metrics["max_abs"] < 0.005 and metrics["relative_rms"] < 0.01, metrics
    # End-to-end selections may differ at score ties/rounding; validate each
    # arm against the reference for its own actual selected token set.
    correctness["chain"] = {}
    for name, select, attend in (
        ("onecat", one_select, one_attention),
        ("sglang", sgl_select, sgl_attention),
        ("onecat_device", one_select, device_attention),
    ):
        picked = select()
        correctness["chain"][name] = error_metrics(
            attend(picked), attention_reference(c, picked)
        )
        assert correctness["chain"][name]["max_abs"] < 0.005
    stages = {
        "score": dict(onecat=one_score, sglang=sgl_score),
        "topk_expand": dict(
            onecat=lambda: one_expand(one_topk(common)),
            sglang=lambda: sgl_expand(ops.sgl_topk(common, c["visible"])),
        ),
        "attention": dict(
            onecat=lambda: one_attention(selected),
            sglang=lambda: sgl_attention(selected),
        ),
        "chain": dict(
            onecat=lambda: one_attention(one_select()),
            sglang=lambda: sgl_attention(sgl_select()),
        ),
        "attention_device": dict(
            onecat=lambda: device_attention(selected),
            sglang=lambda: sgl_attention(selected),
        ),
        "chain_device": dict(
            onecat=lambda: device_attention(one_select()),
            sglang=lambda: sgl_attention(sgl_select()),
        ),
    }
    result = dict(
        context=context,
        requests=requests,
        query_width=width,
        rows=c["m"],
        correctness=correctness,
        timings={},
        metadata_bytes={
            "onecat_main_table": c["one_kv_table"].numel() * 4,
            "sglang_main_table": c["request_to_token"].numel() * 4,
        },
        onecat_device_workspace_bytes=(
            c["device_partial"].numel() + c["device_state"].numel()
        )
        * 4,
        scope=(
            "FP16 score -> topk512 -> expand2048+tail -> sparse attention; "
            "no projection/cache-write/model"
        ),
        onecat_score_route="native_shared_key"
        if requests == 1 and 2 <= c["m"] <= 8
        else "triton",
        onecat_attention_route="triton_split_merge",
        sglang_attention_route="native_split+native_merge"
        if c["m"] <= 4
        else "native_split+tilelang_merge",
    )
    for stage, functions in stages.items():
        timing, graphs, outputs = paired_timing(
            functions, args.groups, args.replays, args.unroll, args.cold_cache
        )
        result["timings"][stage] = timing
        # Each arm must remain valid after a graph's input contents change.
        c["qa"].mul_(0.99)
        for name in graphs:
            graphs[name].replay()
            torch.accelerator.synchronize()
            eager = functions[name]()
            if stage == "score":
                # The shared-key scorer deliberately leaves whole masked
                # tail tiles unwritten; consumers are bounded by visible.
                torch.testing.assert_close(
                    outputs[name][valid], eager[valid], rtol=0, atol=0
                )
            elif eager.dtype == torch.int32:
                assert same_set(outputs[name], eager)
            else:
                torch.testing.assert_close(
                    outputs[name], eager, rtol=0.005, atol=0.0005, equal_nan=True
                )
        del graphs, outputs
        print(
            json.dumps(
                dict(
                    context=context,
                    requests=requests,
                    width=width,
                    stage=stage,
                    **timing,
                )
            ),
            flush=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onecat", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[8192, 32768, 131072]
    )
    parser.add_argument("--cases", nargs="+", default=["1x1", "4x1", "1x5", "4x5"])
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--replays", type=int, default=40)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--onecat-index-page", type=int, default=4)
    parser.add_argument("--onecat-kv-page", type=int, default=16)
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="Scrub 32 MiB before each timed graph region; excludes scrub time",
    )
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    ops = load_operators(
        args.onecat.resolve(), args.sglang.resolve(), args.cache.resolve()
    )
    report = dict(
        onecat_commit="c39f53abae51df6bf8ef122e34975a9f3596a93c",
        sglang_commit="73abfe1dd7e7aae7a62b222a201e48f532fb298e",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        settings=vars(args)
        | {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
        device_status=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,clocks.sm,clocks.mem,temperature.gpu,power.limit",
                "--format=csv",
            ],
            text=True,
        ),
        cases=[],
    )
    for context in args.contexts:
        for case in args.cases:
            requests, width = map(int, case.split("x"))
            print(f"CASE {context=} {requests=} {width=}", flush=True)
            report["cases"].append(run_case(ops, context, requests, width, args))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            gc.collect()
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
