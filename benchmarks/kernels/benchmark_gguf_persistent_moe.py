# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only complete routed-chain screen on real TP4 GGUF weights.

The extension option loads an isolated experimental DSO. Its results are
microbenchmarks, not source-complete model or production performance evidence.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np
import regex as re
import torch
import vllm._C as core

from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.model_executor.layers.quantization.gguf_turbomind_moe import GGUFExpertBank
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def captured(operation):
    for _ in range(3):
        operation()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            operation()
    return graph


def elapsed(graph, iterations):
    for _ in range(5):
        graph.replay()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (8 * iterations)


def q8_values(packed):
    scale = packed[..., :2].contiguous().view(torch.float16).float()
    return (packed[..., 4:].view(torch.int8).float() * scale).flatten(-2)


def error(actual, expected):
    diff = actual.float() - expected.float()
    return dict(
        max_abs=diff.abs().max().item(),
        relative_l2=(diff.norm() / expected.float().norm().clamp_min(1e-30)).item(),
        exact_values=int((actual == expected).sum().item()),
        values=actual.numel(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--extension", type=Path, help="Research-only isolated DSO")
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("gate_reuse", "resident_batch", "queued"),
        default=["queued"],
    )
    args = parser.parse_args()
    if args.extension:
        torch.ops.load_library(str(args.extension.resolve()))
        extension = args.extension
    else:
        import vllm._sm70_gguf_persistent_C as native

        extension = Path(native.__file__)
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_tf32 = False
    splits = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "-*-of-*.gguf", args.model.name)
    readers = [GGUFReader(p) for p in sorted(args.model.parent.glob(splits))]
    tensors = {t.name: t for reader in readers for t in reader.tensors}
    routing = json.loads(args.routes.read_text())
    report = dict(
        scope="research-only full local MoE chain; no model performance claim",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        core_sha256=fingerprint(core.__file__),
        extension_sha256=fingerprint(extension),
        benchmark_sha256=fingerprint(__file__),
        route_sha256=fingerprint(args.routes),
        topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        shape=dict(m=5, topk=10, experts=512, n=160, k=2560, tp=4, rank=0),
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for layer in args.layers:
        source = [
            tensors[f"blk.{layer}.ffn_{name}_exps.weight"]
            for name in ("gate", "up", "down")
        ]
        source_type, down_type = int(source[0].tensor_type), int(source[2].tensor_type)
        assert source_type in (18, 21, 22) and int(source[1].tensor_type) == source_type
        assert down_type in (20, 42)
        raw = [
            torch.from_numpy(
                np.stack(
                    [
                        RawGGUFProjection.from_rows(rows, source_type)
                        .tp_slice(0, 4, axis=0)
                        .data
                        for rows in tensor.data
                    ]
                )
            ).cuda()
            for tensor in source[:2]
        ]
        bank = GGUFExpertBank(down_type, 512, torch.device("cuda"), torch.float16)
        for expert, rows in enumerate(source[2].data):
            bank.add(expert, torch.from_numpy(rows.copy()), 0, 4, axis=1)
        bank.finalize()
        torch.manual_seed(20261009 + layer)
        x = (torch.randn((5, 2560), device="cuda") * 0.25).half()
        ids = torch.empty((5, 10), device="cuda", dtype=torch.int32)
        probabilities = torch.softmax(torch.randn((5, 10), device="cuda"), dim=1)
        q8 = torch.empty((5, 80, 36), device="cuda", dtype=torch.uint8)
        hidden = torch.empty((5, 10, 5, 36), device="cuda", dtype=torch.uint8)
        candidate_hidden = torch.empty_like(hidden)
        control, candidate = [torch.empty_like(x) for _ in range(2)]
        routes = torch.empty((5, 10, 2560), device="cuda", dtype=torch.float16)
        ready = torch.zeros(250, device="cuda", dtype=torch.int64)
        epochs = torch.zeros(80, device="cuda", dtype=torch.int64)
        activated = torch.empty((5, 10, 160), device="cuda", dtype=torch.float16)
        work = torch.empty(4096, device="cuda", dtype=torch.uint8)

        def legacy(
            q8=q8,
            x=x,
            hidden=hidden,
            ids=ids,
            raw=raw,
            source_type=source_type,
            control=control,
            probabilities=probabilities,
            bank=bank,
            down_type=down_type,
        ):
            torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
            torch.ops._C.gguf_dp4a_gate_up_sm70_out(
                hidden, q8, ids, *raw, source_type, True
            )
            torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
                control,
                hidden,
                ids,
                probabilities,
                bank.weight_ptrs,
                bank.stat_ptrs,
                down_type,
                512,
            )

        def persistent(
            candidate=candidate,
            x=x,
            ids=ids,
            probabilities=probabilities,
            raw=raw,
            bank=bank,
            candidate_hidden=candidate_hidden,
            routes=routes,
            ready=ready,
            epochs=epochs,
            source_type=source_type,
            down_type=down_type,
        ):
            torch.ops.sm70_gguf_persistent.run(
                candidate,
                x,
                ids,
                probabilities,
                *raw,
                bank.weight_ptrs,
                bank.stat_ptrs,
                candidate_hidden,
                routes,
                ready,
                epochs,
                source_type,
                down_type,
                1,
            )

        def gate_reused(
            q8=q8,
            x=x,
            candidate_hidden=candidate_hidden,
            ids=ids,
            raw=raw,
            source_type=source_type,
            candidate=candidate,
            probabilities=probabilities,
            bank=bank,
            down_type=down_type,
        ):
            torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
            torch.ops.sm70_gguf_persistent.gate_reuse(
                candidate_hidden, q8, ids, *raw, source_type
            )
            torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
                candidate,
                candidate_hidden,
                ids,
                probabilities,
                bank.weight_ptrs,
                bank.stat_ptrs,
                down_type,
                512,
            )

        def queued(
            candidate=candidate,
            x=x,
            ids=ids,
            probabilities=probabilities,
            raw=raw,
            bank=bank,
            q8=q8,
            activated=activated,
            candidate_hidden=candidate_hidden,
            routes=routes,
            work=work,
            source_type=source_type,
            down_type=down_type,
        ):
            torch.ops.sm70_gguf_persistent.queued(
                candidate,
                x,
                ids,
                probabilities,
                *raw,
                bank.weight_ptrs,
                bank.stat_ptrs,
                q8,
                activated,
                candidate_hidden,
                routes,
                work,
                source_type,
                down_type,
            )

        records = routing[str(layer)]["records"]
        # Changing inputs, routes and probabilities between graph replays detects
        # stale flags, stale routed activations and incorrect expert sharing.
        ids.copy_(torch.tensor(records[0]["ids"], device="cuda", dtype=torch.int32))
        operations = dict(
            gate_reuse=gate_reused, resident_batch=persistent, queued=queued
        )
        graphs = {"legacy": captured(legacy)}
        graphs.update({name: captured(operations[name]) for name in args.variants})
        numerics = {name: [] for name in args.variants}
        for ordinal, record in enumerate(records[:8]):
            ids.copy_(torch.tensor(record["ids"], device="cuda", dtype=torch.int32))
            x.copy_((torch.randn_like(x.float()) * (0.1 + ordinal * 0.05)).half())
            probabilities.copy_(torch.softmax(torch.randn_like(probabilities), dim=1))
            graphs["legacy"].replay()
            for name in numerics:
                graphs[name].replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(candidate_hidden, hidden, rtol=0, atol=0)
                torch.testing.assert_close(candidate, control, rtol=0.002, atol=0.002)
                numerics[name].append(error(candidate, control))
        # Official FP32 weights with the same Q8 activation and FP16 boundaries.
        decoded = q8_values(hidden).reshape(5, 10, 160)
        oracle_down = torch.empty((5, 10, 2560), device="cuda", dtype=torch.float16)
        for expert in ids.unique().tolist():
            weight = torch.from_numpy(
                dequantize(source[2].data[expert], down_type)[:, :160]
            ).cuda()
            locations = (ids == expert).nonzero()
            oracle_down[locations[:, 0], locations[:, 1]] = (
                decoded[locations[:, 0], locations[:, 1]] @ weight.T
            ).half()
        oracle = (oracle_down.float() * probabilities[..., None]).sum(1).half()
        torch.testing.assert_close(candidate, oracle, rtol=0.003, atol=0.003)
        if args.profile_only:
            torch.cuda.cudart().cudaProfilerStart()
            operations[args.variants[0]]()
            torch.accelerator.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print("PROFILE_DONE; no accepted timing", flush=True)
            return
        samples = {name: [] for name in graphs}
        for epoch in range(8):
            for name in list(graphs)[:: 1 if epoch % 2 == 0 else -1]:
                samples[name].append(elapsed(graphs[name], args.iterations))
        median = {name: statistics.median(values) for name, values in samples.items()}
        unique = int(ids.unique().numel())
        gu_bytes = sum(v.numel() * v.element_size() for v in raw) // 512
        down_bytes = (
            sum(v.numel() * v.element_size() for v in (bank.weights, bank.stats)) // 512
        )
        case = dict(
            layer=layer,
            source_type=source_type,
            down_type=down_type,
            unique_experts=unique,
            unique_weight_bytes=unique * (gu_bytes + down_bytes),
            route_weight_bytes=50 * (gu_bytes + down_bytes),
            epoch_us=samples,
            median_us=median,
            delta_us={name: median["legacy"] - median[name] for name in numerics},
            effective_compulsory_gbps=unique
            * (gu_bytes + down_bytes)
            / median[args.variants[0]]
            / 1000,
            kernel_count={
                name: 1 if name == "resident_batch" else 3 for name in graphs
            },
            replay_numerics=numerics,
            official_down_error=error(candidate, oracle),
            clocks=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,clocks.sm,clocks.mem",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
        )
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)
        del raw, bank, graphs
        torch.accelerator.empty_cache()
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
