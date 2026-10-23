# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual TP4 expert down weights: integer dot and fused weighted reduction."""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch
import vllm._C as core
from benchmark_gguf_dp4a_expert import graph_time

import vllm
from vllm.model_executor.layers.quantization.gguf_turbomind_moe import (
    GGUFExpertBank,
    _expert_down,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    assert "site-packages" in vllm.__file__, vllm.__file__
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    reader = GGUFReader(args.model)
    tensor = next(
        t for t in reader.tensors if t.name == f"blk.{args.layer}.ffn_down_exps.weight"
    )
    source_type = int(tensor.tensor_type)
    assert source_type in (20, 42)
    experts, n, k = 512, 2560, 160
    bank = GGUFExpertBank(source_type, experts, torch.device("cuda"), torch.float16)
    for expert, source in enumerate(tensor.data):
        bank.add(expert, torch.from_numpy(source.copy()), args.rank, 4, axis=1)
    bank.finalize()
    result = dict(
        version=vllm.__version__,
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layer=args.layer,
        source_type=source_type,
        shape=[experts, n, k],
        activation_std=0.25,
        calls_per_graph=16,
        cases=[],
    )
    for m in args.m:
        torch.manual_seed(20261005 + m)
        ids = torch.randn((m, experts), device="cuda").topk(10, dim=1).indices.int()
        probabilities = torch.softmax(torch.randn((m, 10), device="cuda"), dim=1)
        hidden = (torch.randn((m, 10, k), device="cuda") * 0.25).half()
        sorted_ids, order = ids.flatten().long().sort(stable=True)
        inverse = order.argsort().int()
        offsets = torch.searchsorted(
            sorted_ids, torch.arange(experts + 1, device="cuda")
        ).int()
        routed = hidden.reshape(-1, k)[order].contiguous()
        baseline_down = torch.empty((m * 10, n), device="cuda", dtype=torch.float16)
        out = torch.empty((m, n), device="cuda", dtype=torch.float16)

        def native(
            baseline_down=baseline_down,
            routed=routed,
            offsets=offsets,
            inverse=inverse,
            probabilities=probabilities,
        ):
            _expert_down(
                baseline_down,
                routed,
                offsets,
                bank.weight_ptrs,
                bank.stat_ptrs,
                source_type,
                bank.decoder,
                experts,
                32,
                bank.down_vector_batches,
            )
            return torch.ops.vllm.sm70_small_expert_unroute(
                baseline_down, inverse, probabilities
            )

        def dp4a(out=out, hidden=hidden, ids=ids, probabilities=probabilities):
            torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
                out,
                hidden,
                ids,
                probabilities,
                bank.weight_ptrs,
                bank.stat_ptrs,
                source_type,
                experts,
            )

        fp16 = native()
        dp4a()
        groups = hidden.float().reshape(m, 10, k // 32, 32)
        maximum = groups.abs().amax(-1, keepdim=True)
        d = (maximum.double() / 127).float()
        ratio = torch.where(maximum == 0, 0, (groups.double() / d.double()).float())
        q = (ratio.sign() * (ratio.abs() + 0.5).floor()).to(torch.int8)
        decoded = (q.float() * d.half().float()).reshape(m, 10, k)
        down = torch.empty((m, 10, n), device="cuda", dtype=torch.float16)
        for expert in ids.unique().tolist():
            weight = torch.from_numpy(
                dequantize(tensor.data[expert], source_type)[
                    :, args.rank * k : (args.rank + 1) * k
                ]
            ).cuda()
            locations = (ids == expert).nonzero()
            down[locations[:, 0], locations[:, 1]] = (
                decoded[locations[:, 0], locations[:, 1]] @ weight.T
            ).half()
        expected = (down.float() * probabilities[..., None]).sum(1).half()
        torch.testing.assert_close(out, expected, rtol=0.003, atol=0.003)
        samples = {"native_down_and_unroute": [], "dp4a_down_and_unroute": []}
        operations = {"native_down_and_unroute": native, "dp4a_down_and_unroute": dp4a}
        clocks = []
        for epoch in range(8):
            for name in list(operations)[:: 1 if epoch % 2 == 0 else -1]:
                samples[name].append(graph_time(operations[name], args.iterations))
            clocks.append(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=clocks.sm,clocks.mem",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip()
            )
        median = {name: statistics.median(values) for name, values in samples.items()}
        expert_bytes = (
            bank.weights.numel() * bank.weights.element_size()
            + bank.stats.numel() * bank.stats.element_size()
        ) // experts
        unique_bytes = expert_bytes * ids.unique().numel()
        case = dict(
            m=m,
            active_experts=ids.unique().numel(),
            unique_weight_bytes=unique_bytes,
            route_weight_bytes=expert_bytes * m * 10,
            epoch_us=samples,
            median_us=median,
            clocks=clocks,
            unique_source_gbps=unique_bytes / median["dp4a_down_and_unroute"] / 1000,
            kernel_count={"native_down_and_unroute": 2, "dp4a_down_and_unroute": 1},
            official_q8_relative_l2=(
                (out.float() - expected.float()).norm() / expected.float().norm()
            ).item(),
            fp16_relative_l2=(
                (out.float() - fp16.float()).norm() / fp16.float().norm()
            ).item(),
        )
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
