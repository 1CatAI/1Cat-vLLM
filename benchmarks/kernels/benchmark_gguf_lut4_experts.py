# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real TP4 canonical IQ4 expert layer, cold same-process ABBA."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import vllm._C as core

import vllm
from vllm.model_executor.layers.quantization.gguf_turbomind_moe import GGUFExpertBank
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader

sys.path.append(str(Path(__file__).resolve().parents[2]))
from benchmarks.kernels.benchmark_gguf_q8_intermediate import cold_time  # noqa: E402


@torch.compile
def activate(gate, up):
    return torch.nn.functional.silu(gate) * up


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=47)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--lanes-per-row", type=int, choices=(4, 8, 16), default=4)
    args = parser.parse_args()
    if "site-packages" not in vllm.__file__:
        raise ValueError("Requires an installed source-complete artifact")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    reader = GGUFReader(args.model)
    tensors = [
        next(
            t
            for t in reader.tensors
            if t.name == f"blk.{args.layer}.ffn_{role}_exps.weight"
        )
        for role in ("gate", "up", "down")
    ]
    kinds = [int(t.tensor_type) for t in tensors]
    if kinds[0] not in (20, 23) or kinds[1] not in (20, 23) or kinds[2] != 20:
        raise ValueError("Requires IQ4 gate/up with IQ4_NL down")
    banks = []
    for role, tensor in enumerate(tensors):
        bank = GGUFExpertBank(kinds[role], 512, torch.device("cuda"), torch.float16)
        for expert, row in enumerate(tensor.data):
            bank.add(
                expert, torch.from_numpy(row.copy()), 0, 4, axis=1 if role == 2 else 0
            )
        bank.finalize()
        banks.append(bank)
    eviction = torch.zeros(32 * 1024 * 1024 // 4, device="cuda")
    report = dict(
        version=vllm.__version__,
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        shape=[512, 160, 2560],
        source_types=kinds,
        layer=args.layer,
        lanes_per_row=args.lanes_per_row,
        routing="seeded synthetic random top10; real weights",
        cache="32MiB eviction before each timed operation",
        cases=[],
    )
    for m in (5, 20):
        torch.manual_seed(1000 + m)
        x = torch.randn((m, 2560), device="cuda", dtype=torch.float16) * 0.125
        ids = torch.randn((m, 512), device="cuda").topk(10, dim=1).indices.int()
        probabilities = torch.softmax(torch.randn((m, 10), device="cuda"), 1)
        q8 = torch.empty((m, 80, 36), device="cuda", dtype=torch.uint8)
        h = torch.empty((m, 10, 5, 36), device="cuda", dtype=torch.uint8)
        output = torch.empty_like(x)

        def baseline(x=x, ids=ids, probabilities=probabilities):
            routed, offsets, sorted_ids, inverse = (
                torch.ops.vllm.sm70_small_expert_route(x, ids, 512)
            )
            gate = banks[0](routed, offsets, sorted_ids)
            up = banks[1](routed, offsets, sorted_ids)
            hidden = activate(gate, up)
            down = banks[2](hidden, offsets, sorted_ids)
            return torch.ops.vllm.sm70_small_expert_unroute(
                down, inverse, probabilities
            )

        def candidate(
            x=x, ids=ids, probabilities=probabilities, q8=q8, h=h, output=output
        ):
            torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
            torch.ops._C.gguf_dp4a_lut4_gate_up_sm70_out(
                h,
                q8,
                ids,
                banks[0].weight_ptrs,
                banks[0].stat_ptrs,
                banks[1].weight_ptrs,
                banks[1].stat_ptrs,
                512,
                args.lanes_per_row,
            )
            torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
                output,
                h,
                ids,
                probabilities,
                banks[2].weight_ptrs,
                banks[2].stat_ptrs,
                20,
                512,
            )
            return output

        reference = baseline()
        candidate()
        relative = float(
            (output.float() - reference.float()).norm() / reference.float().norm()
        )
        if not torch.isfinite(output).all() or relative > 0.02:
            raise ValueError(
                f"IQ4 expert numerical screen failed: relative L2={relative}"
            )
        timings = [
            dict(
                arm=arm,
                us=cold_time(
                    baseline if arm == "canonical" else candidate,
                    eviction,
                    args.iterations,
                ),
            )
            for arm in ("canonical", "candidate", "candidate", "canonical")
        ]
        active = int(ids.unique().numel())
        # Unique canonical code/scale bytes, not an NCU DRAM-traffic claim.
        read_bytes = active * (2 * 160 * 2560 + 2560 * 160) * (0.5 + 2 / 32)
        result = dict(
            m=m,
            active_experts=active,
            relative_l2=relative,
            timings=timings,
            unique_weight_bytes=read_bytes,
            candidate_effective_gbs=[
                read_bytes / (v["us"] * 1000)
                for v in timings
                if v["arm"] == "candidate"
            ],
        )
        report["cases"].append(result)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
