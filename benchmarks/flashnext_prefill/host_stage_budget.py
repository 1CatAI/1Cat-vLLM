# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pair existing FP16 draft host/staged attention across page-budget limits.

Real host_kv_forward dispatch, synthetic KV/queries and one shared stage.
Require finite paired max-row L2 <=0.001 and max absolute error <=0.002.
CUDA event timings include stream-ordered staging; not whole-model latency.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from vllm.models.qwen4_exp.nvidia import qsa
from vllm.models.qwen4_exp.nvidia.ops import host_kv
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=513)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(9683)
    device = torch.device("cuda", 0)
    page, blocks, maximum = 816, 322, 262144
    pool = torch.zeros((blocks, 2, page, 1, 256), dtype=torch.half, pin_memory=True)
    history = get_accelerator_view_from_cpu_tensor(pool)
    state = host_kv.HostQSAKV(
        blocks,
        page,
        256,
        device,
        hot_tokens=16384,
        rows=8,
        dtype=torch.float16,
        is_speculative_draft=True,
        history=history,
    )
    table = torch.randperm(blocks, device=device, dtype=torch.int32)[None]
    k = torch.randn(maximum, 1, 256, device=device, dtype=torch.half) * 0.2
    v = torch.randn_like(k) * 0.2
    slots = table[0, torch.arange(maximum, device=device) // page].long() * page
    slots += torch.arange(maximum, device=device) % page
    state.write(k, v, slots)
    del k, v, slots
    report = {"scope": "synthetic FP16 draft QSA host_kv_forward", "cases": []}
    for context in (130560, 130561, 261936, 262144):
        rows = args.rows
        positions = torch.arange(context - rows, context, device=device)
        complete = (context - rows) // 4
        permutation = torch.randperm(complete, device=device)
        starts = torch.randint(complete, (rows, 1), device=device)
        groups = permutation[(starts + torch.arange(512, device=device)) % complete]
        indices = (groups[:, :, None] * 4 + torch.arange(4, device=device)).flatten(1)
        tail = positions[:, None] // 4 * 4 + torch.arange(3, device=device)
        tail = torch.where(
            torch.arange(3, device=device) < (positions[:, None] + 1) % 4,
            tail,
            -1,
        )
        indices = torch.cat((indices, tail), dim=1).int()
        query = torch.randn(rows, 6, 256, device=device, dtype=torch.half) * 0.2
        gate = torch.randn_like(query) * 0.2
        requests = torch.zeros(rows, device=device, dtype=torch.int32)
        lengths = torch.tensor([context], device=device, dtype=torch.int32)
        case = {"context_tokens": context, "rows": rows, "budgets": {}}
        reference = None
        for budget in (262144, 525504):
            owner = NS(
                host_kv=state,
                host_kv_prefill_stage_tokens=budget,
                host_kv_prefill_stage_slots=1,
                host_kv_is_draft=True,
            )
            metadata = NS()
            output = torch.empty_like(query)

            def run(
                owner=owner,
                metadata=metadata,
                output=output,
                query=query,
                indices=indices,
                requests=requests,
                positions=positions,
                lengths=lengths,
                gate=gate,
            ):
                qsa.Qwen4ExpQSAAttention.host_kv_forward(
                    owner,
                    query,
                    indices,
                    table,
                    requests,
                    positions,
                    lengths,
                    output,
                    gate,
                    metadata,
                )

            run()
            torch.accelerator.synchronize()
            error = absolute = 0.0
            assert torch.isfinite(output).all()
            if reference is None:
                reference = output.clone()
            else:
                diff = (output.float() - reference.float()).flatten(1)
                error = float(
                    (diff.norm(dim=1) / reference.float().flatten(1).norm(dim=1)).max()
                )
                absolute = float(diff.abs().max())
                assert error <= 0.001 and absolute <= 0.002, (error, absolute)
            elapsed = []
            for _ in range(3):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                run()
                end.record()
                end.synchronize()
                elapsed.append(start.elapsed_time(end))
            case["budgets"][str(budget)] = dict(
                stage_pages=state._stage_capacity(budget),
                staged=state.stage_plan(table, lengths, budget) is not None,
                cuda_ms=elapsed,
                max_row_l2=error,
                max_absolute=absolute,
            )
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
    assert len(host_kv._STAGES) == 2  # A/B budgets, never one stage per dtype/layer.
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
