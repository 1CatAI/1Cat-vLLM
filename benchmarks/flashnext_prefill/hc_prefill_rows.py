# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen row-partitioned HC with transient, NVLink-gathered weight shards.

This keeps the resident sharded weight format. Each invocation includes two
weight gathers, both projections, all layout copies, mixing and result gather.
No runtime admission is changed. Require finite outputs and maximum paired
row-relative L2 <= 0.001 separately for block output and injection logits.
The repeated control must be beaten by 10% at both large query sizes before
considering integration and representative checkpoint/output qualification.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from collectives import measure
from hc_pipeline import sequential
from moe_tiles import relative_l2

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.models.qwen4_exp.nvidia.ops.hc import (
    hc_gate_mix_gathered,
    hc_packet_silu,
)

ORDER = (0, 1, 2, 3)


def partitioned(nccl, x, down, up, rank):
    rows = x.shape[0]
    step = (rows + 3) // 4
    part = x[rank * step : min((rank + 1) * step, rows)]
    full_down = down.new_empty((4 * down.shape[0], down.shape[1]))
    nccl.all_gather(full_down, down)
    local = torch.mm(part, full_down.T)
    del full_down
    packet = local.reshape(-1, 4, 88).transpose(0, 1).contiguous().flatten(0, 1)
    del local
    lora, injection = hc_packet_silu(packet, ORDER, 4)
    del packet
    full_up = up.new_empty((4 * up.shape[0], up.shape[1]))
    nccl.all_gather(full_up, up)
    gate = torch.mm(lora, full_up.T)
    del lora, full_up
    gate = gate.reshape(-1, 4, 2560).transpose(0, 1).contiguous().flatten(0, 1)
    mixed = hc_gate_mix_gathered(part, gate, ORDER)
    del gate
    packed = x.new_zeros((step, 2564))
    packed[: part.shape[0], :2560].copy_(mixed)
    packed[: part.shape[0], 2560:].copy_(injection)
    del mixed, injection
    gathered = x.new_empty((4 * step, 2564))
    nccl.all_gather(gathered, packed)
    return gathered[:rows, :2560].contiguous(), gathered[:rows, 2560:].contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    nccl = PyNcclCommunicator(dist.group.WORLD, rank)
    assert not nccl.disabled
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    report = {"cases": [], "paired_row_relative_l2_limit": 0.001}
    passed = True
    try:
        for rows in (7344, 8976, 7345):
            torch.manual_seed(rows)
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.half) * 0.3
            torch.manual_seed(100 + rank)
            down = torch.randn(88, 10240, device="cuda", dtype=torch.half) * 0.01
            up = torch.randn(2560, 320, device="cuda", dtype=torch.half) * 0.01

            def control(x=x, down=down, up=up):
                return sequential(nccl, x, down, up, rank)

            def candidate(x=x, down=down, up=up):
                return partitioned(nccl, x, down, up, rank)

            expected, actual = control(), candidate()
            errors = [relative_l2(a, b) for a, b in zip(actual, expected)]
            finite = all(bool(torch.isfinite(a).all()) for a in actual)
            checks = [None] * 4
            dist.all_gather_object(checks, (errors, finite))
            row = {
                "tokens": rows,
                "max_rank_block_error": max(e[0] for e, _ in checks),
                "max_rank_injection_error": max(e[1] for e, _ in checks),
                "numerical_pass": all(max(e) <= 0.001 and f for e, f in checks),
                "control": measure(control),
                "partitioned_with_weight_gathers": measure(candidate),
                "control_repeat": measure(control),
            }
            passed &= row["numerical_pass"]
            report["cases"].append(row)
            if rank == 0:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    finally:
        torch.accelerator.synchronize()
        dist.destroy_process_group()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
