# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen partitioning replicated QSA scoring/selection by query row on TP4.

Launch with torchrun on four idle V100s. Includes the production bounded
scorer, exact top-512 operator, padding, copies and PyNccl index all-gather.
Key gathering and token expansion are common to both arms and excluded.
No model cache writes or runtime selection policy are changed by this screen.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from collectives import measure
from qsa_indexer_tiles import score_and_select, selection_error

from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.config.sm70_sparse import Sm70SparseConfig
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.models.qwen4_exp.nvidia.ops.qsa import expand_qsa_block_indices_cuda


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
    config = SimpleNamespace(
        kernel_config=KernelConfig(sm70_sparse=Sm70SparseConfig(qsa_score_tile_mb=32))
    )
    report = {"cases": []}
    passed = True
    try:
        with set_current_vllm_config(config):
            for m in (7344, 8976, 7345):
                n = 8672
                torch.manual_seed(9683 + m)
                q = torch.randn(m, 4, 128, device="cuda", dtype=torch.half) * 0.125
                keys = torch.randn(n, 128, device="cuda", dtype=torch.half) * 0.125
                valid = torch.ones(n, device="cuda", dtype=torch.bool)
                valid[::127] = False
                positions = (
                    n * 4 - m + torch.arange(m, device="cuda", dtype=torch.int32)
                )
                visible = (positions + 1) // 4
                step = (m + 3) // 4
                first, last = rank * step, min(m, (rank + 1) * step)
                padded = torch.full((step, 512), -1, device="cuda", dtype=torch.int32)
                gathered = torch.empty(4 * step, 512, device="cuda", dtype=torch.int32)
                token_padding = torch.full(
                    (step, 2051), -1, device="cuda", dtype=torch.int32
                )
                token_gathered = torch.empty(
                    4 * step, 2051, device="cuda", dtype=torch.int32
                )
                requests = torch.zeros(m, device="cuda", dtype=torch.int32)
                lengths = torch.tensor([n * 4], device="cuda", dtype=torch.int32)

                def expand(blocks, pos=positions, req=requests, lengths=lengths):
                    return expand_qsa_block_indices_cuda(
                        blocks, pos, lengths, req, 4, 2048
                    )

                def control(inputs=(q, keys, valid, visible)):
                    return score_and_select(*inputs, 32)[0]

                def partitioned(
                    inputs=(q[first:last], keys, valid, visible[first:last]),
                    padded=padded,
                    gathered=gathered,
                    rows=m,
                ):
                    local = score_and_select(*inputs, 32)[0]
                    padded[: local.shape[0]].copy_(local)
                    nccl.all_gather(gathered, padded)
                    return gathered[:rows]

                def control_tokens(control=control, expand=expand):
                    return expand(control())

                def partitioned_tokens(
                    inputs=(q[first:last], keys, valid, visible[first:last]),
                    pos=positions[first:last],
                    req=requests[first:last],
                    lengths=lengths,
                    padded=token_padding,
                    gathered=token_gathered,
                    rows=m,
                ):
                    local = score_and_select(*inputs, 32)[0]
                    tokens = expand_qsa_block_indices_cuda(
                        local, pos, lengths, req, 4, 2048
                    )
                    padded[: tokens.shape[0]].copy_(tokens)
                    nccl.all_gather(gathered, padded)
                    return gathered[:rows]

                expected = control()
                actual = partitioned()
                expanded_same_blocks = expand(actual)
                expanded = partitioned_tokens()
                expansion_equal = torch.equal(expanded_same_blocks, expanded)
                errors = [None] * 4
                dist.all_gather_object(
                    errors,
                    selection_error(actual, expected, visible)
                    | {"expansion_same_blocks_exact": expansion_equal},
                )
                row = {
                    "rows": m,
                    "columns": n,
                    "per_rank_rows": step,
                    "index_gather_total_bytes": gathered.numel() * 4,
                    "selection_per_rank": errors,
                    "replicated": measure(control),
                    "partitioned": measure(partitioned),
                    "replicated_repeat": measure(control),
                    "expanded_total_bytes": token_gathered.numel() * 4,
                    "replicated_with_expansion": measure(control_tokens),
                    "partitioned_expanded_gather": measure(partitioned_tokens),
                    "replicated_expanded_repeat": measure(control_tokens),
                    "expansion_same_blocks_exact": all(
                        error["expansion_same_blocks_exact"] for error in errors
                    ),
                    "passed": all(
                        error["within_micro_bound"]
                        and error["expansion_same_blocks_exact"]
                        for error in errors
                    ),
                }
                passed &= row["passed"]
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
