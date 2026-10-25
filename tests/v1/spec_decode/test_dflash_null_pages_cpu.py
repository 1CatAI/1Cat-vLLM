# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the production DFlash Triton kernel on CPU, before any GPU imports."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test


def test_dflash_null_physical_pages_with_virtual_splitting():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1", HF_HUB_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, __file__, "--worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence == {"cases": 12, "cuda_initialized": False}


def _run_worker():
    from types import SimpleNamespace as NS

    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

    assert not torch.cuda.is_initialized()
    physical_size = 1152
    physical_ids = [0, 7, 0, 11, 0, 13]
    starts = [
        (0, 12),  # First virtual subpage of physical null block 0.
        (58, 122),  # Later virtual subpages within the same null block.
        (1142, 2294),  # Null -> real and real -> null boundaries.
        (1216, 3520),  # Real pages preserve their exact storage offsets.
        (2358, 4726),  # Later evicted physical pages, not just logical page 0.
        (1078, 4534),  # Last virtual subpages before a physical boundary.
    ]
    count = 0
    for kernel_size in [64, physical_size]:
        blocks_per_page = physical_size // kernel_size
        for first, second in starts:
            # Exercise the real physical -> virtual expansion. In particular,
            # null physical IDs become [0, ..., blocks_per_page - 1].
            tables: list[list[int]] = [[], []]
            owner = object.__new__(BlockTables)
            owner.num_kv_cache_groups = 1
            owner.blocks_per_kv_block = [blocks_per_page]
            owner.num_blocks = NS(np=np.zeros((1, 2), dtype=np.int32))
            owner.block_tables = [
                NS(
                    stage_write=lambda req, start, ids, tables=tables: tables[
                        req
                    ].extend(ids)
                )
            ]
            for req in range(2):
                owner.append_block_ids(req, (physical_ids,), overwrite=True)
            block_table = torch.tensor(tables, dtype=torch.int32)
            nctx = [12, 10]
            rejected = [0, 3]
            positions_by_req = [
                list(range(start, start + size))
                for start, size in zip((first, second), nctx)
            ]
            positions = torch.tensor(sum(positions_by_req, []), dtype=torch.int64)
            capacity, max_reqs, nq = 32, 3, 8
            buffers = NS(
                input_ids=torch.full((capacity,), -99, dtype=torch.int32),
                positions=torch.full((capacity,), -99, dtype=torch.int64),
                query_start_loc=torch.full((max_reqs + 1,), -99, dtype=torch.int32),
                seq_lens=torch.full((max_reqs,), -99, dtype=torch.int32),
            )
            batch = NS(
                num_reqs=2,
                num_scheduled_tokens=np.array(nctx),
                positions=positions,
                query_start_loc=torch.tensor([0, 12, 22], dtype=torch.int32),
                idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
            )
            query_slots = torch.full((capacity,), -99, dtype=torch.int64)
            context_positions = query_slots.clone()
            context_slots = query_slots.clone()
            sample_indices = torch.full((max_reqs * 7,), -99, dtype=torch.int64)
            sample_positions = sample_indices.clone()
            sample_mapping = sample_indices.clone()
            prepare_dflash_inputs(
                buffers,
                query_slots,
                context_positions,
                context_slots,
                sample_indices,
                sample_positions,
                sample_mapping,
                torch.zeros(max_reqs),
                torch.zeros(max_reqs, dtype=torch.int64),
                batch,
                torch.tensor([1, 0], dtype=torch.int32),
                torch.tensor(rejected, dtype=torch.int32),
                torch.tensor([40, 41, 42]),
                torch.tensor([50, 51, 52]),
                torch.zeros(max_reqs),
                torch.tensor([17, 18, 19]),
                block_table,
                kernel_size,
                0,
                1,
                1,
                99,
                nq,
                7,
                max_reqs,
                capacity,
                8192,
                blocks_per_kv_block=blocks_per_page,
            )

            def expected_slot(position):
                physical = physical_ids[position // physical_size]
                if physical == 0:
                    return -1
                return physical * physical_size + position % physical_size

            offset = 0
            for req, request_positions in enumerate(positions_by_req):
                valid = request_positions[: nctx[req] - rejected[req]]
                expected_context = [expected_slot(p) for p in valid]
                expected_context += [-1] * rejected[req]
                queries = list(range(valid[-1] + 1, valid[-1] + 1 + nq))
                assert context_slots[offset : offset + nctx[req]].tolist() == (
                    expected_context
                )
                assert query_slots[req * nq : (req + 1) * nq].tolist() == [
                    expected_slot(p) for p in queries
                ]
                assert buffers.positions[req * nq : (req + 1) * nq].tolist() == queries
                assert buffers.seq_lens[req].item() == queries[-1] + 1
                offset += nctx[req]
            assert query_slots[16:].eq(-1).all()
            assert sample_mapping[:14].tolist() == [2] * 7 + [0] * 7
            assert sample_mapping[14:].eq(-1).all()
            assert buffers.input_ids[[0, 8]].tolist() == [42, 50]
            assert buffers.seq_lens[2].item() == 0
            count += 1
    assert not torch.cuda.is_initialized()
    print(json.dumps({"cases": count, "cuda_initialized": False}))


if __name__ == "__main__":
    assert sys.argv[1:] == ["--worker"]
    _run_worker()
