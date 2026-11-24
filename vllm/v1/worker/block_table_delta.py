# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Delta commit of the per-group block tables (VLLM_INKLING2_SM70_BLOCKTABLE).

`BlockTable.commit_block_table` copies whole rows, `max_model_len / block_size`
int32 each, for every KV group that changed. With Inkling's short-conv state
(`SlidingWindowSpec`, block_size 4) the rows are 65,536 wide at 262,144 tokens,
and 12 groups share one runner, so a batch-8 decode step moved ~16 MB of
mostly-unchanged table (4 ms of GPU idle before each CUDA-graph replay).

A decode step only appends a block (or none) to each row. `BlockTable` records
the dirty column range per row; this module packs the changed entries of all
groups as (group, flat index, value) triplets into one pinned buffer, copies it
with a single H2D memcpy and scatters it into the group tables with one Triton
kernel. Columns past a row's live length keep their old GPU contents, as with
`VLLM_V41_LIVE_TABLE`: kernels only read a row's table below its live length.
"""

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Triplets per commit (12 B each). Larger commits (a long prompt entering the
# batch) fall back to the full-row copy.
_CAP_TRIPLETS = 1 << 17
_RING = 4


@triton.jit(do_not_specialize_on_alignment=["n"])
def _scatter_block_table_kernel(
    table_ptrs,  # int64[num_groups]: base address of each group's int32 table
    packed,  # int32[3 * n]: (group, flat index, value) triplets
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gid = tl.load(packed + offs * 3, mask=mask, other=0)
    idx = tl.load(packed + offs * 3 + 1, mask=mask, other=0)
    val = tl.load(packed + offs * 3 + 2, mask=mask, other=0)
    base = tl.load(table_ptrs + gid, mask=mask, other=0).to(tl.pointer_type(tl.int32))
    tl.store(base + idx, val, mask=mask)


class BlockTableDelta:
    def __init__(self, block_tables: list, device: torch.device) -> None:
        self.block_tables = block_tables
        self.device = device
        self._ptrs = torch.tensor(
            [bt.block_table.gpu.data_ptr() for bt in block_tables],
            dtype=torch.int64,
            device=device,
        )
        # Ring of pinned staging buffers: the host may only reuse a slot once
        # the copy issued _RING steps ago has read it, so it never waits on
        # the GPU in steady state.
        self._cpus = [
            torch.zeros(3 * _CAP_TRIPLETS, dtype=torch.int32, pin_memory=True)
            for _ in range(_RING)
        ]
        self._nps = [c.numpy() for c in self._cpus]
        self._events = [torch.cuda.Event() for _ in range(_RING)]
        self._pending = [False] * _RING
        self._slot = 0
        self._gpu = torch.zeros(3 * _CAP_TRIPLETS, dtype=torch.int32, device=device)
        # JIT the kernel now (n = 0 writes nothing).
        _scatter_block_table_kernel[(1,)](self._ptrs, self._gpu, 0, BLOCK=128)

    def commit(self, num_reqs: int) -> bool:
        """Scatter every group's changed entries; False means the caller must
        use the full-row copy (nothing was modified)."""
        small: list[int] = []  # interleaved (group, flat index, value)
        big = []  # (group, flat index array, values) for long ranges
        total = 0
        for gid, bt in enumerate(self.block_tables):
            if not bt._dirty_rows:
                continue
            width = bt.max_num_blocks_per_req
            table = bt.block_table.np
            lo_arr, hi_arr = bt._row_lo, bt._row_hi
            for row in bt._dirty_rows:
                if row >= num_reqs:
                    continue
                lo = int(lo_arr[row])
                hi = int(hi_arr[row])
                n = hi - lo
                if n <= 0:
                    continue
                total += n
                base = row * width + lo
                if n <= 8:
                    for k, v in enumerate(table[row, lo:hi].tolist()):
                        small += (gid, base + k, v)
                else:
                    big.append((gid, np.arange(base, base + n), table[row, lo:hi]))
        if total > _CAP_TRIPLETS:
            return False
        if total:
            slot = self._slot
            self._slot = (slot + 1) % _RING
            if self._pending[slot]:
                self._events[slot].synchronize()  # copy from _RING steps ago
            flat_np = self._nps[slot]
            off = len(small)
            if small:
                flat_np[:off] = small
            staging = flat_np[off : 3 * total].reshape(-1, 3)
            off = 0
            for gid, idx, vals in big:
                n = idx.shape[0]
                staging[off : off + n, 0] = gid
                staging[off : off + n, 1] = idx
                staging[off : off + n, 2] = vals
                off += n
            self._gpu[: 3 * total].copy_(
                self._cpus[slot][: 3 * total], non_blocking=True
            )
            _scatter_block_table_kernel[(triton.cdiv(total, 128),)](
                self._ptrs, self._gpu, total, BLOCK=128
            )
            self._events[slot].record()
            self._pending[slot] = True
        for bt in self.block_tables:
            bt._finish_delta_commit(num_reqs)
        return True
