# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred commit of speculative GDN recurrent state.

Stock MTP verification stores the recurrent state after every verified
position in num_speculative_tokens extra state blocks per request, and the
next round reads the accepted one. Here the CUDA verifier instead reads the
request's committed state, replays the previous round's accepted tokens from
a small per-request buffer (normalized k, v, decay and beta), commits the
result and evaluates the current round without storing per-position states.
Requests then hold no speculative state blocks, and the replay adds no state
traffic.

Each request owns two buffer sides: a verification writes side ``s`` and
replays side ``s ^ 1``; the host flips ``s`` once per verified step.
Aligned prefix-cache checkpoints are stored by the verification itself, at
the in-round position that ends a Mamba block; a rejected position is simply
rewritten by the round that processes it again.
"""

from __future__ import annotations

import numpy as np
import torch

from vllm.triton_utils import tl, triton

TMAX = 8  # the native verifier's per-sequence token capacity
HEAD_DIM = 128

_STATE: SpecReplayState | None = None


def spec_replay_enabled(vllm_config) -> bool:
    """Whether GDN speculative states are committed by replay."""
    kernel = getattr(vllm_config, "kernel_config", None)
    spec = getattr(vllm_config, "speculative_config", None)
    return bool(
        kernel is not None
        and getattr(kernel, "gdn_spec_replay", False)
        and spec is not None
        and spec.method == "mtp"
        and 0 < spec.num_speculative_state_tokens() < TMAX
    )


def get_spec_replay_state() -> SpecReplayState | None:
    return _STATE


def set_spec_replay_state(state: SpecReplayState | None) -> None:
    global _STATE
    _STATE = state


class SpecReplayState:
    """Per-request replay bookkeeping shared by every GDN layer."""

    def __init__(
        self, max_reqs: int, max_rows: int, num_groups: int, device: torch.device
    ):
        i32 = {"dtype": torch.int32, "device": device}
        self.max_reqs = max_reqs
        self.max_rows = max_rows
        self.num_groups = num_groups
        # Per request: accepted count of its last verification, whether its
        # last step verified, and its current buffer side.
        self.accepted = torch.ones(max_reqs, **i32)
        self.verified = torch.zeros(max_reqs, **i32)
        self.side = torch.zeros(max_reqs, **i32)
        # Per verification row (compacted like the GDN speculative rows).
        self.spec_replay = torch.zeros(max_rows, **i32)
        self.spec_map = torch.zeros(max_rows, **i32)
        self.spec_ckpt_token = torch.full((max_rows,), -1, **i32)
        self.spec_ckpt_slot = torch.full((num_groups, max_rows), -1, **i32)
        # Per non-speculative row (decodes first): commit-only replay.
        self.plain_replay = torch.zeros(max_rows, **i32)
        self.plain_map = torch.zeros(max_rows, **i32)
        self.empty_starts = torch.zeros(max_rows + 1, **i32)
        self._rows_host = torch.empty(2 * max_rows, dtype=torch.int32, pin_memory=True)
        self._rows = torch.empty(2 * max_rows, **i32)
        self.group_of: dict[str, int] = {}
        self._buffers: dict[str, torch.Tensor] = {}

    def layer_buffer(self, layer: str, k_heads: int, v_heads: int, device):
        buffer = self._buffers.get(layer)
        if buffer is None:
            seq = TMAX * (k_heads * HEAD_DIM + v_heads * HEAD_DIM + 2 * v_heads)
            buffer = torch.zeros(
                2 * self.max_reqs * seq, dtype=torch.float32, device=device
            )
            self._buffers[layer] = buffer
        return buffer

    def reset_request(self, req_index: int) -> None:
        self.verified[req_index] = 0
        self.accepted[req_index] = 1

    def prepare(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        num_draft_tokens_per_req: np.ndarray | None,
        is_prefilling: np.ndarray | None,
        num_reqs: int,
        block_table_ptrs: torch.Tensor,
        block_table_stride: int,
        mamba_block_size: int,
    ) -> None:
        """Fill this step's row tables before the forward (eager)."""
        if num_reqs == 0:
            return
        drafts = (
            np.zeros(num_reqs, dtype=np.int64)
            if num_draft_tokens_per_req is None
            else np.asarray(num_draft_tokens_per_req[:num_reqs])
        )
        # Exactly the GDN metadata's speculative rows (mamba_hybrid builds
        # num_decode_draft_tokens the same way): the verifier's row n and
        # these tables must name the same request.
        spec = drafts > 0
        if is_prefilling is not None:
            spec &= ~np.asarray(is_prefilling[:num_reqs], dtype=bool)
        spec_rows = np.flatnonzero(spec)
        plain_rows = np.flatnonzero(~spec)
        num_spec, num_plain = spec_rows.size, plain_rows.size
        rows_np = self._rows_host.numpy()
        rows_np[:num_spec] = spec_rows
        rows_np[self.max_rows : self.max_rows + num_plain] = plain_rows
        self._rows.copy_(self._rows_host, non_blocking=True)
        block = triton.next_power_of_2(max(self.max_rows, 16))
        _prepare_replay_rows_kernel[(2,)](
            self._rows,
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            self.accepted,
            self.verified,
            self.side,
            self.spec_replay,
            self.spec_map,
            self.spec_ckpt_token,
            self.spec_ckpt_slot,
            self.plain_replay,
            self.plain_map,
            block_table_ptrs,
            block_table_stride,
            num_spec,
            num_plain,
            MAX_ROWS=self.max_rows,
            # Only the configured groups have block-table pointers.
            NUM_GROUPS=block_table_ptrs.numel(),
            MAMBA_BLOCK_SIZE=mamba_block_size,
            BLOCK=block,
        )

    def record_accepted(self, idx_mapping: torch.Tensor, accepted: torch.Tensor):
        """Snapshot each request's accepted count before align bookkeeping
        rewrites it for the conv state."""
        if idx_mapping.numel():
            index = idx_mapping.long()
            self.accepted.index_copy_(0, index, accepted.index_select(0, index))


@triton.jit
def _prepare_replay_rows_kernel(
    rows_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_ptr,
    accepted_ptr,
    verified_ptr,
    side_ptr,
    spec_replay_ptr,
    spec_map_ptr,
    spec_ckpt_token_ptr,
    spec_ckpt_slot_ptr,
    plain_replay_ptr,
    plain_map_ptr,
    block_table_ptrs_ptr,
    block_table_stride,
    num_spec,
    num_plain,
    MAX_ROWS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    which = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    in_table = offsets < MAX_ROWS
    if which == 0:
        live = offsets < num_spec
        rows = tl.load(rows_ptr + offsets, mask=live, other=0)
        req = tl.load(idx_mapping_ptr + rows, mask=live, other=0)
        verified = tl.load(verified_ptr + req, mask=live, other=0)
        accepted = tl.load(accepted_ptr + req, mask=live, other=0)
        replay = tl.where(live & (verified != 0), accepted, 0)
        side = tl.load(side_ptr + req, mask=live, other=0) ^ 1
        tl.store(side_ptr + req, side, mask=live)
        tl.store(verified_ptr + req, 1, mask=live)
        tl.store(spec_replay_ptr + offsets, replay, mask=in_table)
        tl.store(
            spec_map_ptr + offsets, tl.where(live, 2 * req + side, 0), mask=in_table
        )
        # The in-round position (if any) that completes a Mamba block.
        computed = tl.load(num_computed_ptr + req, mask=live, other=0)
        start = tl.load(query_start_loc_ptr + rows, mask=live, other=0)
        end = tl.load(query_start_loc_ptr + rows + 1, mask=live, other=0)
        boundary = ((computed + end - start) // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
        has_ckpt = live & (boundary > computed)
        ckpt_token = tl.where(has_ckpt, boundary - 1 - computed, -1)
        tl.store(spec_ckpt_token_ptr + offsets, ckpt_token, mask=in_table)
        column = boundary // MAMBA_BLOCK_SIZE - 1
        for group in tl.static_range(NUM_GROUPS):
            table = tl.load(block_table_ptrs_ptr + group).to(tl.pointer_type(tl.int32))
            slot = tl.load(
                table + rows * block_table_stride + column, mask=has_ckpt, other=-1
            )
            tl.store(
                spec_ckpt_slot_ptr + group * MAX_ROWS + offsets,
                tl.where(has_ckpt, slot, -1),
                mask=in_table,
            )
    else:
        live = offsets < num_plain
        rows = tl.load(rows_ptr + MAX_ROWS + offsets, mask=live, other=0)
        req = tl.load(idx_mapping_ptr + rows, mask=live, other=0)
        verified = tl.load(verified_ptr + req, mask=live, other=0)
        accepted = tl.load(accepted_ptr + req, mask=live, other=0)
        replay = tl.where(live & (verified != 0), accepted, 0)
        side = tl.load(side_ptr + req, mask=live, other=0)
        tl.store(verified_ptr + req, 0, mask=live)
        tl.store(plain_replay_ptr + offsets, replay, mask=in_table)
        # No current tokens: replay the side the last verification wrote.
        tl.store(
            plain_map_ptr + offsets,
            tl.where(live, 2 * req + (side ^ 1), 0),
            mask=in_table,
        )
