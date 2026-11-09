# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline parallelism utilities for the V2 model runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """One deferred PP sampled-output update."""

    event: torch.cuda.Event
    sampled_tokens: torch.Tensor
    draft_tokens: torch.Tensor | None
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    need_sampled_mask: np.ndarray
    gen_at_receive_np: np.ndarray


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return rows whose sampled output can feed a later decode step."""
    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    max_seq_len = input_batch.max_seq_len_np
    assert max_seq_len is not None

    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len
    need_sampled_mask = produces_sample & not_finishing
    return need_sampled_mask if need_sampled_mask.any() else None


def _pack_token_packet(
    sampled_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor | None,
    max_sample_len: int,
    max_draft_len: int,
) -> torch.Tensor:
    if sampled_token_ids.dtype != torch.int64 or sampled_token_ids.ndim != 2:
        raise ValueError("sampled_token_ids must be a 2D int64 tensor")
    if sampled_token_ids.shape[1] > max_sample_len:
        raise ValueError("sampled token width exceeds the PP receive contract")
    if max_draft_len:
        if draft_token_ids is None:
            raise ValueError("DFlash PP broadcast requires next-step draft tokens")
        if draft_token_ids.dtype != torch.int64 or draft_token_ids.ndim != 2:
            raise ValueError("draft_token_ids must be a 2D int64 tensor")
        if draft_token_ids.shape != (sampled_token_ids.shape[0], max_draft_len):
            raise ValueError("draft token shape does not match the PP contract")

    packet = sampled_token_ids.new_full(
        (sampled_token_ids.shape[0], max_sample_len + max_draft_len), -1
    )
    packet[:, : sampled_token_ids.shape[1]].copy_(sampled_token_ids)
    if draft_token_ids is not None:
        packet[:, max_sample_len:].copy_(draft_token_ids)
    return packet


@triton.jit
def _scatter_draft_tokens_kernel(
    dst_ptr,
    dst_stride,
    idx_mapping_ptr,
    src_ptr,
    src_stride,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < width
    values = tl.load(src_ptr + row * src_stride + col, mask=mask)
    tl.store(dst_ptr + req_state_idx * dst_stride + col, values, mask=mask)


def scatter_draft_tokens(
    dst: torch.Tensor, idx_mapping: torch.Tensor, src: torch.Tensor | None
) -> None:
    if src is None or src.shape[1] == 0 or idx_mapping.shape[0] == 0:
        return
    width = src.shape[1]
    _scatter_draft_tokens_kernel[(idx_mapping.shape[0],)](
        dst,
        dst.stride(0),
        idx_mapping,
        src,
        src.stride(0),
        width,
        BLOCK_SIZE=triton.next_power_of_2(width),
    )


class PPHandler:
    """Pipeline sampled/draft transport with a PP-depth deferred slot ring."""

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ) -> None:
        pp = get_pp_group()
        self.is_last_rank = pp.is_last_rank
        self.last_rank = pp.last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.max_draft_len = num_speculative_steps
        self.packet_width = self.max_sample_len + self.max_draft_len
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * pp.world_size)
        )
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)
        self.broadcast_group = pp.make_sibling_device_group(
            group_desc="pp_sampled_draft_broadcast"
        )

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor | None] | None:
        """Consume the receive from one full pipeline traversal earlier."""
        if not self.queue:
            return None
        slot = self.queue.popleft()
        self.queue.append(None)
        if slot is None:
            return None

        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                return None
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        return {
            "sampled_tokens": slot.sampled_tokens,
            "draft_tokens": slot.draft_tokens,
            "num_sampled": slot.num_sampled,
            "num_rejected": slot.num_rejected,
            "idx_mapping": idx_mapping,
        }

    def receive(self, input_batch: InputBatch) -> bool:
        """Queue a non-last-rank receive; return whether every row is decode."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            return False

        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]
        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            token_packet = torch.empty(
                num_reqs,
                self.packet_width,
                dtype=torch.int64,
                device=self.device,
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                token_packet, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            sampled_tokens = token_packet[:, : self.max_sample_len]
            draft_tokens = (
                token_packet[:, self.max_sample_len :] if self.max_draft_len else None
            )
            token_packet.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)

        self.queue[-1] = PendingRecv(
            event=event,
            sampled_tokens=sampled_tokens,
            draft_tokens=draft_tokens,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            idx_mapping=input_batch.idx_mapping,
            idx_mapping_np=input_batch.idx_mapping_np,
            need_sampled_mask=need_sampled_mask,
            gen_at_receive_np=gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
        draft_token_ids: torch.Tensor | None = None,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            token_packet = _pack_token_packet(
                sampled_token_ids,
                draft_token_ids,
                self.max_sample_len,
                self.max_draft_len,
            )
            torch.distributed.broadcast(
                token_packet, src=self.last_rank, group=self.broadcast_group
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (
                token_packet,
                sampled_token_ids,
                num_sampled,
                num_rejected,
            ):
                tensor.record_stream(self.broadcast_stream)
            if draft_token_ids is not None:
                draft_token_ids.record_stream(self.broadcast_stream)
