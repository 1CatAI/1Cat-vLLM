# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline parallelism utilities for the V2 model runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


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


@dataclass
class _FeedbackBuffer:
    """One reusable, contiguous NCCL packet and its completion event.

    Tokens occupy the int64 prefix. The suffix is viewed as two int32 count
    arrays, retaining the count dtype without a second collective or padding.
    """

    packet: torch.Tensor
    event: torch.cuda.Event

    def views(
        self, num_reqs: int, token_width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packet = self.packet[: num_reqs * (token_width + 1)]
        tokens_end = num_reqs * token_width
        tokens = packet[:tokens_end].view(num_reqs, token_width)
        counts = packet[tokens_end:].view(torch.int32).view(2, num_reqs)
        return packet, tokens, counts


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
    out: torch.Tensor | None = None,
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

    elif draft_token_ids is not None and draft_token_ids.numel():
        raise ValueError("draft tokens supplied for a non-speculative PP contract")

    shape = (sampled_token_ids.shape[0], max_sample_len + max_draft_len)
    if out is None:
        packet = sampled_token_ids.new_empty(shape)
    else:
        if (
            out.shape != shape
            or out.dtype != torch.int64
            or out.device != sampled_token_ids.device
        ):
            raise ValueError("token packet output does not match the PP contract")
        packet = out
    packet.fill_(-1)
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
    """Sampled/draft transport delayed by the engine's in-flight batch depth."""

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
        feedback_depth: int | None = None,
        packed_feedback: bool = False,
    ) -> None:
        pp = get_pp_group()
        if feedback_depth is None:
            feedback_depth = pp.world_size
        if not 1 <= feedback_depth <= pp.world_size:
            raise ValueError(
                "PP feedback depth must be positive and no larger than "
                f"the physical pipeline size ({pp.world_size})."
            )
        self.feedback_depth = feedback_depth
        self.packed_feedback = packed_feedback
        self.is_last_rank = pp.is_last_rank
        self.last_rank = pp.last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.max_draft_len = num_speculative_steps
        self.packet_width = self.max_sample_len + self.max_draft_len
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)
        # A receive buffer stays live until its delayed feedback is consumed.
        # At most feedback_depth batches are in flight. Skipped receives only
        # increase the number of batch slots before a buffer is reused.
        self._buffers = [
            _FeedbackBuffer(
                torch.empty(
                    max_num_reqs * (self.packet_width + 1),
                    dtype=torch.int64,
                    device=device,
                ),
                torch.cuda.Event(),
            )
            for _ in range(self.feedback_depth if packed_feedback else 0)
        ]
        self._next_buffer = 0

        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * self.feedback_depth)
        )
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)
        self.broadcast_group = pp.make_sibling_device_group(
            group_desc="pp_sampled_draft_broadcast"
        )
        if self.packed_feedback:
            packet = self._buffers[0].packet
            logger.info(
                "PP packed feedback enabled: depth=%d max_requests=%d "
                "packet_bytes=%d; single broadcast",
                self.feedback_depth,
                self.max_num_reqs,
                packet.numel() * packet.element_size(),
            )

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def _acquire_buffer(self, num_reqs: int) -> _FeedbackBuffer:
        if not 0 <= num_reqs <= self.max_num_reqs:
            raise ValueError("request count exceeds the PP feedback buffer capacity")
        buffer = self._buffers[self._next_buffer]
        self._next_buffer = (self._next_buffer + 1) % self.feedback_depth
        return buffer

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor | None] | None:
        """Consume feedback from the configured number of batch slots earlier."""
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
            # This also waits for the main-stream consumers of a reused slot.
            # Discarded slots remain safe because receive/overwrite operations
            # are ordered on this same broadcast stream.
            self.broadcast_stream.wait_stream(self.main_stream)
            if self.packed_feedback:
                buffer = self._acquire_buffer(num_reqs)
                packet, token_packet, combined = buffer.views(
                    num_reqs, self.packet_width
                )
                torch.distributed.broadcast(
                    packet, src=self.last_rank, group=self.broadcast_group
                )
                event = self.broadcast_stream.record_event(buffer.event)
            else:
                token_packet = torch.empty(
                    num_reqs, self.packet_width, dtype=torch.int64, device=self.device
                )
                combined = torch.empty(
                    2, num_reqs, dtype=torch.int32, device=self.device
                )
                torch.distributed.broadcast(
                    token_packet, src=self.last_rank, group=self.broadcast_group
                )
                torch.distributed.broadcast(
                    combined, src=self.last_rank, group=self.broadcast_group
                )
                event = self.broadcast_stream.record_event()
                token_packet.record_stream(self.main_stream)
                combined.record_stream(self.main_stream)
            num_sampled, num_rejected = combined.unbind(dim=0)
            sampled_tokens = token_packet[:, : self.max_sample_len]
            draft_tokens = (
                token_packet[:, self.max_sample_len :] if self.max_draft_len else None
            )

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
        """Enqueue feedback from per-step, immutable sampler outputs.

        The runner passes newly allocated samples/counts and an advanced-index
        copy of request-owned drafts. record_stream protects their allocator
        lifetime; callers must not overwrite these inputs in place while this
        stream is still reading them.
        """
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        num_reqs = input_batch.num_reqs
        for counts in (num_sampled, num_rejected):
            if counts.dtype != torch.int32 or counts.shape != (num_reqs,):
                raise ValueError("sampled/rejected counts must be 1D int32 tensors")
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            if self.packed_feedback:
                buffer = self._acquire_buffer(num_reqs)
                packet, token_packet, combined = buffer.views(
                    num_reqs, self.packet_width
                )
                _pack_token_packet(
                    sampled_token_ids,
                    draft_token_ids,
                    self.max_sample_len,
                    self.max_draft_len,
                    out=token_packet,
                )
                torch.stack((num_sampled, num_rejected), dim=0, out=combined)
                torch.distributed.broadcast(
                    packet, src=self.last_rank, group=self.broadcast_group
                )
            else:
                token_packet = _pack_token_packet(
                    sampled_token_ids,
                    draft_token_ids,
                    self.max_sample_len,
                    self.max_draft_len,
                )
                combined = torch.stack((num_sampled, num_rejected), dim=0)
                torch.distributed.broadcast(
                    token_packet, src=self.last_rank, group=self.broadcast_group
                )
                torch.distributed.broadcast(
                    combined, src=self.last_rank, group=self.broadcast_group
                )
                token_packet.record_stream(self.broadcast_stream)
                combined.record_stream(self.broadcast_stream)
            for tensor in (
                sampled_token_ids,
                num_sampled,
                num_rejected,
            ):
                tensor.record_stream(self.broadcast_stream)
            if draft_token_ids is not None:
                draft_token_ids.record_stream(self.broadcast_stream)
