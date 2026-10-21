# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in metadata-free V2 Qwen MTP decode transport.

Every rank derives the wire length from the same SchedulerOutput. Startup
negotiates the model-owned tensor schema once; prefill and nonuniform tails
retain the dynamic protocol. Wire tensors contain only real token rows, so
different local CUDA graph padding cannot change a peer's receive size.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator, Handle

logger = init_logger(__name__)


def is_complete_mtp_decode(output: Any, query_len: int, capacity: int) -> bool:
    counts = output.num_scheduled_tokens
    if (
        not counts
        or output.scheduled_new_reqs
        or output.scheduled_encoder_inputs
        or output.scheduled_cached_reqs.resumed_req_ids
        or output.total_num_scheduled_tokens != query_len * len(counts)
        or output.total_num_scheduled_tokens > capacity
        or any(count != query_len for count in counts.values())
    ):
        return False
    cached = output.scheduled_cached_reqs
    output_counts = dict(zip(cached.req_ids, cached.num_output_tokens))
    return all(
        output_counts.get(req_id, 0) > 0
        and len(output.scheduled_spec_decode_tokens.get(req_id, ())) == query_len - 1
        for req_id in counts
    )


@dataclass
class _ReceiveSlot:
    tensors: dict[str, torch.Tensor]
    consumed: torch.cuda.Event | None = None


class StaticPPDecodeTransfer:
    """D receive slots protected until their model-input copy has completed."""

    pp: "GroupCoordinator"
    device: torch.device
    shape: tuple[int, ...]
    capacity: int
    query_len: int
    depth: int
    slots: list[_ReceiveSlot]
    slot_index: int
    active_slot: _ReceiveSlot | None
    static_steps: int
    dynamic_steps: int
    receive_stream: torch.cuda.Stream | None

    @classmethod
    def create(
        cls, config: Any, runner: Any, pp: "GroupCoordinator", *, use_v2: bool
    ) -> "StaticPPDecodeTransfer | None":
        additional = getattr(config, "additional_config", None)
        enabled = (
            additional.get("pp_decode_static_transfer", False)
            if isinstance(additional, dict)
            else False
        )
        if enabled is False:
            return None
        if enabled is not True:
            raise ValueError("pp_decode_static_transfer must be a boolean")
        parallel = config.parallel_config
        spec = config.speculative_config
        text = config.model_config.hf_text_config
        # Only common configuration participates in admission. Never use a
        # local GPU architecture or rank-local model state to choose a protocol.
        if (
            not use_v2
            or parallel.pipeline_parallel_size <= 1
            or parallel.tensor_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.enable_dbo
            or parallel.ubatch_size > 1
            or config.compilation_config.pass_config.enable_sp
            or config.model_config.dtype != torch.float16
            or text.model_type not in ("qwen4_exp", "qwen4_exp_text")
            or spec is None
            or spec.method != "mtp"
            or not 1 <= spec.num_speculative_tokens <= 4
        ):
            raise ValueError(
                "pp_decode_static_transfer requires V2 Qwen4Exp FP16, PP>1, "
                "TP=DP=1, MTP1..4, and no SP/DBO/microbatching"
            )
        query_len = spec.num_speculative_tokens + 1
        capacity = min(
            runner.max_num_tokens,
            max(
                config.scheduler_config.max_num_seqs * query_len,
                max(config.compilation_config.cudagraph_capture_sizes or [0]),
            ),
        )
        signature = None
        error = None
        try:
            schema = runner.model.make_empty_intermediate_tensors(
                batch_size=1, dtype=torch.float16, device=torch.device("meta")
            ).tensors
            if tuple(schema) != ("hidden_states",):
                raise ValueError("expected one model-owned hidden_states tensor")
            hidden = schema["hidden_states"]
            shape = tuple(hidden.shape[1:])
            allowed_shapes = (
                (text.hidden_size * text.hc_count,),
                (text.hc_count, text.hidden_size),
            )
            if (
                hidden.shape[0] != 1
                or shape not in allowed_shapes
                or hidden.dtype != torch.float16
                or not hidden.is_contiguous()
                or runner.device.type != "cuda"
                or capacity < query_len
            ):
                raise ValueError("unsupported model-owned decode tensor schema")
            if not pp.is_first_rank:
                buffers = runner.intermediate_tensors
                if buffers is None or tuple(buffers.tensors) != tuple(schema):
                    raise ValueError("missing persistent model input buffer")
                local = buffers.tensors["hidden_states"]
                if (
                    local.shape[0] < capacity
                    or tuple(local.shape[1:]) != shape
                    or local.dtype != hidden.dtype
                    or local.device != runner.device
                    or not local.is_contiguous()
                ):
                    raise ValueError("persistent model input violates the schema")
            signature = (
                shape,
                str(hidden.dtype),
                capacity,
                query_len,
                config.max_concurrent_batches,
            )
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        # Rank-local validation errors participate in the same collective, then
        # every rank fails together. No peer can silently choose dynamic while
        # another has already selected the metadata-free receive protocol.
        reports: list[Any] = [None] * pp.world_size
        torch.distributed.all_gather_object(
            reports, (signature, error), group=pp.cpu_group
        )
        if any(report[1] is not None for report in reports) or any(
            report[0] != signature for report in reports
        ):
            raise RuntimeError(
                f"PP decode transport schema agreement failed: {reports}"
            )
        assert signature is not None
        result = cls.__new__(cls)
        result.pp = pp
        result.device = runner.device
        result.shape = signature[0]
        result.capacity = capacity
        result.query_len = query_len
        result.depth = config.max_concurrent_batches
        result.slots = []
        result.slot_index = 0
        result.active_slot = None
        result.static_steps = 0
        result.dynamic_steps = 0
        result.receive_stream = None
        if not pp.is_first_rank:
            result.receive_stream = torch.cuda.Stream(runner.device)
            result.slots = [
                _ReceiveSlot(
                    {
                        "hidden_states": torch.empty(
                            (capacity, *result.shape),
                            dtype=torch.float16,
                            device=runner.device,
                        )
                    }
                )
                for _ in range(result.depth)
            ]
            # CUDA's caching allocator may recycle storage whose previous use
            # is still queued on its allocation stream. Establish ownership
            # before the first receive, even though torch.empty has no payload.
            result.receive_stream.wait_stream(torch.cuda.current_stream(runner.device))
        logger.info(
            "PP decode static transfer ready: shape=%s capacity=%d slots=%d "
            "query_len=%d; prefill/nonuniform decode use dynamic transfer",
            result.shape,
            capacity,
            result.depth,
            query_len,
        )
        return result

    def matches(self, output: Any) -> bool:
        hit = is_complete_mtp_decode(output, self.query_len, self.capacity)
        if hit:
            self.static_steps += 1
            if self.static_steps == 1:
                logger.info(
                    "PP decode static transfer first hit: requests=%d tokens=%d",
                    len(output.num_scheduled_tokens),
                    output.total_num_scheduled_tokens,
                )
        elif output.total_num_scheduled_tokens:
            self.dynamic_steps += 1
        return hit

    def receive(
        self, num_tokens: int
    ) -> tuple[dict[str, torch.Tensor], list["Handle"]]:
        if self.active_slot is not None:
            raise RuntimeError("previous PP decode receive has not been consumed")
        slot = self.slots[self.slot_index]
        self.slot_index = (self.slot_index + 1) % self.depth
        self.active_slot = slot
        assert self.receive_stream is not None
        with torch.cuda.stream(self.receive_stream):
            if slot.consumed is not None:
                self.receive_stream.wait_event(slot.consumed)
            # Padding is local and deterministic. Do not send graph-padding
            # rows: capture sizes can differ between heterogeneous PP stages.
            slot.tensors["hidden_states"][num_tokens:].zero_()
            wire = {key: tensor[:num_tokens] for key, tensor in slot.tensors.items()}
            handles = self.pp.irecv_tensor_dict_static(wire)
        return slot.tensors, handles

    def mark_consumed(self) -> None:
        if self.active_slot is None:
            return
        # AsyncIntermediateTensors.wait_for_comm establishes NCCL -> current
        # stream ordering before the runner copies this slot to graph inputs.
        # Reuse waits on this event in the receive stream (no CPU synchronize).
        self.active_slot.consumed = torch.cuda.current_stream(
            self.device
        ).record_event()
        self.active_slot = None

    def send(self, tensors: dict[str, torch.Tensor], num_tokens: int) -> list["Handle"]:
        if tuple(tensors) != ("hidden_states",):
            raise RuntimeError("PP decode output keys violate negotiated schema")
        hidden = tensors["hidden_states"]
        if (
            tuple(hidden.shape[1:]) != self.shape
            or not num_tokens <= hidden.shape[0] <= self.capacity
            or hidden.dtype != torch.float16
            or hidden.device != self.device
            or not hidden.is_contiguous()
        ):
            raise RuntimeError("PP decode output tensor violates negotiated schema")
        # Worker waits for its previous send before reusing graph output memory.
        return self.pp.isend_tensor_dict_static({"hidden_states": hidden[:num_tokens]})
