# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-tensor torchrun worker for the real PP communication implementations.

Run through test_pp_nccl_lifecycle.py for a process-tree timeout. No model
weights, tokenizer, mocked CUDA stream, or mocked collective is involved.
"""

import argparse
import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.distributed.parallel_state import GroupCoordinator, Handle
from vllm.model_executor.models.pp_decode_transfer import StaticPPDecodeTransfer
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pp_utils import PPHandler, compute_need_sampled_mask
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

_Snapshot = tuple[str, torch.Tensor, torch.Tensor]
_MAX_REQS = 8
_SPEC_STEPS = 4
_CAPACITY = _MAX_REQS * (_SPEC_STEPS + 1)
_HIDDEN = 32


@dataclass
class _ExpectedFeedback:
    indices: npt.NDArray[np.int32]
    generations: npt.NDArray[np.int32]
    need_sampled: npt.NDArray[np.bool_]
    values: dict[str, torch.Tensor]


def _retire_busy_storage(
    shape: tuple[int, ...], dtype: torch.dtype, depth: int, device: torch.device
) -> tuple[set[int], torch.cuda.Event]:
    """Make cached blocks whose old writes are still on the allocation stream.

    The native allocator can immediately reuse these blocks on the same stream.
    A new receiving stream must acquire that allocation-stream dependency before
    overwriting them. The assertion at the call site proves reuse actually ran.
    """
    retired = [torch.empty(shape, dtype=dtype, device=device) for _ in range(depth)]
    pointers = {tensor.data_ptr() for tensor in retired}
    torch.cuda._sleep(200_000_000)
    for tensor in retired:
        tensor.fill_(-313)
    completed = torch.cuda.Event()
    completed.record()
    return pointers, completed


def _allocator_pressure(device: torch.device, num_reqs: int, width: int) -> None:
    # Match the sampler/count sizes rather than allocating a large model-sized
    # tensor. record_stream must keep in-flight sender sources out of this pool.
    for shape, dtype in (
        ((num_reqs, width), torch.int64),
        ((num_reqs, _SPEC_STEPS), torch.int64),
        ((num_reqs,), torch.int32),
        ((_CAPACITY, _HIDDEN), torch.float16),
    ):
        scratch = torch.empty(shape, dtype=dtype, device=device)
        scratch.fill_(-91)
        del scratch


def _check_snapshots(snapshots: list[_Snapshot]) -> None:
    # Synchronize only at the end of a case. Per-step .cpu()/assert_close would
    # serialize the streams and hide a ring-buffer overwrite or lifetime bug.
    torch.accelerator.synchronize()
    for label, actual, expected in snapshots:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, msg=label)


def _feedback_batch(step: int, device: torch.device) -> InputBatch:
    mode = step % 7
    indices = (
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0],
        [],
        [0, 2, 3, 7],
        [7, 2, 0, 4, 3],
        [2, 0, 1],
        [7, 0, 6, 1, 5, 2, 4, 3],
    )[mode]
    n = len(indices)
    computed = np.full(n, 8, dtype=np.int32)
    if mode == 3:
        computed[0] = 0  # Mixed unfinished prefill and decode.
    elif mode == 4:
        computed[-1] = 99  # A finishing row does not need future feedback.
    elif mode == 5:
        computed[:] = 0  # Entirely unfinished prefill: no collective.
    return cast(
        InputBatch,
        SimpleNamespace(
            num_reqs=n,
            idx_mapping_np=np.array(indices, dtype=np.int32),
            idx_mapping=torch.tensor(indices, dtype=torch.int32, device=device),
            num_computed_tokens_np=computed,
            num_scheduled_tokens=np.ones(n, dtype=np.int32),
            prefill_len_np=np.full(n, 8, dtype=np.int32),
            max_seq_len_np=np.full(n, 100, dtype=np.int32),
        ),
    )


def _feedback_case(
    pp: GroupCoordinator, device: torch.device, depth: int, packed: bool, steps: int
) -> dict[str, Any]:
    batches = [_feedback_batch(step, device) for step in range(steps)]
    torch.accelerator.synchronize()
    pp.barrier()
    old_pointers, busy_event = _retire_busy_storage(
        (_MAX_REQS * (2 * _SPEC_STEPS + 2),), torch.int64, depth, device
    )
    handler = PPHandler(_MAX_REQS, _SPEC_STEPS, device, depth, packed)
    buffer_pointers = {buffer.packet.data_ptr() for buffer in handler._buffers}
    reused = len(old_pointers & buffer_pointers)
    old_write_pending = not busy_event.query()
    if packed:
        assert reused, "allocator did not exercise first-use packet ownership"
        assert old_write_pending, "old packet write finished before the ownership test"
    generations = np.zeros(_MAX_REQS, dtype=np.int32)
    expected: deque[_ExpectedFeedback | None] = deque([None] * depth)
    snapshots: list[_Snapshot] = []
    counts = {"delivered": 0, "partial_discard": 0, "all_discard": 0}

    def consume() -> None:
        pending = expected.popleft()
        output = handler.get_prev_sampled_outputs()
        if pp.is_last_rank:
            assert output is None
            return
        if pending is None:
            assert output is None
            return
        keep = (
            generations[pending.indices] == pending.generations
        ) & pending.need_sampled
        if not keep.any():
            counts["all_discard"] += 1
            assert output is None
            return
        counts["delivered"] += 1
        counts["partial_discard"] += int(not keep.all())
        assert output is not None
        wanted = dict(pending.values)
        wanted["idx_mapping"] = torch.from_numpy(np.where(keep, pending.indices, -1))
        # The next receive can be queued before these consumers finish. It must
        # wait on the main stream before reusing a packed receive slot.
        torch.cuda._sleep(1_000_000)
        for name, value in wanted.items():
            actual = output[name]
            assert actual is not None
            snapshots.append((name, actual.clone(), value))

    for step, batch in enumerate(batches):
        # Cancel pending generations before consuming them, then immediately
        # reuse those same request indices in this/next batch.
        freed = range(_MAX_REQS) if step % 13 == 0 else (0, 2) if step % 5 == 0 else ()
        for index in freed:
            handler.on_req_idx_freed(index)
            generations[index] += 1
        consume()
        n = batch.num_reqs
        width = 1 + step % (_SPEC_STEPS + 1)
        need = compute_need_sampled_mask(batch)
        sampled_cpu = torch.arange(n * width, dtype=torch.int64).view(n, width)
        sampled_cpu += step * 100
        draft_cpu = torch.arange(n * _SPEC_STEPS, dtype=torch.int64).view(
            n, _SPEC_STEPS
        )
        draft_cpu += step * 1000
        accepted_cpu = torch.full((n,), width, dtype=torch.int32)
        rejected_cpu = torch.full((n,), _SPEC_STEPS + 1 - width, dtype=torch.int32)
        if need is None:
            expected.append(None)
        else:
            padded = torch.full((n, _SPEC_STEPS + 1), -1, dtype=torch.int64)
            padded[:, :width] = sampled_cpu
            expected.append(
                _ExpectedFeedback(
                    batch.idx_mapping_np.copy(),
                    generations[batch.idx_mapping_np].copy(),
                    need.copy(),
                    {
                        "sampled_tokens": padded,
                        "draft_tokens": draft_cpu,
                        "num_sampled": accepted_cpu,
                        "num_rejected": rejected_cpu,
                    },
                )
            )
        if pp.is_last_rank:
            sampled = torch.arange(n * width, dtype=torch.int64, device=device).view(
                n, width
            )
            sampled.add_(step * 100)
            draft = torch.arange(
                n * _SPEC_STEPS, dtype=torch.int64, device=device
            ).view(n, _SPEC_STEPS)
            draft.add_(step * 1000)
            accepted = torch.full((n,), width, dtype=torch.int32, device=device)
            rejected = torch.full(
                (n,), _SPEC_STEPS + 1 - width, dtype=torch.int32, device=device
            )
            handler.broadcast(sampled, accepted, rejected, batch, draft)
            del sampled, draft, accepted, rejected
        else:
            assert handler.receive(batch) == (need is not None and bool(need.all()))
        _allocator_pressure(device, n, width)

    for _ in range(depth):
        consume()
        expected.append(None)
    _check_snapshots(snapshots)
    if not pp.is_last_rank:
        assert counts["delivered"] > depth
        assert counts["partial_discard"] > 0
        assert counts["all_discard"] > 0
        assert all(slot is None for slot in handler.queue)
    assert {buffer.packet.data_ptr() for buffer in handler._buffers} == buffer_pointers
    pp.barrier()
    dist.destroy_process_group(handler.broadcast_group)
    return {
        "case": "feedback",
        "depth": depth,
        "packed": packed,
        "steps": steps,
        "reused_allocation_blocks": reused,
        "first_use_pending_write": old_write_pending,
        **counts,
    }


def _static_config(world_size: int, depth: int) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={"pp_decode_static_transfer": True},
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=world_size,
            tensor_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
            ubatch_size=0,
        ),
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=4),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            hf_text_config=SimpleNamespace(
                model_type="qwen4_exp_text", hidden_size=8, hc_count=4
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=_MAX_REQS),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False),
            cudagraph_capture_sizes=[5, _CAPACITY],
        ),
        max_concurrent_batches=depth,
    )


def _hidden_schema(
    batch_size: int, dtype: torch.dtype, device: torch.device
) -> IntermediateTensors:
    return IntermediateTensors(
        {
            "hidden_states": torch.empty(
                (batch_size, _HIDDEN), dtype=dtype, device=device
            )
        }
    )


def _decode_output(step: int) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    n = 1 + step % _MAX_REQS
    ids = [f"r{i}" for i in range(n)]
    output.num_scheduled_tokens = dict.fromkeys(ids, _SPEC_STEPS + 1)
    output.total_num_scheduled_tokens = n * (_SPEC_STEPS + 1)
    output.scheduled_spec_decode_tokens = dict.fromkeys(ids, [-1] * _SPEC_STEPS)
    cached = output.scheduled_cached_reqs
    cached.req_ids = ids
    cached.num_output_tokens = [1] * n
    if step % 4 == 3:
        # A partial tail takes the ordinary metadata-bearing protocol.
        output.num_scheduled_tokens[ids[-1]] = 2
        output.total_num_scheduled_tokens -= 3
        output.scheduled_spec_decode_tokens[ids[-1]] = [-1]
    return output


def _static_case(
    pp: GroupCoordinator, device: torch.device, depth: int, steps: int
) -> dict[str, Any]:
    persistent_input = _hidden_schema(_CAPACITY, torch.float16, device)
    outgoing = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.float16, device=device)
    runner = SimpleNamespace(
        device=device,
        max_num_tokens=_CAPACITY,
        model=SimpleNamespace(make_empty_intermediate_tensors=_hidden_schema),
        intermediate_tensors=persistent_input,
    )
    torch.accelerator.synchronize()
    pp.barrier()
    old_pointers, busy_event = _retire_busy_storage(
        (_CAPACITY, _HIDDEN), torch.float16, depth, device
    )
    transfer = StaticPPDecodeTransfer.create(
        _static_config(pp.world_size, depth), runner, pp, use_v2=True
    )
    assert transfer is not None
    slot_pointers = {
        slot.tensors["hidden_states"].data_ptr() for slot in transfer.slots
    }
    reused = len(old_pointers & slot_pointers)
    old_write_pending = not busy_event.query()
    if not pp.is_first_rank:
        assert reused, "allocator did not exercise first-use static receive ownership"
        assert old_write_pending, "old hidden write finished before the ownership test"
    snapshots: list[_Snapshot] = []
    pending_send: list[Handle] = []
    for step in range(steps):
        # Same output-buffer ownership rule as Worker.execute_model.
        for handle in pending_send:
            handle.wait()
        pending_send = []
        output = _decode_output(step)
        static = transfer.matches(output)
        assert static == (step % 4 != 3)
        n = output.total_num_scheduled_tokens
        if pp.is_first_rank:
            outgoing[:n].fill_(step * 10)
        else:
            if static:
                tensors, handles = transfer.receive(n)
                postprocess = []
            else:
                tensors, handles, postprocess = pp.irecv_tensor_dict()
                assert tensors is not None
            incoming = AsyncIntermediateTensors(tensors, handles, postprocess)
            ready = incoming.tensors["hidden_states"]
            # Delayed consumers plus cycling slots exercise the consumed event.
            torch.cuda._sleep(1_000_000)
            persistent_input.tensors["hidden_states"][:n].copy_(ready[:n])
            snapshots.append(
                (
                    f"hidden rank={pp.rank} step={step}",
                    persistent_input.tensors["hidden_states"][:n].clone(),
                    torch.full(
                        (n, _HIDDEN), step * 10 + pp.rank - 1, dtype=torch.float16
                    ),
                )
            )
            if static:
                snapshots.append(
                    (
                        f"padding rank={pp.rank} step={step}",
                        ready[n:].clone(),
                        torch.zeros((_CAPACITY - n, _HIDDEN), dtype=torch.float16),
                    )
                )
                transfer.mark_consumed()
            outgoing[:n].copy_(persistent_input.tensors["hidden_states"][:n]).add_(1)
            del ready, incoming, tensors
        if not pp.is_last_rank:
            if static:
                # Different ranks expose different local graph padding, but
                # the wire length remains the common scheduler token count.
                multiple = 8 if pp.rank % 2 == 0 else 16
                padded = min(_CAPACITY, ((n + multiple - 1) // multiple) * multiple)
                outgoing[n:padded].fill_(-17)
                pending_send = transfer.send({"hidden_states": outgoing[:padded]}, n)
            else:
                pending_send = pp.isend_tensor_dict({"hidden_states": outgoing[:n]})
        _allocator_pressure(device, 8, 5)
    for handle in pending_send:
        handle.wait()
    _check_snapshots(snapshots)
    assert transfer.static_steps == steps - steps // 4
    assert transfer.dynamic_steps == steps // 4
    assert {slot.tensors["hidden_states"].data_ptr() for slot in transfer.slots} == (
        slot_pointers
    )
    pp.barrier()
    return {
        "case": "static_dynamic",
        "depth": depth,
        "static_steps": transfer.static_steps,
        "dynamic_steps": transfer.dynamic_steps,
        "reused_allocation_blocks": reused,
        "first_use_pending_write": old_write_pending,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--collective-timeout", type=int, default=45)
    args = parser.parse_args()
    if args.steps < 40 or args.steps % 4:
        parser.error("--steps must be a multiple of four and at least 40")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size >= 2
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    assert torch.cuda.get_allocator_backend() == "native", (
        "allocation-stream reuse regression requires the native CUDA allocator"
    )
    timeout = timedelta(seconds=args.collective_timeout)
    dist.init_process_group("nccl", timeout=timeout)
    config = VllmConfig()
    config.parallel_config.distributed_timeout_seconds = args.collective_timeout
    config.parallel_config.cpu_distributed_timeout_seconds = args.collective_timeout
    with set_current_vllm_config(config):
        pp = GroupCoordinator(
            [list(range(world_size))],
            local_rank=local_rank,
            torch_distributed_backend="nccl",
            use_device_communicator=False,
            group_name="pp_lifecycle_test",
        )
    assert parallel_state._PP is None
    parallel_state._PP = pp
    results = []
    for depth in (2, 4):
        if depth > world_size:
            continue
        for packed in (False, True):
            results.append(_feedback_case(pp, device, depth, packed, args.steps))
        results.append(_static_case(pp, device, depth, args.steps))
    pp.barrier()
    # Serialize only the final report, so rank output cannot interleave in the
    # launcher's shared log. No barrier is inserted into the exercised loops.
    for reporting_rank in range(world_size):
        if rank == reporting_rank:
            print(
                "PP_NCCL_LIFECYCLE_OK "
                + json.dumps(
                    {"rank": rank, "world_size": world_size, "cases": results}
                ),
                flush=True,
            )
        pp.barrier()
    parallel_state._PP = None
    pp.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
