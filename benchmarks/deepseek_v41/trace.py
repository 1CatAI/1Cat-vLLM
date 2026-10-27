# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in qualification hooks; never installed by the serving model.

CUDA events measure each rank's stream intervals, including dependency waits.
Nested expert/Engram intervals are part of target time, not additive stages.
No event is synchronized until the caller drains a completed request.
"""

import time
from collections import defaultdict

import torch


def install(worker):
    runner = worker.model_runner
    if hasattr(worker, "_v41_trace"):
        raise RuntimeError("V4.1 qualification hooks are already installed")
    trace = worker._v41_trace = dict(
        events=[], commits=[], target_forwards=[], sampling=[]
    )
    torch.accelerator.synchronize(worker.device)

    def wrap(owner, name, stage):
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            if torch.cuda.is_current_stream_capturing():
                # Lazy draft/context/head graphs can capture this hook.
                # Internal events express graph dependencies and do not
                # have standalone timestamps for elapsed_time on drain.
                # The enclosing graph call is timed outside capture.
                return original(*args, **kwargs)
            if name == "_model_forward":
                from vllm.forward_context import get_forward_context

                ids = kwargs.get("input_ids")
                if ids is None:
                    ids = args[0]
                trace["target_forwards"].append(
                    dict(
                        rows=ids.numel(),
                        mode=get_forward_context().cudagraph_runtime_mode.name,
                    )
                )
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            cpu_start = time.perf_counter()
            result = original(*args, **kwargs)
            if name == "_sample":
                metadata = kwargs.get("spec_decode_metadata")
                if metadata is None and len(args) >= 3:
                    metadata = args[2]
                if metadata is not None:
                    # State commit clips this same tensor in-place for EOS
                    # and the output budget. Preserve the verifier's actual
                    # result before clipping so final request tails do not
                    # masquerade as draft rejection.
                    remember_sampling(trace, metadata.num_draft_tokens, result)
            cpu_ms = (time.perf_counter() - cpu_start) * 1000
            end.record()
            trace["events"].append((stage, begin, end, cpu_ms))
            return result

        setattr(owner, name, measured)

    # Instrument above CUDAGraphWrapper: the model's Python forward is not
    # invoked by replay. The same boundary also covers ordinary forwards.
    wrap(runner, "_model_forward", "target")
    wrap(runner.model, "compute_logits", "target_head")
    wrap(runner, "_sample", "target_sampling")
    target = runner.model.model
    cache = target.layers[0].ffn.experts.quant_method.expert_cache
    cache.runtime.statistics.zero_()
    cache.tm.cold_transcodes.zero_()
    target.engram_io.runtime.statistics.zero_()
    wrap(cache, "resolve", "expert_transfer_and_group_wait")
    wrap(cache.tm, "prepare", "prefill_expert_transfer_and_transcode")
    wrap(target.engram_io.runtime, "lookup", "engram_wait_and_dequant")
    original_commit = runner.model.commit_speculative_state

    def commit(counts):
        original_commit(counts)
        trace["commits"].append(
            (
                counts.clone(),
                target.pending_position.clone(),
                target.pending_length.clone(),
                target.pending_prefill.clone(),
                target.hash_state.history.clone(),
                target.hash_state.metadata.clone(),
                [
                    target.layers[i].attn.compressor.state.position.clone()
                    for i in (2, 8, 14)
                ],
            )
        )

    runner.model.commit_speculative_state = commit
    wrap(runner.model, "commit_speculative_state", "commit")
    drafter = getattr(runner, "drafter", None)
    if drafter is not None:
        wrap(drafter.model, "forward", "draft")
        wrap(drafter.model, "compute_logits", "draft_head")
        wrap(drafter.model, "combine_hidden_states", "draft_main_projection")
        wrap(drafter.model, "precompute_and_store_context_kv", "draft_context")
        wrap(drafter, "_sample_draft_tokens", "draft_markov_sampling")
    torch.accelerator.synchronize(worker.device)


def drain(worker):
    from vllm.distributed import get_tp_group

    torch.accelerator.synchronize(worker.device)
    trace = worker._v41_trace
    stages = defaultdict(lambda: dict(calls=0, gpu_ms=0.0, cpu_enqueue_ms=0.0))
    for stage, begin, end, cpu_ms in trace["events"]:
        record = stages[stage]
        record["calls"] += 1
        record["gpu_ms"] += begin.elapsed_time(end)
        record["cpu_enqueue_ms"] += cpu_ms
    commits = []
    for counts, start, length, prefill, history, metadata, c2 in trace["commits"]:
        commits.append(
            dict(
                accepted_inputs=int(counts.item()),
                start=int(start.item()),
                rows=int(length.item()),
                prefill=bool(prefill.item()),
                hash_history=history.tolist(),
                hash_metadata=metadata.tolist(),
                c2_positions=[int(t.item()) for t in c2],
            )
        )
    trace["events"].clear()
    trace["commits"].clear()
    sampling = []
    for proposed, tensor in trace["sampling"]:
        for count, ids in zip(proposed, tensor.tolist(), strict=True):
            prefix = next((i for i, value in enumerate(ids) if value < 0), len(ids))
            sampling.append(
                dict(
                    proposed=count,
                    accepted_drafts=max(0, min(count, prefix - 1)),
                    raw_sampled_token_ids=ids,
                )
            )
    trace["sampling"].clear()
    forwards = trace["target_forwards"][:]
    trace["target_forwards"].clear()
    drafter = getattr(worker.model_runner, "drafter", None)
    graph = getattr(getattr(drafter, "model", None), "query_graph", None)
    greedy = getattr(getattr(drafter, "model", None), "greedy_graph", None)
    contexts = getattr(getattr(drafter, "model", None), "context_graphs", None)
    communicator = get_tp_group().device_communicator
    custom = getattr(communicator, "ca_comm", None)
    return dict(
        rank=worker.local_rank,
        stages=dict(stages),
        commits=commits,
        sampling_before_stop_clipping=sampling,
        custom_allreduce_enabled=bool(
            communicator.use_custom_allreduce
            and custom is not None
            and not custom.disabled
        ),
        target_forwards=forwards,
        draft_graph=(
            dict(captures=graph.captures, replays=graph.replays)
            if graph is not None
            else None
        ),
        greedy_graph=(
            dict(captures=greedy.captures, replays=greedy.replays)
            if greedy is not None
            else None
        ),
        context_graphs=(
            dict(
                captures=contexts.captures,
                replays=contexts.replays,
                rows=sorted(contexts.entries),
            )
            if contexts is not None
            else None
        ),
    )


def remember_sampling(trace, proposed, result):
    trace["sampling"].append((list(proposed), result.sampled_token_ids.clone()))


def set_draft_cap(worker, cap):
    runner = worker.model_runner
    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        if cap is not None:
            raise ValueError("A draft cap requires DSpark")
        return
    if cap is not None and cap not in range(6):
        raise ValueError("V4.1 draft cap must be in 0..5")
    enabled = cap is not None and cap < 5
    # The runner normally initializes these from the launch configuration.
    # Changing only the proposer leaves its returned lengths unused, so a
    # claimed M1 comparison would still verify the complete M6 proposal.
    runner.dspark_confidence_scheduling = enabled
    if enabled and runner._dspark_verification_lengths_cpu is None:
        runner._dspark_verification_lengths_cpu = torch.empty(
            runner.max_num_reqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=runner.pin_memory,
        )
    drafter._max_verification_tokens = 5 if cap is None else cap
    drafter.confidence_scheduling_enabled = enabled


def force_draft(worker, continuation, reject_at):
    """Keep the real draft forward, replace only greedy proposals for testing.

    A previously recorded M1 continuation supplies a known accepted prefix.
    The deliberate mismatch at reject_at exercises all six selectors against
    the actual target M6 and runner. This is not a quality or speed workload.
    """
    drafter = worker.model_runner.drafter
    if not hasattr(drafter, "_v41_original_sample"):
        drafter._v41_original_sample = drafter._sample_draft_tokens
    original = drafter._v41_original_sample
    if continuation is None:
        drafter._sample_draft_tokens = original
        return
    if reject_at not in range(6) or len(continuation) < 6:
        raise ValueError("Forced draft requires six continuation tokens and 0..5")
    values = torch.tensor(continuation[1:6], device=worker.device, dtype=torch.int64)
    if reject_at < 5:
        values[reject_at] = (values[reject_at] + 1) % 129280

    def sample(*args, **kwargs):
        # Execute native Markov sampling too, then override its proposal IDs.
        original(*args, **kwargs)
        return values.clone(), None

    drafter._sample_draft_tokens = sample
