# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Setup-time plan qualification, whole-call tuning and exact-identity caching."""

import hashlib
import math
import os
import statistics
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .collective_planner import load_selection, save_selection, select_measured, solve
from .planned_collectives import PlannedChannel, load_native


def backend_identity():
    import subprocess

    import torch

    root = Path(__file__).parent
    return dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        hardware=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,driver_version,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=15,
        ),
        sources={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "collective_planner.py",
                "plan.py",
                "planner_probe.py",
                "collective_setup.py",
                "planned_collectives.py",
                "planned_codegen.py",
                "planned_reduce.cpp",
            )
        },
        arch=os.environ.get("TORCH_CUDA_ARCH_LIST", "native"),
    )


def measure_channel(channel, *, repeats=9, warmup=3):
    """One untimed ready call, then dependent complete calls, max rank per sample.

    Separate local graphs avoid cross-device event fork/join. Every replay
    changes exact inputs and checks the entire final result on every rank.
    Caller can supply a different layout-coordinated measurement callback.
    """
    import torch
    import torch.distributed as dist

    if not 3 <= repeats <= 15 or not 1 <= warmup <= 5:
        raise ValueError("Bounded timing counts required")
    rank = dist.get_rank(channel.group)
    world = dist.get_world_size(channel.group)
    if world == 1:
        return dict(correct=True, samples_us=[0.000001] * repeats, local=True)
    chain = max(1, min(8, int(26 / max(1, math.log2(world))) - 1))
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    # Allocate under the communicator's IPC allocator policy, then register
    # captured graph buffers on context exit before the first replay.
    with getattr(channel, "capture", nullcontext)():
        x = torch.empty(
            channel.capacity // 2, dtype=torch.float16, device=channel.device
        )
        with torch.cuda.graph(graph):
            current = channel.all_reduce(x)
            storage = [current]
            start.record()
            for _ in range(chain):
                current = channel.all_reduce(current)
                storage.append(current)
            end.record()
    samples: list[float] = []
    hashes = []
    for trial in range(warmup + repeats):
        # Positive integer fractions give an exact common answer on every tree;
        # rank-varying patterns detect missing or stale input contributions.
        values = [
            (
                torch.arange(channel.capacity // 2, dtype=torch.int32) % 3
                + r
                + 1
                + trial % 3
            ).half()
            / 16384
            for r in range(world)
        ]
        x.copy_(values[rank])
        for out in storage:
            out.fill_(float("nan"))
        torch.accelerator.synchronize(channel.device)
        dist.barrier(group=channel.group)
        graph.replay()
        torch.accelerator.synchronize(channel.device)
        # Validate every dependency. The existing backend propagates actual
        # rank outputs to allow legal rounding; generated ranks share an exact
        # reference and do not need additional CPU tensor collectives.
        expected = values
        passed = True
        for out in storage:
            by_node = dict(zip(channel.plan["members"], expected))
            cache = [
                by_node.get(node, torch.zeros_like(expected[0]))
                for node in channel.plan["devices"]
            ]
            for value in channel.plan["values"][len(cache) :]:
                left, right = value["children"]
                total = cache[left].float() + cache[right].float()
                cache.append(total if value["dtype"] == "fp32" else total.half())
            ref = cache[channel.plan["outputs"][0]].half()
            if not torch.isfinite(ref).all():
                raise RuntimeError("Timing oracle overflow")
            actual = out.cpu()
            passed &= channel.result_matches(actual, ref, expected)
            expected = channel.next_values(ref, actual)
        states = [None] * world
        dist.all_gather_object(states, passed, group=channel.group)
        if not all(states):
            raise RuntimeError("Measured dependent graph oracle failed")
        local = start.elapsed_time(end) * 1000 / chain
        times = [0.0] * world
        dist.all_gather_object(times, local, group=channel.group)
        if trial >= warmup:
            samples.append(max(times))
            hashes.append(
                hashlib.sha256(
                    current.cpu().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
            )
    return dict(
        correct=True,
        samples_us=samples,
        chain=chain,
        warmup=warmup,
        output_sha256=hashes,
        timing=(
            "max-rank local external-event elapsed / dependent call count; "
            "one untimed complete ready call; no baseline subtraction"
        ),
    )


class PreparedCollective:
    def __init__(self, channel, receipt, fallback=None):
        self.channel, self.receipt, self.fallback = channel, receipt, fallback

    def all_reduce(self, x):
        if self.channel is not None and self.channel.accepts(x):
            return self.channel.all_reduce(x)
        if self.fallback is not None:
            return self.fallback.all_reduce(x)
        raise ValueError("No qualified peer route for this operation signature")

    def close(self):
        if self.channel is not None:
            self.channel.close()


def prepare(
    group,
    device,
    topology=None,
    *,
    payload_bytes=5120,
    precision="fp16",
    helpers=(),
    budget=2048,
    shortlist=8,
    grids=None,
    cache=None,
    context=None,
    fallback=None,
    measure=None,
    verify_live=True,
):
    """All group members call before capture; returns an immutable measured route.

    Graphs never invoke this function or rediscover a topology. For concurrent
    layouts, context must describe those groups and measure can coordinate their
    real joint workload. Default tuning measures this group alone. Existing
    fallback is retained for disconnected graphs and unqualified signatures.
    """
    import torch
    import torch.distributed as dist

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Setup cannot run during CUDA graph capture")
    from .planner_probe import refresh, visible_devices

    if verify_live:
        topology = refresh(topology)
    if topology is None:
        raise ValueError("A topology is required for offline setup")
    if grids is None:
        grids = tuple(
            (t, min(32, max(1, (payload_bytes // 16 + t - 1) // t)))
            for t in (64, 128, 256)
        )
    if (
        not grids
        or len(grids) > 9
        or any(t not in (64, 128, 256) or not 1 <= b <= 32 for t, b in grids)
    ):
        raise ValueError("Bounded launch grid list required")
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    visible = visible_devices()
    members = [None] * world
    dist.all_gather_object(members, visible[device.index], group=group)
    eligibility = [None] * world
    dist.all_gather_object(
        eligibility, set(members) | set(helpers) <= set(visible), group=group
    )
    if not all(eligibility):
        return PreparedCollective(
            None,
            dict(
                status="fallback-required",
                reason=(
                    "Selected members/helpers must be visible "
                    "to every native participant"
                ),
            ),
            fallback,
        )
    results: list[Any] = [None] * world
    local = (
        solve(
            topology,
            members,
            helpers=helpers,
            payload_bytes=payload_bytes,
            precision=precision,
            budget=budget,
            shortlist=shortlist,
        )
        if rank == 0
        else None
    )
    dist.all_gather_object(results, local, group=group)
    result = results[0]
    if not result["candidates"]:
        return PreparedCollective(
            None, dict(status="fallback-required", planner=result), fallback
        )
    if world == 1:
        return PreparedCollective(
            PlannedChannel(group, device, result["candidates"][0]["plan"], topology),
            dict(status="local", planner_key=result["key"]),
            fallback,
        )
    extra = dict(concurrent_groups=[members])
    extra.update(context or {})
    configuration = dict(
        extra,
        backend=backend_identity(),
        grids=[list(v) for v in grids],
        visible=visible,
        device_names=[torch.cuda.get_device_name(visible.index(v)) for v in members],
    )
    loaded = load_selection(cache, result, configuration) if cache else None
    # Cache is read by rank zero only for the decision, then shared. Different
    # local filesystems must never cause different ranks to execute new routes.
    choices: list[Any] = [None] * world
    dist.all_gather_object(choices, loaded if rank == 0 else None, group=group)
    loaded = choices[0]
    if loaded:
        grid = loaded["receipt"].get("launch", {})
        if (grid.get("threads"), grid.get("blocks")) not in grids:
            loaded = None
    if loaded:
        plans = [
            entry["plan"]
            for entry in result["candidates"]
            if len(entry["plan"]["devices"]) == len(loaded["plan"]["devices"])
        ]
        candidate = next(
            i
            for i, p in enumerate(plans)
            if p["fingerprint"] == loaded["plan"]["fingerprint"]
        )
        if rank == 0:
            load_native(plans, topology, payload_bytes)
        dist.barrier(group=group)
        native = load_native(plans, topology, payload_bytes)
        channel = PlannedChannel(
            group,
            device,
            loaded["plan"],
            topology,
            threads=grid["threads"],
            blocks=grid["blocks"],
            native=native,
            candidate=candidate,
        )
        channel.qualify()
        return PreparedCollective(
            channel, dict(loaded["receipt"], cache_hit=True), fallback
        )
    # One compilation per node count; a relay plan can have a different number
    # of nodes. Static kernels retain independent plan-specific rank programs.
    batches: dict[int, list[dict[str, Any]]] = {}
    locations = {}
    for entry in result["candidates"]:
        p = entry["plan"]
        batch = batches.setdefault(len(p["devices"]), [])
        locations[p["fingerprint"]] = (len(p["devices"]), len(batch))
        batch.append(p)
    natives = {}
    for n, plans in sorted(batches.items()):
        if rank == 0:
            load_native(plans, topology, payload_bytes)
        dist.barrier(group=group)
        natives[n] = load_native(plans, topology, payload_bytes)
    channels = {}
    timings = {}
    measure = measure or measure_channel
    try:
        for entry in result["candidates"]:
            plan = entry["plan"]
            n, candidate = locations[plan["fingerprint"]]
            options = []
            for threads, blocks in grids:
                channel = PlannedChannel(
                    group,
                    device,
                    plan,
                    topology,
                    threads=threads,
                    blocks=blocks,
                    native=natives[n],
                    candidate=candidate,
                )
                qualification = channel.qualify()
                receipt = measure(channel)
                if receipt.get("correct") is not True:
                    channel.close()
                    continue
                options.append(
                    (
                        statistics.median(receipt["samples_us"]),
                        threads,
                        blocks,
                        channel,
                        dict(receipt, qualification=qualification),
                    )
                )
            if not options:
                continue
            winner = min(options, key=lambda v: (v[0], v[1], v[2]))
            for option in options:
                if option is not winner:
                    option[3].close()
            channels[plan["fingerprint"]] = winner[3]
            timings[plan["fingerprint"]] = dict(
                winner[4], threads=winner[1], blocks=winner[2]
            )
        selected = select_measured(
            result,
            lambda p: timings.get(p["fingerprint"], dict(correct=False)),
            context=configuration,
        )
        if selected["status"] != "measured":
            return PreparedCollective(None, selected, fallback)
        key = selected["winner"]
        channel = channels.pop(key)
        selected["launch"] = {k: timings[key][k] for k in ("threads", "blocks")}
        if cache and rank == 0:
            save_selection(cache, result, selected)
        dist.barrier(group=group)
        return PreparedCollective(channel, dict(selected, cache_hit=False), fallback)
    finally:
        for channel in channels.values():
            channel.close()
