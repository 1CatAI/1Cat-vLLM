# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare the measured generated route against an existing transport backend."""

import copy
import hashlib
import inspect
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

from .collective_planner import fingerprint
from .collective_setup import (
    PreparedCollective,
    backend_identity,
    measure_channel,
    prepare,
)
from .planned_collectives import PlannedChannel


def choose(samples, margin=0.01):
    """Prefer the existing backend when the measured difference is within margin."""
    if not 0 <= margin < 0.5 or not {"native", "existing"} <= set(samples):
        raise ValueError("Two comparable backends required")
    for values in samples.values():
        if len(values) < 3 or any(
            isinstance(v, bool)
            or not isinstance(v, (float, int))
            or not math.isfinite(v)
            or v <= 0
            for v in values
        ):
            raise ValueError("Invalid latency samples")
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return (
        "native"
        if medians["native"] < medians["existing"] * (1 - margin)
        else "existing"
    )


class Existing(PlannedChannel):
    """Out-of-place backend facade for independent AST qualification and timing."""

    def __init__(self, backend, group, device, plan):
        import torch.distributed as dist

        self.backend, self.group, self.device = backend, group, device
        self.plan = copy.deepcopy(plan)
        self.capacity = plan["payload_bytes"]
        self.native = True
        self.rank = dist.get_rank(group)
        # Existing FP32-accumulating implementations are checked against FP32
        # sums, not falsely required to match a new FP16 expression bitwise.
        for value in self.plan["values"]:
            if value["children"] is not None:
                value["dtype"] = "fp32"
        self.plan["precision"] = "fp32"

    def accepts(self, x):
        import torch

        return (
            x.device == self.device
            and x.dtype == torch.float16
            and x.is_contiguous()
            and x.nbytes == self.capacity
        )

    def all_reduce(self, x):
        return self.backend.all_reduce(x)

    def qualify(self, repeats=3):
        return PlannedChannel.qualify(self, repeats)

    def close(self):
        pass  # The caller owns the existing backend.


class BoundBackend:
    """Snapshot the old method before installing a new communicator wrapper."""

    def __init__(self, backend):
        self.function = backend.all_reduce

    def all_reduce(self, x):
        return self.function(x)


def prepare_best(
    group,
    device,
    topology=None,
    *,
    existing=None,
    cache=None,
    margin=0.01,
    measure=None,
    context=None,
    **options,
):
    """Tune generated candidates and retain the existing backend if it wins.

    Existing backend must have out-of-place all_reduce(Tensor)->Tensor, like
    1Cat's channels and vLLM PyNccl. Caller owns its lifetime. The default
    measurement is per-group; a layout coordinator can provide measure(channel).
    """
    if existing is None:
        return prepare(
            group,
            device,
            topology,
            cache=cache,
            context=context,
            measure=measure,
            **options,
        )
    import torch.distributed as dist

    try:
        source = inspect.getsourcefile(type(existing))
    except TypeError:
        source = None
    native = getattr(getattr(existing, "extension", None), "__file__", None)
    external = dict(
        type=type(existing).__module__ + "." + type(existing).__qualname__,
        source=hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if source and Path(source).is_file()
        else None,
        native=hashlib.sha256(Path(native).read_bytes()).hexdigest()
        if native and Path(native).is_file()
        else None,
    )
    # vLLM's communicator delegates to native custom AR / NCCL wrappers.
    # A rebuilt native library must invalidate the comparison receipt too.
    import sys

    runtime = sys.modules.get("vllm._C")
    runtime_file = getattr(runtime, "__file__", None)
    external["vllm_native"] = (
        hashlib.sha256(Path(runtime_file).read_bytes()).hexdigest()
        if runtime_file and Path(runtime_file).is_file()
        else None
    )
    existing = BoundBackend(existing)
    extra = dict(
        context or {},
        existing_backend=external,
        tuner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        margin=margin,
    )
    # Native selection's own exact topology/shape/group/ABI cache remains the
    # authority; this second receipt decides between it and the old backend.
    native_cache = Path(str(cache) + ".native") if cache else None
    prepared = prepare(
        group,
        device,
        topology,
        cache=native_cache,
        context=extra,
        measure=measure,
        **options,
    )
    if prepared.channel is None:
        return PreparedCollective(None, prepared.receipt, existing)
    if len(prepared.channel.plan["members"]) == 1:
        return prepared
    adapter = Existing(existing, group, device, prepared.channel.plan)
    identity = fingerprint(
        dict(
            native_receipt=prepared.receipt["key"],
            backend=backend_identity(),
            context=extra,
        )
    )
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    cached = None
    if cache and rank == 0:
        try:
            data = json.loads(Path(cache).read_text())
            if (
                data.get("identity") == identity
                and data.get("winner")
                in (
                    "native",
                    "existing",
                )
                and (
                    choose(
                        {k: v["samples_us"] for k, v in data["measurements"].items()},
                        margin,
                    )
                    == data["winner"]
                )
            ):
                cached = data
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
    receipts = [None] * world
    dist.all_gather_object(receipts, cached, group=group)
    cached = receipts[0]
    if cached:
        if cached["winner"] == "existing":
            adapter.qualify()
            prepared.close()
            return PreparedCollective(adapter, dict(cached, cache_hit=True), existing)
        return PreparedCollective(
            prepared.channel, dict(cached, cache_hit=True), existing
        )
    adapter.qualify()
    measure = measure or measure_channel
    # Both are remeasured together with the same boundary and inputs, rather
    # than comparing a warm baseline with a historical native measurement.
    measurements = dict(native=measure(prepared.channel), existing=measure(adapter))
    if any(r.get("correct") is not True for r in measurements.values()):
        prepared.close()
        raise RuntimeError("Backend comparison failed qualification")
    winner = choose({k: r["samples_us"] for k, r in measurements.items()}, margin)
    receipt = dict(
        status="measured",
        identity=identity,
        winner=winner,
        measurements=measurements,
        context=extra,
        cache_hit=False,
        native_selection=prepared.receipt,
        margin=margin,
    )
    if cache and rank == 0:
        path = Path(cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", delete=False, mode="w"
        ) as file:
            temporary = Path(file.name)
            json.dump(receipt, file, allow_nan=False)
            file.write("\n")
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    dist.barrier(group=group)
    if winner == "existing":
        prepared.close()
        return PreparedCollective(adapter, receipt, existing)
    return PreparedCollective(prepared.channel, receipt, existing)
