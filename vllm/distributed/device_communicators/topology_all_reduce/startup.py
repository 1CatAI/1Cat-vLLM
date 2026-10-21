# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collective setup before first model compilation and CUDA graph capture."""

import hashlib
import json
import socket
import time
from pathlib import Path
from typing import Any


def payloads(model_config, speculative_config=None):
    """Derive FP16 single-token and single-sequence verification/draft widths."""
    widths = {2 * model_config.get_hidden_size()}
    if speculative_config is not None:
        rows = speculative_config.num_speculative_tokens + 1
        widths.add(2 * model_config.get_hidden_size() * rows)
        draft = getattr(speculative_config, "draft_model_config", None)
        if draft is not None:
            widths.add(2 * draft.get_hidden_size())
            widths.add(2 * draft.get_hidden_size() * rows)
    if any(w <= 0 or w % 16 for w in widths):
        raise ValueError("Unsupported activation width")
    return sorted(widths)


def install(worker):
    """All TP ranks call once; freeze qualified dispatch before model capture.

    Only the current single-node TP group is measured. Other operation signatures
    keep their existing backend. Do not call this from an already captured graph.
    """
    import torch
    import torch.distributed as dist

    from vllm.distributed import get_tp_group
    from vllm.logger import init_logger

    from .collective_tuner import BoundBackend, prepare_best

    logger = init_logger(__name__)
    tp: Any = get_tp_group()
    comm: Any = tp.device_communicator
    if tp.world_size == 1 or comm is None:
        return {}
    hosts = [None] * tp.world_size
    dist.all_gather_object(hosts, socket.gethostname(), group=tp.cpu_group)
    if len(set(hosts)) != 1:
        logger.warning("Topology all-reduce requires a single-node TP group; skipped")
        return {}
    config = worker.vllm_config
    # Derive exactly the same widths on every rank, even with heterogeneous
    # draft/target hidden dimensions. No model names or physical GPU IDs here.
    widths = payloads(config.model_config, config.speculative_config)
    previous = BoundBackend(comm)
    options = config.kernel_config.topology_all_reduce
    precision = options.precision
    if precision not in ("fp16", "fp32"):
        raise ValueError("topology_all_reduce.precision must be fp16 or fp32")
    if options.cache_dir is None:
        from torch.utils.cpp_extension import get_default_build_root

        cache = Path(get_default_build_root()) / "topology-selections"
    else:
        cache = Path(options.cache_dir)
    context = dict(
        model=config.model_config.model,
        speculation=getattr(config.speculative_config, "method", None),
        max_sequences=config.scheduler_config.max_num_seqs,
        group_ranks=list(tp.ranks),
    )
    key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:20]
    started = time.monotonic()
    routes = {}
    receipts = []
    for width in widths:
        begin = time.monotonic()
        route = prepare_best(
            tp.cpu_group,
            worker.device,
            existing=comm,
            payload_bytes=width,
            precision=precision,
            cache=cache / f"{key}-{width}-{precision}.json",
            context=context,
        )
        torch.accelerator.synchronize(worker.device)
        times = [0.0] * tp.world_size
        dist.all_gather_object(times, time.monotonic() - begin, group=tp.cpu_group)
        routes[width] = route
        receipts.append(
            dict(bytes=width, setup_seconds=max(times), selection=route.receipt)
        )

    def reduce(x):
        route = routes.get(x.nbytes)
        if (
            route is not None
            and x.dtype == torch.float16
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
        ):
            return route.all_reduce(x)
        return previous.all_reduce(x)

    # Capture the old bound method; never recurse through the patched method.
    comm.all_reduce = reduce
    # GroupCoordinator destroys its CPU group before its communicator. Close
    # IPC channels while that group still exists for coordinated buffer release.
    destroy = tp.destroy

    def close():
        try:
            for route in routes.values():
                route.close()
        finally:
            destroy()

    tp.destroy = close
    logger.info(
        "Topology all-reduce rank=%d setup=%.3fs precision=%s widths=%s",
        tp.rank_in_group,
        time.monotonic() - started,
        precision,
        [
            dict(
                bytes=r["bytes"],
                seconds=r["setup_seconds"],
                winner=r["selection"].get("winner"),
                cache_hit=r["selection"].get("cache_hit"),
            )
            for r in receipts
        ],
    )
    return routes
