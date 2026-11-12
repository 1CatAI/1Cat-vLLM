# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_PROPOSE_GRAPH=1 (default off): the drafter step of a decode round
as one CUDA graph.

The drafter step (padded input preparation, context insertion, query layers,
draft head, greedy pick) is ~1 ms of host launch work per round on the last PP
stage, which with PP micro-batches is the stage that paces the pipeline. Per
key (request count and the shapes of every per-step tensor) the step runs
eagerly a few times, so every inner graph exists, then is captured once and
replayed: each round copies its per-step tensors into the graph's own static
copies and launches one graph.

A key whose capture fails stays eager. Drafts only steer verification, so a
wrong draft costs acceptance, never output tokens.
"""

import logging
import traceback
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.distributed import graph_capture

logger = logging.getLogger(__name__)

_WARMUP = 3
# relaxed: the round's host-side helpers may allocate pinned memory or query
# events while the stream captures.
_MODE = envs.VLLM_V41_PROPOSE_GRAPH_MODE


class V41ProposeGraph:
    def __init__(self) -> None:
        self.entries: dict[tuple, dict] = {}
        self.pool = None

    def __call__(
        self,
        key: tuple,
        fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        key = key + tuple(
            (name, tuple(t.shape), t.dtype) for name, t in sorted(inputs.items())
        )
        entry = self.entries.get(key)
        if entry is None:
            entry = self.entries[key] = {"seen": 0, "graph": None, "failed": False}
        if entry["failed"] or entry["seen"] < _WARMUP:
            entry["seen"] += 1
            return fn(inputs)
        if entry["graph"] is None and not self._capture(key, entry, fn, inputs):
            return fn(inputs)
        static = entry["static"]
        for name, tensor in inputs.items():
            static[name].copy_(tensor, non_blocking=True)
        entry["graph"].replay()
        return entry["out"].clone()

    def _capture(self, key, entry, fn, inputs) -> bool:
        static = {name: t.clone() for name, t in inputs.items()}
        graph = torch.cuda.CUDAGraph()
        try:
            # vLLM's graph_capture: TP custom all-reduce inside the round
            # (sharded draft head) needs its capture-registered buffers.
            with (
                graph_capture(device=next(iter(static.values())).device) as context,
                torch.cuda.graph(
                    graph,
                    pool=self.pool,
                    stream=context.stream,
                    capture_error_mode=_MODE,
                ),
            ):
                out = fn(static)
        except Exception as e:  # noqa: BLE001 - any capture failure: stay eager
            logger.warning(
                "V4.1 propose graph: capture failed for %s: %r\n%s",
                key[:3],
                e,
                traceback.format_exc(),
            )
            entry["failed"] = True
            torch.accelerator.synchronize()
            return False
        if self.pool is None:
            self.pool = graph.pool()
        entry.update(graph=graph, static=static, out=out)
        logger.info("V4.1 propose graph: captured %s", key[:3])
        return True
