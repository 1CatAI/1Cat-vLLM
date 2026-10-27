# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed M5 draft graph captured with the real noncausal KV metadata.

The generic DFlash dummy forward supplies no attention metadata. Capturing
that forward would bake the draft's profiling bypass into every replay.
V4.1 instead warms and captures its first real M5 query. Its forward only
reads persistent KV; the proposer inserts context before invoking this graph.
"""

import torch

from vllm.distributed import graph_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .cache import V41CacheMetadata

logger = init_logger(__name__)


def _storage(tensor):
    if tensor is None:
        return None
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.stride(),
        tensor.dtype,
        tensor.device,
    )


class V41QueryGraph:
    def __init__(self, forward):
        self.forward = forward
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: torch.Tensor | None = None
        self.signature: list[object] | None = None
        self.references: tuple | None = None
        self.captures = self.replays = 0

    def __call__(self, input_ids, positions, inputs_embeds=None):
        metadata = get_forward_context().attn_metadata
        if positions.shape != (5,) or not isinstance(metadata, dict) or not metadata:
            raise ValueError("V4.1 draft graph requires M5 with real KV metadata")
        signature = [_storage(t) for t in (input_ids, positions, inputs_embeds)]
        for name, item in sorted(metadata.items()):
            if not isinstance(item, V41CacheMetadata):
                raise ValueError("V4.1 draft graph requires V4.1 cache metadata")
            signature.append((name, item.block_size, item.causal))
            signature.extend(
                _storage(t)
                for t in (item.block_table, item.slot_mapping, item.is_prefilling)
            )
        if self.graph is None:
            # Warm the attention kernels with real metadata. This forward
            # does not insert query KV or advance accepted-prefix state.
            self.forward(input_ids, positions, inputs_embeds)
            torch.accelerator.synchronize(positions.device)
            self.references = input_ids, positions, inputs_embeds, metadata
            self.signature = signature
            self.graph = torch.cuda.CUDAGraph()
            with (
                graph_capture(device=positions.device) as context,
                torch.cuda.graph(
                    self.graph,
                    pool=current_platform.get_global_graph_pool(),
                    stream=context.stream,
                ),
            ):
                self.output = self.forward(input_ids, positions, inputs_embeds)
            self.captures += 1
            logger.info("Captured V4.1 DSpark M5 query graph with real KV metadata")
        elif signature != self.signature:
            raise ValueError("V4.1 M5 graph input storage or metadata geometry changed")
        self.graph.replay()
        self.replays += 1
        return self.output


class V41GreedyDraftGraph:
    """Capture the shared head and five dependent Markov/argmax steps.

    The proposer gathers hidden rows into a fresh allocation each round.
    Copy those values and the current anchor into owned graph inputs; never
    capture an address belonging to that temporary gather or a previous token.
    """

    def __init__(self, forward):
        self.forward = forward
        self.graph: torch.cuda.CUDAGraph | None = None
        self.hidden: torch.Tensor | None = None
        self.anchor: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.captures = self.replays = 0

    def __call__(self, hidden, anchor):
        if (
            hidden.ndim != 2
            or hidden.shape[0] != 5
            or not hidden.is_cuda
            or anchor.shape != (1,)
            or anchor.device != hidden.device
            or anchor.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("V4.1 greedy draft graph requires M5 and one anchor")
        if self.graph is None:
            self.hidden, self.anchor = hidden.clone(), anchor.clone()
            self.forward(self.hidden, self.anchor)
            torch.accelerator.synchronize(hidden.device)
            self.graph = torch.cuda.CUDAGraph()
            with (
                graph_capture(device=hidden.device) as context,
                torch.cuda.graph(
                    self.graph,
                    pool=current_platform.get_global_graph_pool(),
                    stream=context.stream,
                ),
            ):
                self.output = self.forward(self.hidden, self.anchor)
            self.captures += 1
        else:
            assert self.hidden is not None and self.anchor is not None
            if _storage(hidden)[1:] != _storage(self.hidden)[1:] or (
                _storage(anchor)[1:] != _storage(self.anchor)[1:]
            ):
                raise ValueError("V4.1 greedy graph input geometry changed")
            self.hidden.copy_(hidden)
            self.anchor.copy_(anchor)
        self.graph.replay()
        self.replays += 1
        return self.output


class V41ContextGraphs:
    """Bounded M1..6 context-insertion graphs with fresh owned inputs."""

    def __init__(self, forward):
        self.forward = forward
        self.entries: dict[int, tuple] = {}
        self.captures = self.replays = 0

    def __call__(self, states, positions, slots):
        rows = states.shape[0]
        if (
            not 1 <= rows <= 6
            or states.ndim != 2
            or not states.is_cuda
            or states.dtype != torch.float16
            or positions.shape != (rows,)
            or positions.device != states.device
            or positions.dtype not in (torch.int32, torch.int64)
            or len(slots) != 3
            or any(
                value.shape != (rows,)
                or value.device != states.device
                or value.dtype != torch.int64
                for value in slots.values()
            )
        ):
            raise ValueError("V4.1 context graph requires M1..6 and three slot maps")
        signature = (
            _storage(states)[1:],
            _storage(positions)[1:],
            [(name, _storage(value)[1:]) for name, value in sorted(slots.items())],
        )
        if rows not in self.entries:
            values, at = states.clone(), positions.clone()
            maps = {name: value.clone() for name, value in slots.items()}
            # Repeating the same writes is idempotent. No accepted-prefix
            # state or query KV is advanced by this context-only operation.
            self.forward(values, at, maps)
            torch.accelerator.synchronize(states.device)
            graph = torch.cuda.CUDAGraph()
            with (
                graph_capture(device=states.device) as context,
                # Context runs before the query although it may be captured
                # later. Keep its small temporaries in a private graph pool.
                torch.cuda.graph(graph, stream=context.stream),
            ):
                self.forward(values, at, maps)
            self.entries[rows] = graph, values, at, maps, signature
            self.captures += 1
        graph, values, at, maps, expected = self.entries[rows]
        if signature != expected:
            raise ValueError("V4.1 context graph input geometry changed")
        values.copy_(states)
        at.copy_(positions)
        for name in maps:
            maps[name].copy_(slots[name])
        graph.replay()
        self.replays += 1
