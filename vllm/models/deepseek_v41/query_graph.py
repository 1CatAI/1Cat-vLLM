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

from .cache import BATCH, V41CacheMetadata

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
    """One graph per query count: M5 for one request; VLLM_V41_BATCH adds
    M = 5 R for R requests (captured on first use, keyed by M)."""

    def __init__(self, forward):
        self.forward = forward
        self.entries: dict[int, tuple] = {}
        self.captures = self.replays = 0

    def __call__(self, input_ids, positions, inputs_embeds=None):
        if torch.cuda.is_current_stream_capturing():
            # Inside VLLM_V41_PROPOSE_GRAPH's round capture: record the kernels
            # (a graph cannot be replayed while another one is captured).
            return self.forward(input_ids, positions, inputs_embeds)
        metadata = get_forward_context().attn_metadata
        rows = positions.shape[0]
        allowed = rows % 5 == 0 and rows > 0 if BATCH else rows == 5
        if (
            positions.ndim != 1
            or not allowed
            or not isinstance(metadata, dict)
            or (not metadata)
        ):
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
            if item.layout is not None:
                signature.append(("requests", item.layout.num_reqs))
        entry = self.entries.get(rows)
        if entry is None:
            # Warm the attention kernels with real metadata. This forward
            # does not insert query KV or advance accepted-prefix state.
            self.forward(input_ids, positions, inputs_embeds)
            torch.accelerator.synchronize(positions.device)
            graph = torch.cuda.CUDAGraph()
            with (
                graph_capture(device=positions.device) as context,
                torch.cuda.graph(
                    graph,
                    pool=current_platform.get_global_graph_pool(),
                    stream=context.stream,
                ),
            ):
                output = self.forward(input_ids, positions, inputs_embeds)
            references = input_ids, positions, inputs_embeds, metadata
            entry = self.entries[rows] = (graph, output, signature, references)
            self.captures += 1
            logger.info(
                "Captured V4.1 DSpark M%d query graph with real KV metadata", rows
            )
        elif signature != entry[2]:
            raise ValueError(
                "V4.1 draft graph input storage or metadata geometry changed"
            )
        entry[0].replay()
        self.replays += 1
        return entry[1]


class V41GreedyDraftGraph:
    """Capture the shared head and five dependent Markov/argmax steps.

    The proposer gathers hidden rows into a fresh allocation each round.
    Copy those values and the current anchor into owned graph inputs; never
    capture an address belonging to that temporary gather or a previous token.
    One graph per request count R (M = 5 R, R anchors; R > 1 needs
    VLLM_V41_BATCH and the sharded Markov head).
    """

    def __init__(self, forward):
        self.forward = forward
        self.entries: dict[int, tuple] = {}
        self.captures = self.replays = 0

    def __call__(self, hidden, anchor):
        if torch.cuda.is_current_stream_capturing():
            return self.forward(hidden, anchor)
        reqs = anchor.numel()
        if (
            hidden.ndim != 2
            or hidden.shape[0] != 5 * reqs
            or not hidden.is_cuda
            or anchor.ndim != 1
            or (reqs != 1 and not BATCH)
            or anchor.device != hidden.device
            or anchor.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("V4.1 greedy draft graph requires M5 per anchor")
        entry = self.entries.get(reqs)
        if entry is None:
            owned_hidden, owned_anchor = hidden.clone(), anchor.clone()
            self.forward(owned_hidden, owned_anchor)
            torch.accelerator.synchronize(hidden.device)
            graph = torch.cuda.CUDAGraph()
            with (
                graph_capture(device=hidden.device) as context,
                torch.cuda.graph(
                    graph,
                    pool=current_platform.get_global_graph_pool(),
                    stream=context.stream,
                ),
            ):
                output = self.forward(owned_hidden, owned_anchor)
            entry = self.entries[reqs] = (graph, owned_hidden, owned_anchor, output)
            self.captures += 1
        else:
            _, owned_hidden, owned_anchor, _ = entry
            if _storage(hidden)[1:] != _storage(owned_hidden)[1:] or (
                _storage(anchor)[1:] != _storage(owned_anchor)[1:]
            ):
                raise ValueError("V4.1 greedy graph input geometry changed")
            owned_hidden.copy_(hidden)
            owned_anchor.copy_(anchor)
        entry[0].replay()
        self.replays += 1
        return entry[3]


# VLLM_V41_BATCH: context rows of several requests (padded by the caller to
# a few sizes; padding rows carry slot -1 and store nothing).
_CONTEXT_MAX = 192


class V41ContextGraphs:
    """Bounded M1..6 context-insertion graphs with fresh owned inputs."""

    def __init__(self, forward):
        self.forward = forward
        self.entries: dict[int, tuple] = {}
        self.captures = self.replays = 0

    def __call__(self, states, positions, slots):
        if torch.cuda.is_current_stream_capturing():
            return self.forward(states, positions, slots)
        rows = states.shape[0]
        if (
            not 1 <= rows <= (_CONTEXT_MAX if BATCH else 6)
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
