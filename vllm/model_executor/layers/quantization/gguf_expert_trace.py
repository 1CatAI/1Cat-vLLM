# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe routing traces for sizing a routed-expert cache.

Decode-sized calls append their expert ids to a per-layer ring through a
device-side write counter, so captured CUDA graphs record every replay.
Prefill-sized calls add to per-layer expert histograms. Tracing is a
diagnostic: it is off unless ``sm70_gguf.expert_trace_tokens`` is set.
"""

import regex as re
import torch

from vllm.triton_utils import tl, triton

_LAYER = re.compile(r"layers\.(\d+)\.")


@triton.jit
def _append_decode(
    ids,
    ring,
    counter,
    layer,
    ring_base,
    tokens,
    TOPK: tl.constexpr,
    RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < tokens * TOPK
    values = tl.load(ids + offsets, mask=mask, other=-1)
    start = tl.atomic_add(counter + layer, tokens)
    rows = (start + offsets // TOPK) % RING
    target = ring + ring_base + rows * TOPK + offsets % TOPK
    tl.store(target, values.to(tl.int16), mask=mask)


@triton.jit
def _add_prefill(
    ids, hist, hist_base, count, EXPERTS: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    values = tl.load(ids + offsets, mask=mask, other=0)
    valid = mask & (values >= 0) & (values < EXPERTS)
    tl.atomic_add(hist + hist_base + values, 1, mask=valid)


class ExpertTrace:
    """Per-rank buffers shared by every traced MoE layer."""

    DECODE_MAX_TOKENS = 20

    def __init__(self, device, num_layers, ring_tokens, top_k, num_experts):
        self.ring_tokens, self.top_k = ring_tokens, top_k
        self.num_experts = num_experts
        self.ring = torch.full(
            (num_layers, ring_tokens, top_k), -1, dtype=torch.int16, device=device
        )
        self.counter = torch.zeros(num_layers, dtype=torch.int64, device=device)
        self.prefill = torch.zeros(
            (num_layers, num_experts), dtype=torch.int64, device=device
        )

    def record(self, layer_index: int, ids: torch.Tensor) -> None:
        tokens, top_k = ids.shape
        if not tokens or top_k != self.top_k:
            return
        if tokens <= self.DECODE_MAX_TOKENS:
            _append_decode[(1,)](
                ids,
                self.ring,
                self.counter,
                layer_index,
                layer_index * self.ring_tokens * top_k,
                tokens,
                TOPK=top_k,
                RING=self.ring_tokens,
                BLOCK=triton.next_power_of_2(self.DECODE_MAX_TOKENS * top_k),
            )
        else:
            count = tokens * top_k
            _add_prefill[(triton.cdiv(count, 1024),)](
                ids,
                self.prefill,
                layer_index * self.num_experts,
                count,
                EXPERTS=self.num_experts,
                BLOCK=1024,
            )

    @torch.inference_mode()
    def reset(self) -> None:
        self.ring.fill_(-1)
        self.counter.zero_()
        self.prefill.zero_()

    def snapshot(self) -> dict[str, torch.Tensor]:
        torch.accelerator.synchronize(self.ring.device)
        return {
            "ring": self.ring.cpu(),
            "counter": self.counter.cpu(),
            "prefill": self.prefill.cpu(),
        }


_TRACE: ExpertTrace | None = None


def layer_index(layer) -> int:
    cached = getattr(layer, "_expert_trace_layer", None)
    if cached is None:
        name = str(getattr(layer, "layer_name", "") or getattr(layer, "prefix", ""))
        match = _LAYER.search(name)
        cached = int(match.group(1)) if match else -1
        layer._expert_trace_layer = cached
    return cached


def trace_routing(layer, ids, ring_tokens, num_layers, num_experts) -> None:
    """Record ``ids`` for ``layer`` when tracing is enabled (ring_tokens > 0)."""
    global _TRACE
    index = layer_index(layer)
    if index < 0 or index >= num_layers:
        return
    if _TRACE is None:
        if torch.cuda.is_current_stream_capturing():
            return  # buffers must exist before capture; the eager warmup creates them
        _TRACE = ExpertTrace(
            ids.device, num_layers, ring_tokens, ids.shape[1], num_experts
        )
    _TRACE.record(index, ids)


def current_trace() -> ExpertTrace | None:
    return _TRACE
