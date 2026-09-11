# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B instrumentation only; candidate dispatch and operators are runtime code."""

from collections import Counter

import torch

from benchmarks.sm70_dflash2_prefill_route_audit import (
    PrefillRouteAuditExtension as Base,
)
from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl
from vllm.v1.attention.ops import sm70_e4m3_long as grouped
from vllm.v1.attention.ops import sm70_e4m3_scalar as scalar
from vllm.v1.worker.gpu import cudagraph_utils as cg

MODE = "candidate"
COUNTS = Counter()
MANAGERS = []
original_init = cg.ModelCudaGraphManager.__init__
original_dispatch = cg.CudaGraphManager.dispatch
original_select = cg.ModelCudaGraphManager.select_attention_graph
original_scalar = FlashAttnV100Impl._call_flash_attn_decode_paged
original_scalar_load = scalar.load_scalar_tail_attention
original_grouped_load = grouped.load_long_attention


def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if self._sm70_dflash2_tail_graphs:
        MANAGERS.append(self)


def dispatch(self, num_reqs, num_tokens, uniform_token_count):
    tail = (
        self in MANAGERS
        and num_reqs == 1
        and num_tokens == uniform_token_count
        and 1 <= num_tokens < 8
    )
    if tail and MODE == "control" and self._graphs_captured:
        COUNTS[f"control_q{num_tokens}_NONE"] += 1
        return cg.BatchExecutionDescriptor(
            cg_mode=cg.CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs
        )
    desc = original_dispatch(self, num_reqs, num_tokens, uniform_token_count)
    if tail and self._graphs_captured:
        COUNTS[f"{MODE}_q{num_tokens}_{desc.cg_mode.name}"] += 1
    return desc


def scalar_call(self, *args, **kwargs):
    saved = self._sm70_scalar_tail_attention
    if MODE == "control":
        self._sm70_scalar_tail_attention = None
    try:
        return original_scalar(self, *args, **kwargs)
    finally:
        self._sm70_scalar_tail_attention = saved


def select(self, desc, upper):
    selected = original_select(self, desc, upper)
    if self._graphs_captured and desc.num_reqs == 1 and 1 <= desc.num_tokens < 8:
        COUNTS[
            f"{MODE}_q{desc.num_tokens}_bucket_{selected.attention_context_bucket}"
        ] += 1
    return selected


def scalar_load(*args, **kwargs):
    op = original_scalar_load(*args, **kwargs)

    def run(*a, **kw):
        hit = op(*a, **kw)
        if hit:
            COUNTS["scalar_native_bindings"] += 1
        return hit

    return run


def grouped_load(*args, **kwargs):
    op, manifest = original_grouped_load(*args, **kwargs)

    def run(*a, **kw):
        COUNTS[f"grouped_q{a[0].shape[0]}_bindings"] += 1
        return op(*a, **kw)

    return run, manifest


cg.ModelCudaGraphManager.__init__ = init
cg.CudaGraphManager.dispatch = dispatch
cg.ModelCudaGraphManager.select_attention_graph = select
FlashAttnV100Impl._call_flash_attn_decode_paged = scalar_call
scalar.load_scalar_tail_attention = scalar_load
grouped.load_long_attention = grouped_load


class TailAuditExtension(Base):
    def tail_switch(self, mode):
        global MODE
        assert mode in ("control", "candidate")
        torch.accelerator.synchronize()
        MODE = mode
        return self.tail_snapshot()

    def long_attention_switch(self, mode):
        assert mode == "candidate"
        return self.tail_snapshot()

    def tail_snapshot(self):
        return dict(
            rank=torch.distributed.get_rank(),
            mode=MODE,
            counts=dict(COUNTS),
            graphs=[str(d) for m in MANAGERS for d in m.graphs],
            contract=grouped.long_attention_graph_contract(),
        )
