# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit worker-extension probe for an independently built long q1 operator.

Import only with VLLM_SM70_SCALAR_Q1_PROBE_MANIFEST set. Both A/B arms retain
this wrapper, and only eligible eager calls select the candidate. It installs
no default serving route and never reads a GPU sequence length on the host.
"""

import functools
import hashlib
import os
from collections import Counter
from pathlib import Path

import flash_attn_v100_cuda as native
import torch

from benchmarks.kernels.benchmark_sm70_grouped_attention_long import load_operator
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

operator, manifest = load_operator(
    Path(os.environ["VLLM_SM70_SCALAR_Q1_PROBE_MANIFEST"])
)
assert manifest["share_kv_six_heads"] and manifest["max_context"] == 262144
MODE = "control"
_ELIGIBLE = False
SEEN = Counter()
HITS = Counter()
_original_call = FlashAttnV100Impl._call_flash_attn_decode_paged
_original_native = native.decode_paged_fwd


def cpu_context_bound(kwargs):
    bound = kwargs.get("max_seq_len_hint")
    if type(bound) is int:
        return bound
    if is_forward_context_available():
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            bounds = [getattr(m, "max_seq_len", None) for m in metadata.values()]
            bounds = [b for b in bounds if type(b) is int and b > 0]
            if bounds:
                return max(bounds)
    return None


@functools.wraps(_original_call)
def call(self, query, key_cache, value_cache, *args, **kwargs):
    global _ELIGIBLE
    previous = _ELIGIBLE
    _ELIGIBLE = False
    if (
        query.shape == (1, 6, 256)
        and key_cache.dtype == torch.uint8
        and not torch.cuda.is_current_stream_capturing()
    ):
        bound = cpu_context_bound(kwargs)
        SEEN[str(bound)] += 1
        if type(bound) is int and 131072 <= bound <= 262144:
            _ELIGIBLE = MODE == "candidate"
    try:
        return _original_call(self, query, key_cache, value_cache, *args, **kwargs)
    finally:
        _ELIGIBLE = previous


def run(*args, **kwargs):
    if _ELIGIBLE and not kwargs and len(args) == 20:
        q, k, v, out, table, lengths, partial, maximum, sums, active = args[:10]
        (
            scale,
            partition,
            count,
            dtype,
            k_scale,
            v_scale,
            left,
            right,
            anchor,
            window,
        ) = args[10:]
        if (
            q.shape == (1, 6, 256)
            and k.shape[1] == 3296
            and k.shape[2:] == (1, 256)
            and k.dtype == v.dtype == torch.uint8
            and partition == 1024
            and count == 256
            and dtype in ("fp8", "fp8_e4m3")
            and left == right == -1
            and anchor is None
            and window == 0
            and partial.shape == (1, 6, 256, 256)
            and partial.dtype == torch.float32
            and maximum.shape == sums.shape == (1, 6, 256)
        ):
            HITS["eager_long_q1"] += 1
            return operator(*args[:10], scale, k_scale, v_scale)
    return _original_native(*args, **kwargs)


FlashAttnV100Impl._call_flash_attn_decode_paged = call
native.decode_paged_fwd = run


class ScalarQ1ProbeExtension:
    def scalar_attention_switch(self, mode):
        global MODE
        assert mode in ("control", "candidate")
        MODE = mode
        return self.scalar_attention_snapshot()

    def scalar_attention_snapshot(self):
        result = dict(
            rank=torch.distributed.get_rank(),
            mode=MODE,
            seen_cpu_context_bounds=dict(SEEN),
            hits=dict(HITS),
            manifest=manifest,
            native_library=str(Path(native.__file__).resolve()),
            native_library_sha256=hashlib.sha256(
                Path(native.__file__).read_bytes()
            ).hexdigest(),
            scope="Explicit eager long q1 probe; other calls retain the original entry",
        )
        return result
