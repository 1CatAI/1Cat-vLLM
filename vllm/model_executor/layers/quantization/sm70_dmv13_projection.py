# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch small-M GGUF projection with an appended FP16 projection (SM70).

A merged GGUF linear (one or more shards, mixed block formats) is repacked into
dense_mv tile planes; an FP16 side projection on the same input (GDN a/b, QSA
indexer q/k) runs as extra CTAs of the same launch instead of its own GEMV.
"""

from __future__ import annotations

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import gguf_dmv13_dense as dense
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

MAX_M = 8
MAX_SEGMENTS = 4
_DENSE_TYPES = {8, 12, 13, 14, 20, 23}
_F16_TYPES = {1}


def _raw_shards(layer) -> list[tuple[torch.Tensor, int]] | None:
    qweight = getattr(layer, "qweight", None)
    qtype = getattr(layer, "qweight_type", None)
    if qweight is None or qtype is None:
        return None
    container = getattr(qweight, "data_container", None)
    if container:
        ids = ["q", "k", "v"] if "q" in qweight.shard_id else sorted(qweight.shard_id)
        return [
            (container[qweight.shard_id_map[i]], qtype.shard_weight_type[i])
            for i in ids
        ]
    return [(qweight.data, qtype.weight_type)]


def _fp16_weight(layer) -> torch.Tensor | None:
    weight = getattr(layer, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.dtype == torch.float16:
        return weight.data if weight.ndim == 2 else None
    shards = _raw_shards(layer)
    if shards and len(shards) == 1 and shards[0][1] in _F16_TYPES:
        raw = shards[0][0]
        return raw.view(torch.float16) if raw.dtype == torch.uint8 else raw
    return None


_PROJECTIONS: dict[str, "Dmv13Projection"] = {}


class Dmv13Projection:
    def __init__(self, layer, extra=None):
        self.ready = False
        self.name = getattr(layer, "prefix", "") or str(id(layer))
        shards = _raw_shards(layer)
        if not shards or len(shards) > MAX_SEGMENTS:
            return
        if any(t not in _DENSE_TYPES for _, t in shards):
            return
        device = None
        self.codes, self.high, self.scale = [], [], []
        self.formats, self.widths = [], []
        k = None
        for raw, qtype in shards:
            device = raw.device if raw.is_cuda else device
            fmt, q, s, m, gs = dense.decode(raw.detach().cpu().numpy(), qtype)
            if q.shape[0] % 32 or q.shape[1] % 128 or (k is not None and k != q.shape[1]):
                return
            k = q.shape[1]
            codes, high, scale = dense.pack(fmt, q, s, m, gs)
            if fmt == dense.LUT4:
                from vllm.model_executor.layers.quantization.gguf_dmv_formats import (
                    compact_lut4_scale,
                )

                scale = compact_lut4_scale(scale)
            self.codes.append(torch.from_numpy(np.ascontiguousarray(codes)))
            self.high.append(torch.from_numpy(np.ascontiguousarray(high)))
            self.scale.append(torch.from_numpy(np.ascontiguousarray(scale)))
            self.formats.append(int(fmt))
            self.widths.append(int(q.shape[0]))
        device = device or torch.device("cuda", torch.cuda.current_device())
        for name in ("codes", "high", "scale"):
            setattr(self, name, [t.to(device) for t in getattr(self, name)])
        self.k = k
        self.n = sum(self.widths)
        self.extra = None
        if extra is not None:
            weight = _fp16_weight(extra)
            if weight is None or weight.shape[1] != k:
                return
            self.extra = weight.contiguous()
        tiles = sum((w + 31) // 32 for w in self.widths)
        self.workspace = torch.zeros(tiles * 256, device=device)
        self.counters = torch.zeros(tiles, device=device, dtype=torch.int32)
        self.ready = hasattr(torch.ops._C, "sm70_dmv13_out")

    def __call__(self, x: torch.Tensor):
        return torch.ops.vllm.sm70_dmv13_side_projection(x, self.name)

    def run(self, x: torch.Tensor):
        m = x.shape[0]
        out = x.new_empty((m, self.n))
        views, start = [], 0
        for width in self.widths:
            views.append(out[:, start : start + width])
            start += width
        extra_out = x.new_empty((m, self.extra.shape[0] if self.extra is not None else 0))
        torch.ops._C.sm70_dmv13_out(
            x.contiguous(),
            self.codes,
            self.high,
            self.scale,
            views,
            self.formats,
            self.widths,
            self.k,
            1,
            4,
            self.workspace,
            self.counters,
            1,
            self.extra,
            extra_out if self.extra is not None else None,
        )
        return out, extra_out


def _side_projection(x: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    return _PROJECTIONS[name].run(x)


def _side_projection_fake(x: torch.Tensor, name: str):
    projection = _PROJECTIONS[name]
    width = projection.extra.shape[0] if projection.extra is not None else 0
    return x.new_empty((x.shape[0], projection.n)), x.new_empty((x.shape[0], width))


direct_register_custom_op(
    op_name="sm70_dmv13_side_projection",
    op_func=_side_projection,
    fake_impl=_side_projection_fake,
)


def attach(layer, extra, attr: str) -> bool:
    projection = Dmv13Projection(layer, extra)
    if not projection.ready:
        return False
    _PROJECTIONS[projection.name] = projection
    setattr(layer, attr, projection)
    return True
