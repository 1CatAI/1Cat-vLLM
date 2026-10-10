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
_F16_TYPES = {1, 30}  # F16 and BF16 use the existing FP16 dense contract.


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
    if not shards or any(kind not in _F16_TYPES for _, kind in shards):
        return None
    parts = []
    for raw, kind in shards:
        dtype = torch.bfloat16 if kind == 30 else torch.float16
        if raw.ndim != 2:
            return None
        if raw.dtype == torch.uint8:
            if raw.shape[1] % 2:
                return None
            raw = raw.contiguous().view(dtype)
        elif raw.dtype != dtype:
            return None
        if dtype == torch.bfloat16:
            # Preparation runs during loading. Preserve the existing FP16
            # weight contract; decline coefficients that cannot be represented.
            if not bool((raw.float().abs() <= torch.finfo(torch.float16).max).all()):
                return None
            raw = raw.to(torch.float16)
        parts.append(raw)
    if len({part.shape[1] for part in parts}) != 1:
        return None
    return torch.cat(parts) if len(parts) > 1 else parts[0]


_PROJECTIONS: dict[str, Dmv13Projection] = {}


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
            if (
                q.shape[0] % 32
                or q.shape[1] % 128
                or (k is not None and k != q.shape[1])
            ):
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
        device = device or torch.device(
            "cuda", torch.accelerator.current_device_index()
        )
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
        extra_out = x.new_empty(
            (m, self.extra.shape[0] if self.extra is not None else 0)
        )
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
    _register_banks(layer, attr, projection)
    return True


def _register_banks(layer, attr, projection):
    """Expose backing storage to normal model accounting without copying it."""
    if not isinstance(layer, torch.nn.Module):
        return
    for family in ("codes", "high", "scale"):
        for index, tensor in enumerate(getattr(projection, family)):
            layer.register_buffer(f"_{attr}_{family}_{index}", tensor, persistent=False)
    for family in ("workspace", "counters", "extra"):
        layer.register_buffer(
            f"_{attr}_{family}", getattr(projection, family), persistent=False
        )


def share_prepared_banks(layer):
    """Share resident segment planes with the fused side projection.

    Both readers use the same row and bit-plane layout. LUT4's fused reader
    uses compact half scales; only that small coefficient stream is separate.
    No weight is reconstructed or requantized here.
    """
    prepared = getattr(layer, "gguf_tm_projections", None)
    if prepared is None:
        return 0
    projections = [
        (attr, value)
        for attr, value in vars(layer).items()
        if isinstance(value, Dmv13Projection) and value.ready
    ]
    reused = 0
    for attr, projection in projections:
        banks = []
        for p in prepared:
            if not hasattr(p, "segment_format") or p.input_layout_restored:
                break
            k, n = p.kernel.config.partition_weight_shape
            if k != projection.k or p.output_padding:
                break
            groups = (k + 127) // 128
            strides = (
                groups * 2048 * (2 if p.segment_format == dense.Q8 else 1),
                groups * 512 * (2 if p.segment_format == dense.Q6K else 1)
                if p.segment_format in (dense.Q5K, dense.Q6K)
                else 0,
                groups * 512,
            )
            tile = 0
            for width in p.source_output_sizes:
                if width % 32:
                    break
                end = tile + width // 32
                planes = [
                    tensor[tile * stride : end * stride]
                    for tensor, stride in zip(
                        (p.codes, p.segment_high, p.stats), strides
                    )
                ]
                banks.append((width, p.segment_format, planes))
                tile = end
            if tile * 32 != n:
                break
        else:
            if [(b[0], b[1]) for b in banks] != list(
                zip(projection.widths, projection.formats)
            ):
                continue
            for family, index in (("codes", 0), ("high", 1), ("scale", 2)):
                values = []
                for _, fmt, planes in banks:
                    tensor = planes[index].detach()
                    if family == "scale" and fmt == dense.LUT4:
                        tensor = (
                            tensor.view(torch.int16)
                            .reshape(-1, 4, 2)[:, :, 0]
                            .contiguous()
                            .view(torch.uint8)
                            .reshape(-1)
                        )
                    values.append(tensor)
                setattr(projection, family, values)
            _register_banks(layer, attr, projection)
            reused += 1
    return reused
