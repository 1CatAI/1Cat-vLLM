# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 dense projections on the TurboMind route.

Selected Q5_K/Q8_0 linears move from original GGUF blocks to the TurboMind
segment planes after loading (the original bytes are released, so there is
one resident copy). Prefill keeps the prepared projection (canonical restore
plus GEMM); M<=8 decode runs the dmv13 small-M kernel on the same planes,
with same-input projections merged into one launch.
"""

from __future__ import annotations

from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = envs.VLLM_V41_DENSE_TM == "1"
MAX_M = 8
_TYPES = {8, 13}  # Q8_0, Q5_K


def _segment_shapes(model, extra=()):
    shapes = set()
    for layer in _targets(model, extra):
        for module in layer:
            q = getattr(module, "qweight", None)
            t = getattr(getattr(module, "qweight_type", None), "weight_type", None)
            if q is None or t not in _TYPES or q.ndim != 2:
                continue
            from vllm.transformers_utils.gguf_tensor_reader import quant_size

            block, size = quant_size(t)
            shapes.add((q.shape[1] // size * block, q.shape[0]))
    return shapes


def _roles(model, extra=()):
    """(role, group) per decoder layer; each group of linears shares one input.

    The role names key VLLM_V41_DMV_X_CONFIG overrides.
    """
    for group in extra:
        yield "extra", group
    for block in model.layers:
        attn = block.attn
        yield "qkv", (attn.wq_a, attn.wkv)
        yield "q_b", (attn.wq_b,)
        yield "wo_a", (attn.wo_a,)
        yield "wo_b", (attn.wo_b,)
        shared = block.ffn.shared_experts
        yield "shared", (shared.gate_proj, shared.up_proj)
        engram = getattr(block, "engram", None)
        if engram is not None and hasattr(engram, "wkv"):
            yield "engram", (engram.wkv,)


def _targets(model, extra=()):
    """Groups of linears per decoder layer; each group shares one input."""
    for _, group in _roles(model, extra):
        yield group


def _convert(module) -> bool:
    """Original GGUF storage -> one prepared TurboMind segment projection."""
    method = module.quant_method
    q = module.qweight
    t = module.qweight_type.weight_type
    if (
        t not in _TYPES
        or q.ndim != 2
        or q.dtype != torch.uint8
        or getattr(method, "canonical_projections", ())
        or getattr(method, "layout", None) is not None
        or q.shard_id
    ):
        return False
    from vllm.model_executor.layers.quantization.gguf_turbomind import (
        prepare_gguf_projections,
    )

    projections = prepare_gguf_projections(
        [(q.data, t)], torch.float16, True, method.prefill_min_m
    )
    if len(projections) != 1 or not hasattr(projections[0], "segment_format"):
        return False
    module.gguf_tm_projections = torch.nn.ModuleList(projections)
    method.canonical_projections = module.gguf_tm_projections
    method.v41_q5_hmma = False
    method.max_dequant_bytes = 0
    from vllm.model_executor.utils import set_weight_attrs

    empty = torch.nn.Parameter(
        torch.empty(0, dtype=torch.uint8, device=q.device), False
    )
    set_weight_attrs(empty, vars(q))
    module.register_parameter("qweight", empty)
    return True


def _planes(projection, tile0, tile1):
    from vllm.model_executor.layers.quantization import gguf_dmv13_dense as dense

    k = projection.kernel.config.partition_weight_shape[0]
    fmt = projection.segment_format
    groups = (k + 127) // 128
    strides = (
        groups * 2048 * (2 if fmt == dense.Q8 else 1),
        groups * 512 * (2 if fmt == dense.Q6K else 1)
        if fmt in (dense.Q5K, dense.Q6K)
        else 0,
        groups * 512,
    )
    tensors = (projection.codes, projection.segment_high, projection.stats)
    return [t[tile0 * s : tile1 * s] for t, s in zip(tensors, strides)]


_TARGET_CTAS = int(envs.VLLM_V41_DMV_CTAS)
# (k, widths) -> (warps, tp, split) overrides from the dmv13 sweep.
_SMALL_M_CONFIG: Any = {}
_GROUP_VIEWS = envs.VLLM_V41_WOA_VIEWS == "1"

# DMV_X (VLLM_V41_DMV_X): run the runners on the v41k dmv13_x kernel. With
# no VLLM_V41_DMV_X_CONFIG override the warps/split are dmv13's, so outputs
# are bit-identical; an override (warps, split, acc) changes the FP32
# summation order and needs the near-tie gate.
_DMV_X = envs.VLLM_V41_DMV_X == "1"
_X_ROLES = ("qkv", "q_b", "wo_a", "wo_b", "shared", "head", "engram", "extra")
_X_FORMATS = (1, 4)  # gguf_dmv13_dense Q5K, Q8
_X_READY: dict = {}


def parse_x_config(text):
    """'role=warps,split,acc;...' -> {role: (warps, split, acc)}."""
    config = {}
    for item in filter(None, (part.strip() for part in text.split(";"))):
        role, _, values = item.partition("=")
        role = role.strip()
        try:
            warps, split, acc = (int(v) for v in values.split(","))
        except ValueError:
            raise ValueError(f"VLLM_V41_DMV_X_CONFIG: bad entry {item!r}") from None
        if role not in _X_ROLES or warps not in (2, 4, 8):
            raise ValueError(f"VLLM_V41_DMV_X_CONFIG: bad entry {item!r}")
        if not 1 <= split <= 8 or acc not in (1, 2):
            raise ValueError(f"VLLM_V41_DMV_X_CONFIG: bad entry {item!r}")
        config[role] = (warps, split, acc)
    return config


_X_CONFIG = parse_x_config(envs.VLLM_V41_DMV_X_CONFIG)


def _x_ready():
    if "ok" not in _X_READY:
        from vllm.model_executor.layers.quantization.gguf_v41_moe import (
            _expert_major_ready,
        )

        _X_READY["ok"] = bool(
            _expert_major_ready() and hasattr(torch.ops.v41k, "dmv13_x")
        )
        if not _X_READY["ok"]:
            logger.warning(
                "VLLM_V41_DMV_X=1 but the loaded v41k library has no dmv13_x "
                "(rebuild v41k); dense projections stay on dmv13"
            )
    return _X_READY["ok"]


def default_config(k, widths):
    """dmv13's (warps, tp, split) for one launch over `widths` rows at K=k."""
    tiles = sum((w + 31) // 32 for w in widths)
    groups = (k // 32 + 3) // 4
    # Small outputs leave most SMs idle at split 1: split K so the launch
    # covers about two CTAs per SM, keeping at least 4 K-groups per split.
    split = max(1, min(8, round(_TARGET_CTAS / tiles), groups // 4))
    split = max(c for c in (1, 2, 3, 4, 6, 8) if c <= split)
    config = (8, 1, split) if split > 1 else (4, 1, 1)
    return _SMALL_M_CONFIG.get((k, tuple(widths)), config)


class SmallM:
    """dmv13 over segment planes: one launch for 1..4 same-input outputs.

    x_offsets (DMV_X only): first input column of each segment, for
    block-diagonal launches (wo_a). config_widths: the widths whose dmv13
    config this launch keeps (the per-group widths for a merged wo_a, so its
    reduction order is the per-group launches').
    """

    def __init__(self, segments, k, role=None, config_widths=None, x_offsets=None):
        # segments: [(planes, format, width)]
        self.codes = [p[0] for p, _, _ in segments]
        self.high = [p[1] for p, _, _ in segments]
        self.scale = [p[2] for p, _, _ in segments]
        self.formats = [f for _, f, _ in segments]
        self.widths = [w for _, _, w in segments]
        self.k = k
        self.role = role
        self.x_offsets = list(x_offsets) if x_offsets else [0] * len(segments)
        self.acc = 1
        self.use_x = False
        self.configure(*default_config(k, config_widths or self.widths))
        if _DMV_X:
            self._enable_x()

    def _enable_x(self):
        if (
            self.tp != 1
            or len(self.widths) > 4
            or len(set(self.formats)) != 1
            or self.formats[0] not in _X_FORMATS
            or not _x_ready()
        ):
            return
        override = _X_CONFIG.get(self.role)
        if override is not None:
            warps, split, acc = override
            self.configure(warps, 1, split)
            self.acc = acc
        self.use_x = True

    def configure(self, warps, tp, split):
        device = self.codes[0].device
        self.warps, self.tp, self.split = warps, tp, split
        tiles = sum((w + 32 * tp - 1) // (32 * tp) for w in self.widths)
        self.workspace = torch.zeros(
            tiles * split * tp * 256, device=device, dtype=torch.float32
        )
        self.counters = torch.zeros(tiles, device=device, dtype=torch.int32)

    def __call__(self, x, out=None):
        # Separate contiguous outputs (callers feed them to contiguous ops).
        views = (
            out
            if out is not None
            else [x.new_empty((x.shape[0], width)) for width in self.widths]
        )
        if self.use_x:
            torch.ops.v41k.dmv13_x(
                x if x.stride(-1) == 1 else x.contiguous(),
                self.codes,
                self.high,
                self.scale,
                views,
                self.x_offsets,
                self.formats[0],
                self.k,
                self.split,
                self.warps,
                self.acc,
                self.workspace,
                self.counters,
                0,
            )
            return views
        torch.ops._C.sm70_dmv13_out(
            x if x.stride(-1) == 1 else x.contiguous(),
            self.codes,
            self.high,
            self.scale,
            views,
            self.formats,
            self.widths,
            self.k,
            self.split,
            self.warps,
            self.workspace,
            self.counters,
            self.tp,
            None,
            None,
        )
        return views


def attach_lm_head(lm_head) -> bool:
    """dmv13 small-M logits from CPU-packed planes (no device packing peak).

    The original GGUF head stays for any M>8 call; dmv13 serves verify/draft.
    """
    if not ENABLED or getattr(lm_head, "v41_small_m", None) is not None:
        return False
    from vllm.model_executor.layers.quantization import sm70_dmv13_projection as d13

    projection = d13.Dmv13Projection(lm_head)
    if not projection.ready or len(projection.codes) != 1:
        return False
    lm_head.v41_small_m = SmallM(
        [
            (
                (projection.codes[0], projection.high[0], projection.scale[0]),
                projection.formats[0],
                projection.widths[0],
            )
        ],
        projection.k,
        role="head",
    )
    lm_head._v41_dmv13_projection = projection  # keep plane storage alive
    # Logits are only ever requested for a few rows; larger M runs in 8-row
    # dmv13 chunks, so the original head bytes can be released.
    lm_head.v41_small_m_chunked = True
    q = lm_head.qweight
    from vllm.model_executor.utils import set_weight_attrs

    empty = torch.nn.Parameter(
        torch.empty(0, dtype=torch.uint8, device=q.device), False
    )
    set_weight_attrs(empty, vars(q))
    lm_head.register_parameter("qweight", empty)
    torch.accelerator.empty_cache()
    return True


def prepare(model, extra_groups=()) -> dict:
    """Convert and attach small-M runners; returns a short admission report."""
    report = dict(enabled=ENABLED, converted=0, fused_groups=0, skipped=[])
    if not ENABLED:
        return report
    from vllm.model_executor.layers.quantization import gguf_dense_hmma

    extra = tuple(tuple(g) for g in extra_groups)
    try:  # diagnostics: where the shared expert down weights live
        d = model.layers[0].ffn.shared_experts.down_proj
        names = {
            n: (tuple(t.shape), str(t.dtype))
            for n, t in list(d.named_parameters()) + list(d.named_buffers())
        }
        attrs = [a for a in vars(d) if "gguf" in a or "weight" in a or "v41" in a]
        print(
            "V41 shared down:", type(d.quant_method).__name__, names, attrs, flush=True
        )
    except Exception as error:  # noqa: BLE001
        print("V41 shared down diag failed", error)
    gguf_dense_hmma.EXTRA_SEGMENT_SHAPES.update(_segment_shapes(model, extra))
    for role, group in _roles(model, extra):
        if not all(
            _convert(m) or getattr(m, "gguf_tm_projections", None) for m in group
        ):
            report["skipped"].append(getattr(group[0], "prefix", "?"))  # type: ignore[attr-defined]
            continue
        report["converted"] += len(group)  # type: ignore[operator]
        projections = [m.gguf_tm_projections[0] for m in group]
        k = projections[0].kernel.config.partition_weight_shape[0]
        if k % 128 or any(
            p.kernel.config.partition_weight_shape[0] != k
            or p.logical_output_size % 32
            or p.output_padding
            for p in projections
        ):
            report["skipped"].append(getattr(group[0], "prefix", "?") + ":shape")  # type: ignore[attr-defined]
            continue
        segments = [
            (
                _planes(p, 0, p.logical_output_size // 32),
                p.segment_format,
                p.logical_output_size,
            )
            for p in projections
        ]
        group[0].v41_small_m = SmallM(segments, k, role=role)
        if len(group) > 1:
            report["fused_groups"] += 1  # type: ignore[operator]
            for m in group[1:]:
                m.v41_small_m_peer = group[0]
    # Block-diagonal wo_a: one runner per O group over its own 1024 rows.
    for block in model.layers:
        wo_a = block.attn.wo_a
        if getattr(wo_a, "gguf_tm_projections", None):
            p = wo_a.gguf_tm_projections[0]
            k = p.kernel.config.partition_weight_shape[0]
            groups = p.logical_output_size // 1024
            wo_a.v41_group_runners = [
                SmallM(
                    [(_planes(p, g * 32, (g + 1) * 32), p.segment_format, 1024)],
                    k,
                    role="wo_a",
                )
                for g in range(groups)
            ]
            if _DMV_X and 1 < groups <= 4:
                # One block-diagonal launch: group g reads input columns
                # [g * k, (g + 1) * k) and writes its own 1024 rows, with the
                # per-group launch config (same reduction order).
                merged = SmallM(
                    [
                        (_planes(p, g * 32, (g + 1) * 32), p.segment_format, 1024)
                        for g in range(groups)
                    ],
                    k,
                    role="wo_a",
                    config_widths=[1024],
                    x_offsets=[g * k for g in range(groups)],
                )
                if merged.use_x:
                    wo_a.v41_group_merged = merged
    # Prefill restores a whole projection into the shared workspace; size it
    # for the largest converted V4.1 shape (e.g. Engram wkv K6144/N6400).
    need_w = need_s = 0
    # The LM head only ever sees a few sampled rows (never the M>64 restore).
    for group in _targets(model):
        for m in group:
            projs = getattr(m, "gguf_tm_projections", None)
            if not projs or not hasattr(projs[0], "segment_format"):
                continue
            p = projs[0]
            k, n = p.kernel.config.partition_weight_shape
            fmt = p.segment_format
            need_w = max(need_w, k * n // (4 if fmt == 4 else 8) * 4)
            group_size = 16 if fmt == 2 else 32
            width = 2 if fmt == 3 else 8 if fmt in (1, 2) else 4
            need_s = max(need_s, k // group_size * n * width)
    gguf_dense_hmma.MIN_WORKSPACE_BYTES.update(weight=need_w, stats=need_s)
    gguf_dense_hmma._workspaces.clear()
    report["restore_workspace_mb"] = round((need_w + need_s) / 2**20, 1)
    torch.accelerator.empty_cache()
    logger.info("V4.1 TurboMind dense: %s", report)
    return report


def small_m(module, x):
    """Output of `module` for M<=8 via its dmv13 runner, or None."""
    if not 1 <= x.shape[0] <= MAX_M:
        return None
    runner = getattr(module, "v41_small_m", None)
    if runner is not None:
        return runner(x)[0]
    return None


def fused(first, x):
    """All same-input outputs of a fused group (e.g. wq_a+wkv), or None."""
    runner = getattr(first, "v41_small_m", None)
    if runner is None or not 1 <= x.shape[0] <= MAX_M or len(runner.widths) < 2:
        return None
    return runner(x)


def grouped_wo_a(wo_a, output):
    runners = getattr(wo_a, "v41_group_runners", None)
    if runners is None or not 1 <= output.shape[0] <= MAX_M:
        return None
    merged = getattr(wo_a, "v41_group_merged", None)
    if merged is not None and output.shape[1] == len(merged.widths):
        # DMV_X: every group in one launch over the [tokens, groups * k] rows.
        width = merged.widths[0]
        result = output.new_empty((output.shape[0], width * len(merged.widths)))
        merged(
            output.reshape(output.shape[0], -1),
            out=[
                result[:, g * width : (g + 1) * width]
                for g in range(len(merged.widths))
            ],
        )
        return result
    if not _GROUP_VIEWS:
        return torch.cat(
            [r(output[:, g].contiguous())[0] for g, r in enumerate(runners)], dim=-1
        )
    # dmv13 reads rows at x.stride(0) and writes rows at out.stride(0): feed
    # each group's strided slice and write into its columns of one buffer.
    width = runners[0].widths[0]
    result = output.new_empty((output.shape[0], width * len(runners)))
    for g, r in enumerate(runners):
        r(output[:, g], out=[result[:, g * width : (g + 1) * width]])
    return result
