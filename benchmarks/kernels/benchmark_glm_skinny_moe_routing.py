# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph microbenchmark of GLM-5.3-Flash's skinny NVFP4 MoE on real layers.

One PP stage's MoE layers (real checkpoint weights, real prepack path) run
back to back in ONE CUDA graph, each layer with the routing a verify round
(1 anchor + 4 draft tokens = 5 rows x top-8) really produces: the number of
distinct experts per round is drawn from per-layer histograms recorded while
serving (``--histograms``: one ``rank<stage>.json`` per PP stage, holding
``{"layers": {<module name with "layers.N.">: [counts for 1 row, 2 rows, ...]}}``
with at least five entries (five rows are read), each list counting decode
steps by the number of distinct
experts (the list index)). Weights of a layer are
GBs, far larger than L2, and the layers alternate, so every layer streams its
experts from DRAM like the serving decode does.

Production path used (nothing rebuilt by hand except where stated):
- total mode: ``Nvfp4SkinnySm70Experts._apply_grouped``
  (vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py, the
  function body that runs ``group_routing``, the W13 ``_grouped_projection``,
  ``self.activation``, the W2 ``_grouped_projection`` and
  ``_mtp_weighted_reduce``), called on a harness subclass that only skips
  ``__init__`` (no vLLM layer is built).
- split mode: the same calls in the same order, one CUDA graph per
  (layer, phase) so the time of route / up / act / down / reduce is read from
  events. The split output is checked byte for byte against the total mode;
  a mismatch aborts, which ties this orchestration to the production body.
- prepack: ``qpn_prepack`` (skinny_sm70_moe.py), csf scales decoded by the
  reference ``decode_csf32`` (csf32_scales.py; the production loader runs the
  GPU expander ``csf32_expand`` for the same bytes), then the production
  decision of ``_csf_to_packed``: the smallest width ``scale_bits_for`` gives
  for the projection's ``scale_spread`` -> five- or six-bit planes
  (``encode_scales``, raster dropped), else the raster is served as is.
  Not measured here: the per-call ``csf32_expand`` of csf projections.

Run on a GPU that nothing else uses (the weights of seven layers take about
27 GB). CPU only: ``--selftest`` (checks the routing generator against the
histograms, needs no GPU and no weights).

Example (stage 1 runs on a V100, stages 0 and 4 on a RTX 8000):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python \\
      benchmarks/kernels/benchmark_glm_skinny_moe_routing.py --stage 1 \\
      --checkpoint <checkpoint dir> --histograms <histogram dir> \\
      --layers 7 --mode dist --iters 200 --timing both
"""

import argparse
import json
import statistics
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import regex as re
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    Nvfp4SkinnySm70Experts,
    QpnFragmentOrder,
    _ScaleCaches,
    grouped_splitk,
    nvfp4_skinny_scale_reason,
    qpn_fragment_order,
    qpn_prepack,
)
from vllm.model_executor.layers.fused_moe.sm70.reduction import _mtp_weighted_reduce
from vllm.model_executor.layers.fused_moe.sm70.routing import group_routing
from vllm.model_executor.layers.quantization.utils.csf32_scales import decode_csf32
from vllm.model_executor.layers.quantization.utils.s5_scales import (
    PackedScales,
    encode_scales,
    scale_bits_for,
    scale_spread,
)

SWEEP_DISTINCT = (16, 20, 24, 28, 32, 36, 40)
ROWS = 5  # verify round: anchor + 4 draft tokens (cudagraph capture size 5)
TOP_K = 8
SLOTS = ROWS * TOP_K
PHASES = ("route", "up", "act", "down", "reduce")
CSF_PARTS = ("bases", "codes", "page_bases", "offsets", "exceptions")
# Zipf exponent of the repeat weights; the self-test reports the resulting
# number of rows the busiest expert sits in (the real verify step: 3 to 5).
DEFAULT_ZIPF = 1.5
LAYER_NAME = re.compile(r"layers\.(\d+)\.")


# ---------------------------------------------------------------- routing ----


def load_histograms(directory: Path, stage: int) -> dict[int, np.ndarray]:
    """layer index -> counts of steps with d distinct experts at 5 rows."""
    layers = json.loads((directory / f"rank{stage}.json").read_text())["layers"]
    histograms: dict[int, np.ndarray] = {}
    for name, per_rows in layers.items():
        match = LAYER_NAME.search(name)
        assert match is not None, name
        counts = np.asarray(per_rows[ROWS - 1], dtype=np.float64)
        if counts.sum() > 0:
            histograms[int(match.group(1))] = counts
    return dict(sorted(histograms.items()))


def draw_distinct(rng: np.random.Generator, histogram: np.ndarray) -> int:
    return int(rng.choice(histogram.size, p=histogram / histogram.sum()))


def make_topk_ids(
    rng: np.random.Generator, distinct: int, num_experts: int, zipf: float
) -> np.ndarray:
    """int32 [5, 8] routing with exactly ``distinct`` different experts.

    Draw the experts, give each one random row
    first (at most 8 per row), fill the remaining 40 - d places per row with
    experts the row does not hold yet, Zipf-weighted over a random expert
    ranking, so the leading experts recur in several rows like the anchor and
    draft tokens of a real verify step. Every row ends with eight different
    experts.
    """
    if not TOP_K <= distinct <= SLOTS:
        raise ValueError(f"{distinct} distinct experts cannot fill {ROWS}x{TOP_K}")
    experts = rng.choice(num_experts, size=distinct, replace=False)
    rows: list[list[int]] = [[] for _ in range(ROWS)]
    for expert in experts:
        open_rows = [r for r in range(ROWS) if len(rows[r]) < TOP_K]
        rows[int(rng.choice(open_rows))].append(int(expert))
    weights = 1.0 / np.arange(1, distinct + 1, dtype=np.float64) ** zipf
    for row in rows:
        need = TOP_K - len(row)
        if need:
            candidates = [i for i in range(distinct) if int(experts[i]) not in row]
            probs = weights[candidates] / weights[candidates].sum()
            picked = rng.choice(len(candidates), size=need, replace=False, p=probs)
            row.extend(int(experts[candidates[i]]) for i in picked)
        rng.shuffle(row)
    return np.asarray(rows, dtype=np.int32)


def selftest(directory: Path, samples: int, zipf: float, seed: int) -> None:
    """The generator against the histograms, per layer of every stage found."""
    rng = np.random.default_rng(seed)
    failures = 0
    for path in sorted(directory.glob("rank*.json")):
        stage = int(re.search(r"rank(\d+)", path.name).group(1))  # type: ignore[union-attr]
        for layer, histogram in load_histograms(directory, stage).items():
            probs = histogram / histogram.sum()
            observed = np.zeros(histogram.size)
            top_rows, repeat_mean = [], []
            for _ in range(samples):
                ids = make_topk_ids(rng, draw_distinct(rng, histogram), 288, zipf)
                assert ids.shape == (ROWS, TOP_K) and ids.min() >= 0
                assert ids.max() < 288 and ids.size == SLOTS
                assert all(len(set(row.tolist())) == TOP_K for row in ids)
                values, counts = np.unique(ids, return_counts=True)
                observed[values.size] += 1
                top_rows.append(counts.max())
                repeat_mean.append(SLOTS / values.size)
            # d bins with an expected count >= 10: |observed - expected| within
            # 4.5 standard deviations of a binomial; rarer bins are pooled.
            expected = probs * samples
            sigma = np.sqrt(expected * (1 - probs))
            checked = expected >= 10
            z = np.abs(observed - expected)[checked] / sigma[checked]
            rare_expected = expected[~checked].sum()
            rare_observed = observed[~checked].sum()
            rare_ok = abs(rare_observed - rare_expected) <= 4.5 * np.sqrt(
                max(rare_expected, 1.0)
            )
            mean_d = float((np.arange(histogram.size) * probs).sum())
            mean_obs = float((np.arange(histogram.size) * observed).sum() / samples)
            ok = bool((z < 4.5).all()) and bool(rare_ok)
            failures += not ok
            print(
                f"stage {stage} layer {layer:2d}: mean d histogram {mean_d:5.2f} "
                f"generated {mean_obs:5.2f}, max z {z.max():4.2f}, rare bins "
                f"{rare_observed:.0f}/{rare_expected:.1f}, top expert in "
                f"{np.mean(top_rows):.2f} rows (max {max(top_rows)}), places per "
                f"expert {np.mean(repeat_mean):.2f} -> {'ok' if ok else 'FAIL'}"
            )
    if failures:
        raise SystemExit(f"selftest: {failures} layers deviate from their histogram")
    print("selftest ok")


# ---------------------------------------------------------------- weights ----


class Checkpoint:
    def __init__(self, directory: Path):
        self.directory = directory
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.config = json.loads((directory / "config.json").read_text())
        self._open: dict[str, object] = {}

    def tensor(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map[key]
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.directory / shard), "pt")
        return self._open[shard].get_tensor(key)  # type: ignore[attr-defined]

    def expert(self, layer: int, expert: int, name: str) -> dict[str, torch.Tensor]:
        prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{name}"
        parts = {
            part: self.tensor(f"{prefix}.weight_scale.csf32_{part}")
            for part in CSF_PARTS
        }
        parts["weight"] = self.tensor(f"{prefix}.weight")
        parts["global_scale"] = self.tensor(f"{prefix}.weight_scale_2")
        return parts


def decode_scales(parts: dict[str, torch.Tensor], k: int) -> torch.Tensor:
    rows = parts["bases"].numel()
    return decode_csf32(
        parts["bases"],
        parts["codes"],
        parts["page_bases"],
        parts["offsets"],
        parts["exceptions"],
        rows,
        k // 16,
    )


@dataclass
class Layer:
    index: int
    w1: torch.Tensor
    w2: torch.Tensor
    harness: Nvfp4SkinnySm70Experts
    x: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor  # static input of the graphs
    output: torch.Tensor
    packed: tuple[PackedScales | None, PackedScales | None]
    scales: _ScaleCaches

    def projection_bytes(self) -> tuple[int, int]:
        """Bytes one expert streams for W13 and W2 (codes plus scale planes)."""
        experts = self.w1.size(0)
        result = []
        for w, packed, raster in (
            (self.w1, self.packed[0], self.scales.w1_scales_u8),
            (self.w2, self.packed[1], self.scales.w2_scales_u8),
        ):
            codes = w[0].numel()
            if packed is None:
                scale_bytes = raster[0].numel()
            else:
                scale_bytes = (
                    packed.base[0].numel()
                    + packed.nib[0].numel()
                    + 4 * packed.hi[0].numel()
                )
            result.append(codes + scale_bytes)
            assert w.size(0) == experts
        return result[0], result[1]


class _Harness(Nvfp4SkinnySm70Experts):
    """Production ``_apply_grouped`` and its helpers without a vLLM layer.

    Only the attributes those methods read are set; ``__init__`` of the real
    class needs a FusedMoE config and quant config the benchmark has no use for.
    """

    def __init__(  # noqa: D107
        self,
        order: QpnFragmentOrder,
        scales: _ScaleCaches,
        packed: tuple[PackedScales | None, PackedScales | None],
        swiglu_limit: float,
    ):
        self._fragment_order = order
        self._triton_tensor_cores = False
        self._scale_mode = 0
        self._scale_caches = scales
        self._csf_planes = None
        self._csf_scale_entries = None
        self._packed_scales = packed
        # TritonExperts.activation reads the clamp limit from here.
        self.quant_config = SimpleNamespace(gemm1_clamp_limit=swiglu_limit)  # type: ignore[assignment]


def build_layer(
    ckpt: Checkpoint,
    layer: int,
    device: torch.device,
    order: QpnFragmentOrder,
    force_raster: bool,
    generator: torch.Generator,
) -> Layer:
    text = ckpt.config.get("text_config", ckpt.config)
    num_experts, hidden = text["n_routed_experts"], text["hidden_size"]
    inter = text["moe_intermediate_size"]
    w1 = torch.empty(
        num_experts, 2 * inter, hidden // 2, dtype=torch.uint8, device=device
    )
    w2 = torch.empty(num_experts, hidden, inter // 2, dtype=torch.uint8, device=device)
    tiles13, tiles2 = 2 * inter // 32, hidden // 32
    s13 = torch.empty(
        num_experts, tiles13 * (hidden // 16) * 32, dtype=torch.uint8, device=device
    )
    s2 = torch.empty(
        num_experts, tiles2 * (inter // 16) * 32, dtype=torch.uint8, device=device
    )
    g1 = torch.empty(num_experts, 2, dtype=torch.float32)
    g2 = torch.empty(num_experts, dtype=torch.float32)
    for e in range(num_experts):
        gate, up, down = (
            ckpt.expert(layer, e, name)
            for name in ("gate_proj", "up_proj", "down_proj")
        )
        codes13 = torch.cat([gate["weight"], up["weight"]]).to(device)
        scales13 = torch.cat(
            [decode_scales(gate, hidden), decode_scales(up, hidden)]
        ).to(device)
        qc, qs = qpn_prepack(codes13, scales13, 16, order)
        w1[e].view(-1).copy_(qc)
        s13[e].copy_(qs)  # type: ignore[arg-type]
        qc, qs = qpn_prepack(
            down["weight"].to(device), decode_scales(down, inter).to(device), 16, order
        )
        w2[e].view(-1).copy_(qc)
        s2[e].copy_(qs)  # type: ignore[arg-type]
        g1[e, 0], g1[e, 1] = gate["global_scale"], up["global_scale"]
        g2[e] = down["global_scale"]
    reason = nvfp4_skinny_scale_reason(g1, g2)
    if reason is not None:
        raise RuntimeError(f"layer {layer}: {reason}")
    placeholder = s13.new_empty(0)
    packed_sets: list[PackedScales | None] = []
    rasters = []
    for raster, groups in ((s13, hidden // 16), (s2, inter // 16)):
        bits = None if force_raster else scale_bits_for(scale_spread(raster, groups))
        packed_sets.append(
            None if bits is None else encode_scales(raster, groups, bits)
        )
        rasters.append(raster if bits is None else placeholder)
    scales = _ScaleCaches(
        g1=[],
        g2=[],
        g1_t=g1[:, 0].contiguous().to(device),
        g2_t=g2.to(device),
        w1_scales_u8=rasters[0],
        w2_scales_u8=rasters[1],
    )
    packed = (packed_sets[0], packed_sets[1])
    harness = _Harness(order, scales, packed, float(text["swiglu_limit"]))
    x = (torch.randn(ROWS, hidden, generator=generator, dtype=torch.float32) * 0.25).to(
        device=device, dtype=torch.float16
    )
    weights = torch.rand(ROWS, TOP_K, generator=generator)
    weights = (weights / weights.sum(dim=1, keepdim=True)).to(device)
    return Layer(
        index=layer,
        w1=w1,
        w2=w2,
        harness=harness,
        x=x,
        topk_weights=weights,
        topk_ids=torch.zeros(ROWS, TOP_K, dtype=torch.int32, device=device),
        output=torch.empty(ROWS, hidden, dtype=torch.float16, device=device),
        packed=packed,
        scales=scales,
    )


# ------------------------------------------------------------- the chains ----


def apply_total(layer: Layer) -> None:
    """Production path of one layer: ``_apply_grouped``."""
    layer.harness._apply_grouped(
        layer.output,
        layer.x,
        layer.w1,
        layer.w2,
        layer.topk_weights,
        layer.topk_ids,
        MoEActivation.SILU,
        None,
        False,
    )


class SplitLayer:
    """The body of ``_apply_grouped`` cut into the phases of ``PHASES``.

    Same calls, same order, same buffers sizes as skinny_sm70_moe.py
    ``_apply_grouped`` (no csf expander: raster layers read the raster as is).
    """

    def __init__(self, layer: Layer):
        self.layer = layer
        h = layer.harness
        self.top_k = TOP_K
        self.num_slots = layer.topk_ids.numel()
        self.inter_dim = h.adjust_N_for_activation(layer.w1.size(1), MoEActivation.SILU)
        self.routing: tuple[torch.Tensor, ...] = ()
        self.y13 = torch.empty(
            self.num_slots, layer.w1.size(1), dtype=torch.float16, device=layer.x.device
        )
        self.inter = torch.empty(
            self.num_slots, self.inter_dim, dtype=torch.float16, device=layer.x.device
        )
        self.y2 = torch.empty(
            self.num_slots, layer.w2.size(1), dtype=torch.float16, device=layer.x.device
        )
        self.output = torch.empty_like(layer.output)

    def route(self) -> None:
        self.routing = group_routing(self.layer.topk_ids)

    def up(self) -> None:
        layer = self.layer
        perm, gids, goff = self.routing
        layer.harness._grouped_projection(
            layer.x,
            layer.w1,
            layer.scales.w1_scales_u8,
            layer.scales.g1_t,
            perm,
            gids,
            goff,
            self.top_k,
            self.y13,
            False,
            ROWS,
            layer.packed[0],
        )

    def act(self) -> None:
        self.layer.harness.activation(MoEActivation.SILU, self.inter, self.y13)

    def down(self) -> None:
        layer = self.layer
        perm, gids, goff = self.routing
        layer.harness._grouped_projection(
            self.inter,
            layer.w2,
            layer.scales.w2_scales_u8,
            layer.scales.g2_t,
            perm,
            gids,
            goff,
            self.top_k,
            self.y2,
            True,
            ROWS,
            layer.packed[1],
        )

    def reduce(self) -> None:
        _mtp_weighted_reduce(self.y2, self.layer.topk_weights, self.output)


def capture(fn) -> torch.cuda.CUDAGraph:
    """Warm up ``fn`` (compiles Triton kernels), then capture it."""
    for _ in range(2):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def set_routing(
    layers: Sequence[Layer], routing: Sequence[np.ndarray], device: torch.device
) -> None:
    for layer, ids in zip(layers, routing):
        layer.topk_ids.copy_(torch.from_numpy(ids).to(device))
    torch.accelerator.synchronize()


class Draws:
    """Routing draws of one run: ids [draw][layer] and the distinct counts."""

    def __init__(
        self,
        mode: str,
        layers: Sequence[Layer],
        histograms: dict[int, np.ndarray],
        count: int,
        rng: np.random.Generator,
        zipf: float,
        num_experts: int,
    ):
        self.ids: list[list[np.ndarray]] = []
        for draw in range(count):
            row = []
            for layer in layers:
                if mode == "dist":
                    distinct = draw_distinct(rng, histograms[layer.index])
                    distinct = max(distinct, TOP_K)
                else:
                    distinct = SWEEP_DISTINCT[draw % len(SWEEP_DISTINCT)]
                row.append(make_topk_ids(rng, distinct, num_experts, zipf))
            self.ids.append(row)
        self.distinct = [[np.unique(ids).size for ids in row] for row in self.ids]


def memory_used() -> str:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return "; ".join(f"GPU {line.replace(', ', ': ')} MiB" for line in out.splitlines())


def time_total(
    graph: torch.cuda.CUDAGraph,
    layers: Sequence[Layer],
    draws: Draws,
    device: torch.device,
) -> list[float]:
    """Graph time in us per draw."""
    for _ in range(3):
        graph.replay()
    events = []
    for ids in draws.ids:
        set_routing(layers, ids, device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(True)
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    torch.accelerator.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in events]


def time_split(
    graphs: Sequence[dict[str, torch.cuda.CUDAGraph]],
    layers: Sequence[Layer],
    draws: Draws,
    device: torch.device,
) -> list[list[dict[str, float]]]:
    """us per [draw][layer][phase]; the events enclose one graph launch each."""
    for per_layer in graphs:
        for phase in PHASES:
            per_layer[phase].replay()
    result = []
    for ids in draws.ids:
        set_routing(layers, ids, device)
        events = []
        for per_layer in graphs:
            marks = {}
            for phase in PHASES:
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                per_layer[phase].replay()
                end.record()
                marks[phase] = (start, end)
            events.append(marks)
        torch.accelerator.synchronize()
        result.append(
            [
                {p: s.elapsed_time(e) * 1000.0 for p, (s, e) in marks.items()}
                for marks in events
            ]
        )
    return result


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(values: Sequence[float]) -> str:
    return (
        f"mean {statistics.fmean(values):8.1f}  median {statistics.median(values):8.1f}"
        f"  p10 {quantile(values, 0.1):8.1f}  p90 {quantile(values, 0.9):8.1f}"
    )


def regression(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.asarray(x), np.asarray(y), 1)
    return float(slope), float(intercept)


def report_total(
    times: list[float], draws: Draws, layers: Sequence[Layer], mode: str
) -> None:
    per_layer = [t / len(layers) for t in times]
    bytes13 = [sum(layer.projection_bytes()) for layer in layers]
    print(
        f"\n[total] {len(layers)} layers in one graph, {len(times)} draws, us per layer"
    )
    print(f"  {summarize(per_layer)}")
    expert_bytes = [sum(d * b for d, b in zip(row, bytes13)) for row in draws.distinct]
    rate = [bytes_ / 1e3 / t for bytes_, t in zip(expert_bytes, times)]
    print(
        f"  GB/s (distinct experts x bytes per expert / graph time): {summarize(rate)}"
    )
    ratio = sum(expert_bytes) / 1e3 / sum(times)
    print(f"  GB/s ratio of sums: {ratio:.1f}")
    mean_d = statistics.fmean(statistics.fmean(row) for row in draws.distinct)
    print(f"  mean distinct experts per layer: {mean_d:.2f}")
    if mode == "sweep":
        for distinct in SWEEP_DISTINCT:
            sel = [
                t / len(layers)
                for t, row in zip(times, draws.distinct)
                if statistics.fmean(row) == distinct
            ]
            if sel:
                print(f"  d={distinct:2d}: {summarize(sel)} us per layer")
    slope, intercept = regression(
        [statistics.fmean(row) for row in draws.distinct], per_layer
    )
    print(
        f"  fit us per layer = {intercept:.1f} + {slope:.2f} * d "
        f"(slope = {sum(bytes13) / len(bytes13) / 1e3 / slope:.0f} GB/s limit rate)"
    )


def report_split(
    split: list[list[dict[str, float]]], draws: Draws, layers: Sequence[Layer]
) -> None:
    print("\n[split] us per layer and phase (one graph per layer and phase)")
    w13 = statistics.fmean(layer.projection_bytes()[0] for layer in layers)
    w2 = statistics.fmean(layer.projection_bytes()[1] for layer in layers)
    for phase in PHASES:
        values = [m[phase] for draw in split for m in draw]
        print(f"  {phase:7s} {summarize(values)}")
    for phase, per_expert in (("up", w13), ("down", w2)):
        rate = [
            d * per_expert / 1e3 / m[phase]
            for draw, dist in zip(split, draws.distinct)
            for m, d in zip(draw, dist)
        ]
        print(
            f"  {phase} GB/s ({per_expert / 1e6:.3f} MB per expert): {summarize(rate)}"
        )
    total = [sum(m.values()) for draw in split for m in draw]
    print(f"  sum of phases per layer {summarize(total)}")
    print("  per layer (mean us): index d_mean " + " ".join(PHASES))
    for i, layer in enumerate(layers):
        cells = [statistics.fmean(draw[i][p] for draw in split) for p in PHASES]
        d_mean = statistics.fmean(row[i] for row in draws.distinct)
        print(
            f"   layer {layer.index:2d} d {d_mean:5.1f}: "
            + " ".join(f"{c:8.1f}" for c in cells)
        )


# --------------------------------------------------------- dump / compare ----


def bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def run_reference(
    layers: Sequence[Layer],
    splits: Sequence[SplitLayer],
    ids_per_draw: Sequence[Sequence[np.ndarray]],
    device: torch.device,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Eager (y_slots, output) per [draw][layer] of the split orchestration."""
    result = []
    for ids in ids_per_draw:
        set_routing(layers, ids, device)
        row = []
        for split in splits:
            for phase in PHASES:
                getattr(split, phase)()
            row.append((split.y2.clone().cpu(), split.output.clone().cpu()))
        result.append(row)
    return result


def check_split_equals_total(
    layers: Sequence[Layer],
    splits: Sequence[SplitLayer],
    ids: Sequence[np.ndarray],
    device: torch.device,
) -> None:
    set_routing(layers, ids, device)
    for layer, split in zip(layers, splits):
        apply_total(layer)
        for phase in PHASES:
            getattr(split, phase)()
        torch.accelerator.synchronize()
        if not bytes_equal(layer.output, split.output):
            raise RuntimeError(
                f"layer {layer.index}: split phases differ from _apply_grouped"
            )
    print("split orchestration == _apply_grouped byte for byte (1 draw, all layers)")


def dump_output(path: Path, layers, splits, ids_per_draw, device) -> None:
    results = run_reference(layers, splits, ids_per_draw, device)
    payload = {
        "layers": [layer.index for layer in layers],
        "topk_ids": [[ids.copy() for ids in row] for row in ids_per_draw],
        "x": [layer.x.cpu() for layer in layers],
        "topk_weights": [layer.topk_weights.cpu() for layer in layers],
        "y_slots": [[y for y, _ in row] for row in results],
        "output": [[out for _, out in row] for row in results],
    }
    torch.save(payload, path)
    print(f"dumped {len(ids_per_draw)} draws x {len(layers)} layers to {path}")


def compare_output(path: Path, layers, splits, device) -> bool:
    ref = torch.load(path, weights_only=False)
    if ref["layers"] != [layer.index for layer in layers]:
        raise SystemExit(f"{path} holds layers {ref['layers']}")
    for layer, x, w in zip(layers, ref["x"], ref["topk_weights"]):
        if not (
            bytes_equal(layer.x.cpu(), x) and bytes_equal(layer.topk_weights.cpu(), w)
        ):
            raise SystemExit(
                "inputs differ from the dump: use the same --seed/--layer-ids"
            )
    results = run_reference(layers, splits, ref["topk_ids"], device)
    all_equal = True
    for i, layer in enumerate(layers):
        bad_slots = sum(
            not bytes_equal(results[d][i][0], ref["y_slots"][d][i])
            for d in range(len(results))
        )
        bad_out = sum(
            not bytes_equal(results[d][i][1], ref["output"][d][i])
            for d in range(len(results))
        )
        all_equal &= bad_slots == 0 and bad_out == 0
        print(
            f"  layer {layer.index}: y_slots differ in {bad_slots}/{len(results)} "
            f"draws, output in {bad_out}/{len(results)}"
        )
    print("compare: " + ("BYTE-IDENTICAL" if all_equal else "DIFFERENT"))
    return all_equal


# ------------------------------------------------------------------- main ----


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage", type=int, help="PP stage 0-4 (layers of rank<stage>.json)"
    )
    parser.add_argument("--layers", type=int, default=7, help="number of layers")
    parser.add_argument(
        "--layer-ids", type=int, nargs="+", help="explicit layer indices"
    )
    parser.add_argument("--mode", choices=("dist", "sweep"), default="dist")
    parser.add_argument("--timing", choices=("total", "split", "both"), default="both")
    parser.add_argument("--iters", type=int, default=200, help="routing draws timed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zipf", type=float, default=DEFAULT_ZIPF)
    parser.add_argument(
        "--force-raster",
        action="store_true",
        help="serve the raster even where five-bit planes fit",
    )
    parser.add_argument("--dump-output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument(
        "--dump-draws", type=int, default=16, help="draws stored/compared"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="GLM-5.3-Flash NVFP4 checkpoint directory"
    )
    parser.add_argument(
        "--histograms",
        type=Path,
        required=True,
        help="directory of the per-stage rank<stage>.json expert histograms",
    )
    parser.add_argument(
        "--selftest", action="store_true", help="CPU only: generator vs histograms"
    )
    parser.add_argument("--selftest-samples", type=int, default=3000)
    args = parser.parse_args()
    if not args.selftest and args.checkpoint is None:
        parser.error("--checkpoint is required unless --selftest is given")
    return args


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest(args.histograms, args.selftest_samples, args.zipf, args.seed)
        return
    if args.stage is None:
        raise SystemExit("--stage is required")
    histograms = load_histograms(args.histograms, args.stage)
    ids = args.layer_ids or list(histograms)[: args.layers]
    for index in ids:
        if index not in histograms:
            raise SystemExit(f"layer {index} is not on stage {args.stage}")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    order = qpn_fragment_order(device.index)
    ckpt = Checkpoint(args.checkpoint)
    num_experts = ckpt.config.get("text_config", ckpt.config)["n_routed_experts"]
    free, total = torch.cuda.mem_get_info(device)
    need = len(ids) * 4.0e9
    print(
        f"device {device.index} {torch.cuda.get_device_name(device)} order "
        f"{order.name}, layers {ids}, free {free / 1e9:.1f} of {total / 1e9:.1f} GB "
        f"(weights about {need / 1e9:.1f} GB)"
    )
    if free < need + 1.5e9:
        raise SystemExit("not enough free GPU memory for this many layers")
    print("memory.used before:", memory_used())

    generator = torch.Generator().manual_seed(args.seed)
    started = time.time()
    layers = []
    for index in ids:
        layers.append(
            build_layer(ckpt, index, device, order, args.force_raster, generator)
        )
        kinds = ["raster" if s is None else f"S{s.bits}" for s in layers[-1].packed]
        w13, w2 = layers[-1].projection_bytes()
        print(
            f"layer {index}: W13 {kinds[0]} {w13 / 1e6:.3f} MB/expert, W2 {kinds[1]} "
            f"{w2 / 1e6:.3f} MB/expert, splitk "
            f"{grouped_splitk(4096, order)}/{grouped_splitk(2048, order)} "
            f"({time.time() - started:.0f} s)"
        )
    torch.accelerator.synchronize()
    print("memory.used after load:", memory_used())

    rng = np.random.default_rng(args.seed)
    count = max(args.iters, len(SWEEP_DISTINCT) if args.mode == "sweep" else 1)
    draws = Draws(args.mode, layers, histograms, count, rng, args.zipf, num_experts)
    splits = [SplitLayer(layer) for layer in layers]
    set_routing(layers, draws.ids[0], device)
    check_split_equals_total(layers, splits, draws.ids[0], device)

    if args.dump_output or args.compare:
        rng_ref = np.random.default_rng(args.seed + 1)
        reference = Draws(
            "dist", layers, histograms, args.dump_draws, rng_ref, args.zipf, num_experts
        )
        if args.dump_output:
            dump_output(args.dump_output, layers, splits, reference.ids, device)
        if args.compare and not compare_output(args.compare, layers, splits, device):
            raise SystemExit(1)

    if args.timing in ("total", "both"):

        def whole_chain() -> None:
            for layer in layers:
                apply_total(layer)

        graph = capture(whole_chain)
        print("memory.used after total graph:", memory_used())
        report_total(time_total(graph, layers, draws, device), draws, layers, args.mode)
    if args.timing in ("split", "both"):
        graphs = []
        for split in splits:
            per_layer = {}
            for phase in PHASES:
                per_layer[phase] = capture(getattr(split, phase))
                # Later phases warm up on the routing this graph produces.
                per_layer[phase].replay()
                torch.accelerator.synchronize()
            graphs.append(per_layer)
        print("memory.used after split graphs:", memory_used())
        report_split(time_split(graphs, layers, draws, device), draws, layers)
    print("memory.used end:", memory_used())
    peak = torch.accelerator.max_memory_reserved(device) / 2**20
    print(f"peak reserved by this process: {peak:.0f} MiB")


if __name__ == "__main__":
    main()
