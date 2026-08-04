#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen router-first shared-expert stream scheduling on SM70.

The fixture uses the DeepSeek-V4-Flash batch-one gate, top-k, shared FP8, and
routed MXFP4 decode shapes. All schedules execute identical arithmetic and
only move the auxiliary-stream dependency from the MoE input to the completed
router output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from benchmark_sm70_mxfp4_moe_active_experts import STAGES, _prepare_experts

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)


@dataclass(frozen=True)
class Schedule:
    name: str
    overlap: bool
    router_first: bool


SCHEDULES = (
    Schedule("serial", False, False),
    Schedule("overlap_root_release", True, False),
    Schedule("overlap_router_first", True, True),
)


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _prepare_fp8(
    n: int,
    k: int,
    *,
    gated_silu: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = torch.randn((n, k), device="cuda", dtype=torch.float16).to(
        torch.float8_e4m3fn
    )
    scales = torch.ones(
        ((n + 127) // 128, (k + 127) // 128),
        device="cuda",
        dtype=torch.float32,
    )
    return tuple(sm70_ops.fp8_sm70_prepare(weight, scales, 128, gated_silu))


class Fixture:
    def __init__(self, seed: int):
        torch.manual_seed(seed)
        device = torch.device("cuda")
        self.x = torch.randn(1, 4096, dtype=torch.float16, device=device) * 0.01
        self.gate_weight = torch.randn(256, 4096, dtype=torch.float16, device=device)
        self.e_score_bias = torch.randn(256, dtype=torch.float32, device=device)

        self.shared_gate_up = _prepare_fp8(512, 4096, gated_silu=True)
        self.shared_down = _prepare_fp8(4096, 256, gated_silu=False)
        self.shared_intermediate = torch.empty(
            1, 256, dtype=torch.float16, device=device
        )
        self.shared_output = torch.empty(1, 4096, dtype=torch.float16, device=device)
        self.scaled_shared_output = torch.empty_like(self.shared_output)
        self.shared_scale = 1.0 / 1.5

        num_experts = 256
        top_k = 6
        w13_storage, w13_scales, w13_ptrs_w, w13_ptrs_s = _prepare_experts(
            STAGES["w13"], num_experts, device
        )
        w2_storage, w2_scales, w2_ptrs_w, w2_ptrs_s = _prepare_experts(
            STAGES["w2"], num_experts, device
        )
        self._routed_storage = (
            w13_storage,
            w13_scales,
            w2_storage,
            w2_scales,
        )
        self.w13_ptrs_w = w13_ptrs_w
        self.w13_ptrs_s = w13_ptrs_s
        self.w2_ptrs_w = w2_ptrs_w
        self.w2_ptrs_s = w2_ptrs_s
        self.routed_output = torch.empty(1, 4096, dtype=torch.float16, device=device)
        self.permuted_input = torch.empty(
            top_k, 4096, dtype=torch.float16, device=device
        )
        self.gate_up = torch.empty(top_k, 512, dtype=torch.float16, device=device)
        self.intermediate = torch.empty(top_k, 256, dtype=torch.float16, device=device)
        self.sorted_output = torch.empty(
            top_k, 4096, dtype=torch.float16, device=device
        )
        self.compact_offsets = torch.arange(top_k + 1, dtype=torch.int32, device=device)
        self.compact_offsets64 = self.compact_offsets.to(torch.int64)
        self.inv_permuted_idx = torch.empty(1, top_k, dtype=torch.int32, device=device)
        self.permuted_experts_id = torch.empty(top_k, dtype=torch.int32, device=device)
        self.combined_output = torch.empty_like(self.routed_output)

    def router_call(self) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = F.linear(self.x, self.gate_weight).float()
        return fused_topk_bias(
            hidden_states=self.x,
            gating_output=router_logits,
            scoring_func="sqrtsoftplus",
            e_score_correction_bias=self.e_score_bias,
            topk=6,
            renormalize=True,
            routed_scaling_factor=1.5,
        )

    def shared_call(self) -> None:
        gate_weight, gate_scales, gate_meta = self.shared_gate_up
        sm70_ops.fp8_gemm_sm70_out_meta(
            self.shared_intermediate,
            self.x,
            gate_weight,
            gate_scales,
            gate_meta,
            True,
        )
        down_weight, down_scales, down_meta = self.shared_down
        sm70_ops.fp8_gemm_sm70_out_meta(
            self.shared_output,
            self.shared_intermediate,
            down_weight,
            down_scales,
            down_meta,
            False,
        )

    def routed_call(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        sm70_ops.mxfp4_moe_single_token_prepare_w13_sm70_out(
            self.gate_up,
            self.permuted_input,
            self.x,
            topk_ids,
            self.w13_ptrs_w,
            self.w13_ptrs_s,
            self.compact_offsets,
            self.inv_permuted_idx,
            self.permuted_experts_id,
            4096,
            512,
            32,
            4096,
        )
        torch.ops._C.silu_and_mul_with_clamp(
            self.intermediate,
            self.gate_up,
            10.0,
        )
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            self.sorted_output,
            self.intermediate,
            self.compact_offsets,
            self.permuted_experts_id,
            self.w2_ptrs_w,
            self.w2_ptrs_s,
            6,
            256,
            4096,
            32,
        )
        torch.ops._moe_C.moe_unpermute(
            self.sorted_output,
            topk_weights,
            self.inv_permuted_idx,
            self.compact_offsets64,
            6,
            self.routed_output,
        )

    def combine(self) -> None:
        torch.mul(
            self.shared_output,
            self.shared_scale,
            out=self.scaled_shared_output,
        )
        torch.add(
            self.scaled_shared_output,
            self.routed_output,
            out=self.combined_output,
        )


def _make_body(
    fixture: Fixture,
    schedule: Schedule,
    main: torch.cuda.Stream,
    auxiliary: torch.cuda.Stream,
) -> Callable[[], None]:
    def body() -> None:
        if not schedule.overlap:
            topk_weights, topk_ids = fixture.router_call()
            fixture.shared_call()
            fixture.routed_call(topk_weights, topk_ids)
            fixture.combine()
            return

        if not schedule.router_first:
            auxiliary.wait_stream(main)
        topk_weights, topk_ids = fixture.router_call()
        if schedule.router_first:
            auxiliary.wait_stream(main)
        fixture.routed_call(topk_weights, topk_ids)
        with torch.cuda.stream(auxiliary):
            fixture.shared_call()
        main.wait_stream(auxiliary)
        fixture.combine()

    return body


def _capture(
    fixture: Fixture,
    schedule: Schedule,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]:
    parent = torch.cuda.current_stream()
    main = torch.cuda.Stream()
    auxiliary = torch.cuda.Stream()
    main.wait_stream(parent)
    auxiliary.wait_stream(parent)
    body = _make_body(fixture, schedule, main, auxiliary)
    with torch.cuda.stream(main):
        for _ in range(4):
            body()
    main.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=main):
        body()
    graph.replay()
    main.synchronize()
    return graph, main


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    replays: int,
    repeats: int,
) -> list[float]:
    for _ in range(10):
        graph.replay()
    stream.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / replays)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires an NVIDIA V100/SM70 GPU.")

    fixture = Fixture(args.seed)
    results: list[dict[str, object]] = []
    reference: torch.Tensor | None = None
    for schedule in SCHEDULES:
        graph, stream = _capture(fixture, schedule)
        graph.replay()
        stream.synchronize()
        output = fixture.combined_output.clone()
        if reference is None:
            reference = output
        assert reference is not None
        samples_ms = _time_graph(graph, stream, args.replays, args.repeats)
        median_ms = statistics.median(samples_ms)
        results.append(
            {
                "name": schedule.name,
                "overlap": schedule.overlap,
                "router_first": schedule.router_first,
                "samples_ms": samples_ms,
                "median_ms": median_ms,
                "equal_to_serial": torch.equal(output, reference),
                "output_sha256": _digest(output),
            }
        )

    root_ms = float(results[1]["median_ms"])
    for result in results:
        result_ms = float(result["median_ms"])
        result["speedup_vs_root_release"] = root_ms / result_ms
        result["projected_saving_ms_per_token"] = (root_ms - result_ms) * 43

    payload = {
        "contract": {
            "model": "DeepSeek-V4-Flash",
            "batch": 1,
            "layers": 43,
            "gate": "K4096/N256 + sqrtsoftplus top-6",
            "shared": "FP8 K4096/N512 + K256/N4096",
            "routed": "MXFP4 top-6 K4096/N512 + K256/N4096",
            "cuda_graph": True,
            "replays": args.replays,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2)
    args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if all(bool(result["equal_to_serial"]) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
