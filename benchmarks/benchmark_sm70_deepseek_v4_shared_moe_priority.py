# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen CUDA stream priorities for DeepSeek-V4 shared-MoE overlap on SM70.

The fixture combines the exact TP8 batch-one shared-expert FP8 shapes with the
exact top-6 routed-expert MXFP4 shapes. It compares serial execution with
multi-stream CUDA Graph execution while keeping every arithmetic operation and
output buffer unchanged.
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
from benchmark_sm70_mxfp4_moe_active_experts import STAGES, _prepare_experts

from vllm import _sm70_ops as sm70_ops


@dataclass(frozen=True)
class Schedule:
    name: str
    overlap: bool
    main_priority: int
    aux_priority: int = 0
    fast_reduce: bool = False
    fused_reduce_add: bool = False


SCHEDULES = (
    Schedule("serial", False, 0),
    Schedule("overlap_p0", True, 0),
    Schedule("overlap_main_p1", True, -1),
    Schedule("overlap_main_p2", True, -2),
    Schedule("overlap_fast_reduce", True, 0, fast_reduce=True),
    Schedule("overlap_fused_reduce_add", True, 0, fused_reduce_add=True),
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
    qweight = torch.randn((n, k), device="cuda", dtype=torch.float16).to(
        torch.float8_e4m3fn
    )
    scales = torch.ones(
        ((n + 127) // 128, (k + 127) // 128),
        device="cuda",
        dtype=torch.float32,
    )
    return tuple(sm70_ops.fp8_sm70_prepare(qweight, scales, 128, gated_silu))


class Fixture:
    def __init__(self, seed: int):
        torch.manual_seed(seed)
        device = torch.device("cuda")
        self.x = torch.randn(1, 4096, dtype=torch.float16, device=device) * 0.01

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

        route = [255, 3, 128, 17, 99, 42]
        self.topk_ids = torch.tensor([route], dtype=torch.int32, device=device)
        self.topk_weights = torch.rand(1, top_k, dtype=torch.float32, device=device)
        self.topk_weights /= self.topk_weights.sum(dim=-1, keepdim=True)
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

    def routed_call(self, fast_reduce: bool, fused_reduce_add: bool) -> None:
        sm70_ops.mxfp4_moe_single_token_prepare_w13_sm70_out(
            self.gate_up,
            self.permuted_input,
            self.x,
            self.topk_ids,
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
        if fused_reduce_add:
            return
        if fast_reduce:
            sm70_ops.awq_moe_single_token_weighted_reduce_out(
                self.sorted_output,
                self.topk_weights,
                self.inv_permuted_idx,
                self.routed_output,
                6,
                4096,
            )
        else:
            torch.ops._moe_C.moe_unpermute(
                self.sorted_output,
                self.topk_weights,
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

    def fused_reduce_add(self) -> None:
        sm70_ops.sm70_moe_single_token_weighted_reduce_add_out(
            self.sorted_output,
            self.topk_weights,
            self.inv_permuted_idx,
            self.shared_output,
            self.combined_output,
            self.shared_scale,
            6,
            4096,
        )


def _make_body(
    fixture: Fixture,
    schedule: Schedule,
    main: torch.cuda.Stream,
    aux: torch.cuda.Stream,
) -> Callable[[], None]:
    def body() -> None:
        if not schedule.overlap:
            fixture.shared_call()
            fixture.routed_call(schedule.fast_reduce, schedule.fused_reduce_add)
            if schedule.fused_reduce_add:
                fixture.fused_reduce_add()
            else:
                fixture.combine()
            return

        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            fixture.shared_call()
        fixture.routed_call(schedule.fast_reduce, schedule.fused_reduce_add)
        main.wait_stream(aux)
        if schedule.fused_reduce_add:
            fixture.fused_reduce_add()
        else:
            fixture.combine()

    return body


def _capture(
    fixture: Fixture,
    schedule: Schedule,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]:
    parent = torch.cuda.current_stream()
    main = torch.cuda.Stream(priority=schedule.main_priority)
    aux = torch.cuda.Stream(priority=schedule.aux_priority)
    main.wait_stream(parent)
    aux.wait_stream(parent)
    body = _make_body(fixture, schedule, main, aux)
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
    main: torch.cuda.Stream,
    replays: int,
    repeats: int,
) -> list[float]:
    for _ in range(10):
        graph.replay()
    main.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(main):
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
        raise RuntimeError("This benchmark requires an NVIDIA V100 (SM70).")

    fixture = Fixture(args.seed)
    results: list[dict[str, object]] = []
    reference: torch.Tensor | None = None
    reference_name = "serial"
    for schedule in SCHEDULES:
        graph, main_stream = _capture(fixture, schedule)
        graph.replay()
        main_stream.synchronize()
        output = fixture.combined_output.clone()
        if reference is None:
            reference = output
        assert reference is not None
        diff = (output.float() - reference.float()).abs()
        samples_ms = _time_graph(graph, main_stream, args.replays, args.repeats)
        median_ms = statistics.median(samples_ms)
        results.append(
            {
                "name": schedule.name,
                "overlap": schedule.overlap,
                "main_priority": main_stream.priority,
                "aux_priority": schedule.aux_priority,
                "fast_reduce": schedule.fast_reduce,
                "fused_reduce_add": schedule.fused_reduce_add,
                "samples_ms": samples_ms,
                "median_ms": median_ms,
                "output_sha256": _digest(output),
                "equal_to_serial": torch.equal(output, reference),
                "max_abs_vs_serial": float(diff.max().item()),
                "mean_abs_vs_serial": float(diff.mean().item()),
            }
        )

    serial_ms = float(results[0]["median_ms"])
    for result in results:
        result_ms = float(result["median_ms"])
        result["speedup_vs_serial"] = serial_ms / result_ms
        result["projected_savings_ms_per_token"] = (serial_ms - result_ms) * 43

    payload = {
        "contract": {
            "model": "DeepSeek-V4-Flash",
            "tp": 8,
            "batch": 1,
            "layers": 43,
            "shared": "FP8 K4096/N512 gated-SiLU + K256/N4096",
            "routed": "MXFP4 top-6 K4096/N512 + K256/N4096",
            "cuda_graph": True,
            "reference": reference_name,
            "replays": args.replays,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if all(bool(item["equal_to_serial"]) for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
