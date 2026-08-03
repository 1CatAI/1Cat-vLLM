# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the DeepSeek V4 M=1 attention-input multi-stream stage on SM70."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass

import torch

from vllm import _sm70_ops as sm70_ops


@dataclass(frozen=True)
class LayerKind:
    name: str
    layers: int
    fp16_aux_n: tuple[int, ...]
    fp8_aux_n: tuple[int, ...]


LAYER_KINDS = (
    LayerKind("c4", 21, (2048, 512), (64,)),
    LayerKind("c128", 20, (1024,), ()),
    LayerKind("swa", 2, (), ()),
)


def _prepare_fp8(
    qweight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return sm70_ops.fp8_sm70_prepare(qweight, scales, 128, False)


def _fp8_call(
    out: torch.Tensor,
    x: torch.Tensor,
    prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    weight, scales, meta = prepared
    sm70_ops.fp8_gemm_sm70_out_meta(out, x, weight, scales, meta, False)


class CapturedStage:
    def __init__(
        self,
        *,
        x: torch.Tensor,
        main_call,
        aux_calls: list,
    ) -> None:
        self.capture_stream = torch.cuda.Stream()
        self.aux_streams = [torch.cuda.Stream() for _ in aux_calls]
        self.start_event = torch.cuda.Event()
        self.done_events = [torch.cuda.Event() for _ in aux_calls]

        def stage_call() -> None:
            main_stream = torch.cuda.current_stream()
            self.start_event.record(main_stream)
            for stream, done, aux_call in zip(
                self.aux_streams, self.done_events, aux_calls, strict=True
            ):
                with torch.cuda.stream(stream):
                    stream.wait_event(self.start_event)
                    aux_call()
                    done.record(stream)
            main_call()
            for done in self.done_events:
                main_stream.wait_event(done)

        self.capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.capture_stream):
            for _ in range(4):
                stage_call()
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            stage_call()
        self.graph.replay()
        torch.cuda.synchronize()
        self.x = x

    def time(self, replays: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.capture_stream):
            start.record(self.capture_stream)
            for _ in range(replays):
                self.graph.replay()
            end.record(self.capture_stream)
        end.synchronize()
        return start.elapsed_time(end) / replays


def _make_aux_calls(
    x: torch.Tensor, kind: LayerKind
) -> tuple[list, list[torch.Tensor]]:
    calls = []
    keepalive: list[torch.Tensor] = []
    for n in kind.fp16_aux_n:
        weight = torch.randn((n, x.shape[1]), device=x.device, dtype=torch.float16)
        out = torch.empty((1, n), device=x.device, dtype=torch.float32)
        keepalive.extend((weight, out))

        def fp16_call(weight=weight, out=out) -> None:
            torch.mm(x, weight.T, out=out, out_dtype=torch.float32)

        calls.append(fp16_call)

    for n in kind.fp8_aux_n:
        qweight = torch.randn((n, x.shape[1]), device=x.device, dtype=torch.float16).to(
            torch.float8_e4m3fn
        )
        scales = torch.ones(
            ((n + 127) // 128, x.shape[1] // 128),
            device=x.device,
            dtype=torch.float32,
        )
        prepared = _prepare_fp8(qweight, scales)
        out = torch.empty((1, n), device=x.device, dtype=torch.float16)
        keepalive.extend((qweight, scales, *prepared, out))

        def fp8_call(prepared=prepared, out=out) -> None:
            _fp8_call(out, x, prepared)

        calls.append(fp8_call)
    return calls, keepalive


def benchmark_kind(kind: LayerKind, *, replays: int, repeats: int) -> dict[str, object]:
    k, fused_n, split_n = 4096, 1536, 1024
    x = torch.randn((1, k), device="cuda", dtype=torch.float16)
    qweight = torch.randn((fused_n, k), device="cuda", dtype=torch.float16).to(
        torch.float8_e4m3fn
    )
    scales = torch.ones((fused_n // 128, k // 128), device="cuda", dtype=torch.float32)
    fused_prepared = _prepare_fp8(qweight, scales)
    split_prepared = (
        _prepare_fp8(qweight[:split_n].contiguous(), scales[: split_n // 128]),
        _prepare_fp8(qweight[split_n:].contiguous(), scales[split_n // 128 :]),
    )
    fused_out = torch.empty((1, fused_n), device="cuda", dtype=torch.float16)
    split_out = torch.empty_like(fused_out)

    def fused_call() -> None:
        _fp8_call(fused_out, x, fused_prepared)

    def split_call() -> None:
        _fp8_call(split_out[:, :split_n], x, split_prepared[0])
        _fp8_call(split_out[:, split_n:], x, split_prepared[1])

    fused_aux, fused_keepalive = _make_aux_calls(x, kind)
    split_aux, split_keepalive = _make_aux_calls(x, kind)
    fused_stage = CapturedStage(x=x, main_call=fused_call, aux_calls=fused_aux)
    split_stage = CapturedStage(x=x, main_call=split_call, aux_calls=split_aux)

    fused_samples = []
    split_samples = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            fused_samples.append(fused_stage.time(replays))
            split_samples.append(split_stage.time(replays))
        else:
            split_samples.append(split_stage.time(replays))
            fused_samples.append(fused_stage.time(replays))

    fused_stage.graph.replay()
    split_stage.graph.replay()
    torch.cuda.synchronize()
    fused_median = statistics.median(fused_samples)
    split_median = statistics.median(split_samples)
    _keepalive = (fused_keepalive, split_keepalive)
    return {
        "name": kind.name,
        "layers": kind.layers,
        "fused_samples_ms": fused_samples,
        "split_samples_ms": split_samples,
        "fused_median_ms": fused_median,
        "split_median_ms": split_median,
        "split_minus_fused_ms": split_median - fused_median,
        "projected_split_delta_ms_per_token": (split_median - fused_median)
        * kind.layers,
        "main_output_bitwise_equal": torch.equal(fused_out, split_out),
        "main_output_max_abs": float(
            (fused_out.float() - split_out.float()).abs().max().item()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires an NVIDIA V100 (SM70).")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    results = [
        benchmark_kind(kind, replays=args.replays, repeats=args.repeats)
        for kind in LAYER_KINDS
    ]
    payload = {
        "contract": {
            "model": "DeepSeek-V4-Flash",
            "tp": 8,
            "m": 1,
            "cuda_graph": True,
            "streams": 4,
            "replays": args.replays,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "results": results,
        "projected_split_delta_ms_per_token": sum(
            float(result["projected_split_delta_ms_per_token"]) for result in results
        ),
        "all_main_outputs_bitwise_equal": all(
            bool(result["main_output_bitwise_equal"]) for result in results
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_main_outputs_bitwise_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
