# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark exact SM70 M=1 compressor state-save fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)


@dataclass(frozen=True)
class Case:
    name: str
    head_dim: int
    compress_ratio: int
    block_size: int
    calls_per_token: int
    boundary: bool

    @property
    def overlap(self) -> bool:
        return self.compress_ratio == 4

    @property
    def state_width(self) -> int:
        return (1 + self.overlap) * self.head_dim

    @property
    def rope_dim(self) -> int:
        return 64

    @property
    def token_stride(self) -> int:
        return 576 if self.head_dim == 512 else 128

    @property
    def scale_dim(self) -> int:
        return 8 if self.head_dim == 512 else 4

    @property
    def quant_block(self) -> int:
        return 64 if self.head_dim == 512 else 128

    @property
    def frequency(self) -> float:
        if self.boundary:
            return 1.0 / self.compress_ratio
        return 1.0 - 1.0 / self.compress_ratio


CASES = (
    Case("main_c4_nonboundary", 512, 4, 4, 21, False),
    Case("main_c4_boundary", 512, 4, 4, 21, True),
    Case("index_c4_nonboundary", 128, 4, 4, 21, False),
    Case("index_c4_boundary", 128, 4, 4, 21, True),
    Case("main_c128_nonboundary", 512, 128, 8, 20, False),
    Case("main_c128_boundary", 512, 128, 8, 20, True),
)


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _capture(run: Callable[[], None]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    return graph


def _time_graph(
    graph: torch.cuda.CUDAGraph, *, replays: int, repeats: int
) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / replays)
    return samples


def _position(case: Case) -> int:
    if case.boundary:
        return case.compress_ratio - 1
    return case.compress_ratio


def _make_buffers(case: Case, seed: int) -> dict[str, torch.Tensor | object]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    position = _position(case)
    num_pages = position // case.block_size + 2
    state = torch.randn(
        num_pages,
        case.block_size,
        2 * case.state_width,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    kv = torch.randn(
        (1, case.state_width),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    score = torch.randn_like(kv, generator=generator)
    ape = torch.randn(
        case.compress_ratio,
        case.state_width,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    positions = torch.tensor([position], dtype=torch.int64, device="cuda")
    state_slot = torch.tensor([position], dtype=torch.int64, device="cuda")
    block_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").view(1, -1)
    token_to_req = torch.zeros(1, dtype=torch.int32, device="cuda")
    rms_weight = torch.randn(
        case.head_dim,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    cos_sin = torch.randn(
        position + 1,
        case.rope_dim,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    kv_block_size = 256
    k_cache = torch.zeros(
        2,
        kv_block_size,
        case.token_stride + case.scale_dim,
        dtype=torch.uint8,
        device="cuda",
    )
    kv_slot = torch.zeros(1, dtype=torch.int64, device="cuda")
    return {
        "state": state,
        "kv": kv,
        "score": score,
        "ape": ape,
        "positions": positions,
        "state_slot": state_slot,
        "block_table": block_table,
        "token_to_req": token_to_req,
        "rms_weight": rms_weight,
        "cos_sin": cos_sin,
        "k_cache": k_cache,
        "kv_slot": kv_slot,
        "kv_metadata": SimpleNamespace(slot_mapping=kv_slot),
    }


def _compress(
    case: Case, buffers: dict[str, torch.Tensor | object], *, fused: bool
) -> None:
    state = buffers["state"]
    kv = buffers["kv"]
    score = buffers["score"]
    ape = buffers["ape"]
    assert isinstance(state, torch.Tensor)
    assert isinstance(kv, torch.Tensor)
    assert isinstance(score, torch.Tensor)
    assert isinstance(ape, torch.Tensor)
    if not fused:
        save_partial_states(
            kv=kv,
            score=score,
            ape=ape,
            positions=buffers["positions"],
            state_cache=state,
            slot_mapping=buffers["state_slot"],
            block_size=case.block_size,
            state_width=case.state_width,
            compress_ratio=case.compress_ratio,
        )

    compress_norm_rope_store_triton(
        state_cache=state,
        num_actual=1,
        token_to_req_indices=buffers["token_to_req"],
        positions=buffers["positions"],
        slot_mapping=buffers["state_slot"],
        block_table=buffers["block_table"],
        block_size=case.block_size,
        state_width=case.state_width,
        cos_sin_cache=buffers["cos_sin"],
        kv_cache=buffers["k_cache"],
        k_cache_metadata=buffers["kv_metadata"],
        pdl_kwargs={},
        head_dim=case.head_dim,
        rope_head_dim=case.rope_dim,
        compress_ratio=case.compress_ratio,
        overlap=case.overlap,
        use_fp4_cache=False,
        rms_norm_weight=buffers["rms_weight"],
        rms_norm_eps=1e-6,
        quant_block=case.quant_block,
        token_stride=case.token_stride,
        scale_dim=case.scale_dim,
        fresh_kv=kv if fused else None,
        fresh_score=score if fused else None,
        fresh_ape=ape if fused else None,
    )


def _measure_case(
    case: Case, *, seed: int, replays: int, repeats: int
) -> dict[str, object]:
    baseline = _make_buffers(case, seed)
    candidate = _make_buffers(case, seed)
    baseline_graph = _capture(lambda: _compress(case, baseline, fused=False))
    candidate_graph = _capture(lambda: _compress(case, candidate, fused=True))
    baseline_graph.replay()
    candidate_graph.replay()
    torch.cuda.synchronize()

    baseline_state = baseline["state"]
    candidate_state = candidate["state"]
    baseline_cache = baseline["k_cache"]
    candidate_cache = candidate["k_cache"]
    assert isinstance(baseline_state, torch.Tensor)
    assert isinstance(candidate_state, torch.Tensor)
    assert isinstance(baseline_cache, torch.Tensor)
    assert isinstance(candidate_cache, torch.Tensor)

    baseline_samples = _time_graph(baseline_graph, replays=replays, repeats=repeats)
    candidate_samples = _time_graph(candidate_graph, replays=replays, repeats=repeats)
    baseline_ms = statistics.median(baseline_samples)
    candidate_ms = statistics.median(candidate_samples)
    return {
        **asdict(case),
        "frequency": case.frequency,
        "baseline_samples_ms": baseline_samples,
        "candidate_samples_ms": candidate_samples,
        "baseline_median_ms": baseline_ms,
        "candidate_median_ms": candidate_ms,
        "saving_ms": baseline_ms - candidate_ms,
        "projected_saving_ms_per_token": (
            (baseline_ms - candidate_ms) * case.calls_per_token * case.frequency
        ),
        "state_equal": torch.equal(baseline_state, candidate_state),
        "cache_equal": torch.equal(baseline_cache, candidate_cache),
        "state_sha256": _digest(candidate_state),
        "cache_sha256": _digest(candidate_cache),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires NVIDIA V100/SM70.")

    results = [
        _measure_case(
            case,
            seed=args.seed + index,
            replays=args.replays,
            repeats=args.repeats,
        )
        for index, case in enumerate(CASES)
    ]
    projected = sum(float(row["projected_saving_ms_per_token"]) for row in results)
    exact = all(bool(row["state_equal"] and row["cache_equal"]) for row in results)
    payload = {
        "contract": {
            "model": "DeepSeek-V4-Flash",
            "tp": 8,
            "m": 1,
            "cuda_graph": True,
            "replays": args.replays,
            "repeats": args.repeats,
        },
        "summary": {
            "projected_saving_ms_per_token": projected,
            "all_bitwise_exact": exact,
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
