# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen an original-FP16 fused Qwen3.8 MTP5 HyperConnection mix.

The candidate is adapted from SGLang's Qwen3.8 persistent HC-mix kernel.  It
keeps checkpoint FP16 weights and activations; no weight or activation
quantization is performed.  The down projection also emits the four delayed
HC injection logits used by this vLLM tree.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from vllm.models.qwen4_exp.nvidia.ops.hc import (
    _hc_gate_mix_kernel,
    _hc_silu_kernel,
)
from vllm.triton_utils import tl, triton

MODEL = Path("/data/models/RadixArk/Qwen3.8-Flash-Next-NVFP4")
PREFIX = "model.language_model.layers.0.attn_hyper_connection"
TOKENS = 5
HC = 4
HIDDEN = 2560
HYPER_HIDDEN = HC * HIDDEN
LOWRANK = 320
INJECTION = 4
PADDED_DOWN = 336
PADDED_ROWS = 16


@triton.jit
def _grid_barrier(counter_ptr, num_ctas):
    tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
        pass


@triton.jit
def _fp16_hc_mix_mtp5_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    partial_ptr,
    block_out_ptr,
    injection_out_ptr,
    counters_ptr,
    num_rows,
    num_ctas,
    inv_hc,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    DOWN_N: tl.constexpr,
    LOWRANK_N: tl.constexpr,
    HS: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_m = tl.arange(0, ROWS)
    mask_m = offsets_m < num_rows

    zero_span = ROWS * DOWN_N
    offsets_zero = tl.arange(0, 256)
    for zero_start in range(pid * 256, zero_span, num_ctas * 256):
        indices = zero_start + offsets_zero
        tl.store(partial_ptr + indices, 0.0, mask=indices < zero_span)
    _grid_barrier(counters_ptr, num_ctas)

    offsets_k = tl.arange(0, BLOCK_K)
    offsets_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(DOWN_N, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    for tile in range(pid, n_blocks * k_chunks, num_ctas):
        n_block = tile % n_blocks
        k_chunk = tile // n_blocks
        n = n_block * BLOCK_N + offsets_n
        k = k_chunk * BLOCK_K + offsets_k
        mask_n = n < DOWN_N
        x = tl.load(
            x_ptr + offsets_m[:, None] * K + k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        weight = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        accumulation = tl.dot(x, tl.trans(weight))
        tl.atomic_add(
            partial_ptr + offsets_m[:, None] * DOWN_N + n[None, :],
            accumulation,
            mask=mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )
    _grid_barrier(counters_ptr + 1, num_ctas)

    if pid == 0:
        offsets_injection = tl.arange(0, HC_COUNT)
        injection = tl.load(
            partial_ptr
            + offsets_m[:, None] * DOWN_N
            + LOWRANK_N
            + offsets_injection[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        tl.store(
            injection_out_ptr
            + offsets_m[:, None] * HC_COUNT
            + offsets_injection[None, :],
            injection,
            mask=mask_m[:, None],
        )

    offsets_j = tl.arange(0, BLOCK_J)
    offsets_r = tl.arange(0, BLOCK_R)
    offsets_hc = tl.arange(0, HC_COUNT)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    for j_block in range(pid, j_blocks, num_ctas):
        j = j_block * BLOCK_J + offsets_j
        mask_j = j < HS
        hc_j = offsets_hc[:, None] * HS + j[None, :]
        hc_j_flat = tl.reshape(hc_j, (HC_COUNT * BLOCK_J,))
        mask_hc_j = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC_COUNT, BLOCK_J)),
            (HC_COUNT * BLOCK_J,),
        )
        accumulation = tl.zeros((ROWS, HC_COUNT * BLOCK_J), dtype=tl.float32)
        for r_start in range(0, LOWRANK_N, BLOCK_R):
            r = r_start + offsets_r
            mask_r = r < LOWRANK_N
            down = tl.load(
                partial_ptr + offsets_m[:, None] * DOWN_N + r[None, :],
                mask=mask_m[:, None] & mask_r[None, :],
                other=0.0,
            )
            # Preserve the production GEMM -> FP16 materialization boundary
            # before SiLU; only the split-K accumulation order may differ.
            down = down.to(x_ptr.dtype.element_ty).to(tl.float32) * inv_hc
            activated = (down * tl.sigmoid(down)).to(x_ptr.dtype.element_ty)
            weight = tl.load(
                w_up_ptr + hc_j_flat[:, None] * LOWRANK_N + r[None, :],
                mask=mask_hc_j[:, None] & mask_r[None, :],
                other=0.0,
            )
            accumulation = tl.dot(activated, tl.trans(weight), accumulation)
        gate_logits = tl.reshape(accumulation, (ROWS, HC_COUNT, BLOCK_J))
        gate_logits = gate_logits.to(x_ptr.dtype.element_ty).to(tl.float32)
        gate = tl.sigmoid(gate_logits)
        x_grouped = tl.load(
            x_ptr
            + offsets_m[:, None, None] * (HC_COUNT * HS)
            + offsets_hc[None, :, None] * HS
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        block_out = tl.sum(gate * x_grouped, axis=1) * inv_hc
        tl.store(
            block_out_ptr + offsets_m[:, None] * HS + j[None, :],
            block_out,
            mask=mask_m[:, None] & mask_j[None, :],
        )

    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        tl.store(counters_ptr, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


def _graph_us(
    launch: Callable[[], None], *, unroll: int, replays: int
) -> tuple[float, list[float]]:
    for _ in range(8):
        launch()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            launch()
    for _ in range(8):
        graph.replay()
    torch.accelerator.synchronize()
    samples: list[float] = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / (unroll * replays))
    return statistics.median(samples), samples


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = actual.float() - expected.float()
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(
            difference.norm() / expected.float().norm().clamp_min(1e-12)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unroll", type=int, default=16)
    parser.add_argument("--replays", type=int, default=100)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("benchmark requires exact SM70")

    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def load(suffix: str) -> torch.Tensor:
        key = f"{PREFIX}.{suffix}"
        with safe_open(
            args.model / weight_map[key], framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(key).half().cuda().contiguous()

    down_weight = load("input_mix_weight_down.weight")
    injection_weight = load("block_inject_weight.weight")
    up_weight = load("input_mix_weight_up.weight")
    combined_down = torch.cat(
        (
            down_weight,
            injection_weight,
            down_weight.new_zeros(PADDED_DOWN - LOWRANK - INJECTION, HYPER_HIDDEN),
        )
    ).contiguous()

    torch.manual_seed(args.seed)
    x = torch.randn(TOKENS, HYPER_HIDDEN, device="cuda", dtype=torch.float16)
    baseline_down = torch.empty(TOKENS, PADDED_DOWN, device="cuda", dtype=torch.float16)
    baseline_lora = torch.empty(TOKENS, LOWRANK, device="cuda", dtype=torch.float16)
    baseline_gate = torch.empty(
        TOKENS, HYPER_HIDDEN, device="cuda", dtype=torch.float16
    )
    baseline_block = torch.empty(TOKENS, HIDDEN, device="cuda", dtype=torch.float16)
    candidate_block = torch.empty_like(baseline_block)
    candidate_injection = torch.empty(TOKENS, HC, device="cuda", dtype=torch.float16)
    partials = torch.empty(PADDED_ROWS, PADDED_DOWN, device="cuda", dtype=torch.float32)
    counters = torch.zeros(3, device="cuda", dtype=torch.int32)
    num_ctas = torch.cuda.get_device_properties(x.device).multi_processor_count

    def baseline_launch() -> None:
        torch.mm(x, combined_down.t(), out=baseline_down)
        _hc_silu_kernel[(TOKENS,)](
            baseline_down,
            baseline_lora,
            baseline_down.stride(0),
            baseline_lora.stride(0),
            DIM=LOWRANK,
            HC=HC,
            launch_pdl=False,
        )
        torch.mm(baseline_lora, up_weight.t(), out=baseline_gate)
        _hc_gate_mix_kernel[(TOKENS, triton.cdiv(HIDDEN, 512))](
            x,
            baseline_gate,
            baseline_block,
            x.stride(0),
            baseline_gate.stride(0),
            baseline_block.stride(0),
            HYPER_HIDDEN,
            HC,
            512,
            launch_pdl=False,
        )

    def candidate_launch() -> None:
        _fp16_hc_mix_mtp5_kernel[(num_ctas,)](
            x,
            combined_down,
            up_weight,
            partials,
            candidate_block,
            candidate_injection,
            counters,
            TOKENS,
            num_ctas,
            1.0 / HC,
            ROWS=PADDED_ROWS,
            K=HYPER_HIDDEN,
            DOWN_N=PADDED_DOWN,
            LOWRANK_N=LOWRANK,
            HS=HIDDEN,
            HC_COUNT=HC,
            BLOCK_N=32,
            BLOCK_K=256,
            BLOCK_J=32,
            BLOCK_R=64,
            num_warps=8,
        )

    baseline_launch()
    candidate_launch()
    torch.accelerator.synchronize()
    block_error = _error(candidate_block, baseline_block)
    injection_error = _error(
        candidate_injection, baseline_down[:, LOWRANK : LOWRANK + HC]
    )

    replay_outputs = []
    for _ in range(16):
        candidate_launch()
        replay_outputs.append(candidate_block.clone())
    torch.accelerator.synchronize()
    replay_variation = max(
        float((output.float() - replay_outputs[0].float()).abs().max())
        for output in replay_outputs[1:]
    )

    baseline_us, baseline_samples = _graph_us(
        baseline_launch, unroll=args.unroll, replays=args.replays
    )
    candidate_us, candidate_samples = _graph_us(
        candidate_launch, unroll=args.unroll, replays=args.replays
    )
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": [TOKENS, HYPER_HIDDEN],
        "precision_contract": {
            "weights": "checkpoint FP16 unchanged",
            "activations": "FP16",
            "accumulation": "FP32 Tensor Core with FP32 atomic down reduction",
            "quantization": "none",
        },
        "baseline_us": baseline_us,
        "candidate_us": candidate_us,
        "saved_us_per_hc_mix": baseline_us - candidate_us,
        "projected_saved_ms_per_96_calls": (baseline_us - candidate_us) * 96 / 1000,
        "speedup": baseline_us / candidate_us,
        "block_output_error": block_error,
        "injection_error": injection_error,
        "candidate_replay_max_abs_variation": replay_variation,
        "baseline_samples_us": baseline_samples,
        "candidate_samples_us": candidate_samples,
        "unroll": args.unroll,
        "replays": args.replays,
        "source_reference": "sgl-project/sglang#36497 hc_mix_triton.py",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
