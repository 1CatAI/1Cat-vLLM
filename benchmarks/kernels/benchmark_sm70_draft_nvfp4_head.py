# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen a private NVFP4 draft head; never modify the target head.

Use original FP8 checkpoint codes/channel scales and the production FP8 head
as control. Both arms use a cold-L2 CUDA graph. Synthetic hidden states only
screen throughput: they do not qualify draft acceptance or model quality.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _sm70_ops as ops


@torch.inference_mode()
def quantize_private_head(weight: torch.Tensor, channel_scale: torch.Tensor):
    dense = weight.float() * channel_scale.float()
    groups = dense.reshape(weight.shape[0], -1, 16)
    block_scale = groups.abs().amax(-1) / 6.0
    global_scale = float(block_scale.max()) / 448.0
    if global_scale == 0:
        global_scale = 1.0
    scales = (block_scale / global_scale).to(torch.float8_e4m3fn)
    # QPN2 rounds the effective scale before multiplying the E2M1 value.
    effective = (scales.float() * global_scale).half().float()
    normalized = groups / effective.clamp_min(2.0**-24)[..., None]
    magnitude = normalized.abs()
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=weight.device
    )
    code = torch.bucketize(magnitude, boundaries)
    # Nearest, ties to the code with an even low bit.
    boundary = boundaries[code.clamp_max(6)]
    code += ((magnitude == boundary) & (code < 7) & ((code & 1) != 0)).long()
    code = (code | (normalized.signbit().long() << 3)).to(torch.uint8)
    code = code.reshape_as(weight)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    native_codes, native_scales = ops.nvfp4_qpn2_prepare_sm70(
        packed, scales.contiguous()
    )
    return native_codes, native_scales, global_scale


def capture(fn, eviction):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        eviction.zero_()
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        begin.record()
        fn()
        end.record()
    return graph, begin, end


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--iters", type=int, default=60)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(123)
    assert torch.cuda.get_device_capability() == (7, 0)
    assert 0 <= args.rank < 4
    start, end = args.rank * 62080, (args.rank + 1) * 62080
    with safe_open(
        args.model / "model.safetensors", framework="pt", device="cpu"
    ) as checkpoint:
        weight = checkpoint.get_slice("lm_head.weight")[start:end].cuda()
        scale = checkpoint.get_slice("lm_head.weight_scale")[start:end].cuda()
    fp8_codes, fp8_scales, metadata = ops.fp8_sm70_prepare(
        weight, scale.float(), 128, False
    )
    codes, scales, global_scale = quantize_private_head(weight, scale)
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.1
    reference_out = torch.empty(8, 62080, device="cuda", dtype=torch.float16)
    candidate_out = torch.empty(8, 62080, device="cuda", dtype=torch.float16)
    eviction = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    # Resolve metadata outside graph capture: .item() is a host transfer.
    k_stride, q_stride = int(metadata[0]), int(metadata[1])

    def reference():
        ops.fp8_gemm_sm70_out(
            reference_out, x, fp8_codes, fp8_scales, 128, k_stride, q_stride
        )

    ref_graph, ref_begin, ref_end = capture(reference, eviction)
    rows = []
    for split in (8, 16, 32):
        for nacc in (1, 2):

            def candidate(split=split, nacc=nacc):
                ops.nvfp4_qpn2_gemm_sm70_out(
                    candidate_out, x, codes, scales, global_scale, split, nacc
                )

            graph, begin, end = capture(candidate, eviction)
            timings = [[], []]
            for iteration in range(args.iters):
                arms = (0, 1) if iteration % 2 == 0 else (1, 0)
                for arm in arms:
                    if arm == 0:
                        ref_graph.replay()
                        ref_end.synchronize()
                        timings[0].append(ref_begin.elapsed_time(ref_end) * 1000)
                    else:
                        graph.replay()
                        end.synchronize()
                        timings[1].append(begin.elapsed_time(end) * 1000)
            row = {
                "split": split,
                "nacc": nacc,
                "baseline_us": statistics.median(timings[0]),
                "candidate_us": statistics.median(timings[1]),
                "max_abs_logit_difference_synthetic_only": float(
                    (reference_out.float() - candidate_out.float()).abs().max()
                ),
            }
            row["saved_ms_one_head"] = (row["baseline_us"] - row["candidate_us"]) / 1000
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "model": str(args.model),
        "checkpoint_bytes": (args.model / "model.safetensors").stat().st_size,
        "rank": args.rank,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_weight_bytes": codes.numel() + scales.numel(),
        "cold_l2_bytes": eviction.numel(),
        "hidden_states": "synthetic; no acceptance/teacher-forcing claim",
        "sampling_and_collectives": "excluded",
        "baseline_output_dtype": str(reference_out.dtype),
        "timings_include": "CUDA event interval around one head projection",
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
