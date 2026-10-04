# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-L2 graph timing of DFlash2 query preparation on retained real QKV.

Use an untimed attention-input capture containing raw_qkv, q_weight, k_weight,
and positions. This benchmark does not replace a complete-layer/model gate.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-fused", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    values = torch.load(args.inputs, weights_only=True, map_location="cuda")
    raw, qw, kw, positions = (
        values[key] for key in ("raw_qkv", "q_weight", "k_weight", "positions")
    )
    assert raw.shape == (8, 1536) and raw.dtype == torch.float16
    with set_current_vllm_config(VllmConfig()):
        rope = RotaryEmbedding(128, 128, 262144, 1e7, True, raw.dtype).cuda()
    pages = (int(positions.max()) + 2048) // 2048
    kc = torch.empty(pages, 2048, 2, 128, device="cuda", dtype=raw.dtype)
    vc = torch.empty_like(kc)
    scale = torch.ones(1, device="cuda")
    slots = positions.clone()

    def baseline():
        q = torch.empty(8, 8, 128, device="cuda", dtype=raw.dtype)
        k = torch.empty(8, 2, 128, device="cuda", dtype=raw.dtype)
        ops.rms_norm(q, raw[:, :1024].reshape_as(q), qw, 1e-6)
        ops.rms_norm(k, raw[:, 1024:1280].reshape_as(k), kw, 1e-6)
        q, k = rope.forward_cuda(positions, q.view(8, 1024), k.view(8, 256))
        ops.reshape_and_cache_flash(
            k.view(8, 2, 128), raw[:, 1280:].view(8, 2, 128), kc, vc,
            slots, "auto", scale, scale,
        )
        return q, k

    routes = [("baseline", baseline)]
    if args.compare_fused:
        from vllm.model_executor.layers.attention.sm70_dflash2_qk_rope import (
            qk_norm_rope_cache,
        )
        routes.append(("fused", lambda: qk_norm_rope_cache(
            raw, qw, kw, positions, rope.cos_sin_cache, kc, vc, slots,
        )))
    eviction = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    records = []
    for name, fn in routes:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            eviction.fill_(1)
            begin.record()
            fn()
            end.record()
        samples = []
        for repeat in range(35):
            for _ in range(5):
                graph.replay()
            end.synchronize()
            if repeat >= 5:
                samples.append(begin.elapsed_time(end) * 1000)
        records.append({"route": name, "mean_us": statistics.mean(samples)})
        graph.reset()
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
