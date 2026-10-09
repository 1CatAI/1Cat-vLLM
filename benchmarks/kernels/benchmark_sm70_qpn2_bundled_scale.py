# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research bundled code/scale groups with unchanged QPN2 arithmetic.

Use the original M8 grids, split, native lane mapping and logical payload. Pack
256 code bytes followed by 32 E4M3 scale bytes per K16/N32 group. Packing is
outside graph timing. This is not a registered serving layout.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import runpy
import statistics
from functools import partial
from pathlib import Path

import torch
from benchmark_sm70_qpn2_stage_skeleton import source_for_screen
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def generate(source):
    text = source_for_screen(source)
    reader = """
template <bool TurboMindLayout, bool CacheCodes = false>
struct BundledCodeReader {
  static_assert(!TurboMindLayout, "Native QPN2 only");
  const uint8_t* base;
  __device__ __forceinline__ BundledCodeReader(
      const uint8_t* codes, int tile, int groups_k16, int lane) {
    base = codes + static_cast<size_t>(tile) * groups_k16 * 288 + lane * 8;
  }
  __device__ __forceinline__ uint2 load(int group) const {
    return __ldcs(reinterpret_cast<const uint2*>(base + group * 288));
  }
};
"""
    text = text.replace('#include "nvfp4_qpn2_layout.cuh"', reader)
    text = text.replace("Nvfp4Qpn2CodeReader", "BundledCodeReader")
    old = "group_scales + static_cast<size_t>(tile) * groups_k16 * 32 + lane;"
    assert text.count(old) == 6
    text = text.replace(
        old, "codes + static_cast<size_t>(tile) * groups_k16 * 288 + 256 + lane;"
    )
    scale_address = "scale_ptr + static_cast<size_t>(group) * 32"
    assert text.count(scale_address) == 6
    return text.replace(scale_address, "scale_ptr + static_cast<size_t>(group) * 288")


def extension_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph_pair(control, candidate, eviction, iterations):
    graphs = []
    for run in (control, candidate):
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            eviction.fill_(1)
            start.record()
            run()
            end.record()
        for _ in range(10):
            graph.replay()
        graphs.append((graph, start, end))
    samples = [[], []]
    rng = random.Random(123)
    for _ in range(iterations):
        order = [0, 1]
        rng.shuffle(order)
        for index in order:
            graph, start, end = graphs[index]
            graph.replay()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000)
    differences = [a - b for a, b in zip(*samples)]
    boot = sorted(
        statistics.mean(rng.choices(differences, k=len(differences)))
        for _ in range(2000)
    )
    return {
        "control_mean_us": statistics.mean(samples[0]),
        "candidate_mean_us": statistics.mean(samples[1]),
        "paired_saving_us": statistics.mean(differences),
        "paired_saving_ci95_us": [boot[50], boot[1949]],
        "samples_us": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--control-extension", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source_root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
    generated = args.out / "bundled_scale.cu"
    text = generate(source)
    if not generated.exists() or generated.read_text() != text:
        generated.write_text(text)
    candidate = load(
        name="round12_qpn2_bundled_scale",
        sources=[str(generated)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )
    if args.control_extension is None:
        control_source = args.out / "control_scale.cu"
        control_text = source_for_screen(source)
        if not control_source.exists() or control_source.read_text() != control_text:
            control_source.write_text(control_text)
        control = load(
            name="round12_qpn2_bundled_control",
            sources=[str(control_source)],
            extra_include_paths=[str(source.parent)],
            extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
            verbose=True,
        )
    else:
        control = extension_from_file(
            "round12_qpn2_stage_skeleton", args.control_extension
        )
    if args.compile_only:
        return
    helpers = runpy.run_path(
        str(Path(__file__).with_name("benchmark_sm70_nvfp4_qpn2.py"))
    )
    torch.manual_seed(123)
    torch.set_grad_enabled(False)
    assert torch.cuda.get_device_capability() == (7, 0)
    projections = helpers["_load_projection_shards"](args.model, 0, 0, 4)
    prepared, records = [], []
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    for projection in projections:
        codes, scales = ops.nvfp4_qpn2_prepare_sm70(
            projection.packed.cuda(), projection.scales.cuda()
        )
        n, k = projection.packed.shape[0], projection.packed.shape[1] * 2
        width = n // 2 if projection.gated_silu else n
        tiles, groups = n // 32, k // 16
        bundle = torch.cat(
            (codes.view(tiles, groups, 256), scales.view(tiles, groups, 32)),
            dim=-1,
        ).contiguous()
        assert bundle.numel() == codes.numel() + scales.numel()
        assert torch.equal(bundle[..., :256].reshape_as(codes), codes)
        assert torch.equal(bundle[..., 256:].reshape_as(scales), scales)
        x = torch.randn(8, k, device="cuda", dtype=torch.float16)
        output = torch.empty(8, width, device="cuda", dtype=torch.float16)
        reference = torch.empty_like(output)
        sink = torch.empty(width // 32, 512, device="cuda", dtype=torch.int32)
        config = (projection.inverse_global_scale, projection.gated_silu)
        installed = torch.empty_like(output)
        production = (
            ops.nvfp4_qpn2_gated_sm70_out
            if projection.gated_silu
            else ops.nvfp4_qpn2_gemm_sm70_out
        )
        for amplitude in (0.01, 0.1, 1.0, 4.0):
            trial = x * amplitude
            control.launch(reference, trial, codes, scales, *config, 2, sink)
            candidate.launch(output, trial, bundle, scales, *config, 2, sink)
            production(
                installed,
                trial,
                codes,
                scales,
                projection.inverse_global_scale,
                8 if projection.gated_silu else 16,
                1 if projection.gated_silu else 2,
            )
            assert torch.equal(reference.view(torch.int16), installed.view(torch.int16))
            assert torch.equal(output.view(torch.int16), reference.view(torch.int16))
        x.mul_(0.1)
        for stage, label in ((0, "read"), (1, "decode"), (2, "full")):
            result = graph_pair(
                partial(
                    control.launch, reference, x, codes, scales, *config, stage, sink
                ),
                partial(
                    candidate.launch, output, x, bundle, scales, *config, stage, sink
                ),
                eviction,
                args.iters,
            )
            result.update(
                projection="gate_up" if projection.gated_silu else "down",
                stage=label,
                weight_bytes=bundle.numel(),
                output_bitwise=True,
            )
            records.append(result)
            print(
                json.dumps({k: v for k, v in result.items() if k != "samples_us"}),
                flush=True,
            )
        prepared.append((projection, codes, scales, bundle))
    gate, down = prepared
    hidden = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.1
    middle = [torch.empty(8, 4352, device="cuda", dtype=hidden.dtype) for _ in range(2)]
    output = [torch.empty_like(hidden) for _ in range(2)]
    sink = torch.empty(160, 512, device="cuda", dtype=torch.int32)

    def mlp(module, index, bundled):
        for item, x, y in (
            (gate, hidden, middle[index]),
            (down, middle[index], output[index]),
        ):
            projection, codes, scales, bundle = item
            module.launch(
                y,
                x,
                bundle if bundled else codes,
                scales,
                projection.inverse_global_scale,
                projection.gated_silu,
                2,
                sink,
            )

    mlp(control, 0, False)
    mlp(candidate, 1, True)
    assert torch.equal(middle[0].view(torch.int16), middle[1].view(torch.int16))
    assert torch.equal(output[0].view(torch.int16), output[1].view(torch.int16))
    mlp_result = graph_pair(
        partial(mlp, control, 0, False),
        partial(mlp, candidate, 1, True),
        eviction,
        args.iters,
    )
    mlp_result["projection"] = "mlp_chain"
    mlp_result["extrapolated_56_layer_saving_ms"] = (
        mlp_result["paired_saving_us"] * 56 / 1000
    )
    records.append(mlp_result)
    proof = {
        "research_only": True,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "generated_sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
        "records": records,
        "scope": (
            "Original M8/TP4 layer0, same bytes/grid/split; "
            "bundled native codes+scales; no serving dispatch."
        ),
    }
    (args.out / "result.json").write_text(json.dumps(proof, indent=2))
    print(
        json.dumps({k: v for k, v in mlp_result.items() if k != "samples_us"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
