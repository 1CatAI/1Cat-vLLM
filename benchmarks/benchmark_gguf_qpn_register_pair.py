# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only exact IQ3_S expansion in the NVFP4 register-pair skeleton."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
import uuid
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load


def native_pack(root, raw):
    path = root / "vllm/model_executor/layers/quantization/gguf_dmv_formats.py"
    spec = importlib.util.spec_from_file_location("pair_formats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fmt, codes, scales = module.pack(raw, 21)
    assert fmt == 5
    return codes, scales, module.tables()


def bundle_pack(raw):
    import gguf.quants as quant

    quant.IQ3_S.init_grid()
    grid = np.asarray(quant.IQ3_S.grid).reshape(-1, 4).astype(np.int16)
    n = raw.shape[0]
    b = raw.reshape(n, -1, 110)
    d = np.ascontiguousarray(b[..., :2]).view(np.float16)[..., 0].astype(np.float32)
    indices = b[..., 2:66].reshape(n, -1, 8, 8).astype(np.uint16)
    high = b[..., 66:74].astype(np.uint16)
    indices |= ((high[..., None] >> np.arange(8, dtype=np.uint16)) & 1) << 8
    values = grid[indices].reshape(n, -1, 8, 32)
    signs = b[..., 74:106].reshape(n, -1, 8, 4)
    signs = ((signs[..., None] >> np.arange(8, dtype=np.uint8)) & 1).reshape(
        values.shape
    )
    values = values * (1 - 2 * signs.astype(np.int16))
    codes = ((values + 15) // 2).astype(np.uint8).reshape(n, -1)
    sc = b[..., 106:110]
    nib = np.stack([sc & 15, sc >> 4], -1).reshape(n, -1, 8)
    scale = (
        (d[..., None] * (1 + 2 * nib.astype(np.float32)))
        .astype(np.float16)
        .reshape(n, -1)
    )
    # Preserve coefficient bits in storage; match native signed-zero cancellation.
    reference_scale = scale.copy()
    reference_scale[reference_scale == 0] = np.float16(0)
    weights = (
        values.reshape(n, -1).astype(np.float32)
        * np.repeat(reference_scale, 32, axis=1).astype(np.float32)
    ).astype(np.float16)
    k = codes.shape[1]
    lane = np.arange(32)
    column = ((lane >> 2) & 3) * 8 + (lane & 3) + np.where(lane & 16, 4, 0)
    physical = np.arange(16)
    logical = (physical & 8) + ((physical & 3) << 1) + ((physical & 7) >> 2)
    ordered = codes.reshape(n // 32, 32, k // 16, 16).transpose(0, 2, 1, 3)[
        :, :, column
    ][:, :, :, logical]
    packed = ordered[..., 0::2] | (ordered[..., 1::2] << 4)
    metadata = (
        np.repeat(scale, 2, axis=1)
        .reshape(n // 32, 32, k // 16)
        .transpose(0, 2, 1)[:, :, column]
    )
    bundle = np.empty((n // 32, k // 16, 320), dtype=np.uint8)
    bundle[..., :256] = packed.reshape(n // 32, k // 16, 256)
    bundle[..., 256:] = (
        np.ascontiguousarray(metadata).view(np.uint8).reshape(n // 32, k // 16, 64)
    )
    return bundle, weights


def source(root, output):
    path = root / "csrc/sm70_turbomind/ops"
    original = (path / "gguf_dmv_sm70.cu").read_text()
    device = original[: original.index("}  // namespace")]
    nv = (path / "nvfp4_qpn2_sm70.cu").read_text()
    a = nv.index("template <bool Interleave>\n__global__")
    b = nv.index("template <int SplitK, int NAcc, int RowTiles", a)
    kernel = nv[a:b].replace(
        "nvfp4_qpn2_m8_paired_gated_sm70_kernel", "signed_qpn_pair"
    )
    kernel = kernel.replace("Nvfp4PairReader<false, true>", "SignedPairReader")
    kernel = kernel.replace("VLLM_SM70_QPN2_MMA", "mma")
    kernel = kernel.replace(
        "  const half silu = __float2half(gf / (1.0f + expf(-gf)));",
        "  const float silu = gf / (1.0f + expf(-gf));",
    )
    kernel = kernel.replace(
        "      __hmul(silu, uh);", "      __float2half_rn(silu * __half2float(uh));"
    )
    header = (root / "benchmarks/csrc/gguf_qpn_register_pair_sm70.cuh").read_text()
    reader, footer = header.split("// HOST_BINDINGS\n", 1)
    generated = output / "qpn_register_pair.cu"
    content = device + reader + kernel + "}  // namespace\n" + footer
    if not generated.exists() or generated.read_text() != content:
        generated.write_text(content)
    return generated


def worker(rank, args, generated):
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{args.rendezvous}", rank=rank, world_size=4
    )
    ext = load(
        name="gguf_qpn_register_pair_research",
        sources=[str(generated)],
        build_directory=str(args.output / "build"),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--ptxas-options=-v",
            "-gencode=arch=compute_70,code=sm_70",
        ],
    )
    reader = gguf.GGUFReader(str(args.model))
    tensors = {t.name: t for t in reader.tensors}
    chain, cases = [], []
    for layer in (6, 23, 24, 51):
        codes, scales, bundles, weights, official = [], [], [], [], []
        for role in ("ffn_gate", "ffn_up"):
            t = tensors[f"blk.{layer}.{role}.weight"]
            k, n = map(int, t.shape)
            raw = np.ascontiguousarray(
                t.data.reshape(n, k // 256, 110)[
                    rank * (n // 4) : (rank + 1) * (n // 4)
                ]
            ).reshape(n // 4, -1)
            c, sc, tab = native_pack(args.root, raw)
            bundle, expected = bundle_pack(raw)
            codes.append(torch.from_numpy(c).cuda())
            scales.append(torch.from_numpy(sc).cuda())
            bundles.append(torch.from_numpy(bundle).cuda())
            expected = torch.from_numpy(expected).cuda()
            decoded = torch.empty_like(expected)
            ext.decode(bundles[-1], decoded)
            torch.accelerator.synchronize()
            assert torch.equal(decoded.view(torch.uint8), expected.view(torch.uint8)), (
                rank,
                layer,
                role,
                "weight bits",
            )
            weights.append(expected)
            official.append(
                torch.from_numpy(gguf.quants.dequantize(raw, t.tensor_type)).cuda()
            )
        bundle = torch.cat(bundles, dim=0).contiguous()
        tab = torch.from_numpy(tab).cuda()
        x = torch.randn(8, 5120, device=rank, dtype=torch.float16)
        out = torch.empty(8, 4352, device=rank, dtype=torch.float16)
        errors = []
        for amp in (0.125, 0.5, 1.0):
            x.normal_(std=amp)
            ext.run(x, codes, scales, tab, bundle, out, False)
            control = out.clone()
            ext.run(x, codes, scales, tab, bundle, out, True)
            torch.accelerator.synchronize()
            gate = (x.float() @ official[0].float().T).half().float()
            up = (x.float() @ official[1].float().T).half().float()
            oracle = (torch.nn.functional.silu(gate) * up).half()
            errors.append(
                dict(
                    official_relative_l2=float(
                        (out.float() - oracle.float()).norm() / oracle.float().norm()
                    ),
                    control_relative_l2=float(
                        (out.float() - control.float()).norm() / control.float().norm()
                    ),
                    max_abs=float((out.float() - control.float()).abs().max()),
                )
            )
            assert (
                errors[-1]["official_relative_l2"] < 0.01
                and errors[-1]["control_relative_l2"] < 0.005
                and torch.isfinite(out).all()
            ), (rank, layer, errors[-1])
        cases.append(
            dict(
                layer=layer,
                decoded_weights_bitwise=True,
                errors=errors,
                native_weight_bytes=sum(
                    c.numel() + s.numel() for c, s in zip(codes, scales)
                ),
                candidate_weight_bytes=bundle.numel(),
            )
        )
        chain.append((x, codes, scales, tab, bundle, out))
        del weights, official, bundles
    graphs = []
    for arm in (False, True):

        def call(arm=arm):
            for x, c, sc, tab, bundle, out in chain:
                ext.run(x, c, sc, tab, bundle, out, arm)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs.append(graph)
    replay_errors = []
    for _ in range(20):
        for x, *_ in chain:
            x.normal_(std=0.5)
        graphs[0].replay()
        torch.accelerator.synchronize()
        reference = [c[-1].clone() for c in chain]
        graphs[1].replay()
        torch.accelerator.synchronize()
        err = [
            float((c[-1].float() - r.float()).norm() / r.float().norm())
            for c, r in zip(chain, reference)
        ]
        assert max(err) < 0.005 and all(torch.isfinite(c[-1]).all() for c in chain)
        replay_errors.append(err)
    samples = {0: [], 1: []}
    intervals = []
    if not args.numeric_only:
        until = time.monotonic() + 5
        while time.monotonic() < until:
            for i in range(512):
                graphs[i % 2].replay()
            torch.accelerator.synchronize()
        for arm in (0, 1, 1, 0) * 5:
            dist.barrier()
            for _ in range(10):
                graphs[arm].replay()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            before = time.time()
            start.record()
            for _ in range(100):
                graphs[arm].replay()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000 / 100
            samples[arm].append(us)
            intervals.append(
                dict(arm=arm, us=us, wall_start=before, wall_end=time.time())
            )
    record = dict(
        research_only=True,
        serving_runtime=False,
        rank=rank,
        source_sha=args.source_sha,
        generated_cuda_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
        cases=cases,
        changed_input_graph_replays=20,
        replay_relative_l2=replay_errors,
        samples_us=samples,
        timing_intervals=intervals,
        original_chain_us=statistics.mean(samples[0]) if samples[0] else None,
        candidate_chain_us=statistics.mean(samples[1]) if samples[1] else None,
        accumulation="FP32; contiguous eight-warp split, measured reassociation",
    )
    (args.output / f"rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--numeric-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "build").mkdir(exist_ok=True)
    args.rendezvous = args.output / ("gloo-" + uuid.uuid4().hex)
    generated = source(args.root, args.output)
    torch.multiprocessing.spawn(worker, args=(args, generated), nprocs=4, join=True)
