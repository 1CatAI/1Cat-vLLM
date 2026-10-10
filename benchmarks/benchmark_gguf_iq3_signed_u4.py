# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only exact signed-u4 expansion of IQ3_S.

Run under shared GPU locks. The original control kernel is unchanged.
Existing tile geometry, accumulation order and epilogue stay unchanged.
Weight coefficients are expanded exactly as the existing native reader.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
import uuid
from pathlib import Path

import gguf
import numpy as np
import torch
from torch.utils.cpp_extension import load


def pack(root, raw, kind):
    directory = root / "vllm/model_executor/layers/quantization"
    modules = []
    for name in ("gguf_dmv_formats", "gguf_dense_hmma_formats"):
        spec = importlib.util.spec_from_file_location(name, directory / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    iq, affine = modules
    if kind in (18, 21):
        fmt, codes, scales = iq.pack(raw, kind)
    else:
        fmt, q, scale, minimum, group = affine.decode(raw, kind)
        codes, _, scales = affine.pack(fmt, q, scale, minimum, group)
        if fmt == 3:
            scales = iq.compact_lut4_scale(scales)
    return fmt, codes, scales, iq.tables()


def signed_pack(root, raw):
    directory = root / "vllm/model_executor/layers/quantization"
    spec = importlib.util.spec_from_file_location(
        "signed_affine", directory / "gguf_dense_hmma_formats.py"
    )
    affine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(affine)
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
    assert np.all((values & 1) == 1) and np.max(np.abs(values)) <= 15
    codes = ((values + 15) // 2).astype(np.uint8).reshape(n, -1)
    sc = b[..., 106:110]
    nib = np.stack([sc & 15, sc >> 4], -1).reshape(n, -1, 8)
    scales = (d[..., None] * (1 + 2 * nib.astype(np.float32))).astype(np.float16)
    group_scales = scales.reshape(n, -1)
    # Q4 word order places even/odd weights in the two halfwords; compact
    # LUT4 metadata still supplies four exact FP16 coefficients per K128.
    packed, _, metadata = affine.pack(
        0, codes, group_scales, np.zeros_like(group_scales), 32
    )
    spec = importlib.util.spec_from_file_location(
        "signed_iq", directory / "gguf_dmv_formats.py"
    )
    iq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(iq)
    return packed, iq.compact_lut4_scale(metadata)


def source(root, directory):
    original = (root / "csrc/sm70_turbomind/ops/gguf_dmv_sm70.cu").read_text()
    device = original[: original.index("}  // namespace")]
    start = device.index("template <int FMT, int KW, int TN, bool LegacyIq2 = false>")
    end = device.index("__device__ __forceinline__ void write_out", start)
    body = device[start:end].replace("void body6(", "void signed_body6(")
    body = body.replace(
        "decode<FMT, LegacyIq2>(A, st, hw, lut)", "signed_decode(A, st, hw)"
    )
    marker = (
        "template <int KW, int TN, int FA, int FB, int FC = -1, bool Clocked = false>"
    )
    a = device.index(marker + "\n__global__")
    b = device.index(marker + "\nvoid launch", a)
    kernel = device[a:b].replace("dense_mv(Segs", "signed_dense_mv(Segs")
    a = kernel.index("  if (FA == FB || sg.fmt == FA)")
    b = kernel.index("  __syncthreads();", a)
    kernel = (
        kernel[:a]
        + """  signed_body6<LUT4, KW, TN>(sg, t, on, kslot, tin, g0, g1, S, G,
        x, ldx, M, xs, lut, acc, nullptr, segs.gdn_heads, pair);
"""
        + kernel[b:]
    )
    decoder = """
__device__ __forceinline__ void signed_decode(const Ld<LUT4>& weights,
    int st, uint32_t (&hw)[16]) {
  const uint32_t sw = (st & 2) ? weights.sc.y : weights.sc.x;
  uint32_t scale_bits = (st & 1) ? sw >> 16 : sw & 65535u;
  // The native cancellation FMA produces +0 before applying its sign mask
  // for either signed zero coefficient. Preserve coefficient bits in storage.
  if ((scale_bits & 32767u) == 0) scale_bits = 0;
  const half2 scale = h2(scale_bits | (scale_bits << 16));
  const uint4 q = weights.c[st][0];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint32_t bits = word(q, c);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t shifted = j == 0 ? bits << 1 : bits >> (4 * j - 1);
      const uint32_t value = lop_or(shifted, 0x001e001eu, 0x64006400u);
      hw[4 * c + j] = u32(__hmul2(__hsub2(h2(value), h2(0x640f640fu)), scale));
    }
  }
}
"""
    header = (root / "benchmarks/csrc/gguf_iq3_signed_u4_sm70.cuh").read_text()
    target = directory / "signed_u4.cu"
    content = device + decoder + body + kernel + "}  // namespace\n" + header
    if not target.exists() or target.read_text() != content:
        target.write_text(content)
    return target


def single_screen(rank, args, extension, tensors):
    cases = []
    for role in ("ffn_down", "ssm_out", "attn_output"):
        tensor = next(
            t
            for t in tensors.values()
            if t.name.startswith("blk.")
            and t.name.endswith(f".{role}.weight")
            and int(t.tensor_type) == 21
        )
        global_k, n = map(int, tensor.shape)
        raw = tensor.data.reshape(n, global_k // 256, 110)
        if role == "ssm_out":
            raw = (
                raw.reshape(n, 3, 16, -1)[:, :, rank * 4 : (rank + 1) * 4]
                .copy()
                .reshape(n, -1)
            )
        else:
            blocks = global_k // 256 // 4
            raw = np.ascontiguousarray(
                raw[:, rank * blocks : (rank + 1) * blocks]
            ).reshape(n, -1)
        k = global_k // 4
        _, original_c, original_s, table = pack(args.root, raw, 21)
        new_c, new_s = signed_pack(args.root, raw)
        c = [torch.from_numpy(v).cuda() for v in (original_c, new_c)]
        sc = [torch.from_numpy(v).cuda() for v in (original_s, new_s)]
        tab = torch.from_numpy(table).cuda()
        mismatch = torch.zeros(1, device=rank, dtype=torch.int32)
        extension.compare_weights(c[0], sc[0], c[1], sc[1], tab, mismatch, n, k)
        assert mismatch.item() == 0
        x = torch.randn(8, k, device=rank, dtype=torch.float16)
        if role == "ssm_out":
            # GGUF value tiles span three stored head planes.
            oracle_x = x.reshape(8, 4, 3, 128).transpose(1, 2).reshape(8, k)
        else:
            oracle_x = x
        oracle_w = torch.from_numpy(
            gguf.quants.dequantize(raw, tensor.tensor_type)
        ).cuda()
        oracle = (oracle_x.float() @ oracle_w.float().T).half()
        banks = []
        split = 1 if role == "ffn_down" else 2
        tiles = (n + 63) // 64
        for _ in range(8):
            banks.append(
                (
                    x.clone(),
                    [v.clone() for v in c],
                    [v.clone() for v in sc],
                    torch.empty(8, n, device=rank, dtype=torch.float16),
                    torch.empty(tiles * split * 2 * 256, device=rank),
                    torch.zeros(tiles, device=rank, dtype=torch.int32),
                )
            )

        def call(arm, banks=banks, tab=tab, role=role, split=split):
            for bx, bc, bs, out, ws, cnt in banks:
                extension.single(
                    bx,
                    bc[arm],
                    bs[arm],
                    tab,
                    out,
                    ws,
                    cnt,
                    bool(arm),
                    role == "ssm_out",
                    split,
                )

        call(0)
        expected = [b[3].clone() for b in banks]
        call(1)
        torch.accelerator.synchronize()
        for b, e in zip(banks, expected):
            assert torch.equal(b[3], e)
        relative = float(
            (banks[0][3].float() - oracle.float()).norm() / oracle.float().norm()
        )
        assert relative < 0.01
        samples = {0: [], 1: []}
        graphs = []
        timeline = []
        if not args.numeric_only:
            for arm in (0, 1):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call(arm)
                graphs.append(graph)
            # Warm hardware before recording short cold-working-set samples.
            for iteration in range(512):
                graphs[iteration % 2].replay()
            torch.accelerator.synchronize()
            for arm in (0, 1, 1, 0) * 5:
                torch.distributed.barrier()
                for _ in range(10):
                    graphs[arm].replay()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                before = time.time()
                start.record()
                for _ in range(100):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                us = start.elapsed_time(end) * 1000 / 800
                samples[arm].append(us)
                timeline.append(
                    dict(arm=arm, wall_start=before, wall_end=time.time(), us=us)
                )
            for _ in range(20):
                for b in banks:
                    b[0].normal_()
                graphs[0].replay()
                expected = [b[3].clone() for b in banks]
                graphs[1].replay()
                torch.accelerator.synchronize()
                for b, e in zip(banks, expected):
                    assert torch.equal(b[3], e) and not torch.count_nonzero(b[5])
        cases.append(
            dict(
                role=role,
                tensor=tensor.name,
                n=n,
                k=k,
                split=split,
                decoded_weights_bitwise=True,
                output_bitwise=True,
                official_relative_l2=relative,
                original_us=statistics.mean(samples[0]) if samples[0] else None,
                candidate_us=statistics.mean(samples[1]) if samples[1] else None,
                original_bytes=original_c.nbytes + original_s.nbytes,
                candidate_bytes=new_c.nbytes + new_s.nbytes,
                samples_us=samples,
                timing_intervals=timeline,
            )
        )
    return cases


def worker(rank, args, generated):
    torch.accelerator.set_device_index(rank)
    extension = load(
        name="gguf_iq3_signed_u4_research",
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
    if args.singles:
        cases = single_screen(rank, args, extension, tensors)
        clocks = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={rank}",
                "--query-gpu=clocks.sm,clocks.mem",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(
                dict(
                    research_only=True,
                    serving_runtime=False,
                    cases=cases,
                    clocks_after_timing=clocks,
                    source_sha=args.source_sha,
                ),
                indent=2,
            )
            + "\n"
        )
        torch.distributed.destroy_process_group()
        return
    chain, cases = [], []
    for layer in (6, 23, 24, 51):
        planes, oracles, kinds = [], [], []
        for role in ("ffn_gate", "ffn_up"):
            t = tensors[f"blk.{layer}.{role}.weight"]
            k, n = map(int, t.shape)
            block, size = gguf.GGML_QUANT_SIZES[t.tensor_type]
            raw = t.data.reshape(n, k // block, size)
            raw = np.ascontiguousarray(raw[rank * (n // 4) : (rank + 1) * (n // 4)])
            raw = raw.reshape(n // 4, -1)
            fmt, codes, scales, table = pack(args.root, raw, int(t.tensor_type))
            planes.append(
                (fmt, torch.from_numpy(codes).cuda(), torch.from_numpy(scales).cuda())
            )
            oracles.append(
                torch.from_numpy(gguf.quants.dequantize(raw, t.tensor_type)).cuda()
            )
            kinds.append(int(t.tensor_type))
        tab = torch.from_numpy(table).cuda()
        x = torch.randn(8, 5120, device=rank, dtype=torch.float16)
        out = torch.empty(8, 4352, device=rank, dtype=torch.float16)
        c, sc = [p[1] for p in planes], [p[2] for p in planes]
        fa, fb = [p[0] for p in planes]
        candidate_c, candidate_sc = [], []
        for role in ("ffn_gate", "ffn_up"):
            t = tensors[f"blk.{layer}.{role}.weight"]
            k, n = map(int, t.shape)
            raw = np.ascontiguousarray(
                t.data.reshape(n, k // 256, 110)[
                    rank * (n // 4) : (rank + 1) * (n // 4)
                ]
            ).reshape(n // 4, -1)
            pc, ps = signed_pack(args.root, raw)
            candidate_c.append(torch.from_numpy(pc).cuda())
            candidate_sc.append(torch.from_numpy(ps).cuda())
        mismatch = torch.zeros(1, device=rank, dtype=torch.int32)
        for i in range(2):
            extension.compare_weights(
                c[i], sc[i], candidate_c[i], candidate_sc[i], tab, mismatch, 4352, 5120
            )
        assert int(mismatch.item()) == 0, (rank, layer, "weight bits", mismatch.item())
        zero_meta = sc[0].clone().view(torch.int32).reshape(-1, 2)
        zero_meta[:, 1] = (zero_meta[:, 1] & -65536) | 32768
        zero_expanded = candidate_sc[0].clone().view(torch.int16).fill_(-32768)
        extension.compare_weights(
            c[0],
            zero_meta.view(torch.uint8),
            candidate_c[0],
            zero_expanded.view(torch.uint8),
            tab,
            mismatch,
            4352,
            5120,
        )
        assert int(mismatch.item()) == 0, (rank, layer, "negative-zero bits")
        errors = []
        for amp in (0.125, 0.5, 1.0):
            x.normal_(std=amp)
            extension.run(x, c, sc, tab, out, False)
            ref = out.clone()
            extension.run(x, candidate_c, candidate_sc, tab, out, True)
            torch.accelerator.synchronize()
            assert torch.equal(out, ref), (rank, layer, "pair bits")
            gate = (x.float() @ oracles[0].float().T).half().float()
            up = (x.float() @ oracles[1].float().T).half().float()
            oracle = (torch.nn.functional.silu(gate) * up).half()
            error = float((out.float() - oracle.float()).norm() / oracle.float().norm())
            assert torch.isfinite(out).all() and error < 0.01
            errors.append(error)
        cases.append(
            dict(
                layer=layer,
                types=kinds,
                formats=[fa, fb],
                bitwise_equal=True,
                decoded_weights_bitwise=True,
                negative_zero_weights_bitwise=True,
                official_relative_l2=errors,
            )
        )
        chain.append((x, c, sc, candidate_c, candidate_sc, tab, out))
        del oracles
    if not args.numeric_only:
        # Four independent real layer pairs exceed L2 in both arms.
        graphs = []
        for common in (False, True):

            def call(common=common):
                for x, c, sc, cc, cs, tab, out in chain:
                    extension.run(
                        x, cc if common else c, cs if common else sc, tab, out, common
                    )

            call()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            graphs.append(graph)
        samples = {False: [], True: []}
        timeline = []
        for arm in (False, True, True, False) * 5:
            torch.distributed.barrier()
            graph = graphs[int(arm)]
            for _ in range(10):
                graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.time()
            start.record()
            for _ in range(50):
                graph.replay()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000 / 50
            samples[arm].append(us)
            timeline.append(
                dict(
                    common=arm, chain_us=us, wall_start=wall_start, wall_end=time.time()
                )
            )
        for _ in range(20):
            for x, *_ in chain:
                x.normal_(std=0.5)
            graphs[0].replay()
            expected = [out.clone() for _, _, _, _, _, _, out in chain]
            graphs[1].replay()
            torch.accelerator.synchronize()
            for (_, _, _, _, _, _, out), ref in zip(chain, expected):
                assert torch.equal(out, ref)
        performance = dict(
            original_chain_us=statistics.mean(samples[False]),
            common_chain_us=statistics.mean(samples[True]),
            samples_us={str(k): v for k, v in samples.items()},
            timing_intervals=timeline,
            changed_input_graph_replays=20,
        )
    else:
        performance = None
    clocks = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={rank}",
            "--query-gpu=clocks.sm,clocks.mem",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    record = dict(
        research_only=True,
        serving_runtime=False,
        rank=rank,
        source_sha=args.source_sha,
        generated_cuda_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
        cases=cases,
        performance=performance,
        clocks_after_timing=clocks,
    )
    (args.output / f"rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    torch.distributed.destroy_process_group()


def distributed_worker(rank, args, generated):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{args.rendezvous}",
        rank=rank,
        world_size=args.ranks,
    )
    worker(rank, args, generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--singles", action="store_true")
    parser.add_argument("--ranks", type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "build").mkdir(exist_ok=True)
    args.rendezvous = args.output / f"rendezvous-{uuid.uuid4().hex}"
    generated = source(args.root, args.output)
    load(
        name="gguf_iq3_signed_u4_research",
        sources=[str(generated)],
        build_directory=str(args.output / "build"),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--ptxas-options=-v",
            "-gencode=arch=compute_70,code=sm_70",
        ],
        verbose=True,
    )
    torch.multiprocessing.spawn(
        distributed_worker, args=(args, generated), nprocs=args.ranks
    )
