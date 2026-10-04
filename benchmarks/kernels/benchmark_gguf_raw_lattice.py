# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare original IQ blocks against current canonical dispatch in CUDA graphs."""

import argparse
import hashlib
import json
import math
from functools import partial
from pathlib import Path

import torch
import vllm._C as core
from benchmark_gguf_turbomind import (
    canonical_dense_call,
    elapsed,
    prepare_projection,
)

import vllm
from vllm.model_executor.kernels.linear.mixed_precision.sm70_gguf_lattice import (
    _LATTICE_BLAS_BANDS,
)
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def errors(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / expected.float().norm()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gguf", type=Path)
    p.add_argument("--tensor", required=True)
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--tp-axis", type=int, choices=(0, 1), required=True)
    p.add_argument("--tp-rank", type=int, default=0)
    p.add_argument("--m", type=int, nargs="+", default=[1, 5, 8, 16, 512])
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument(
        "--weight-banks",
        type=int,
        help="Distinct banks per graph; default exceeds twice L2",
    )
    p.add_argument("--include-compact", action="store_true")
    p.add_argument(
        "--mma-cta-per-sm",
        type=int,
        nargs="+",
        default=[4],
        help="Target CTA count per SM for compact MMA split-K candidates",
    )
    p.add_argument(
        "--mma-row-tiles",
        type=int,
        nargs="+",
        default=[0],
        help="Compact MMA row tiles: 0 selects automatically",
    )
    p.add_argument("--include-occupancy7", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=("canonical", "raw"))
    p.add_argument(
        "--profile-route", help="Force the unprofiled raw winner for counters"
    )
    a = p.parse_args()
    if any(target < 1 for target in a.mma_cta_per_sm):
        p.error("--mma-cta-per-sm values must be positive")
    if any(tile not in (0, 8, 16, 32) for tile in a.mma_row_tiles):
        p.error("--mma-row-tiles values must be 0, 8, 16, or 32")
    assert "site-packages" in vllm.__file__, vllm.__file__
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    reader = GGUFReader(str(a.gguf))
    tensor = next(t for t in reader.tensors if t.name == a.tensor)
    source = tensor.data[a.expert] if tensor.data.ndim == 3 else tensor.data
    kind = int(tensor.tensor_type)
    raw = RawGGUFProjection.from_rows(source, kind).tp_slice(
        a.tp_rank, 4, axis=a.tp_axis
    )
    payload = raw.data[:, : raw.payload_bytes_per_row]
    canonical = transcode_lattice(payload, kind)
    w, stats, meta = prepare_projection(canonical)
    k_ld, q_ld = meta.tolist()
    original = torch.from_numpy(raw.data).cuda()
    n, k = raw.shape
    compact = None
    if a.include_compact:
        compact = torch.empty(
            (payload.nbytes + 7) // 8 * 8, dtype=torch.uint8, device="cuda"
        )
        torch.ops._C.gguf_lattice_compact_reorder_sm70_out(compact, original, kind, k)
    l2_bytes = getattr(
        torch.cuda.get_device_properties(0), "L2_cache_size", 6 * 1024 * 1024
    )
    raw_bytes = original.numel()
    canonical_bytes = (
        w.numel() * w.element_size() + stats.numel() * stats.element_size()
    )
    banks = (
        a.weight_banks
        if a.weight_banks is not None
        else max(
            1,
            math.ceil(
                2
                * l2_bytes
                / min(
                    raw_bytes,
                    canonical_bytes,
                    compact.numel() if compact is not None else raw_bytes,
                )
            ),
        )
    )
    if banks < 1:
        p.error("--weight-banks must be positive")
    # Separate addresses keep small expert tests from measuring only L2 hits.
    weight_banks = [(original, w, stats, compact)] + [
        (
            original.clone(),
            w.clone(),
            stats.clone(),
            compact.clone() if compact is not None else None,
        )
        for _ in range(banks - 1)
    ]

    def banked_call(call, out):
        calls = []
        for rw, cw, cs, cp in weight_banks:
            replacements = {id(original): rw, id(w): cw, id(stats): cs}
            if compact is not None:
                replacements[id(compact)] = cp
            calls.append(
                partial(
                    call.func,
                    *(replacements.get(id(arg), arg) for arg in call.args),
                    **call.keywords,
                )
            )

        def run():
            for bank_call in calls:
                bank_call()
            return out

        return run

    ref = torch.from_numpy(dequantize(payload, kind)).cuda()
    decoded = torch.empty_like(ref)
    torch.ops._C.gguf_lattice_raw_dequantize_sm70_out(decoded, original, kind)
    if compact is not None:
        torch.ops._C.gguf_lattice_compact_dequantize_sm70_out(decoded, compact, kind)
        torch.testing.assert_close(decoded, ref, rtol=0, atol=0)
        torch.ops._C.gguf_lattice_raw_dequantize_sm70_out(decoded, original, kind)
    weight_error = errors(decoded, ref)
    torch.testing.assert_close(decoded, ref, rtol=0, atol=0)
    # Scratch belongs to the invocation, not persistent weight storage.
    raw_scratch = torch.empty((k, n), dtype=torch.float16, device="cuda")
    canon_scratch = torch.empty((k, n), dtype=torch.float16, device="cuda")
    report = {
        "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "origin": vllm.__file__,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "checkpoint": a.gguf.name,
        "tensor": a.tensor,
        "expert": a.expert,
        "source_type": kind,
        "tp": {"size": 4, "rank": a.tp_rank, "axis": a.tp_axis},
        "n": n,
        "k": k,
        "graph": "FULL",
        "profiled_run": bool(a.profile),
        "timing_cache_state": "distinct_weight_banks",
        "timing_weight_banks": banks,
        "l2_bytes": l2_bytes,
        "counter_cache_state": "64_MiB_eviction_before_profiled_replay",
        "accumulation": "FP32",
        "mma_cta_per_sm": a.mma_cta_per_sm,
        "mma_row_tiles": a.mma_row_tiles,
        "row_padding_bytes": raw.padding_bytes_per_row,
        "original_payload_bytes": n * raw.payload_bytes_per_row,
        "raw_weight_bytes": original.numel(),
        "compact_weight_bytes": compact.numel() if compact is not None else None,
        "canonical_weight_bytes": w.numel() * w.element_size()
        + stats.numel() * stats.element_size(),
        "weight_error": weight_error,
        "timings": [],
        "complete": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for m in a.m:
        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        out = torch.empty((m, n), dtype=torch.float16, device="cuda")
        expected = x.float() @ (ref if m == 1 else ref.half().float()).T
        old = canonical_dense_call(canonical, out, x, w, stats, k_ld, q_ld)
        old_name = "gguf_lattice_gemm_sm70_out"
        if any(
            lo <= m and (hi is None or m <= hi)
            for lo, hi in _LATTICE_BLAS_BANDS.get((kind, k, n), ())
        ):
            old = partial(
                torch.ops._C.gguf_lattice_blas_sm70_out,
                out,
                x,
                w,
                stats,
                kind,
                canon_scratch,
                canonical.group_size,
            )
            old_name = "gguf_lattice_blas_sm70_out"
        candidates = []
        if m == 1:
            auto = min(
                k // 256, max(1, math.ceil(4 * report["sm_count"] / math.ceil(n / 4)))
            )
            for split in sorted({1, auto}):
                tmp = torch.empty((split, n), dtype=torch.float32, device="cuda")
                candidates.append(
                    (
                        f"vec_split{split}",
                        partial(
                            torch.ops._C.gguf_lattice_raw_vec_sm70_out,
                            out,
                            x,
                            original,
                            kind,
                            tmp,
                            split,
                        ),
                    )
                )
                candidates.append(
                    (
                        f"vec_prefetch_split{split}",
                        partial(
                            torch.ops._C.gguf_lattice_raw_vec_sm70_out,
                            out,
                            x,
                            original,
                            kind,
                            tmp,
                            split,
                            True,
                        ),
                    )
                )
                for prefetch in (False, True):
                    prefix = "vec_factor_scale" + ("_prefetch" if prefetch else "")
                    candidates.append(
                        (
                            f"{prefix}_split{split}",
                            partial(
                                torch.ops._C.gguf_lattice_raw_vec_sm70_out,
                                out,
                                x,
                                original,
                                kind,
                                tmp,
                                split,
                                prefetch,
                                True,
                            ),
                        )
                    )
        elif m <= 64:
            for tile in (8, 32):
                mt = 8 if m <= 8 else 16 if m <= 16 else 32
                auto = min(
                    k // 256,
                    max(
                        1,
                        math.ceil(
                            4
                            * report["sm_count"]
                            / (math.ceil(n / tile) * math.ceil(m / mt))
                        ),
                    ),
                )
                for split in sorted({1, auto}):
                    tmp = torch.empty((split, m, n), dtype=torch.float32, device="cuda")
                    candidates.append(
                        (
                            f"mma_n{tile}_split{split}",
                            partial(
                                torch.ops._C.gguf_lattice_raw_mma_sm70_out,
                                out,
                                x,
                                original,
                                kind,
                                tmp,
                                split,
                                tile,
                            ),
                        )
                    )
        else:
            candidates.append(
                (
                    "raw_dequant_cublas",
                    partial(
                        torch.ops._C.gguf_lattice_raw_blas_sm70_out,
                        out,
                        x,
                        original,
                        kind,
                        raw_scratch,
                    ),
                )
            )
        if compact is not None:
            if m == 1:
                auto = min(
                    k // 256,
                    max(1, math.ceil(4 * report["sm_count"] / math.ceil(n / 128))),
                )
                for split in sorted({1, auto}):
                    tmp = torch.empty((split, n), dtype=torch.float32, device="cuda")
                    candidates.append(
                        (
                            f"compact_vec_split{split}",
                            partial(
                                torch.ops._C.gguf_lattice_compact_vec_sm70_out,
                                out,
                                x,
                                compact,
                                kind,
                                tmp,
                                split,
                            ),
                        )
                    )
                row_auto = min(
                    k // 256,
                    max(1, math.ceil(4 * report["sm_count"] / math.ceil(n / 4))),
                )
                for split in sorted({1, row_auto}):
                    tmp = torch.empty((split, n), dtype=torch.float32, device="cuda")
                    candidates.append(
                        (
                            f"compact_row_vec_split{split}",
                            partial(
                                torch.ops._C.gguf_lattice_compact_vec_sm70_out,
                                out,
                                x,
                                compact,
                                kind,
                                tmp,
                                split,
                                True,
                            ),
                        )
                    )
            elif m <= 64:
                for row_tile in a.mma_row_tiles:
                    mt = row_tile or (8 if m <= 8 else 16 if m <= 16 else 32)
                    ctas = math.ceil(n / 32) * math.ceil(m / mt)
                    split_candidates = {1} | {
                        min(
                            k // 256,
                            max(1, math.ceil(target * report["sm_count"] / ctas)),
                        )
                        for target in a.mma_cta_per_sm
                    }
                    for split in sorted(split_candidates):
                        tmp = torch.empty(
                            (split, m, n), dtype=torch.float32, device="cuda"
                        )
                        variants = [("", False, False), ("_prefetch", True, False)]
                        if n % 32 == 0:
                            variants.append(("_staged", True, True))
                        variants = [(*variant, False) for variant in variants]
                        if a.include_occupancy7 and mt == 16:
                            variants += [(*variant[:3], True) for variant in variants]
                        variants = [(*variant, False) for variant in variants]
                        if mt == 16:
                            variants += [
                                ("_activation", False, False, False, True),
                                ("_prefetch_activation", True, False, False, True),
                            ]
                        for (
                            name,
                            prefetch,
                            staged,
                            bounded,
                            stage_activation,
                        ) in variants:
                            tile_name = f"_rows{row_tile}" if row_tile else ""
                            if bounded:
                                tile_name += "_occupancy7"
                            candidates.append(
                                (
                                    f"compact_mma{name}{tile_name}_split{split}",
                                    partial(
                                        torch.ops._C.gguf_lattice_compact_mma_sm70_out,
                                        out,
                                        x,
                                        compact,
                                        kind,
                                        tmp,
                                        split,
                                        prefetch,
                                        staged,
                                        row_tile,
                                        bounded,
                                        stage_activation,
                                    ),
                                )
                            )
            else:
                candidates.append(
                    (
                        "compact_dequant_cublas",
                        partial(
                            torch.ops._C.gguf_lattice_compact_blas_sm70_out,
                            out,
                            x,
                            compact,
                            kind,
                            raw_scratch,
                        ),
                    )
                )
                if n % 32 == 0:
                    tm_scratch = torch.empty((n, k), device="cuda", dtype=torch.float16)
                    offsets = torch.tensor([0, m], device="cuda", dtype=torch.int32)
                    tm_ptrs, _ = torch.ops._C.awq_moe_build_strided_ptrs(
                        tm_scratch.unsqueeze(0),
                        tm_scratch.unsqueeze(0),
                        k * 32,
                        k * 32,
                        1,
                    )
                    candidates.append(
                        (
                            "compact_dequant_turbomind_f16",
                            partial(
                                torch.ops._C.gguf_lattice_compact_tm_f16_sm70_out,
                                out,
                                x,
                                compact,
                                kind,
                                tm_scratch,
                                offsets,
                                tm_ptrs,
                            ),
                        )
                    )
                natural_scratch = torch.empty(
                    (n, k), device="cuda", dtype=torch.float16
                )
                candidates.append(
                    (
                        "compact_dequant_cublas_tn",
                        partial(
                            torch.ops._C.gguf_lattice_compact_blas_sm70_out,
                            out,
                            x,
                            compact,
                            kind,
                            natural_scratch,
                            True,
                        ),
                    )
                )
                for natural, scratch in ((False, raw_scratch), (True, natural_scratch)):
                    candidates.append(
                        (
                            "compact_dequant_cublas"
                            + ("_tn" if natural else "")
                            + "_algo2",
                            partial(
                                torch.ops._C.gguf_lattice_compact_blas_sm70_out,
                                out,
                                x,
                                compact,
                                kind,
                                scratch,
                                natural,
                                102,
                            ),
                        )
                    )
                for partitions in (2, 4):
                    candidates.append(
                        (
                            f"compact_dequant_cublas_algo2_parts{partitions}",
                            partial(
                                torch.ops._C.gguf_lattice_compact_blas_sm70_out,
                                out,
                                x,
                                compact,
                                kind,
                                raw_scratch,
                                False,
                                102,
                                partitions,
                            ),
                        )
                    )
                    if n % 32 == 0:
                        candidates.append(
                            (
                                f"compact_dequant_turbomind_f16_parts{partitions}",
                                partial(
                                    torch.ops._C.gguf_lattice_compact_tm_f16_sm70_out,
                                    out,
                                    x,
                                    compact,
                                    kind,
                                    tm_scratch,
                                    offsets,
                                    tm_ptrs,
                                    partitions,
                                ),
                            )
                        )
                if n % 32 == 0:
                    lt_algorithms, lt_sizes = (
                        torch.ops._C.gguf_lattice_compact_lt_sm70_prepare(x, out)
                    )
                    lt_workspace = torch.empty(
                        32 * 1024 * 1024, device="cuda", dtype=torch.uint8
                    )
                    for index, algorithm in enumerate(lt_algorithms):
                        candidates.append(
                            (
                                f"compact_dequant_blaslt_algo{index}_parts2",
                                partial(
                                    torch.ops._C.gguf_lattice_compact_lt_sm70_out,
                                    out,
                                    x,
                                    compact,
                                    kind,
                                    raw_scratch,
                                    lt_workspace,
                                    algorithm,
                                    2,
                                ),
                            )
                        )
                fp32_result = torch.empty((m, n), device="cuda", dtype=torch.float32)

                def fp32_blas(
                    output,
                    activation,
                    packets,
                    source_type,
                    workspace,
                    natural,
                    temporary,
                ):
                    torch.ops._C.gguf_lattice_compact_blas_sm70_out(
                        temporary,
                        activation,
                        packets,
                        source_type,
                        workspace,
                        natural,
                    )
                    output.copy_(temporary)

                for natural, scratch in ((False, raw_scratch), (True, natural_scratch)):
                    candidates.append(
                        (
                            "compact_dequant_cublas_f32" + ("_tn" if natural else ""),
                            partial(
                                fp32_blas,
                                out,
                                x,
                                compact,
                                kind,
                                scratch,
                                natural,
                                fp32_result,
                            ),
                        )
                    )
        old()
        old_error = errors(out, expected)

        old_us = elapsed(banked_call(old, out), a.iterations, capture=True) / banks
        measured = []
        for name, call in candidates:
            call()
            error = errors(out, expected)
            assert error["finite"] and error["relative_l2"] < 0.003, error
            us = elapsed(banked_call(call, out), a.iterations, capture=True) / banks
            measured.append({"route": name, "us": us, "error": error})
        best = min(measured, key=lambda r: r["us"])
        report["timings"].append(
            {
                "m": m,
                "canonical_operator": old_name,
                "canonical_us": old_us,
                "canonical_error": old_error,
                "raw": measured,
                "best_raw": best["route"],
                "best_raw_us": best["us"],
                "raw_over_canonical": best["us"] / old_us,
                "saved_us": old_us - best["us"],
            }
        )
        save()
        if a.profile:
            selected = a.profile_route or best["route"]
            if a.profile == "raw" and selected not in {name for name, _ in candidates}:
                p.error(f"Unknown raw profile candidate: {selected}")
            call = (
                old
                if a.profile == "canonical"
                else next(call for name, call in candidates if name == selected)
            )
            report["profile_operator"] = (
                old_name if a.profile == "canonical" else selected
            )
            save()
            # TurboMind workspaces are keyed by stream. Warm the same stream
            # used for capture, otherwise workspace zeroing enters the graph.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                call()
            graph.replay()
            torch.accelerator.synchronize()
            # Small experts fit in V100 L2. Evict before counters so their
            # DRAM traffic is not hidden by the repeated-shape timing warmup.
            eviction = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            eviction.zero_()
            torch.accelerator.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            graph.replay()
            torch.accelerator.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
