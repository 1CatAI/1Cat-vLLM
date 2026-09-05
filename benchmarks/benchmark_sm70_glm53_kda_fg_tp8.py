# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark one strided-batched cuBLAS launch for GLM TP8 KDA f/g B."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1
_CUDA_R_16F = 2
_CUBLAS_COMPUTE_32F = 68
_CUBLAS_TENSOR_OP_MATH = 1
_CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99


def _digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class _Cublas:
    def __init__(self, stream: int) -> None:
        self.lib = ctypes.CDLL("libcublas.so.12")
        self.handle = ctypes.c_void_p()
        self.lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cublasCreate_v2.restype = ctypes.c_int
        self.lib.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
        self.lib.cublasDestroy_v2.restype = ctypes.c_int
        self.lib.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cublasSetStream_v2.restype = ctypes.c_int
        self.lib.cublasSetMathMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.cublasSetMathMode.restype = ctypes.c_int
        self.lib.cublasGemmStridedBatchedEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cublasGemmStridedBatchedEx.restype = ctypes.c_int
        self._check(self.lib.cublasCreate_v2(ctypes.byref(self.handle)), "create")
        self._check(
            self.lib.cublasSetStream_v2(self.handle, ctypes.c_void_p(stream)),
            "set stream",
        )
        self._check(
            self.lib.cublasSetMathMode(self.handle, _CUBLAS_TENSOR_OP_MATH),
            "set math mode",
        )

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"cuBLAS {operation} failed with status {status}")

    def fg_b(
        self,
        output: torch.Tensor,
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        batch, _ = output.shape
        output_stride = output.stride(0)
        input_stride = inputs.stride(0)
        pair_count, rows, cols = weights.shape
        if pair_count != 2 or output_stride != 2 * rows:
            raise ValueError("Expected packed f/g weights and output")
        self._check(
            self.lib.cublasSetStream_v2(
                self.handle,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            ),
            "set stream",
        )
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        status = self.lib.cublasGemmStridedBatchedEx(
            self.handle,
            _CUBLAS_OP_T,
            _CUBLAS_OP_N,
            rows,
            batch,
            cols,
            ctypes.byref(alpha),
            ctypes.c_void_p(weights.data_ptr()),
            _CUDA_R_16F,
            cols,
            rows * cols,
            ctypes.c_void_p(inputs.data_ptr()),
            _CUDA_R_16F,
            input_stride,
            cols,
            ctypes.byref(beta),
            ctypes.c_void_p(output.data_ptr()),
            _CUDA_R_16F,
            output_stride,
            rows,
            2,
            _CUBLAS_COMPUTE_32F,
            _CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
        self._check(status, "strided batched GEMM")

    def close(self) -> None:
        self._check(self.lib.cublasDestroy_v2(self.handle), "destroy")


def _measure_graph_us(
    launch: Callable[[], None], *, warmups: int, repeats: int, trials: int
) -> dict[str, Any]:
    for _ in range(warmups):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    samples = []
    for _ in range(trials):
        for _ in range(warmups):
            graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()

    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires SM70/V100")
    torch.manual_seed(20260902)
    tokens, projected_size, rows, cols = 8, 3336, 1024, 128
    projected = torch.randn(
        (tokens, projected_size), device="cuda", dtype=torch.float16
    ).mul_(0.1)
    inputs = projected[:, -2 * cols :]
    weights = torch.randn((2, rows, cols), device="cuda", dtype=torch.float16).mul_(
        0.01
    )
    output = torch.empty((tokens, 2 * rows), device="cuda", dtype=torch.float16)
    reference = torch.cat(
        [
            F.linear(inputs[:, :cols], weights[0]),
            F.linear(inputs[:, cols:], weights[1]),
        ],
        dim=-1,
    )
    cublas = _Cublas(torch.cuda.current_stream().cuda_stream)
    try:
        cublas.fg_b(output, inputs, weights)
        torch.cuda.synchronize()
        difference = (output.float() - reference.float()).abs()
        candidate_timing = _measure_graph_us(
            lambda: cublas.fg_b(output, inputs, weights),
            warmups=args.warmups,
            repeats=args.repeats,
            trials=args.trials,
        )

        separate_f = torch.empty((tokens, rows), device="cuda", dtype=torch.float16)
        separate_g = torch.empty_like(separate_f)

        def separate() -> None:
            torch.mm(inputs[:, :cols], weights[0].t(), out=separate_f)
            torch.mm(inputs[:, cols:], weights[1].t(), out=separate_g)

        baseline_timing = _measure_graph_us(
            separate,
            warmups=args.warmups,
            repeats=args.repeats,
            trials=args.trials,
        )
    finally:
        cublas.close()

    payload = {
        "contract": {
            "tokens": tokens,
            "projected_size": projected_size,
            "rows": rows,
            "cols": cols,
            "cuda_graph": True,
            "algorithm": _CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        },
        "quality": {
            "exact": torch.equal(output, reference),
            "different": int((output != reference).sum().item()),
            "max_abs": float(difference.max().item()),
            "candidate_sha256": _digest(output),
            "reference_sha256": _digest(reference),
        },
        "timing": {
            "separate": baseline_timing,
            "strided_batched": candidate_timing,
            "speedup": baseline_timing["median_us"] / candidate_timing["median_us"],
            "projected_ms_per_verification": (
                baseline_timing["median_us"] - candidate_timing["median_us"]
            )
            * 34
            / 1000.0,
        },
    }
    encoded = json.dumps(payload, indent=2)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if payload["quality"]["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
