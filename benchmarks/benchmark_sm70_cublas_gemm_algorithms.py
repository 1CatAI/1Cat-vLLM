# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen legacy cuBLAS Tensor Core algorithms on exact SM70 QKV shapes."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

_CUBLAS_STATUS_SUCCESS = 0
_CUBLAS_OP_N = 0
_CUDA_R_16F = 2
_CUBLAS_COMPUTE_32F = 68
_CUBLAS_TENSOR_OP_MATH = 1
_CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
_CUBLAS_GEMM_ALGO0_TENSOR_OP = 100


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m", type=int, default=8000)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--n", type=int, nargs="+", default=[4096, 3584])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--reference-mode",
        choices=("batch", "rowwise"),
        default="batch",
        help="Compare against one batched GEMM or independent M1 GEMMs.",
    )
    return parser.parse_args()


def _event_trials(
    launch: Callable[[], None], warmup: int, iters: int, trials: int
) -> dict[str, Any]:
    for _ in range(warmup):
        launch()
    torch.accelerator.synchronize()
    samples: list[float] = []
    for _ in range(trials):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            launch()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) / iters))
    return {
        "median_ms": float(statistics.median(samples)),
        "mean_ms": float(statistics.mean(samples)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


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
        self.lib.cublasGemmEx.argtypes = [
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
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cublasGemmEx.restype = ctypes.c_int
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
        if status != _CUBLAS_STATUS_SUCCESS:
            raise RuntimeError(f"cuBLAS {operation} failed with status {status}.")

    def gemm(
        self,
        output: torch.Tensor,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        algo: int,
    ) -> int:
        m, k = inputs.shape
        weight_k, n = weight.shape
        if weight_k != k or output.shape != (m, n):
            raise ValueError("Incompatible row-major GEMM shapes.")
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        # Row-major C=MxN is column-major C^T=NxM. Compute
        # C^T = weight^T(NxK) * inputs^T(KxM) without materializing transposes.
        return int(
            self.lib.cublasGemmEx(
                self.handle,
                _CUBLAS_OP_N,
                _CUBLAS_OP_N,
                n,
                m,
                k,
                ctypes.byref(alpha),
                ctypes.c_void_p(weight.data_ptr()),
                _CUDA_R_16F,
                n,
                ctypes.c_void_p(inputs.data_ptr()),
                _CUDA_R_16F,
                k,
                ctypes.byref(beta),
                ctypes.c_void_p(output.data_ptr()),
                _CUDA_R_16F,
                n,
                _CUBLAS_COMPUTE_32F,
                algo,
            )
        )

    def close(self) -> None:
        if self.handle.value is not None:
            self._check(self.lib.cublasDestroy_v2(self.handle), "destroy")
            self.handle = ctypes.c_void_p()


def _run_shape(
    args: argparse.Namespace, device: torch.device, cublas: _Cublas, n: int
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(args.seed + n)
    inputs = torch.randn(
        (args.m, args.k), generator=generator, device=device, dtype=torch.float16
    ).mul_(0.02)
    weight = torch.randn(
        (args.k, n), generator=generator, device=device, dtype=torch.float16
    ).mul_(0.02)
    reference = torch.empty((args.m, n), device=device, dtype=torch.float16)
    if args.reference_mode == "rowwise":
        for row_idx in range(args.m):
            torch.mm(
                inputs[row_idx : row_idx + 1],
                weight,
                out=reference[row_idx : row_idx + 1],
            )
    else:
        torch.mm(inputs, weight, out=reference)
    torch.accelerator.synchronize()

    algorithms = [_CUBLAS_GEMM_DEFAULT_TENSOR_OP] + list(
        range(_CUBLAS_GEMM_ALGO0_TENSOR_OP, _CUBLAS_GEMM_ALGO0_TENSOR_OP + 16)
    )
    results: list[dict[str, Any]] = []
    for algo in algorithms:
        output = torch.empty_like(reference)

        def launch(output: torch.Tensor = output, algo: int = algo) -> None:
            status = cublas.gemm(output, inputs, weight, algo)
            if status != _CUBLAS_STATUS_SUCCESS:
                raise RuntimeError(f"cuBLAS algo {algo} returned status {status}.")

        try:
            timing = _event_trials(launch, args.warmup, args.iters, args.trials)
            launch()
            torch.accelerator.synchronize()
            difference = (output.float() - reference.float()).abs()
            results.append(
                {
                    "algo": algo,
                    "status": "success",
                    "timing": timing,
                    "useful_tflops": float(
                        2 * args.m * n * args.k / (timing["median_ms"] * 1e-3) / 1e12
                    ),
                    "max_abs_diff": float(difference.max().item()),
                    "mean_abs_diff": float(difference.mean().item()),
                    "equal_elements": int((output == reference).sum().item()),
                    "elements": output.numel(),
                }
            )
        except RuntimeError as error:
            torch.accelerator.synchronize()
            results.append({"algo": algo, "status": "rejected", "error": str(error)})
    return {
        "m": args.m,
        "n": n,
        "k": args.k,
        "algorithms": sorted(
            results,
            key=lambda item: item.get("timing", {}).get("median_ms", float("inf")),
        ),
    }


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This benchmark requires SM70/V100.")
    torch.accelerator.set_device_index(device)
    cublas = _Cublas(torch.cuda.current_stream(device).cuda_stream)
    try:
        shapes = [_run_shape(args, device, cublas, n) for n in args.n]
    finally:
        cublas.close()
    payload = {
        "environment": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "m": args.m,
            "k": args.k,
            "n": args.n,
            "warmup": args.warmup,
            "iters": args.iters,
            "trials": args.trials,
            "seed": args.seed,
            "reference_mode": args.reference_mode,
        },
        "shapes": shapes,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
