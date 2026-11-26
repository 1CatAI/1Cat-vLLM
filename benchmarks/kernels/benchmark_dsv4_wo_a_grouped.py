# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DeepSeek-V4 wo_a: FP16 dequantized einsum against the grouped FP8 routes.

wo_a is a grouped matmul, 8 groups of [1024, 4096] block-FP8 rows. Arms:

  fp16_einsum        the former Turing route: the weight dequantized to FP16 at
                     load (2 bytes per weight), torch.einsum per call (cuBLAS)
  qpn8_grouped       the grouped QPN8 route (FP8 pack, one launch for M <= 8);
                     runs on Volta and Turing
  qpn8_dense_bmm     the large-M route of the grouped QPN8 layer forced for every
                     M: the whole weight dequantized into the shared workspace
                     on each call, then one batched product (FP32 reduction)
  turbomind_grouped  the Volta production route (TurboMind one-launch grouped
                     GEMM on FP8); Volta only

Every arm is timed as CUDA-graph replay of ``--calls`` back-to-back calls, so
the figure per call has no launch gaps beyond what a graph has. GB/s counts the
bytes of the weight the arm streams (FP16 arm 67.1 MB, FP8 arms 33.6 MB), decimal.

Run once per card (a Quadro RTX 8000 and a V100 differ):

    CUDA_DEVICE_ORDER=PCI_BUS_ID python \
        benchmarks/kernels/benchmark_dsv4_wo_a_grouped.py --device cuda:0

The accuracy column is the largest error against an FP32 product with the exactly
dequantized weight, in units of the FP16 output rounding step (2**-11 of the
value). It is diagnostic: the numerical gate is
tests/kernels/quantization/test_sm70_qpn8_grouped_bmm.py.
"""

from __future__ import annotations

import argparse
import json

import torch

BLOCK = 128


def _weights(groups: int, rows: int, k: int):
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = (
        torch.randn(groups * rows, k, device="cuda", generator=generator) * 0.5
    ).to(torch.float8_e4m3fn)
    exponents = torch.randint(
        -9,
        -4,
        (groups * rows // BLOCK, k // BLOCK),
        device="cuda",
        generator=generator,
    )
    scales = torch.pow(2.0, exponents.float())
    dense = weight.float() * scales.repeat_interleave(BLOCK, 0).repeat_interleave(
        BLOCK, 1
    )
    return weight, scales, dense.view(groups, rows, k)


def _layer(weight: torch.Tensor, scales: torch.Tensor, groups: int):
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(weight.clone(), requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scales.clone(), requires_grad=False)
    )
    layer.orig_dtype = torch.float16
    layer.is_bmm = True
    layer.bmm_batch_size = groups
    return layer


def _native_ops():
    from vllm._sm70.policy import NativeBindings
    from vllm.config.sm70_native import capture_linear_native_config

    return NativeBindings(capture_linear_native_config("fp8").values)


def _fp16_einsum_arm(weight, scales, groups):
    """The former Turing route: dequantize at load, einsum per call."""
    dense = weight.to(torch.float16) * (
        scales.to(torch.float16).repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    )
    dense = dense.view(groups, -1, weight.shape[1])

    def apply(x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...gk,grk->...gr", x, dense)

    return apply, dense.numel() * dense.element_size()


def _qpn8_kernel():
    from vllm.config.kernel import Sm70Fp8Config
    from vllm.model_executor.kernels.linear.qpn.fp8_block import (
        QPN8Fp8BlockScaledMMLinearKernel,
    )

    kernel = object.__new__(QPN8Fp8BlockScaledMMLinearKernel)
    kernel.policy = Sm70Fp8Config()
    kernel.policy.resolve()
    kernel.native_ops = _native_ops()
    return kernel


def _qpn8_arm(weight, scales, groups):
    layer = _layer(weight, scales, groups)
    kernel = _qpn8_kernel()
    kernel.process_weights_after_loading(layer)
    codes = layer._sm70_block_fp8_qpn8_packed_codes
    return (lambda x: kernel.apply_weights(layer, x)), codes.numel()


def _qpn8_dense_bmm_arm(weight, scales, groups):
    """Dequantize-per-call plus one bmm, at every M (finds the M crossover)."""
    from vllm import _sm70_ops as sm70_ops
    from vllm.model_executor.kernels.linear.qpn import fp8_block

    layer = _layer(weight, scales, groups)
    kernel = _qpn8_kernel()
    kernel.process_weights_after_loading(layer)
    codes = layer._sm70_block_fp8_qpn8_packed_codes
    scales_packed = layer._sm70_block_fp8_qpn8_packed_scales
    n_group = codes.shape[2]

    def apply(x: torch.Tensor) -> torch.Tensor:
        rows = x.shape[0]
        out = x.new_empty(rows, groups * n_group)
        dense = fp8_block._ensure_block_qpn8_prefill_workspace(x.device, codes.numel())
        sm70_ops.fp8_qpn8_prefill_sm70_out(
            out,
            dense.data_ptr(),
            x.reshape(rows, -1),
            codes,
            scales_packed,
            False,
        )
        return out.view(rows, groups, n_group)

    return apply, codes.numel()


def _turbomind_arm(weight, scales, groups):
    from vllm.config.kernel import Sm70Fp8Config
    from vllm.model_executor.kernels.linear.qpn.fp8 import TurboMindFp8LinearKernel

    layer = _layer(weight, scales, groups)
    kernel = object.__new__(TurboMindFp8LinearKernel)
    kernel.policy = Sm70Fp8Config(qpn8=False, qpn8_pp2_tp4=False)
    kernel.policy.resolve()
    kernel.native_ops = _native_ops()
    kernel.weight_block_size = [BLOCK, BLOCK]
    kernel.is_scale_e8m0 = False
    kernel.process_weights_after_loading(layer)
    if not getattr(layer, "sm70_fp8_bmm_grouped_decode", False):
        raise RuntimeError("the one-launch grouped decode path is not enabled")
    return (lambda x: kernel.apply_weights(layer, x)), layer.weight.numel()


def _graph_us(call, x: torch.Tensor, calls: int, replays: int) -> float:
    """Microseconds per call, from CUDA-graph replays of ``calls`` calls."""
    for _ in range(3):
        call(x)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            call(x)
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    stop.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(stop) * 1000.0 / (calls * replays)


def _error_steps(out: torch.Tensor, x: torch.Tensor, dense: torch.Tensor) -> float:
    reference = torch.einsum("mgk,gnk->mgn", x.float(), dense)
    step = reference.abs() * 2.0**-11 + 1e-12
    return float(((out.float() - reference).abs() / step).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--rows-per-group", type=int, default=1024)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 96, 128, 192, 256, 512, 1024],
        help="tokens per group (rows of the GEMM)",
    )
    parser.add_argument("--calls", type=int, default=20, help="calls per graph")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--json", help="also write the table as JSON")
    args = parser.parse_args()

    torch.accelerator.set_device_index(args.device)
    import vllm._C  # noqa: F401

    from vllm.utils.torch_utils import set_high_precision_cuda_matmul_defaults

    # The numerical policy a worker sets before loading: no reduced-precision
    # FP16 reductions, so the cuBLAS arms measure what production runs.
    set_high_precision_cuda_matmul_defaults()

    name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    arms = {
        "fp16_einsum": _fp16_einsum_arm,
        "qpn8_grouped": _qpn8_arm,
        "qpn8_dense_bmm": _qpn8_dense_bmm_arm,
    }
    if capability == (7, 0):
        arms["turbomind_grouped"] = _turbomind_arm

    weight, scales, dense = _weights(args.groups, args.rows_per_group, args.k)
    prepared = {
        label: build(weight, scales, args.groups) for label, build in arms.items()
    }
    print(
        f"{name} sm{capability[0]}{capability[1]}, wo_a {args.groups} x "
        f"[{args.rows_per_group}, {args.k}] block FP8"
    )
    print(
        f"{'M':>5} {'arm':<18} {'us/call (less is better)':>26} "
        f"{'GB/s':>8} {'error (FP16 steps)':>20}"
    )
    rows = []
    for m in args.m:
        x = torch.randn(m, args.groups, args.k, device="cuda", dtype=torch.float16)
        for label, (apply, weight_bytes) in prepared.items():
            if label == "turbomind_grouped" and m > 16:
                continue
            microseconds = _graph_us(apply, x, args.calls, args.replays)
            error = _error_steps(apply(x), x, dense)
            gbps = weight_bytes / microseconds / 1e3
            rows.append(
                {
                    "device": name,
                    "m": m,
                    "arm": label,
                    "us_per_call": microseconds,
                    "gb_per_s": gbps,
                    "error_fp16_steps": error,
                }
            )
            print(
                f"{m:>5} {label:<18} {microseconds:>26.1f} {gbps:>8.0f} {error:>20.2f}"
            )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
