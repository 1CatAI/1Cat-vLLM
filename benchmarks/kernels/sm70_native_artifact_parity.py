# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fresh-process before/after native artifact checks (operator scope).

Run once in the baseline installation with --output base.pt, then in the
candidate with --output head.pt --reference base.pt and identical environment.
Each installation must supply its normal packaged native libraries. This
synthetic operator test provides no model throughput or text-quality evidence."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from torch.nn import Parameter
from torch.utils._python_dispatch import TorchDispatchMode

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.quantization import sm70_turbomind as tm

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference", type=Path)
parser.add_argument(
    "--gguf-cache",
    type=Path,
    help="Export baseline GGUF tuning choices and reuse them after candidate warmup",
)
args = parser.parse_args()
assert not os.getenv("LD_PRELOAD")
assert torch.cuda.get_device_capability() == (7, 0)


class Trace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.names = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func._schema.name.split("::")[0] in ("_C", "_moe_C", "_C_gguf"):
            self.names.append(func._schema.name)
        return result


def parameter(x):
    return Parameter(x, requires_grad=False)


def rand(shape):
    return (torch.randn(shape, device="cuda") * 0.1).half()


def fixture(family):
    torch.manual_seed(54633)
    n = k = 256
    layer = torch.nn.Module()
    layer.prefix = "parity." + family
    bias = rand((n,))
    if family in ("nvfp4", "mxfp4"):
        packed = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
        group = 16 if family == "nvfp4" else 32
        scales = torch.full((n, k // group), 0.0625, device="cuda", dtype=torch.float16)
        if family == "nvfp4":
            layer.weight = parameter(packed)
            layer.weight_scale = parameter(scales)
            layer.weight_global_scale = parameter(torch.tensor(0.25, device="cuda"))
            tm.prepare_nvfp4_linear(layer)
        else:
            layer.weight_packed = parameter(packed)
            layer.weight_scale = parameter(
                torch.full((n, k // group), 123, device="cuda", dtype=torch.uint8)
            )
            tm.prepare_mxfp4_linear(layer)
        return k, lambda x: tm.apply_prepared_linear(layer, x, bias)
    if family == "awq":
        from vllm.model_executor.kernels.linear.mixed_precision.sm70_awq import (
            Sm70AwqLinearLayerConfig,
            TurboMindAwqLinearKernel,
        )
        from vllm.scalar_type import scalar_types

        cfg = Sm70AwqLinearLayerConfig(
            (k, n), (k, n), scalar_types.uint4, torch.float16, 32, True, False
        )
        kernel = TurboMindAwqLinearKernel(cfg, "qweight", "scales", "qzeros")
        layer.qweight = parameter(
            torch.randint(0, 2**31 - 1, (k, n // 8), device="cuda", dtype=torch.int32)
        )
        layer.qzeros = parameter(
            torch.full((k // 32, n // 8), 0x77777777, device="cuda", dtype=torch.int32)
        )
        layer.scales = parameter(
            torch.full((k // 32, n), 0.0125, device="cuda", dtype=torch.float16)
        )
        layer.output_partition_sizes = [n]
        layer.output_size_per_partition = n
        kernel.process_weights_after_loading(layer)
        return k, lambda x: kernel.apply_weights(layer, x, bias)
    if family in ("fp8", "qpn8"):
        from vllm.model_executor.kernels.linear.scaled_mm.qpn8_blk import (
            QPN8Fp8BlockScaledMMLinearKernel,
        )
        from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
            FP8ScaledMMLinearLayerConfig,
        )
        from vllm.model_executor.kernels.linear.scaled_mm.sm70_fp8 import (
            Sm70Fp8LinearLayerConfig,
            TurboMindFp8LinearKernel,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8Dynamic128Sym,
            kFp8Static128BlockSym,
        )

        layer.weight = parameter(rand((n, k)).to(torch.float8_e4m3fn))
        layer.weight_scale_inv = parameter(
            torch.full((n // 128, k // 128), 0.125, device="cuda")
        )
        layer.output_partition_sizes = [n]
        layer.output_size_per_partition = n
        layer.orig_dtype = torch.float16
        layer.input_size_per_partition = k
        if family == "qpn8":
            config = FP8ScaledMMLinearLayerConfig(
                kFp8Static128BlockSym,
                kFp8Dynamic128Sym,
                (n, k),
                torch.float16,
                torch.float16,
            )
            kernel = QPN8Fp8BlockScaledMMLinearKernel(config)
        else:
            config = Sm70Fp8LinearLayerConfig(
                kFp8Static128BlockSym,
                kFp8Dynamic128Sym,
                (n, k),
                torch.float16,
                torch.float16,
            )
            kernel = TurboMindFp8LinearKernel(config, ())
        kernel.process_weights_after_loading(layer)
        return k, lambda x: kernel.apply_weights(layer, x, bias)
    if family == "gguf":
        from vllm.model_executor.kernels.linear.mixed_precision.sm70_gguf import (
            Sm70GgufAffineConfig,
            TurboMindGgufAffineKernel,
        )
        from vllm.scalar_type import scalar_types

        config = Sm70GgufAffineConfig(
            (k, n),
            (k, n),
            scalar_types.uint4,
            torch.float16,
            32,
            True,
            False,
            source_type=3,
        )
        kernel = TurboMindGgufAffineKernel(config, "codes", "scales", "mins")
        layer.codes = parameter(
            torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
        )
        layer.scales = parameter(
            torch.full((n, k // 32), 0.0125, device="cuda", dtype=torch.float16)
        )
        layer.mins = parameter(
            torch.full((n, k // 32), 0.00125, device="cuda", dtype=torch.float16)
        )
        kernel.process_weights_after_loading(layer)

        def apply_gguf(x):
            return kernel.apply_weights(layer, x, bias)

        if args.gguf_cache:
            owner = getattr(kernel, "native_ops", ops)
            hint = torch.empty(0, device="cuda")
            if args.reference:
                # GGUF deliberately measures each cold descriptor, regardless
                # of AWQ tuning flags. Populate its warmup set first, then
                # import the same measured choices for deterministic A/B.
                for rows in (1, 2, 8, 9, 32):
                    apply_gguf(torch.zeros(rows, k, device="cuda", dtype=torch.float16))
                torch.accelerator.synchronize()
                assert owner.sm70_gemm_import_cache(hint, str(args.gguf_cache)) > 0
            apply_gguf.export_cache = lambda: owner.sm70_gemm_export_cache(
                hint, str(args.gguf_cache)
            )
        return k, apply_gguf
    fmt = family.removeprefix("moe_")
    from vllm.model_executor.layers.quantization import awq_sm70_moe, fp8_sm70_moe

    module = awq_sm70_moe if fmt == "awq" else fp8_sm70_moe
    cls = module.AWQSM70MoEMethod if fmt == "awq" else module.Fp8SM70MoEMethod
    method = object.__new__(cls)
    method.moe = NS(experts_per_token=2)
    method.group_size = 32 if fmt == "awq" else 128
    method.pack_factor = 8
    method.quant_config = NS(activation_scheme="dynamic")
    method.block_quant = True
    method.compact_compare_reference = False
    method.compact_exact_layout = True
    method.compact_native_unpermute = False
    method.compact_decomposed = False
    method.use_batched_gemm = False
    method.use_permute_with_scratch = True
    method.use_batched_w13_per_expert_dispatch = False
    method.use_batched_w2_per_expert_dispatch = False
    layer.intermediate_size_per_partition = n
    layer.moe_config = NS(tp_size=1)
    layer.local_num_experts = 4
    layer.global_num_experts = 4
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    layer.layer_name = layer.prefix
    method._initialize_sm70_policy(fmt, layer, module.logger)
    for prefix, nc, kc in [("w13", 2 * n, k), ("w2", k, n)]:
        if fmt == "awq":
            setattr(
                layer,
                prefix + "_qweight",
                parameter(
                    torch.randint(
                        0, 2**31 - 1, (4, kc, nc // 8), device="cuda", dtype=torch.int32
                    )
                ),
            )
            setattr(
                layer,
                prefix + "_qzeros",
                parameter(
                    torch.full(
                        (4, kc // 32, nc // 8),
                        0x77777777,
                        device="cuda",
                        dtype=torch.int32,
                    )
                ),
            )
            setattr(
                layer,
                prefix + "_scales",
                parameter(
                    torch.full(
                        (4, kc // 32, nc), 0.0125, device="cuda", dtype=torch.float16
                    )
                ),
            )
        else:
            setattr(
                layer,
                prefix + "_weight",
                parameter(rand((4, nc, kc)).to(torch.float8_e4m3fn)),
            )
            setattr(
                layer,
                prefix + "_weight_scale_inv",
                parameter(torch.full((4, nc // 128, kc // 128), 0.125, device="cuda")),
            )
    method.process_weights_after_loading(layer)
    routing = {}

    def apply(x):
        m = x.shape[0]
        if m not in routing:
            routing[m] = (
                (torch.arange(m * 2, device="cuda").reshape(m, 2) % 4).int(),
                torch.full((m, 2), 0.5, device="cuda"),
            )
        ids, weights = routing[m]
        return method.apply(layer, x, weights, ids, None, None)

    def references():
        for mode, m in (
            ("compact", 1),
            ("batched", 1),
            ("dense", 3),
            ("per_expert", 3),
        ):
            torch.manual_seed(717 + m)
            x = rand((m, k))
            ids = (torch.arange(m * 2, device="cuda").reshape(m, 2) % 4).int()
            weights = torch.full((m, 2), 0.5, device="cuda")
            buffers = method._get_buffers(layer, m * 2, m)
            layer.sm70_fp8_moe_batched_gemm = mode != "dense"
            layer.sm70_fp8_moe_batched_w13_per_expert_dispatch = mode == "per_expert"
            layer.sm70_fp8_moe_batched_w2_per_expert_dispatch = mode == "per_expert"
            operation = (
                method._apply_compact_reference_for_compare
                if mode == "compact"
                else method._apply_batched_reference_for_compare
            )
            with Trace() as trace:
                tensors = operation(layer, x, weights, ids, buffers, 2)
            torch.accelerator.synchronize()
            yield (
                mode,
                {key: value.cpu().clone() for key, value in tensors.items()},
                trace.names,
            )

    if fmt == "fp8":
        apply.references = references

    return k, apply


rows = []
outputs = {}
for family in ("awq", "fp8", "nvfp4", "mxfp4", "qpn8", "gguf", "moe_awq", "moe_fp8"):
    k, apply = fixture(family)
    print("prepare " + family, flush=True)
    rows_to_check = (
        (0, 1, 2, 8, 9, 32, 33, 64, 65, 128)
        if family in ("qpn8", "gguf", "moe_awq", "moe_fp8")
        else (1, 2, 8, 9, 32, 33, 64, 65, 128)
    )
    for m in rows_to_check:
        torch.manual_seed(917 + m)
        x = rand((m, k))
        name = f"{family}:M{m}"
        try:
            with Trace() as trace:
                y = apply(x)
        except ValueError as error:
            if m != 0:
                raise
            rows.append(
                {"case": name, "native_calls": trace.names, "error": str(error)}
            )
            print(name + " rejected: " + str(error), flush=True)
            continue
        torch.accelerator.synchronize()
        assert torch.isfinite(y).all(), name
        outputs[name + ":eager"] = y.cpu().clone()
        row = {"case": name, "native_calls": trace.names, "eager_shape": list(y.shape)}
        if family == "qpn8" and m in (1, 33):
            # TorchDispatchMode stops at this provider's opaque custom op.
            # Observe its nested native launch on both sides of the M boundary.
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                apply(x)
                torch.accelerator.synchronize()
            row["observed_native_operators"] = [
                event.name
                for event in profiler.events()
                if event.name.startswith(("_C::", "_moe_C::"))
            ]
            row["cuda_kernels"] = sorted(
                {
                    event.name
                    for event in profiler.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                }
            )
            assert row["observed_native_operators"] and row["cuda_kernels"]
        if m:
            for _ in range(4):
                apply(x)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gy = apply(x)
            for replay in range(3):
                x.mul_(0.875).add_(0.00125)
                graph.replay()
                torch.accelerator.synchronize()
                outputs[name + f":replay{replay}"] = gy.cpu().clone()
                assert torch.equal(gy, apply(x)), (name, "capture/replay mismatch")
            graph_times = []
            eager_times = []
            for _ in range(5):
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                for _ in range(100):
                    graph.replay()
                end.record()
                end.synchronize()
                graph_times.append(begin.elapsed_time(end) * 10)
                start = time.perf_counter()
                for _ in range(30):
                    apply(x)
                torch.accelerator.synchronize()
                eager_times.append((time.perf_counter() - start) * 1e6 / 30)
            row.update(
                graph_us=statistics.median(graph_times),
                eager_us=statistics.median(eager_times),
            )
        rows.append(row)
        print(name, flush=True)
    if family == "gguf" and args.gguf_cache and not args.reference:
        assert apply.export_cache() > 0
    if family == "moe_fp8":
        for mode, tensors, calls in apply.references():
            name = "fp8_reference:" + mode
            for stage, value in tensors.items():
                assert torch.isfinite(value).all(), (name, stage)
                outputs[name + ":" + stage] = value
            rows.append({"case": name, "native_calls": calls})
            print(name, flush=True)
root = Path(__import__("vllm").__file__).parent
record = {
    "contract": {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "env": {
            k: v for k, v in os.environ.items() if k.startswith(("VLLM_SM70_", "TM_"))
        },
    },
    "native": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.so")
    },
    "rows": rows,
}
if args.gguf_cache:
    record["contract"]["gguf_cache_sha256"] = hashlib.sha256(
        args.gguf_cache.read_bytes()
    ).hexdigest()
torch.save(outputs, args.output)
args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
if args.reference:
    old = torch.load(args.reference, weights_only=True)
    assert outputs.keys() == old.keys()
    mismatches = {
        name: (value - old[name]).abs().max().item()
        for name, value in outputs.items()
        if not torch.equal(value, old[name])
    }
    record["mismatches"] = mismatches
    reference = json.loads(args.reference.with_suffix(".json").read_text())
    assert reference["contract"] == record["contract"]
    for old, new in zip(reference["rows"], rows, strict=True):
        assert (
            old["case"] == new["case"]
            and old["native_calls"] == new["native_calls"]
            and old.get("error") == new.get("error")
            and old.get("observed_native_operators")
            == new.get("observed_native_operators")
        ), (old, new)
        if "graph_us" in new:
            new["graph_change_pct"] = 100 * (new["graph_us"] / old["graph_us"] - 1)
    record["exact_outputs"] = len(outputs) - len(mismatches)
args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
if args.reference:
    assert not mismatches, mismatches
