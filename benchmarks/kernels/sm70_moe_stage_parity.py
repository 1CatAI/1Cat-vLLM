# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same tensors, native bytes and route switches; alternate base/head timing."""

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import Parameter

from vllm import _sm70_ops, envs
from vllm.model_executor.layers.quantization import awq_sm70_moe, fp8_sm70_moe

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--root", type=Path, required=True, help="Directory containing base/ and artifacts/"
)
parser.add_argument(
    "--diagnostics", action="store_true", help="AWQ eager reference-observer parity"
)
parser.add_argument(
    "--decomposed", action="store_true", help="FP8 legacy decomposed route only"
)
args = parser.parse_args()
root = args.root
(root / "artifacts").mkdir(parents=True, exist_ok=True)
assert torch.cuda.get_device_capability() == (7, 0)
base = {}
for fmt in ("awq", "fp8"):
    spec = importlib.util.spec_from_file_location(
        "base_" + fmt,
        root / "base/vllm/model_executor/layers/quantization" / (fmt + "_sm70_moe.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base[fmt] = module

native_calls = []

for name in dir(_sm70_ops):
    native = getattr(_sm70_ops, name)
    if callable(native) and any(
        name.startswith(prefix)
        for prefix in ("awq_moe_", "fp8_moe_", "silu_and_mul_interleaved")
    ):

        def instrument(fn, name):
            @functools.wraps(fn)
            def call(*args, **kwargs):
                result = fn(*args, **kwargs)
                native_calls.append(name)
                return result

            return call

        setattr(_sm70_ops, name, instrument(native, name))

E, H, INTERMEDIATE, K = 4, 256, 256, 2
cases = []
for fmt in ("awq", "fp8"):
    if args.decomposed and fmt != "fp8":
        continue
    if args.diagnostics and fmt != "awq":
        continue
    current = awq_sm70_moe if fmt == "awq" else fp8_sm70_moe
    classname = "AWQSM70MoEMethod" if fmt == "awq" else "Fp8SM70MoEMethod"
    routes = [
        dict(name="dense", batched=False),
        dict(name="batched", batched=True),
        dict(name="indexed", batched=False, indexed=True),
        dict(name="compact", batched=False, indexed=True, compact=True),
        dict(name="legacy", batched=True, legacy=True),
    ]
    if fmt == "awq":
        routes += [
            dict(name="strict", batched=True, strict=True),
            dict(name="exact_w2", batched=True, exact=True),
            dict(name="active_w2", batched=True, active=True),
            dict(name="threshold", batched=True, threshold=2),
        ]
    else:
        routes += [dict(name="per_expert", batched=True, dispatch=True)]
    if args.diagnostics:
        routes = [
            dict(name="compare_batched", batched=True),
            dict(name="compare_active_w2", batched=True, active=True),
        ]
    if args.decomposed:
        routes = [dict(name="legacy_decomposed", batched=True, legacy=True)]
    for route in routes:
        for name in list(os.environ):
            if name.startswith("VLLM_SM70_"):
                del os.environ[name]
        flags = {
            f"VLLM_SM70_{fmt.upper()}_MOE_BATCHED_GEMM": route["batched"],
            f"VLLM_SM70_{fmt.upper()}_MOE_LEGACY_SINGLE_TOKEN_COMPACT": route.get(
                "legacy", False
            ),
            "VLLM_SM70_AWQ_MOE_COMPACT_METADATA": False,
            "VLLM_SM70_AWQ_QWEN38_MOE_INDEXED_PREFILL": False,
            "VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH": route.get(
                "indexed", False
            ),
            "VLLM_SM70_MOE_SINGLE_TOKEN_COMPACT_W13_FASTPATH": route.get(
                "compact", False
            ),
            "VLLM_SM70_AWQ_MOE_BATCHED_SINGLE_TOKEN_DENSE_W13": route.get(
                "strict", False
            ),
            "VLLM_SM70_AWQ_MOE_BATCHED_EXACT_W2": route.get("exact", False),
            "VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2": route.get("active", False),
            "VLLM_SM70_FP8_MOE_BATCHED_W13_PER_EXPERT_DISPATCH": route.get(
                "dispatch", False
            ),
            "VLLM_SM70_FP8_MOE_BATCHED_W2_PER_EXPERT_DISPATCH": route.get(
                "dispatch", False
            ),
        }
        for name, value in flags.items():
            os.environ[name] = str(int(value))
        os.environ["VLLM_SM70_AWQ_MOE_BATCHED_DECODE_MAX_TOKENS"] = str(
            route.get("threshold", 0)
        )
        if args.diagnostics:
            os.environ["VLLM_SM70_AWQ_MOE_COMPARE_DENSE_DIR"] = str(
                root / "artifacts/dense-compare"
            )
            os.environ["VLLM_SM70_AWQ_MOE_COMPARE_DENSE_LAYER_IDS"] = "*"
        if args.decomposed:
            os.environ["VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_DECOMPOSED"] = "1"
        envs.disable_envs_cache()
        torch.manual_seed(54633)
        layers = []
        methods = []
        for index, module in enumerate((base[fmt], current)):
            cls = getattr(module, classname)
            method = object.__new__(cls)
            method.moe = SimpleNamespace(experts_per_token=K)
            method.group_size = 32 if fmt == "awq" else 128
            method.pack_factor = 8
            method.quant_config = SimpleNamespace(activation_scheme="dynamic")
            method.block_quant = True
            method.compact_compare_reference = False
            method.compact_exact_layout = True
            method.compact_native_unpermute = False
            method.compact_decomposed = args.decomposed
            method.use_batched_gemm = route["batched"]
            method.use_batched_w13_per_expert_dispatch = route.get("dispatch", False)
            method.use_batched_w2_per_expert_dispatch = route.get("dispatch", False)
            method.use_permute_with_scratch = True
            layer = torch.nn.Module()
            layer.intermediate_size_per_partition = INTERMEDIATE
            layer.moe_config = SimpleNamespace(tp_size=1)
            layer.local_num_experts = E
            layer.global_num_experts = E
            layer.expert_map = None
            layer.apply_router_weight_on_input = False
            layer.layer_name = "model.layers.0.mlp.experts"
            if hasattr(method, "_initialize_sm70_policy"):
                method._initialize_sm70_policy(fmt, layer, current.logger)
            if index == 0:
                values = {}
                for prefix, n, k in [
                    ("w13", 2 * INTERMEDIATE, H),
                    ("w2", H, INTERMEDIATE),
                ]:
                    if fmt == "fp8":
                        values[prefix + "_weight"] = (
                            torch.randn(E, n, k, device="cuda") * 0.1
                        ).to(torch.float8_e4m3fn)
                        values[prefix + "_weight_scale_inv"] = (
                            torch.rand(E, n // 128, k // 128, device="cuda") * 0.1
                            + 0.05
                        )
                    else:
                        values[prefix + "_qweight"] = torch.randint(
                            0,
                            2**31 - 1,
                            (E, k, n // 8),
                            device="cuda",
                            dtype=torch.int32,
                        )
                        values[prefix + "_qzeros"] = torch.full(
                            (E, k // 32, n // 8),
                            0x77777777,
                            device="cuda",
                            dtype=torch.int32,
                        )
                        values[prefix + "_scales"] = (
                            torch.rand(
                                E, k // 32, n, device="cuda", dtype=torch.float16
                            )
                            * 0.01
                            + 0.01
                        )
            for name, value in values.items():
                setattr(layer, name, Parameter(value.clone(), requires_grad=False))
            method.process_weights_after_loading(layer)
            layers.append(layer)
            methods.append(method)
        for name in ("w13_tm_weight", "w13_tm_scales", "w2_tm_weight", "w2_tm_scales"):
            assert torch.equal(getattr(layers[0], name), getattr(layers[1], name)), (
                fmt,
                route,
                name,
            )
        for tokens in (1, 2) if args.diagnostics else (0, 1, 2, 3, 32, 33, 65):
            x = torch.randn(tokens, H, device="cuda", dtype=torch.float16) * 0.1
            ids = (torch.arange(tokens * K, device="cuda").reshape(tokens, K) % E).int()
            weights = torch.softmax(torch.randn(tokens, K, device="cuda"), dim=-1)
            out = []
            calls = []
            for method, layer in zip(methods, layers):
                native_calls.clear()
                y = method.apply(layer, x, weights, ids, None, None)
                calls.append(native_calls.copy())
                torch.accelerator.synchronize()
                out.append(y.clone())
            assert all(torch.isfinite(y).all() for y in out)
            assert torch.equal(*out), (
                fmt,
                route,
                tokens,
                (out[0] - out[1]).abs().max().item(),
            )
            assert calls[0] == calls[1], (fmt, route, tokens, calls)
            row = dict(
                format=fmt,
                route=route["name"],
                tokens=tokens,
                eager="bit_exact",
                native_calls=calls[1],
            )
            if 0 < tokens <= 32 and not args.diagnostics:
                graphs = []
                ys = []
                for method, layer in zip(methods, layers):
                    for _ in range(3):
                        method.apply(layer, x, weights, ids, None, None)
                    torch.accelerator.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        y = method.apply(layer, x, weights, ids, None, None)
                    graphs.append(g)
                    ys.append(y)
                for replay in range(3):
                    x.normal_(0, 0.1)
                    ids.copy_((ids + 1) % E)
                    for g in graphs:
                        g.replay()
                    torch.accelerator.synchronize()
                    assert torch.isfinite(ys[0]).all() and torch.equal(*ys), (
                        fmt,
                        route,
                        tokens,
                        replay,
                    )
                row["replay"] = "3 bit_exact"
                for g in graphs:
                    for _ in range(100):
                        g.replay()
                torch.accelerator.synchronize()
                times = [[], []]
                for repeat in range(6):
                    for ix in (0, 1) if repeat % 2 == 0 else (1, 0):
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(100):
                            graphs[ix].replay()
                        end.record()
                        end.synchronize()
                        times[ix].append(begin.elapsed_time(end) / 100)
                row["graph_ms"] = times
            cases.append(row)
            print(json.dumps(row), flush=True)
        del layers, methods
        torch.accelerator.empty_cache()
libs = {}
for name, module in list(sys.modules.items()):
    file = getattr(module, "__file__", None)
    if name.startswith("vllm.") and file and file.endswith(".so"):
        libs[name] = dict(
            path=file, sha256=hashlib.sha256(Path(file).read_bytes()).hexdigest()
        )
result = dict(
    device=torch.cuda.get_device_name(),
    torch=torch.__version__,
    cuda=torch.version.cuda,
    native=libs,
    cases=cases,
)
(
    root
    / (
        "artifacts/moe-diagnostics.json"
        if args.diagnostics
        else "artifacts/moe-ab.json"
    )
).write_text(json.dumps(result, indent=2))
