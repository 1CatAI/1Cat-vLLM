# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare immutable/extracted SM70 fused norm/RoPE/cache source kernels.

Explicit AOT compilation does not qualify an installed model or a new codec.
--run additionally compares output/cache bytes and paired CUDA Graph timings.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import statistics
import subprocess
from pathlib import Path

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

BASE_SHA = "c4f6245f841466782752a8c3283e4727565cf17a"
WRITER = "vllm/model_executor/layers/attention/sm70_qwen38_qk_rope.py"


def load_source(path, source, helpers=()):
    tree = ast.Module(
        body=[
            *helpers,
            *[
                node
                for node in ast.parse(source).body
                if getattr(node, "name", None)
                in {"_e4m3_satfinite", "_qk_norm_rope", "qk_norm_rope"}
            ],
        ],
        type_ignores=[],
    )
    path.write_text(
        "import torch\nimport triton\nimport triton.language as tl\n\n"
        + ast.unparse(tree)
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_gpu(modules, constants):
    assert torch.cuda.get_device_capability() == (7, 0)
    tokens = constants["TOKENS"]
    generator = torch.Generator(device="cuda").manual_seed(20261008)
    qkv = torch.randn(
        tokens * 2, 3584, device="cuda", dtype=torch.float16, generator=generator
    )[::2]
    qw = torch.randn(256, device="cuda", dtype=torch.float16, generator=generator)
    kw = torch.randn(256, device="cuda", dtype=torch.float16, generator=generator)
    cache = torch.randn(
        10000, 64, device="cuda", dtype=torch.float16, generator=generator
    )
    positions = torch.arange(3 * (tokens + 5), device="cuda", dtype=torch.int64)
    positions = (positions.reshape(3, tokens + 5) + 1024)[:, :tokens]
    if constants["POS_PLANES"] == 1:
        positions = positions[0]
    positions[..., 0] = -1
    slots = torch.arange(2047, 2047 + tokens, device="cuda", dtype=torch.int64)
    if tokens > 1:
        slots[1] = -1
    scales = (
        torch.tensor([0.7], device="cuda", dtype=torch.float32),
        torch.tensor([1.25], device="cuda", dtype=torch.float32),
    )
    calls, hashes = [], []
    for module in modules:
        key = torch.full((3, 2048, 1, 256), 0x55, device="cuda", dtype=torch.uint8)
        value = torch.full_like(key, 0x55)
        q = torch.empty(tokens, 1536, device="cuda", dtype=torch.float16)
        k = torch.empty(tokens, 256, device="cuda", dtype=torch.float16)
        gate = torch.full_like(q, 5)

        def call(module=module, key=key, value=value, q=q, k=k, gate=gate):
            module.qk_norm_rope(
                qkv,
                qw,
                kw,
                positions,
                cache,
                key_cache=key if constants["STORE_CACHE"] else None,
                value_cache=value if constants["STORE_CACHE"] else None,
                slots=slots,
                k_scale=scales[0],
                v_scale=scales[1],
                q_out=q,
                k_out=k,
                gate_out=gate if constants["STORE_GATE"] else None,
            )

        call()
        torch.accelerator.synchronize()
        hashes.append(
            [
                hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                for t in (q, k, gate, key, value)
            ]
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        calls.append(graph)
    assert hashes[0] == hashes[1]
    for _ in range(50):
        for graph in calls:
            graph.replay()
    samples = [[], []]
    for repeat in range(8):
        for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(200):
                calls[arm].replay()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) * 1000 / 200)
    return {
        "bitwise_equal": True,
        "output_sha256": hashes[0],
        "timing_us": samples,
        "median_us": [statistics.median(values) for values in samples],
        "timing_scope": "paired graph events, warm cache; not model latency",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=True)
    original = subprocess.check_output(
        ["git", "show", f"{BASE_SHA}:{WRITER}"], cwd=root, text=True
    )
    writer = root / WRITER
    codec = root / "vllm/v1/attention/ops/kv_codec.py"
    baseline = load_source(args.out / "baseline.py", original)
    candidate = load_source(
        args.out / "candidate.py",
        writer.read_text(),
        [
            node
            for node in ast.parse(codec.read_text()).body
            if getattr(node, "name", None)
            in {"_e4m3_satfinite", "scale_kv_e4m3_per_tensor"}
        ],
    )
    # Share PTX metadata normalization with the explicit SM70 writer probe.
    spec = importlib.util.spec_from_file_location(
        "writer_probe", Path(__file__).with_name("verify_triton_writer.py")
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    rows = []
    for tokens in (1, 8, 32):
        for planes in (1, 3):
            for store_cache in (False, True):
                for store_gate in (False, True):
                    constants = dict(
                        STORE_GATE=store_gate,
                        STORE_CACHE=store_cache,
                        BLOCK_SIZE=2048 if store_cache else 0,
                        CACHE_BLOCK=524288 if store_cache else 0,
                        CACHE_TOKEN=256 if store_cache else 0,
                        POS_ROW=tokens + 5 if planes == 3 else 0,
                        POS_COL=1,
                        CACHE_ROWS=10000,
                        TOKENS=tokens,
                        NUM_SLOTS=tokens if store_cache else 0,
                        ROW=7168,
                        POS_PLANES=planes,
                        EPS=1e-6,
                    )
                    signature = {
                        p.name: (
                            "*i64"
                            if p.name in {"Pos", "Slots"}
                            else "*fp32"
                            if p.name in {"KScale", "VScale"}
                            else "*u8"
                            if p.name in {"KCache", "VCache"}
                            else "*fp16"
                        )
                        for p in candidate._qk_norm_rope.params
                        if not p.is_constexpr
                    }
                    arms = []
                    for label, module in (
                        ("baseline", baseline),
                        ("candidate", candidate),
                    ):
                        built = triton.compile(
                            ASTSource(module._qk_norm_rope, signature, constants),
                            target=GPUTarget("cuda", 70, 32),
                            options={"num_warps": 2, "enable_fp_fusion": True},
                        )
                        ptx = probe.normalize_ptx(built.asm["ptx"])
                        (args.out / f"{len(rows)}-{label}.ptx").write_text(ptx)
                        cubin = args.out / f"{len(rows)}-{label}.cubin"
                        cubin.write_bytes(built.asm["cubin"])
                        sass = subprocess.check_output(
                            ["cuobjdump", "--dump-sass", str(cubin)], text=True
                        )
                        arms.append(
                            {
                                "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                                "sass_sha256": hashlib.sha256(
                                    sass.encode()
                                ).hexdigest(),
                                "shared": built.metadata.shared,
                            }
                        )
                    assert arms[0] == arms[1], constants
                    row = {"constants": constants, "arms": arms, "code_equal": True}
                    if args.run:
                        row["gpu"] = compare_gpu((baseline, candidate), constants)
                    rows.append(row)
    result = {
        "base_sha": BASE_SHA,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=root)
        ).hexdigest(),
        "sources_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (writer, codec)
        },
        "torch": torch.__version__,
        "triton": triton.__version__,
        "target": "cuda/sm70/warp32; explicit compilation target",
        "device_execution": args.run,
        "scope": "source kernel pairs; installed artifact/model gates separate",
        "rows": rows,
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
