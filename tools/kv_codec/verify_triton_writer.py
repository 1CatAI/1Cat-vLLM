# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare original/extracted cache writers, optionally executing on SM70.

Target SM70 explicitly. A compiler rejection remains an unsupported compiler
path in both arms, not runtime KV support or a waived quality/performance gate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import regex as re
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

BASE_SHA = "c4f6245f841466782752a8c3283e4727565cf17a"
KERNELS = (
    "reshape_and_cache_kernel_flash",
    "reshape_and_cache_kernel_flash_diffkv",
    "_reshape_cache_per_token_head",
)


def compare_gpu_outputs(row, modules):
    """Exercise strided inputs, page boundaries, padding and untouched bytes."""
    import torch

    assert torch.cuda.get_device_capability() == (7, 0)
    name, dim = row["kernel"], row["dim"]
    dtype = {"fp16": torch.float16, "fp8e5": torch.float8_e5m2, "i8": torch.int8}[
        row["output_type"]
    ]
    heads, page, blocks = 2, 16, 3
    slots = torch.tensor([1, 15, 16, 31, 32, -1, 7, 3, 24], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(20261008)
    value_dim = dim // 2 if name == KERNELS[1] else dim
    key = torch.randn(
        (18, heads, dim), device="cuda", dtype=torch.float16, generator=generator
    )[::2]
    value = torch.randn(
        (18, heads, value_dim),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )[::2]
    key[0].zero_()  # Dynamic scale floor; leave other tokens nonzero.
    k_scale = torch.tensor(0.75, device="cuda")
    v_scale = torch.tensor(1.25, device="cuda")
    results = []

    def cache(shape, cache_dtype=dtype):
        # Check all unaddressed cache/scale bytes, not only live slots.
        count = 1
        for extent in shape:
            count *= extent
        item_size = torch.empty((), dtype=cache_dtype).element_size()
        return (
            torch.full((count * item_size,), 0xA5, device="cuda", dtype=torch.uint8)
            .view(cache_dtype)
            .reshape(shape)
        )

    for module in modules:
        kwargs = dict(
            key_ptr=key,
            value_ptr=value,
            slot_mapping_ptr=slots,
            k_scale=k_scale,
            v_scale=v_scale,
            key_stride=key.stride(0),
            value_stride=value.stride(0),
            **row["constexprs"],
        )
        outputs = []
        if name == KERNELS[1]:
            output = cache((blocks, page, heads, dim + value_dim))
            kwargs.update(
                kv_cache_ptr=output,
                block_stride=output.stride(0),
                page_stride=output.stride(1),
            )
            outputs.append(output)
            grid = (slots.numel(), heads)
        else:
            if row["head_major"]:
                kc = cache((blocks, heads, dim // 8, page, 8))
                vc = cache((blocks, heads, dim, page))
            else:
                kc = cache((blocks, page, heads, dim))
                vc = cache((blocks, page, heads, value_dim))
            kwargs.update(key_cache_ptr=kc, value_cache_ptr=vc)
            outputs.extend((kc, vc))
            if name == KERNELS[0]:
                kwargs.update(
                    block_stride=kc.stride(0),
                    head_stride=kc.stride(1 if row["head_major"] else 2),
                    dim_stride_k=kc.stride(2) if row["head_major"] else 0,
                    dim_stride_v=vc.stride(2) if row["head_major"] else 0,
                    page_stride=kc.stride(1),
                )
                grid = (slots.numel(), triton.cdiv(heads * dim, 256))
            else:
                ks = cache((blocks, page, heads), torch.float32)
                vs = cache((blocks, page, heads), torch.float32)
                kwargs.update(k_scale_cache_ptr=ks, v_scale_cache_ptr=vs)
                outputs.extend((ks, vs))
                for prefix, tensor, axes in (
                    ("key", key, ("tok", "head")),
                    ("val", value, ("tok", "head")),
                    ("kc", kc, ("blk", "slot", "head")),
                    ("vc", vc, ("blk", "slot", "head")),
                    ("ks", ks, ("blk", "slot", "head")),
                    ("vs", vs, ("blk", "slot", "head")),
                ):
                    kwargs.update(
                        {
                            f"stride_{prefix}_{axis}": tensor.stride(i)
                            for i, axis in enumerate(axes)
                        }
                    )
                grid = (slots.numel(), heads)
        fn = getattr(module, name)
        fn[grid](
            **{p.name: kwargs[p.name] for p in fn.params},
            num_warps=min(16, max(1, dim // 32)),
        )
        torch.accelerator.synchronize()
        results.append(
            [
                hashlib.sha256(
                    t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest()
                for t in outputs
            ]
        )
    assert results[0] == results[1], row
    return {"bitwise_equal": True, "output_sha256": results[0], "tokens": 9}


def normalize_ptx(text):
    # Only non-executable source/debug metadata; retain all instructions.
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.split(r"\.section\s+\.debug_", text, maxsplit=1)[0]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or re.match(r"\.(?:file|loc)\s", stripped):
            continue
        if re.fullmatch(r"\$L__tmp\d+:", stripped):
            continue
        # Debug marker labels must never be branch/data targets in retained code.
        assert not re.search(r"\$L__tmp\d+", stripped)
        lines.append(line.rstrip())
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--run", action="store_true", help="Execute compiled pairs on SM70"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=True)
    original = subprocess.check_output(
        [
            "git",
            "show",
            f"{BASE_SHA}:vllm/v1/attention/ops/triton_reshape_and_cache_flash.py",
        ],
        cwd=root,
        text=True,
    )
    source_path = root / "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py"
    codec_path = source_path.with_name("kv_codec.py")

    def load_source(arm, source, helpers=()):
        tree = ast.Module(
            body=[
                *helpers,
                *[
                    n
                    for n in ast.parse(source).body
                    if getattr(n, "name", None) in KERNELS
                ],
            ],
            type_ignores=[],
        )
        path = args.out / f"{arm}_writer.py"
        # vLLM supplies non-JIT placeholders without a GPU. Compile the exact
        # kernel/helper ASTs with real Triton for this offline source probe;
        # no production platform/driver behavior is patched.
        path.write_text(
            "import triton\nimport triton.language as tl\n\n" + ast.unparse(tree)
        )
        spec = importlib.util.spec_from_file_location(f"kv_codec_{arm}_writer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    baseline = load_source("baseline", original)
    candidate = load_source(
        "candidate",
        source_path.read_text(),
        helpers=[
            n
            for n in ast.parse(codec_path.read_text()).body
            if isinstance(n, ast.FunctionDef)
        ],
    )
    rows = []
    for name in KERNELS:
        fn = getattr(candidate, name)
        for dim in (64, 256):
            for output_type in ("fp16", "fp8e5", "fp8e4nv", "i8"):
                dynamic = name == "_reshape_cache_per_token_head"
                if (dynamic and output_type == "fp16") or (
                    not dynamic and output_type == "i8"
                ):
                    continue
                for layout in (False, True) if name == KERNELS[0] else (False,):
                    constants = {
                        "num_heads": 2,
                        "head_size": dim,
                        "head_size_k": dim,
                        "head_size_v": dim // 2 if name == KERNELS[1] else dim,
                        "block_size": 16,
                        "x": 8,
                        "USE_HEAD_MAJOR_LAYOUT": layout,
                        "FP8_KV_CACHE": output_type != "fp16",
                        "TILE_SIZE": 256,
                        "HEAD_SIZE_PADDED": dim,
                        "QUANT_MAX": 127.0 if output_type == "i8" else 448.0,
                        "QUANT_MIN": -128.0 if output_type == "i8" else -448.0,
                    }
                    constexprs = {
                        p.name: constants[p.name] for p in fn.params if p.is_constexpr
                    }
                    signature = {}
                    for p in fn.params:
                        if p.is_constexpr:
                            continue
                        if p.name in ("key_ptr", "value_ptr"):
                            signature[p.name] = "*fp16"
                        elif p.name == "slot_mapping_ptr":
                            signature[p.name] = "*i64"
                        elif "scale" in p.name:
                            signature[p.name] = "*fp32"
                        elif "cache_ptr" in p.name:
                            signature[p.name] = f"*{output_type}"
                        else:
                            signature[p.name] = "i64"
                    arms = {}
                    index = len(rows)
                    for arm, module in (
                        ("baseline", baseline),
                        ("candidate", candidate),
                    ):
                        try:
                            built = triton.compile(
                                ASTSource(getattr(module, name), signature, constexprs),
                                target=GPUTarget("cuda", 70, 32),
                                options={"num_warps": min(16, max(1, dim // 32))},
                            )
                        except Exception as error:
                            arms[arm] = {
                                "compiled": False,
                                "exception_type": type(error).__name__,
                                "error": str(error),
                            }
                            (args.out / f"{index}-{arm}.error").write_text(str(error))
                        else:
                            ptx = built.asm["ptx"]
                            (args.out / f"{index}-{arm}.ptx").write_text(ptx)
                            cubin = args.out / f"{index}-{arm}.cubin"
                            cubin.write_bytes(built.asm["cubin"])
                            sass = subprocess.check_output(
                                ["cuobjdump", "--dump-sass", str(cubin)], text=True
                            )
                            (args.out / f"{index}-{arm}.sass").write_text(sass)
                            arms[arm] = {
                                "compiled": True,
                                "ptx_sha256": hashlib.sha256(
                                    normalize_ptx(ptx).encode()
                                ).hexdigest(),
                                "sass_sha256": hashlib.sha256(
                                    sass.encode()
                                ).hexdigest(),
                                "shared": built.metadata.shared,
                                "num_warps": built.metadata.num_warps,
                            }
                    both_compiled = all(a["compiled"] for a in arms.values())
                    same_rejection = (
                        not any(a["compiled"] for a in arms.values())
                        and arms["baseline"]["exception_type"]
                        == arms["candidate"]["exception_type"]
                        and "fp8e4nv" in arms["baseline"]["error"]
                        and "fp8e4nv" in arms["candidate"]["error"]
                    )
                    equal = both_compiled and arms["baseline"] == arms["candidate"]
                    rows.append(
                        dict(
                            kernel=name,
                            dim=dim,
                            output_type=output_type,
                            head_major=layout,
                            constexprs=constexprs,
                            signature=signature,
                            arms=arms,
                            ptx_equal=equal,
                            same_unsupported_e4m3=same_rejection,
                        )
                    )
                    if args.run and equal:
                        rows[-1]["gpu_outputs"] = compare_gpu_outputs(
                            rows[-1], (baseline, candidate)
                        )
    result = {
        "base_sha": BASE_SHA,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=root)
        ).hexdigest(),
        "triton": triton.__version__,
        "cuobjdump": subprocess.check_output(["cuobjdump", "--version"], text=True),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source_path, codec_path)
        },
        "target": "cuda/sm70/warp32; explicit compilation target",
        "device_execution": args.run,
        "gpu_gate": "compiled pairs bitwise equal" if args.run else "pending",
        "rows": rows,
        "all_compiled_pairs_equal": all(
            row["ptx_equal"] or row["same_unsupported_e4m3"] for row in rows
        ),
    }
    if args.run:
        import torch

        result["runtime"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(torch.cuda.get_device_properties(0)),
        }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    if not result["all_compiled_pairs_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
