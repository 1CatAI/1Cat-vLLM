# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile original/extracted cache writers without opening a CUDA device.

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
        "target": "cuda/sm70/warp32; explicit target, no driver/device opened",
        "gpu_gate": "pending",
        "rows": rows,
        "all_compiled_pairs_equal": all(
            row["ptx_equal"] or row["same_unsupported_e4m3"] for row in rows
        ),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    if not result["all_compiled_pairs_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
