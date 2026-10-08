# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the SM70 KV migration inventory without importing vLLM/CUDA.

The broad search deliberately includes weight-quantization false positives.
Keep its full evidence outside Git; review KV-specific consumers separately.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import regex as re

SEARCH = re.compile(r"fp8|e4m3|kv_cache_dtype|_record_route|uint8", re.IGNORECASE)
BACKEND = "vllm/v1/attention/backends/flash_attn_v100.py"
DECODE = "flash-attention-v100/kernel/flash_decode_paged.cu"
SUFFIXES = {".py", ".cu", ".cuh", ".h", ".cpp", ".cc", ".sh", ".cmake"}


def scan_python(source: str) -> dict:
    tree = ast.parse(source)
    route_calls = []
    env_reads = set()
    dtype_tests = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.Assert)):
            condition = ast.unparse(node.test)
            if re.search(r"(?:kv_)?cache_dtype", condition):
                dtype_tests.append({"line": node.lineno, "condition": condition})
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "envs"
            and node.attr.startswith("VLLM_")
        ):
            env_reads.add(node.attr)
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "_record_route":
            route_calls.append(
                {"line": node.lineno, "expression": ast.unparse(node.args[0])}
            )
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"getenv", "get"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("VLLM_")
        ):
            env_reads.add(node.args[0].value)
    literal_routes = sorted(
        {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_record_route"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
    )
    return {
        "literal_routes": literal_routes,
        "route_calls": sorted(route_calls, key=lambda row: row["line"]),
        "env_reads": sorted(env_reads),
        "dtype_tests": sorted(dtype_tests, key=lambda row: row["line"]),
        "dtype_mentions": len(re.findall(r"\bkv_cache_dtype\b", source)),
    }


def collect(root: Path) -> dict:
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"], text=True
    ).split("\0")
    files = []
    for name in sorted(tracked):
        if Path(name).suffix not in SUFFIXES:
            continue
        path = root / name
        if not path.is_file():
            continue
        source = path.read_text(errors="replace")
        hits = [
            {"line": number, "text": line.strip()}
            for number, line in enumerate(source.splitlines(), 1)
            if SEARCH.search(line)
        ]
        if not hits:
            continue
        files.append(
            {
                "path": name,
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "kv_dtype_lines": [
                    hit["line"] for hit in hits if "kv_cache_dtype" in hit["text"]
                ],
                "hits": hits,
            }
        )
    backend_source = (root / BACKEND).read_text()
    attention_modules = {
        path.relative_to(root).as_posix(): scan_python(path.read_text())
        for path in sorted((root / BACKEND).parent.joinpath("flash_v100").glob("*.py"))
    }
    backend = scan_python(backend_source)
    related = [backend, *attention_modules.values()]
    return {
        "source_sha": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "scope": "tracked source; broad matches are candidates, not confirmed KV use",
        "backend_lines": len(backend_source.splitlines()),
        "decode_lines": len((root / DECODE).read_text().splitlines()),
        "backend": backend,
        "attention_modules": attention_modules,
        "backend_and_modules": {
            "env_reads": sorted(
                {name for item in related for name in item["env_reads"]}
            ),
            "dtype_mentions": sum(item["dtype_mentions"] for item in related),
            "dtype_tests": sum(len(item["dtype_tests"]) for item in related),
            "route_calls": sum(len(item["route_calls"]) for item in related),
        },
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    backend = result["backend"]
    print(
        json.dumps(
            {
                "source_sha": result["source_sha"],
                "backend_lines": result["backend_lines"],
                "decode_lines": result["decode_lines"],
                "dtype_mentions": backend["dtype_mentions"],
                "dtype_tests": len(backend["dtype_tests"]),
                "env_reads": len(backend["env_reads"]),
                "backend_and_modules": {
                    **result["backend_and_modules"],
                    "env_reads": len(result["backend_and_modules"]["env_reads"]),
                },
                "literal_routes": len(backend["literal_routes"]),
                "route_calls": len(backend["route_calls"]),
                "candidate_files": len(result["files"]),
                "kv_dtype_files": sum(
                    bool(f["kv_dtype_lines"]) for f in result["files"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
