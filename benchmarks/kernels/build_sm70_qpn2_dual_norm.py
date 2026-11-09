# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive a private dual-store norm; preserve the original expressions."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    W = Path(__file__).resolve().parents[2]
    source_path = W / "vllm/model_executor/layers/layernorm.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    parts = ["from vllm.triton_utils import tl, triton\n"]
    for name in (
        "_sm70_dflash2_fixed_gemma_rms_kernel",
        "_sm70_dflash2_gemma_fused_add_rms_kernel",
    ):
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        segment = "@triton.jit\n" + ast.get_source_segment(source, node)
        segment = segment.replace(name, name + "_dual_store", 1)
        segment = segment.replace(
            "    residual_out,\n", "    residual_out,\n    packed_out,\n", 1
        )
        store = next(n for n in reversed(node.body) if isinstance(n, ast.Expr))
        tail = ast.get_source_segment(source, store)
        old_index = (
            "normalized_out + row * 5120 + cols"
            if "fixed" in name
            else "normalized_out + row * hidden_size + cols"
        )
        assert tail.count(old_index) == 1
        tail = tail.replace(
            old_index, "packed_out + (cols // 16) * 128 + row * 16 + cols % 16"
        )
        segment += "\n    " + tail + "\n"
        parts.append(segment)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts))
    manifest = dict(
        parent_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        scope="private dual-store q8 node candidate, not installed",
    )
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
