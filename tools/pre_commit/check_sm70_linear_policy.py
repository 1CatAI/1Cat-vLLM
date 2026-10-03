# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep migrated NVFP4 environment compatibility in its configuration adapter.

This complements #711's registration check: registered names can still be
incorrectly read at runtime rather than from a resolved per-engine config.
"""

import ast
import sys
from pathlib import Path

NAMES = {
    "VLLM_SM70_NVFP4_QPN2",
    "VLLM_SM70_NVFP4_QPN2_PREFILL",
    "VLLM_SM70_NVFP4_QPN2_SHARED_WEIGHT",
    "VLLM_SM70_NVFP4_QPN2_SHARED_SCALES",
    "VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M",
    "VLLM_SM70_AWQ_MLP_ENGINE",
    "VLLM_SM70_AWQ_PREFILL_EXACT_DENSE",
}
FP8_NAMES = {
    "VLLM_SM70_FP8_TURBOMIND",
    "VLLM_SM70_FP8_DEQUANT_FALLBACK",
    "VLLM_SM70_FP8_QPN8",
    "VLLM_SM70_FP8_QPN8_PP2_TP4",
    "VLLM_SM70_FP8_QPN8_PP2_TP4_SHARED_GATE",
    "VLLM_SM70_FP8_PRESCALED_M1_DECODE",
    "VLLM_SM70_FP8_PRESCALED_M1_SHARED_GATE",
    "VLLM_SM70_FP8_PREFILL_PRESCALED",
    "VLLM_SM70_FP8_PREFILL_EXACT_DENSE",
    "VLLM_SM70_FP8_PREFILL_VISIBLE_DENSE_MM",
    "VLLM_SM70_FP8_DENSE_GATED_SILU",
}
ALLOWED = {"vllm/envs.py", "vllm/config/kernel.py", "vllm/config/sm70_dflash2.py"}
_POLICY = Path(__file__).resolve().parents[2] / "vllm/config/sm70_dflash2.py"
_POLICY_TREE = ast.parse(_POLICY.read_text())
DFLASH_NAMES = set(
    next(
        ast.literal_eval(node.value)
        for node in _POLICY_TREE.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SM70_DFLASH2_LEGACY_FIELDS"
            for target in node.targets
        )
    )
) - {"VLLM_SM70_FP8_QPN8"}


def violations(path: Path) -> list[str]:
    if path.as_posix() in ALLOWED or "vllm" not in path.parts:
        return []
    tree = ast.parse(path.read_text())
    errors = []
    fp8_nodes = set()
    if path.name == "sm70_fp8.py":
        fp8_nodes.update(ast.walk(tree))
    elif path.name == "fp8.py":
        for candidate in tree.body:
            if (
                isinstance(candidate, ast.ClassDef)
                and candidate.name == "Fp8LinearMethod"
            ):
                fp8_nodes.update(ast.walk(candidate))
    legacy_glm_defaults = set()
    if path.as_posix() == "vllm/config/vllm.py":
        for assignment in tree.body:
            if isinstance(assignment, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id == "_SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS"
                for target in assignment.targets
            ):
                legacy_glm_defaults.update(ast.walk(assignment))
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            name = node.value
        if name in DFLASH_NAMES and node not in legacy_glm_defaults:
            errors.append(
                f"{path}:{node.lineno}: {name} belongs to the compatibility "
                "adapter; consume speculative_config.sm70_dflash2 instead"
            )
        if name in NAMES or (node in fp8_nodes and name in FP8_NAMES):
            errors.append(
                f"{path}:{node.lineno}: {name} belongs to the deprecated compatibility "
                "adapter; consume the resolved kernel_config policy instead"
            )
    return errors


def main():
    paths = [Path(name) for name in sys.argv[1:]]
    if not paths:
        paths = list(Path("vllm").rglob("*.py"))
    errors = [
        error for path in paths if path.suffix == ".py" for error in violations(path)
    ]
    print("\n".join(errors))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
