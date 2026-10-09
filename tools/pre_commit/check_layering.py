# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ratchet the coupling between generic vLLM layers and project specifics.

Generic modules (scheduler, runner, config, shared layers, ...) must not grow
knowledge of particular models, of the SM70 platform, or read ``VLLM_*``
environment variables directly. Existing debt is recorded per file in
``tools/pre_commit/layering_baseline.json``; a file may only reduce its counts.
See ``docs/design/architecture/README.md`` for the layering rules.

    python tools/pre_commit/check_layering.py            # check (pre-commit)
    python tools/pre_commit/check_layering.py --update   # accept reductions
    python tools/pre_commit/check_layering.py --report   # print the debt
    python tools/pre_commit/check_layering.py --accept-moves  # code moved files
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tools/pre_commit/layering_baseline.json"

# Coupling kinds counted in generic modules.
PATTERNS: dict[str, re.Pattern[str]] = {
    # Model or feature family names that belong in model/feature packages.
    "model": re.compile(
        r"qwen38|qwen3_8|qwen3\.8|qwen4_exp|qwen4exp|flash_?next|dflash|ddtree"
        r"|quasar|glm5|minimax_h3",
        re.IGNORECASE,
    ),
    # Platform knowledge that belongs behind the platform/kernel boundary.
    "platform": re.compile(
        r"sm_?70|v100|volta|is_device_capability\(\s*70\s*\)|\(\s*7\s*,\s*0\s*\)",
        re.IGNORECASE,
    ),
    # Environment reads that bypass the typed configuration.
    "env": re.compile(
        r"os\.(?:getenv|environ\.get)\(\s*[\"']VLLM_|os\.environ\[\s*[\"']VLLM_"
    ),
}

# Modules that own a model, a platform integration or a feature by design.
OWNER_PATH = re.compile(
    r"^vllm/(?:models|model_executor/models|sm70_profiles)/"
    r"|sm_?70|v100|volta|turbomind|dflash|ddtree|qwen4_exp|flash_?next|gguf"
    r"|_custom_ops\.py$|_sm70_ops\.py$",
    re.IGNORECASE,
)
# Configuration registries own every kind of knob by design.
CONFIG_OWNERS = {
    "vllm/envs.py",
    "vllm/envs_metadata.py",
    "vllm/config/kernel.py",
    "vllm/config/sm70_moe.py",
}


def tracked_python() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "vllm/*.py", "vllm/**/*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(set(out.split()))


def measure(path: str, text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if path in CONFIG_OWNERS:
        return counts
    if (
        path.startswith("vllm/v1/attention/backends/flash_v100/")
        and "/spec/" not in path
    ):
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if any(
                    isinstance(t, ast.Name) and t.id == "ROUTE_SPECS" for t in targets
                ):
                    node.value = ast.Constant("")
        hits = len(re.findall(r"dflash|ddtree|mtp|qwen|glm", ast.unparse(tree), re.I))
        if hits:
            counts["flash_v100_model"] = hits
    owner = bool(OWNER_PATH.search(path))
    for kind, pattern in PATTERNS.items():
        if kind in ("model", "platform") and owner:
            continue
        hits = len(pattern.findall(text))
        if hits:
            counts[kind] = hits
    return counts


def collect() -> dict[str, dict[str, int]]:
    result = {}
    for path in tracked_python():
        file = ROOT / path
        if not file.exists():
            continue
        counts = measure(path, file.read_text(errors="replace"))
        if counts:
            result[path] = counts
    return result


def totals_of(counts_by_file: dict[str, dict[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for counts in counts_by_file.values():
        for kind, n in counts.items():
            totals[kind] = totals.get(kind, 0) + n
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--accept-moves",
        action="store_true",
        help="record per-file growth when no coupling kind grows in total "
        "(code moved between files)",
    )
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    current = collect()
    if args.report:
        totals = totals_of(current)
        print(json.dumps({"files": len(current), "totals": totals}, indent=1))
        for path, counts in sorted(
            current.items(), key=lambda item: -sum(item[1].values())
        )[:40]:
            print(f"{sum(counts.values()):6d}  {path}  {counts}")
        return 0

    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"baseline created: {len(current)} files")
        return 0
    baseline = json.loads(BASELINE.read_text())

    errors = []
    for path, counts in current.items():
        old = baseline.get(path, {})
        for kind, n in counts.items():
            if n > old.get(kind, 0):
                errors.append(f"{path}: {kind} coupling {old.get(kind, 0)} -> {n}")
    if args.accept_moves:
        old_totals = totals_of(baseline)
        new_totals = totals_of(current)
        grown = {
            kind: (old_totals.get(kind, 0), n)
            for kind, n in new_totals.items()
            if n > old_totals.get(kind, 0)
        }
        if grown:
            print(f"total coupling grew, not a move: {grown}")
            return 1
        BASELINE.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"baseline updated after a move: {new_totals}")
        return 0
    if args.update and not errors:
        # Only reductions can be recorded; growth must be fixed instead.
        BASELINE.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"baseline updated: {len(current)} files")
        return 0
    if errors:
        print(
            "Generic modules gained model/platform/env coupling. Move the logic\n"
            "behind a platform hook, a model/feature package or KernelConfig\n"
            "(docs/design/architecture/README.md).\n  " + "\n  ".join(errors)
        )
        return 1
    stale = [
        path
        for path, counts in baseline.items()
        if any(current.get(path, {}).get(k, 0) < n for k, n in counts.items())
    ]
    if stale:
        print(
            f"{len(stale)} file(s) reduced coupling; run "
            "`python tools/pre_commit/check_layering.py --update` to lock it in."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
