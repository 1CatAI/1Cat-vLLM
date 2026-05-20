"""Fleet regression runner — launch each model, run its suite, report.

Usage:
    python -m tests.fleet.run                            # all models, all suites
    python -m tests.fleet.run --models granite,qwen36
    python -m tests.fleet.run --models granite --suites smoke,perf_t1
    python -m tests.fleet.run --report /tmp/fleet.md     # write summary to file
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from tests.fleet import engine, measurements, report
from tests.fleet.measurements import SUITE_FUNCS
from tests.fleet.registry import REGISTRY


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--models", default="",
        help="Comma-separated model keys from registry (default: all in order)",
    )
    p.add_argument(
        "--suites", default="",
        help="Comma-separated suite names (default: each model's full suite list)",
    )
    p.add_argument(
        "--report", default="",
        help="Write markdown summary to this file (also always printed to stdout)",
    )
    p.add_argument(
        "--log-dir", default="/tmp/fleet",
        help="Directory for per-model launch logs (default: /tmp/fleet)",
    )
    return p.parse_args()


def run_model(model_key: str, suites: list[str], log_dir: Path) -> tuple[int | None, list[dict]]:
    """Launch one model, run its suite, return (kv_cache_tokens, list_of_results)."""
    cfg = REGISTRY[model_key]
    log_path = str(log_dir / f"{model_key}_launch.log")
    results: list[dict] = []
    kv_tokens: int | None = None

    print(f"\n========== {model_key} ==========", flush=True)
    print(f"[fleet] launch script: {cfg.launch_script}", flush=True)
    print(f"[fleet] log: {log_path}", flush=True)

    try:
        with engine.running_engine(cfg.launch_script, log_path, cfg.ready_timeout_s):
            kv_tokens = engine.kv_cache_size_tokens(log_path)
            print(f"[fleet] ready. KV cache: {kv_tokens} tokens", flush=True)
            for suite_name in suites:
                func = SUITE_FUNCS.get(suite_name)
                if func is None:
                    print(f"[fleet] unknown suite {suite_name!r}, skipping", flush=True)
                    continue
                print(f"[fleet]   running {suite_name}...", flush=True)
                try:
                    r = func(cfg)
                except Exception as e:
                    r = {
                        "name": suite_name, "passed": False, "elapsed_s": 0.0,
                        "details": {}, "error": f"{type(e).__name__}: {e}",
                    }
                    traceback.print_exc()
                results.append(r)
                tag = "PASS" if r["passed"] else "FAIL"
                print(f"[fleet]   -> {tag} ({r['elapsed_s']:.1f}s)", flush=True)
    except engine.EngineLaunchError as e:
        results.append({
            "name": "engine_launch", "passed": False, "elapsed_s": 0.0,
            "details": {}, "error": str(e),
        })
        print(f"[fleet] LAUNCH FAILED: {e}", flush=True)

    return kv_tokens, results


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    requested_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models else list(REGISTRY.keys())
    )
    unknown = [m for m in requested_models if m not in REGISTRY]
    if unknown:
        print(f"Unknown model(s): {unknown}. Valid: {list(REGISTRY.keys())}",
              file=sys.stderr)
        return 2

    requested_suites = (
        [s.strip() for s in args.suites.split(",") if s.strip()]
        if args.suites else None  # use per-model default
    )

    per_model: list[tuple[str, int | None, list[dict]]] = []
    t_total = time.perf_counter()
    for model_key in requested_models:
        cfg = REGISTRY[model_key]
        suites = list(requested_suites or cfg.suites)
        kv_tokens, results = run_model(model_key, suites, log_dir)
        per_model.append((model_key, kv_tokens, results))

    # Render report
    sections = [report.format_overall([(m, rs) for m, _, rs in per_model])]
    for model_key, kv_tokens, results in per_model:
        sections.append(report.format_model(model_key, kv_tokens, results))
    summary = "\n".join(sections)
    summary += f"\nTotal wall time: {time.perf_counter() - t_total:.1f}s\n"

    print("\n" + summary, flush=True)
    if args.report:
        Path(args.report).write_text(summary)
        print(f"[fleet] summary written to {args.report}", flush=True)

    all_passed = all(r["passed"] for _, _, rs in per_model for r in rs)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
