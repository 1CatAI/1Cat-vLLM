# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete native H3 warmup and three-run performance measurements.

Run with ``python -m vllm.video.benchmark --contract contract.json --output DIR``.
The contract contains the same ``config`` and ``request`` as native run.json.
"""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from vllm.model_executor.models.minimax_h3.config import (
    H3Config,
    H3Request,
    H3SamplingParams,
)
from vllm.video.engine import H3Engine
from vllm.video.metrics import evaluate_performance


def source_provenance():
    import importlib.metadata

    import torch

    import vllm

    package = Path(vllm.__file__).resolve().parent
    root = package.parent
    provenance = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "vllm": vllm.__version__,
        "package_path": str(package),
        "sources_sha256": {},
        "dependencies": {},
    }
    for name in ("cache-dit", "huggingface-hub", "diffusers", "transformers"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        record: dict = {"version": distribution.version}
        if name == "cache-dit":
            record["sources_sha256"] = {
                str(path): hashlib.sha256(
                    Path(str(distribution.locate_file(path))).read_bytes()
                ).hexdigest()
                for path in distribution.files or ()
                if str(path).startswith("cache_dit/") and str(path).endswith(".py")
            }
        provenance["dependencies"][name] = record
    paths = [
        *package.joinpath("video").glob("*.py"),
        *package.joinpath("model_executor/models/minimax_h3").glob("*.py"),
        package / "model_executor/layers/linear.py",
        package / "model_executor/layers/sm70_diffusion.py",
        package / "model_executor/layers/sm70_sparse_attention.py",
    ]
    for path in sorted(paths):
        if path.is_file():
            provenance["sources_sha256"][str(path.relative_to(package))] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
            )
    try:
        provenance["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        provenance["git_status"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        provenance["git_head"] = None
    return provenance


def benchmark(config, request, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance()
    contract = {
        "config": asdict(config),
        "request": asdict(request),
        "provenance": provenance,
        "state": "running",
        "started": time.time(),
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2))
    try:
        runs = []
        with H3Engine(config) as engine:
            for index in range(4):
                directory = output / ("warmup" if index == 0 else f"run-{index}")
                result = engine.generate(request, directory)
                result["measurement"] = {
                    "warmup": index == 0,
                    "profiled": False,
                    "capture": False,
                }
                result["provenance"] = provenance
                (directory / "run.json").write_text(json.dumps(result, indent=2))
                runs.append(result)
        report = evaluate_performance(runs[1:], warmup=runs[0])
        (output / "performance.json").write_text(json.dumps(report, indent=2))
        contract["state"] = "completed"
        return report
    except BaseException as exc:
        contract.update(state="failed", error=repr(exc))
        raise
    finally:
        contract["finished"] = time.time()
        (output / "contract.json").write_text(json.dumps(contract, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    config = H3Config(**contract["config"])
    request = dict(contract["request"])
    request["sampling"] = H3SamplingParams(**request["sampling"])
    report = benchmark(config, H3Request(**request), args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
