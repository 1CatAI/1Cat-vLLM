# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict workload contracts and provenance for the A3 parity runners."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A3 numerical parity requires a CUDA device")
    devices = [
        torch.cuda.get_device_properties(i)
        for i in range(torch.accelerator.device_count())
    ]
    if any((p.major, p.minor) != (7, 0) for p in devices):
        raise RuntimeError("A3 acceptance requires V100/SM70")
    return dict(
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        devices=[
            dict(
                name=p.name,
                memory=p.total_memory,
                uuid=str(getattr(p, "uuid", "unavailable")),
            )
            for p in devices
        ],
        policy_env={
            k: v
            for k, v in os.environ.items()
            if k.startswith("VLLM_")
            and k not in {"VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT", "VLLM_RPC_BASE_PATH"}
        },
    )


def provenance(source_sha):
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("Record the full, source-verified commit SHA")
    return dict(
        source_sha=source_sha,
        harness_sha256={
            p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*parity*.py"))
        },
        gpu_state=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,driver_version,clocks.sm,clocks.mem",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines(),
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("VLLM_", "CUDA_", "TORCH_", "TRITON_"))
        },
    )


def require_same_contract(reference, candidate):
    if reference["contract"] != candidate["contract"]:
        raise ValueError("Workload/runtime contracts differ; parity is not comparable")


def require_routes(snapshot, required):
    observed = {
        name
        for worker in snapshot
        for name, count in worker["routes"].items()
        if count > 0
    }
    missing = set(required) - observed
    if missing:
        raise AssertionError(f"Required attention routes were not observed: {missing}")


def compare_routes(reference, candidate):
    require_same_contract(reference, candidate)
    if not reference["requests"] or not reference["startup"] or not reference["after"]:
        raise AssertionError("Empty route/token evidence cannot establish parity")
    for key in ("requests", "startup", "after"):
        if reference[key] != candidate[key]:
            raise AssertionError(f"Route/token parity differs: {key}")
    return dict(equal=True, requests=len(candidate["requests"]))
