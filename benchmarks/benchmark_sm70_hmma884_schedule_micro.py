# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and measure the isolated native-cubin SM70 HMMA.884 schedule probe."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

K_REQUIRED_SPEEDUP_PCT = 2.0
K_REQUIRED_CLOCK_MHZ = 1200
K_DEFAULT_PHYSICAL_GPU = 1
K_DEFAULT_VISIBLE_DEVICE = "1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, default=K_DEFAULT_PHYSICAL_GPU)
    parser.add_argument(
        "--cuda-visible-devices",
        default=K_DEFAULT_VISIBLE_DEVICE,
        help="One idle physical V100 selected for this isolated probe.",
    )
    parser.add_argument("--clock-mhz", type=int, default=K_REQUIRED_CLOCK_MHZ)
    parser.add_argument("--groups", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=81)
    parser.add_argument("--launches-per-sample", type=int, default=8)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--turingas-dir", type=Path)
    parser.add_argument(
        "--moved-ldg-wait-mask",
        type=lambda value: int(value, 0),
        help="Replace only the moved B1 LDG wait mask during the native patch.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    verbose: bool = False,
) -> subprocess.CompletedProcess[str]:
    if verbose:
        print("run:", " ".join(command), file=sys.stderr)
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} is required for this microbenchmark.")
    return path


def _validate_args(args: argparse.Namespace) -> None:
    if args.device != 0:
        raise ValueError("Use logical --device 0 after CUDA_VISIBLE_DEVICES filtering.")
    if args.cuda_visible_devices != str(args.physical_gpu):
        raise ValueError(
            "This probe accepts exactly one numeric CUDA-visible device matching "
            "--physical-gpu."
        )
    if args.groups < 1 or args.warmup < 0:
        raise ValueError("groups must be positive and warmup cannot be negative.")
    if args.rounds < 5 or args.launches_per_sample < 2:
        raise ValueError("Use at least 5 rounds and 2 launches per sample.")
    if args.clock_mhz <= 0:
        raise ValueError("--clock-mhz must be positive.")


def _verify_idle_gpu(
    physical_gpu: int, required_clock_mhz: int, verbose: bool
) -> dict[str, str | int]:
    nvidia_smi = _require_tool("nvidia-smi")
    gpus = _run(
        [
            nvidia_smi,
            "--query-gpu=index,uuid,name,compute_cap,clocks.current.graphics",
            "--format=csv,noheader",
        ],
        verbose=verbose,
    )
    if gpus.returncode:
        raise RuntimeError(f"nvidia-smi GPU query failed: {gpus.stderr.strip()}")
    selected: dict[str, str] | None = None
    for row in gpus.stdout.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 5 or fields[0] != str(physical_gpu):
            continue
        selected = {
            "index": fields[0],
            "uuid": fields[1],
            "name": fields[2],
            "compute_capability": fields[3],
            "graphics_clock_mhz": fields[4].split()[0],
        }
        break
    if selected is None:
        raise RuntimeError(
            f"Physical GPU {physical_gpu} was not reported by nvidia-smi."
        )
    if selected["compute_capability"] != "7.0":
        raise RuntimeError(
            f"Physical GPU {physical_gpu} is SM{selected['compute_capability']}, "
            "not SM70."
        )
    actual_clock_mhz = int(selected["graphics_clock_mhz"])
    if actual_clock_mhz != required_clock_mhz:
        raise RuntimeError(
            f"Physical GPU {physical_gpu} clock is {actual_clock_mhz} MHz, "
            f"not the required {required_clock_mhz} MHz."
        )
    apps = _run(
        [
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        verbose=verbose,
    )
    if apps.returncode:
        raise RuntimeError(
            f"nvidia-smi compute-app query failed: {apps.stderr.strip()}"
        )
    users = [line for line in apps.stdout.splitlines() if selected["uuid"] in line]
    if users:
        raise RuntimeError(
            f"Physical GPU {physical_gpu} is not idle; refusing to disturb: {users}"
        )
    return {
        "physical_index": physical_gpu,
        **selected,
        "graphics_clock_mhz": actual_clock_mhz,
        "compute_processes": 0,
    }


def _build(
    source: Path, work_dir: Path, verbose: bool
) -> tuple[Path, Path, dict[str, Any]]:
    nvcc = _require_tool("nvcc")
    binary = work_dir / "sm70_hmma884_schedule_micro"
    cubin = work_dir / "sm70_hmma884_schedule_baseline.cubin"
    common = [
        nvcc,
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "-Wno-deprecated-gpu-targets",
        "--generate-code=arch=compute_70,code=sm_70",
        "-Xptxas=-v",
    ]
    executable = _run(
        common + ["-o", str(binary), str(source), "-lcuda"], verbose=verbose
    )
    if executable.returncode:
        raise RuntimeError(
            f"nvcc executable build failed:\n{executable.stdout}\n{executable.stderr}"
        )
    cubin_build = _run(
        common + ["-cubin", "-o", str(cubin), str(source)], verbose=verbose
    )
    if cubin_build.returncode:
        raise RuntimeError(
            f"nvcc cubin build failed:\n{cubin_build.stdout}\n{cubin_build.stderr}"
        )
    return (
        binary,
        cubin,
        {
            "executable_command": common + ["-o", str(binary), str(source), "-lcuda"],
            "cubin_command": common + ["-cubin", "-o", str(cubin), str(source)],
            "ptxas_executable_log": executable.stderr,
            "ptxas_cubin_log": cubin_build.stderr,
        },
    )


def _run_probe(
    cubin: Path,
    candidate: Path,
    work_dir: Path,
    turingas_dir: Path | None,
    moved_ldg_wait_mask: int | None,
    verbose: bool,
) -> dict[str, Any]:
    probe = Path(__file__).resolve().parent / "tools" / "sm70_hmma884_schedule_probe.py"
    command = [
        sys.executable,
        str(probe),
        "--cubin",
        str(cubin),
        "--candidate-cubin",
        str(candidate),
        "--work-dir",
        str(work_dir / "sass"),
    ]
    if turingas_dir is not None:
        command.extend(["--turingas-dir", str(turingas_dir)])
    if moved_ldg_wait_mask is not None:
        command.extend(["--moved-ldg-wait-mask", hex(moved_ldg_wait_mask)])
    result = _run(command, verbose=verbose)
    if result.returncode:
        raise RuntimeError(f"SASS probe failed:\n{result.stdout}\n{result.stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"SASS probe did not emit JSON: {result.stdout}") from error


def _run_candidate_binary(
    binary: Path,
    candidate: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    command = [
        str(binary),
        "--candidate-cubin",
        str(candidate),
        "--device",
        str(args.device),
        "--groups",
        str(args.groups),
        "--warmup",
        str(args.warmup),
        "--rounds",
        str(args.rounds),
        "--launches-per-sample",
        str(args.launches_per_sample),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    result = _run(command, env=environment, verbose=args.verbose)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Candidate binary did not emit JSON (returncode {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        ) from error
    if result.returncode:
        raise RuntimeError(
            f"Candidate binary failed after emitting JSON: {json.dumps(payload)}"
        )
    return payload, command


def _same_instruction_counts(probe: dict[str, Any]) -> bool:
    return (
        probe["baseline"]["instruction_counts"]
        == probe["candidate"]["instruction_counts"]
    )


def _gates(probe: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    exactness = payload.get("exactness", {})
    resources = payload.get("resources", {})
    baseline_resources = resources.get("baseline", {})
    candidate_resources = resources.get("candidate", {})
    speedup = payload.get("timing", {}).get("candidate_speedup_vs_baseline_pct")
    reasons: list[str] = []
    if probe.get("patch", {}).get("changed_real_sass_order") is not True:
        reasons.append("candidate cubin did not prove a changed SASS instruction order")
    if not _same_instruction_counts(probe):
        reasons.append("candidate instruction counts differ from baseline")
    if probe.get("patch", {}).get("b_fragment_instances") != 1:
        reasons.append("candidate requires more than one B fragment instance")
    if exactness.get("bitwise_equal") is not True:
        reasons.append("candidate output is not bitwise equal to baseline")
    if exactness.get("mismatch_words") != 0:
        reasons.append("candidate has nonzero output word mismatches")
    if baseline_resources.get("local_bytes_per_thread") != 0:
        reasons.append("baseline has local memory")
    if candidate_resources.get("local_bytes_per_thread") != 0:
        reasons.append("candidate has local memory")
    if baseline_resources.get("registers_per_thread") != candidate_resources.get(
        "registers_per_thread"
    ):
        reasons.append("candidate register allocation differs from baseline")
    if not isinstance(speedup, (int, float)) or speedup < K_REQUIRED_SPEEDUP_PCT:
        reasons.append(
            f"median speedup={speedup}, requires >= {K_REQUIRED_SPEEDUP_PCT}%"
        )
    return {
        "sass_order_passed": probe.get("patch", {}).get("changed_real_sass_order")
        is True,
        "resource_passed": not any(
            "local" in reason or "register" in reason for reason in reasons
        ),
        "correctness_passed": exactness.get("bitwise_equal") is True
        and exactness.get("mismatch_words") == 0,
        "performance_passed": isinstance(speedup, (int, float))
        and speedup >= K_REQUIRED_SPEEDUP_PCT,
        "required_speedup_pct": K_REQUIRED_SPEEDUP_PCT,
        "accepted": not reasons,
        "reasons": reasons,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    work_dir = args.work_dir or (
        Path(tempfile.gettempdir()) / f"vllm-sm70-hmma884-schedule-{os.getuid()}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    gpu_before = _verify_idle_gpu(args.physical_gpu, args.clock_mhz, args.verbose)
    source = Path(__file__).resolve().parent / "csrc" / "sm70_hmma884_schedule_micro.cu"
    binary, cubin, build = _build(source, work_dir, args.verbose)
    candidate = work_dir / "sm70_hmma884_schedule_candidate.cubin"
    probe = _run_probe(
        cubin,
        candidate,
        work_dir,
        args.turingas_dir,
        args.moved_ldg_wait_mask,
        args.verbose,
    )
    cpp_json, binary_command = _run_candidate_binary(binary, candidate, args)
    gpu_after = _verify_idle_gpu(args.physical_gpu, args.clock_mhz, args.verbose)
    gates = _gates(probe, cpp_json)
    result = {
        "benchmark": "sm70_hmma884_schedule_micro",
        "target": "sm_70",
        "gpu_selection": {
            "cuda_visible_devices": args.cuda_visible_devices,
            "logical_device": args.device,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "clock_mhz": {"required": args.clock_mhz, "strict": True},
            "clock_control": "queried before and after; not changed",
        },
        "build": build,
        "probe": probe,
        "command": binary_command,
        "cpp_json": cpp_json,
        "gates": gates,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        _write_json(args.json_out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
