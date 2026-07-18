# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strict operator-only A/B benchmark for the SM70 real prefill shape.

This benchmark fixes M=8096, N=121440, Hq=6, Hkv=1, D=256, and page=784.
It measures the accepted generic paged BM32 pair-scratch route, the fixed
native pybind entry, and an identity-only dense zero-copy comparator. It does
not create an engine, load a model, or run an end-to-end workload.

Use --static-check for a CPU-only contract check. A GPU run requires an
exclusive SM70 GPU at exactly 1200 MHz; any other GPU process is a failure.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BATCH_SIZE = 1
M = 8096
N = 121440
H_Q = 6
H_KV = 1
HEAD_DIM = 256
PAGE_SIZE = 784
CACHE_PAGES = (N + PAGE_SIZE - 1) // PAGE_SIZE
ALLOCATED_CACHE_TOKENS = CACHE_PAGES * PAGE_SIZE
TAIL_PAGE_VALID_TOKENS = N - (CACHE_PAGES - 1) * PAGE_SIZE
TAIL_PAGE_UNUSED_TOKENS = PAGE_SIZE - TAIL_PAGE_VALID_TOKENS
SOFTMAX_SCALE = HEAD_DIM**-0.5

REQUIRED_GRAPHICS_CLOCK_MHZ = 1200
PAIRED_ROUNDS = 100
DEFAULT_WARMUP_ROUNDS = 20
LOAD_MONITOR_INTERVAL_MS = 250
LOAD_MONITOR_COMMAND_TIMEOUT_SECONDS = 5.0

SOURCE_ROOT = Path(__file__).resolve().parents[1]
FLASH_V100_ROOT = SOURCE_ROOT / "flash-attention-v100"

GENERIC_ROUTE = "generic_paged_bm32_pair_scratch"
FIXED_ROUTE = "fixed_pybind_bm32_pair_scratch"
DENSE_ROUTE = "dense_identity_zero_copy"
ROUTE_NAMES = (GENERIC_ROUTE, FIXED_ROUTE, DENSE_ROUTE)

# These are the accepted generic dispatch controls. The fixed entry is called
# directly and intentionally runs with all generic controls disabled.
PAGED_BM32_ENV = {
    "VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM": "1",
    "VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE": "1",
    "VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P": "1",
    "VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH": "1",
}
DENSE_ENV = {
    "VLLM_FLASH_V100_DENSE_D256_WMMA_QK": "1",
    "VLLM_FLASH_V100_DENSE_D256_LOW_SMEM": "0",
    "VLLM_FLASH_V100_PREFILL_SCALAR_PV": "0",
}
CONTROL_ENV = tuple(sorted({*PAGED_BM32_ENV, *DENSE_ENV}))


def _route_environment(enabled: Mapping[str, str]) -> dict[str, str]:
    environment = {name: "0" for name in CONTROL_ENV}
    environment.update(enabled)
    return environment


GENERIC_RUNTIME_ENV = _route_environment(PAGED_BM32_ENV)
FIXED_RUNTIME_ENV = _route_environment({})
DENSE_RUNTIME_ENV = _route_environment(DENSE_ENV)

# Every pairwise ordering reverses between these two schedules: generic/fixed,
# generic/dense, and fixed/dense all alternate first/second position.
ORDER_SCHEDULE = (
    (GENERIC_ROUTE, FIXED_ROUTE, DENSE_ROUTE),
    (DENSE_ROUTE, FIXED_ROUTE, GENERIC_ROUTE),
)

# Kept lazy so --static-check neither imports torch nor initializes CUDA.
torch: Any = None


@dataclass(frozen=True)
class _Route:
    name: str
    environment: Mapping[str, str]
    launch: Callable[[], Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical-gpu",
        type=int,
        default=0,
        help="Physical GPU index to expose as logical cuda:0.",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=DEFAULT_WARMUP_ROUNDS,
        help="Untimed alternating warmup rounds before the fixed 100 pairs.",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional JSON destination. Without it, emit the full record to stdout.",
    )
    parser.add_argument(
        "--static-check",
        action="store_true",
        help="Validate the fixed contract without importing torch or using a GPU.",
    )
    args = parser.parse_args()
    if args.physical_gpu != 0:
        parser.error("strict benchmark is fixed to physical GPU0")
    if args.warmup_rounds < 0:
        parser.error("--warmup-rounds must be non-negative")
    return args


def _page_specs() -> dict[str, dict[str, Any]]:
    identity = tuple(range(CACHE_PAGES))
    reverse = tuple(reversed(identity))
    tail_relocated = (identity[-1], *identity[:-1])
    return {
        "identity": {
            "page_ids": identity,
            "description": (
                "Physical page IDs are logical page IDs; the only partial page "
                "is the terminal logical page."
            ),
            "dense_zero_copy": {
                "eligible": True,
                "reason": (
                    "cache[P,page,1,D] can be reinterpreted as [1,1,P*page,D] "
                    "and sliced to N without a gather."
                ),
            },
        },
        "reverse": {
            "page_ids": reverse,
            "description": "Physical pages are traversed in reverse logical order.",
            "dense_zero_copy": {
                "eligible": False,
                "reason": (
                    "A contiguous dense view preserves physical order, not this "
                    "reverse logical page order; a gather would be required."
                ),
            },
        },
        "tail": {
            "page_ids": tail_relocated,
            "description": (
                "The physical page that is the identity-map tail is moved to "
                "logical page zero."
            ),
            "dense_zero_copy": {
                "eligible": False,
                "reason": (
                    "The partial terminal logical page no longer matches the "
                    "terminal physical page in the raw view; a dense call would "
                    "change token order or require materialization."
                ),
            },
        },
    }


def _geometry() -> dict[str, int]:
    return {
        "batch_size": BATCH_SIZE,
        "m": M,
        "n": N,
        "heads_q": H_Q,
        "heads_kv": H_KV,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "cache_pages": CACHE_PAGES,
        "allocated_cache_tokens": ALLOCATED_CACHE_TOKENS,
        "tail_page_valid_tokens": TAIL_PAGE_VALID_TOKENS,
        "tail_page_unused_tokens": TAIL_PAGE_UNUSED_TOKENS,
        "q_tiles_bm32": M // 32,
        "ctas": (M // 32) * H_Q,
    }


def _static_contract() -> dict[str, Any]:
    failures: list[str] = []
    if (M, N, H_Q, H_KV, HEAD_DIM, PAGE_SIZE) != (
        8096,
        121440,
        6,
        1,
        256,
        784,
    ):
        failures.append("real-shape constants changed")
    if CACHE_PAGES != 155:
        failures.append(f"cache_pages={CACHE_PAGES}, expected 155")
    if ALLOCATED_CACHE_TOKENS != 121520:
        failures.append(
            f"allocated_cache_tokens={ALLOCATED_CACHE_TOKENS}, expected 121520"
        )
    if TAIL_PAGE_VALID_TOKENS != 704:
        failures.append(
            f"tail_page_valid_tokens={TAIL_PAGE_VALID_TOKENS}, expected 704"
        )
    if M % 32 != 0:
        failures.append("M must be divisible by the BM32 query tile")
    if H_Q % H_KV != 0:
        failures.append("Hq must be divisible by Hkv")
    if PAIRED_ROUNDS != 100:
        failures.append("strict benchmark requires exactly 100 paired rounds")
    if LOAD_MONITOR_INTERVAL_MS <= 0:
        failures.append("load monitor interval must be positive")
    if set(ORDER_SCHEDULE[0]) != set(ROUTE_NAMES):
        failures.append("first timing order does not contain each route once")
    if set(ORDER_SCHEDULE[1]) != set(ROUTE_NAMES):
        failures.append("second timing order does not contain each route once")

    page_specs = _page_specs()
    if not page_specs["identity"]["dense_zero_copy"]["eligible"]:
        failures.append("identity pages must admit the dense zero-copy view")
    for name in ("reverse", "tail"):
        if page_specs[name]["dense_zero_copy"]["eligible"]:
            failures.append(f"{name} pages must reject the dense zero-copy view")

    if failures:
        raise RuntimeError("static contract failed: " + "; ".join(failures))
    return {
        "passed": True,
        "geometry": _geometry(),
        "paired_rounds": PAIRED_ROUNDS,
        "order_schedule": [list(order) for order in ORDER_SCHEDULE],
        "load_monitor_interval_ms": LOAD_MONITOR_INTERVAL_MS,
        "gpu_not_touched": True,
    }


def _configure_cuda_visibility(physical_gpu: int) -> None:
    if "torch" in sys.modules:
        raise RuntimeError(
            "torch was imported before CUDA_VISIBLE_DEVICES could be fixed"
        )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)


def _load_torch() -> None:
    global torch
    if torch is None:
        torch = importlib.import_module("torch")


def _require_cuda_runtime() -> None:
    _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "benchmark must expose exactly one GPU through CUDA_VISIBLE_DEVICES"
        )
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError(
            "fixed entry requires SM70; got "
            f"{torch.cuda.get_device_capability(0)} on {torch.cuda.get_device_name(0)}"
        )


def _nvidia_smi() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required for the strict occupancy gate")
    return executable


def _query_compute_processes(executable: str, uuid: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            executable,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running"):
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected compute-process output: {line!r}")
        process_uuid, pid, process_name, used_memory_mib = fields
        if process_uuid != uuid:
            continue
        processes.append(
            {
                "pid": int(pid),
                "name": process_name,
                "type": "C",
                "used_memory_mib": used_memory_mib,
                "source": "query-compute-apps",
            }
        )
    return processes


def _nvidia_smi_xml_gpu(
    executable: str,
    physical_gpu: int,
    *,
    timeout_seconds: float | None = None,
) -> ElementTree.Element:
    result = subprocess.run(
        [executable, "-q", "-x", "-i", str(physical_gpu)],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    try:
        root = ElementTree.fromstring(result.stdout)
    except ElementTree.ParseError as exc:
        raise RuntimeError("could not parse nvidia-smi XML process list") from exc

    gpus = root.findall("gpu")
    if len(gpus) != 1:
        raise RuntimeError(f"expected one GPU in nvidia-smi XML, got {len(gpus)}")
    return gpus[0]


def _xml_gpu_processes(gpu: ElementTree.Element) -> list[dict[str, Any]]:
    process_root = gpu.find("processes")
    if process_root is None:
        raise RuntimeError("nvidia-smi XML omitted the process list")

    processes: list[dict[str, Any]] = []
    for process in process_root.findall("process_info"):
        pid_text = process.findtext("pid")
        if pid_text is None or not pid_text.isdigit():
            raise RuntimeError(f"unexpected GPU process pid: {pid_text!r}")
        processes.append(
            {
                "pid": int(pid_text),
                "name": process.findtext("process_name") or "unknown",
                "type": process.findtext("type") or "unknown",
                "used_memory_mib": (
                    process.findtext("used_memory")
                    or process.findtext("gpu_instance_id")
                    or "unknown"
                ),
                "source": "nvidia-smi-xml",
            }
        )
    return processes


def _query_all_gpu_processes(
    executable: str,
    physical_gpu: int,
) -> list[dict[str, Any]]:
    gpu = _nvidia_smi_xml_gpu(executable, physical_gpu)
    return _xml_gpu_processes(gpu)


def _gpu_state(physical_gpu: int) -> dict[str, Any]:
    executable = _nvidia_smi()
    query = (
        "index,uuid,name,compute_cap,clocks.current.graphics,clocks.max.graphics,pstate"
    )
    result = subprocess.run(
        [
            executable,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "--id",
            str(physical_gpu),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 7:
        raise RuntimeError(f"unexpected nvidia-smi GPU output: {result.stdout!r}")
    index, uuid, name, capability, current_clock, max_clock, pstate = fields

    compute_processes = _query_compute_processes(executable, uuid)
    all_processes = _query_all_gpu_processes(executable, physical_gpu)
    by_pid = {process["pid"]: process for process in all_processes}
    for process in compute_processes:
        by_pid.setdefault(process["pid"], process)

    return {
        "physical_gpu": int(index),
        "uuid": uuid,
        "name": name,
        "compute_capability": capability,
        "graphics_clock_mhz": int(current_clock),
        "graphics_clock_max_mhz": int(max_clock),
        "pstate": pstate,
        "compute_processes": sorted(
            compute_processes,
            key=lambda process: (process["pid"], process["name"]),
        ),
        "gpu_processes": sorted(
            by_pid.values(),
            key=lambda process: (process["pid"], process["name"]),
        ),
    }


def _require_exclusive_snapshot(
    state: Mapping[str, Any],
    *,
    physical_gpu: int,
    phase: str,
    allowed_pids: set[int],
) -> None:
    if state["physical_gpu"] != physical_gpu:
        raise RuntimeError(
            f"{phase}: nvidia-smi returned GPU {state['physical_gpu']}, "
            f"not requested GPU {physical_gpu}"
        )
    if state["compute_capability"] != "7.0":
        raise RuntimeError(f"{phase}: expected SM70, got {state}")
    if state["graphics_clock_mhz"] != REQUIRED_GRAPHICS_CLOCK_MHZ:
        raise RuntimeError(
            f"{phase}: expected {REQUIRED_GRAPHICS_CLOCK_MHZ} MHz, got "
            f"{state['graphics_clock_mhz']} MHz"
        )
    unexpected = [
        process
        for process in state["gpu_processes"]
        if process["pid"] not in allowed_pids
    ]
    if unexpected:
        raise RuntimeError(
            f"{phase}: concurrent GPU occupancy is disallowed: {unexpected}"
        )


def _record_gpu_snapshot(
    result: dict[str, Any],
    *,
    physical_gpu: int,
    phase: str,
    allowed_pids: set[int],
) -> dict[str, Any]:
    state = _gpu_state(physical_gpu)
    result["gpu"]["snapshots"][phase] = state
    _require_exclusive_snapshot(
        state,
        physical_gpu=physical_gpu,
        phase=phase,
        allowed_pids=allowed_pids,
    )
    return state


def _xml_graphics_clock_mhz(gpu: ElementTree.Element) -> int:
    clock_text = gpu.findtext("clocks/graphics_clock")
    match = re.search(r"\d+", clock_text or "")
    if match is None:
        raise RuntimeError(
            f"nvidia-smi XML omitted a parseable graphics clock: {clock_text!r}"
        )
    return int(match.group())


def _load_monitor_sample(physical_gpu: int) -> dict[str, Any]:
    executable = _nvidia_smi()
    gpu = _nvidia_smi_xml_gpu(
        executable,
        physical_gpu,
        timeout_seconds=LOAD_MONITOR_COMMAND_TIMEOUT_SECONDS,
    )
    return {
        "physical_gpu": physical_gpu,
        "graphics_clock_mhz": _xml_graphics_clock_mhz(gpu),
        "gpu_processes": _xml_gpu_processes(gpu),
    }


class _LoadMonitor:
    """NVML-backed load-window sampler for the clock and active GPU PIDs."""

    def __init__(
        self,
        *,
        physical_gpu: int,
        allowed_pids: set[int],
        interval_ms: int,
    ) -> None:
        self._physical_gpu = physical_gpu
        self._allowed_pids = frozenset(allowed_pids)
        self._interval_seconds = interval_ms / 1000.0
        self._started_at = time.monotonic()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._violations: list[dict[str, Any]] = []
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name="sm70-gpu-load-monitor",
            daemon=True,
        )

    def _record_sample(self, sample: dict[str, Any]) -> None:
        reasons: list[str] = []
        if "error" in sample:
            reasons.append(str(sample["error"]))
        else:
            if sample["graphics_clock_mhz"] != REQUIRED_GRAPHICS_CLOCK_MHZ:
                reasons.append(
                    f"graphics clock={sample['graphics_clock_mhz']} MHz, "
                    f"requires {REQUIRED_GRAPHICS_CLOCK_MHZ} MHz"
                )
            unexpected = [
                process
                for process in sample["gpu_processes"]
                if process["pid"] not in self._allowed_pids
            ]
            if unexpected:
                reasons.append(f"unexpected GPU processes: {unexpected}")

        with self._lock:
            sample["sample_index"] = len(self._samples)
            sample["valid"] = not reasons
            self._samples.append(sample)
            if reasons:
                self._violations.append(
                    {
                        "sample_index": sample["sample_index"],
                        "elapsed_ms": sample["elapsed_ms"],
                        "reasons": reasons,
                        "sample": sample,
                    }
                )

    def _capture_once(self) -> None:
        started_at = time.monotonic()
        try:
            sample = _load_monitor_sample(self._physical_gpu)
        except Exception as exc:
            sample = {"error": f"{type(exc).__name__}: {exc}"}
        completed_at = time.monotonic()
        sample["elapsed_ms"] = (started_at - self._started_at) * 1000.0
        sample["sample_duration_ms"] = (completed_at - started_at) * 1000.0
        self._record_sample(sample)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self._capture_once()

    def start(self) -> None:
        if self._started:
            raise RuntimeError("load monitor was started twice")
        self._started = True
        self._capture_once()
        self._thread.start()

    def assert_healthy(self) -> None:
        with self._lock:
            violations = list(self._violations)
        if violations:
            raise RuntimeError(
                "load monitor detected invalid GPU state "
                f"({len(violations)} sample(s)): {violations[0]}"
            )

    def stop(self) -> dict[str, Any]:
        if self._started:
            self._stop_event.set()
            self._thread.join(timeout=LOAD_MONITOR_COMMAND_TIMEOUT_SECONDS + 1.0)
            if self._thread.is_alive():
                self._record_sample(
                    {
                        "elapsed_ms": (time.monotonic() - self._started_at) * 1000.0,
                        "sample_duration_ms": 0.0,
                        "error": "load monitor thread did not stop",
                    }
                )
        with self._lock:
            samples = list(self._samples)
            violations = list(self._violations)
        return {
            "sampler": "background nvidia-smi XML (NVML-backed)",
            "interval_ms": LOAD_MONITOR_INTERVAL_MS,
            "expected_graphics_clock_mhz": REQUIRED_GRAPHICS_CLOCK_MHZ,
            "allowed_pids": sorted(self._allowed_pids),
            "passed": not violations,
            "samples": samples,
            "violations": violations,
        }


@contextmanager
def _restore_control_environment():
    saved = {name: os.environ.get(name) for name in CONTROL_ENV}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _set_environment(values: Mapping[str, str]) -> None:
    for name, value in values.items():
        os.environ[name] = value


def _load_apis() -> tuple[Any, Any, Any, dict[str, str]]:
    if not FLASH_V100_ROOT.is_dir():
        raise RuntimeError(f"Flash-V100 source tree not found: {FLASH_V100_ROOT}")
    if "flash_attn_v100" in sys.modules:
        raise RuntimeError("flash_attn_v100 was imported before this benchmark")
    sys.path.insert(0, str(FLASH_V100_ROOT))

    flash_attn_v100 = importlib.import_module("flash_attn_v100")
    interface = importlib.import_module("flash_attn_v100.flash_attn_interface")
    generic = getattr(flash_attn_v100, "flash_attn_prefill_paged_bhmd", None)
    dense = getattr(flash_attn_v100, "flash_attn_bhmd_func", None)
    native_extension = getattr(interface, "flash_attn_v100_cuda", None)
    fixed = getattr(
        native_extension,
        "prefill_paged_d256_bm32_allp_pair_scratch_fwd",
        None,
    )
    if not all(callable(entry) for entry in (generic, fixed, dense)):
        raise RuntimeError(
            "Flash-V100 build lacks generic paged, fixed pybind, or dense API"
        )
    metadata = {
        "flash_attn_v100_package": str(Path(flash_attn_v100.__file__).resolve()),
        "flash_attn_interface": str(Path(interface.__file__).resolve()),
        "native_extension": str(Path(native_extension.__file__).resolve()),
    }
    return generic, fixed, dense, metadata


def _runtime_metadata(api_metadata: Mapping[str, str]) -> dict[str, Any]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
        "logical_device_name": torch.cuda.get_device_name(0),
        "logical_device_capability": list(torch.cuda.get_device_capability(0)),
        **api_metadata,
    }


def _make_inputs(seed: int) -> tuple[Any, Any, Any, Any]:
    torch.manual_seed(seed)
    query = torch.randn(
        (BATCH_SIZE, H_Q, M, HEAD_DIM),
        device="cuda",
        dtype=torch.float16,
    )
    key_cache = torch.randn(
        (CACHE_PAGES, PAGE_SIZE, H_KV, HEAD_DIM),
        device="cuda",
        dtype=torch.float16,
    )
    value_cache = torch.randn_like(key_cache)
    sequence_lengths = torch.full(
        (BATCH_SIZE,),
        N,
        device="cuda",
        dtype=torch.int32,
    )
    return query, key_cache, value_cache, sequence_lengths


def _make_identity_dense_view(cache: Any) -> tuple[Any, dict[str, Any]]:
    if H_KV != 1 or cache.shape[2] != 1:
        raise RuntimeError("identity zero-copy dense view requires Hkv=1")
    if not cache.is_contiguous():
        raise RuntimeError("identity zero-copy dense source cache is not contiguous")
    view = cache.view(BATCH_SIZE, H_KV, ALLOCATED_CACHE_TOKENS, HEAD_DIM)[..., :N, :]
    details = {
        "layout": "cache[P,page,1,D].view(1,1,P*page,D)[..., :N, :]",
        "is_contiguous": bool(view.is_contiguous()),
        "aliases_cache_data_ptr": view.data_ptr() == cache.data_ptr(),
        "logical_tokens": N,
        "allocated_tokens": ALLOCATED_CACHE_TOKENS,
        "terminal_tail_valid_tokens": TAIL_PAGE_VALID_TOKENS,
    }
    if not details["is_contiguous"] or not details["aliases_cache_data_ptr"]:
        raise RuntimeError(f"identity dense view copied unexpectedly: {details}")
    return view, details


def _make_cases(
    query: Any,
    page_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for name, specification in page_specs.items():
        page_ids = torch.tensor(
            specification["page_ids"],
            device=query.device,
            dtype=torch.int32,
        )
        case: dict[str, Any] = {
            "block_table": page_ids.view(BATCH_SIZE, CACHE_PAGES),
            "generic_out": torch.empty_like(query),
            "fixed_out": torch.empty_like(query),
            "fixed_lse": torch.empty(
                query.shape[:-1],
                device=query.device,
                dtype=torch.float32,
            ),
            "specification": specification,
        }
        if specification["dense_zero_copy"]["eligible"]:
            case["dense_out"] = torch.empty_like(query)
        cases[name] = case
    return cases


def _assert_alias(returned: Any, preallocated: Any, label: str) -> None:
    if returned.data_ptr() != preallocated.data_ptr():
        raise RuntimeError(f"{label} did not honor its preallocated buffer")


def _comparison_evidence(reference: Any, candidate: Any) -> dict[str, Any]:
    shape_match = tuple(reference.shape) == tuple(candidate.shape)
    dtype_match = reference.dtype == candidate.dtype
    evidence: dict[str, Any] = {
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "bitwise_equal": False,
        "max_abs_diff": None,
    }
    if not shape_match or not dtype_match:
        return evidence

    reference_bytes = reference.view(torch.uint8)
    candidate_bytes = candidate.view(torch.uint8)
    equal = bool(torch.equal(reference_bytes, candidate_bytes))
    evidence["bitwise_equal"] = equal
    if equal:
        evidence["different_bytes"] = 0
        evidence["max_abs_diff"] = 0.0
        return evidence

    evidence["different_bytes"] = int(
        torch.count_nonzero(reference_bytes != candidate_bytes).item()
    )
    difference = (reference.float() - candidate.float()).abs()
    evidence["max_abs_diff"] = float(difference.max().item())
    return evidence


def _run_correctness(
    *,
    cases: Mapping[str, Mapping[str, Any]],
    query: Any,
    key_cache: Any,
    value_cache: Any,
    sequence_lengths: Any,
    generic_entry: Any,
    fixed_entry: Any,
    dense_entry: Any,
    dense_key: Any,
    dense_value: Any,
    health_check: Callable[[], None],
) -> dict[str, Any]:
    correctness: dict[str, Any] = {}
    for name, case in cases.items():
        health_check()
        _set_environment(GENERIC_RUNTIME_ENV)
        generic_out = generic_entry(
            query,
            key_cache,
            value_cache,
            case["block_table"],
            sequence_lengths,
            softmax_scale=SOFTMAX_SCALE,
            out=case["generic_out"],
            causal=True,
        )
        _assert_alias(generic_out, case["generic_out"], f"{name} generic output")

        _set_environment(FIXED_RUNTIME_ENV)
        fixed_out, fixed_lse = fixed_entry(
            query,
            key_cache,
            value_cache,
            case["fixed_out"],
            case["fixed_lse"],
            case["block_table"],
            sequence_lengths,
            SOFTMAX_SCALE,
        )
        torch.cuda.synchronize()
        _assert_alias(fixed_out, case["fixed_out"], f"{name} fixed output")
        _assert_alias(fixed_lse, case["fixed_lse"], f"{name} fixed LSE")

        fixed_evidence = _comparison_evidence(generic_out, fixed_out)
        fixed_evidence["output_preallocated"] = True
        fixed_evidence["lse_preallocated"] = True
        fixed_evidence["lse_finite"] = bool(torch.isfinite(fixed_lse).all().item())
        if not fixed_evidence["bitwise_equal"]:
            raise RuntimeError(f"{name}: fixed pybind output is not bitwise generic")
        if not fixed_evidence["lse_finite"]:
            raise RuntimeError(f"{name}: fixed pybind LSE is non-finite")

        dense_policy = case["specification"]["dense_zero_copy"]
        if dense_policy["eligible"]:
            _set_environment(DENSE_RUNTIME_ENV)
            dense_out = dense_entry(
                query,
                dense_key,
                dense_value,
                softmax_scale=SOFTMAX_SCALE,
                causal=True,
                out=case["dense_out"],
            )
            torch.cuda.synchronize()
            _assert_alias(dense_out, case["dense_out"], f"{name} dense output")
            dense_evidence = _comparison_evidence(generic_out, dense_out)
            dense_evidence["output_preallocated"] = True
            dense_evidence["finite"] = bool(torch.isfinite(dense_out).all().item())
            if not dense_evidence["finite"]:
                raise RuntimeError(f"{name}: dense zero-copy output is non-finite")
        else:
            dense_evidence = {
                "status": "not_run",
                "eligible": False,
                "reason": dense_policy["reason"],
            }

        correctness[name] = {
            "fixed_pybind_vs_generic": fixed_evidence,
            "dense_zero_copy_vs_generic": dense_evidence,
        }
        health_check()
    return correctness


def _timing_stats(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": samples,
        "mean_ms": float(statistics.mean(samples)),
        "p50_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
    }


def _paired_stats(
    raw_rounds: list[dict[str, Any]],
    *,
    reference: str,
    candidate: str,
) -> dict[str, Any]:
    reference_samples = [
        float(round_data["samples_ms"][reference]) for round_data in raw_rounds
    ]
    candidate_samples = [
        float(round_data["samples_ms"][candidate]) for round_data in raw_rounds
    ]
    deltas = [
        candidate_ms - reference_ms
        for reference_ms, candidate_ms in zip(reference_samples, candidate_samples)
    ]
    reference_mean = float(statistics.mean(reference_samples))
    reference_p50 = float(statistics.median(reference_samples))
    candidate_mean = float(statistics.mean(candidate_samples))
    candidate_p50 = float(statistics.median(candidate_samples))
    return {
        "reference": reference,
        "candidate": candidate,
        "reference_mean_ms": reference_mean,
        "reference_p50_ms": reference_p50,
        "candidate_mean_ms": candidate_mean,
        "candidate_p50_ms": candidate_p50,
        "candidate_vs_reference_mean_pct": (
            (candidate_mean / reference_mean - 1.0) * 100.0
        ),
        "candidate_vs_reference_p50_pct": (
            (candidate_p50 / reference_p50 - 1.0) * 100.0
        ),
        "candidate_minus_reference_mean_ms": float(statistics.mean(deltas)),
        "candidate_minus_reference_p50_ms": float(statistics.median(deltas)),
        "candidate_faster_wins": sum(
            candidate_ms < reference_ms
            for reference_ms, candidate_ms in zip(
                reference_samples,
                candidate_samples,
            )
        ),
        "reference_faster_wins": sum(
            reference_ms < candidate_ms
            for reference_ms, candidate_ms in zip(
                reference_samples,
                candidate_samples,
            )
        ),
        "ties": sum(
            reference_ms == candidate_ms
            for reference_ms, candidate_ms in zip(
                reference_samples,
                candidate_samples,
            )
        ),
        "raw_delta_ms": deltas,
    }


def _time_route(route: _Route, events: tuple[Any, Any]) -> float:
    _set_environment(route.environment)
    start, end = events
    start.record()
    route.launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _run_interleaved_timing(
    routes: Mapping[str, _Route],
    *,
    warmup_rounds: int,
    health_check: Callable[[], None],
) -> dict[str, Any]:
    if tuple(sorted(routes)) != tuple(sorted(ROUTE_NAMES)):
        raise RuntimeError(f"unexpected route set: {sorted(routes)}")

    for warmup_round in range(warmup_rounds):
        health_check()
        for name in ORDER_SCHEDULE[warmup_round % len(ORDER_SCHEDULE)]:
            _set_environment(routes[name].environment)
            routes[name].launch()
        health_check()
    torch.cuda.synchronize()

    events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in ROUTE_NAMES
    }
    raw_rounds: list[dict[str, Any]] = []
    for round_index in range(PAIRED_ROUNDS):
        health_check()
        order = ORDER_SCHEDULE[round_index % len(ORDER_SCHEDULE)]
        samples: dict[str, float] = {}
        for name in order:
            samples[name] = _time_route(routes[name], events[name])
        raw_rounds.append(
            {
                "round": round_index,
                "route_order": list(order),
                "samples_ms": samples,
            }
        )
        health_check()

    route_samples = {
        name: [float(round_data["samples_ms"][name]) for round_data in raw_rounds]
        for name in ROUTE_NAMES
    }
    return {
        "paired_rounds": PAIRED_ROUNDS,
        "warmup_rounds": warmup_rounds,
        "order_schedule": [list(order) for order in ORDER_SCHEDULE],
        "raw_rounds": raw_rounds,
        "route_stats": {
            name: _timing_stats(route_samples[name]) for name in ROUTE_NAMES
        },
        "paired_comparisons": {
            "fixed_vs_generic": _paired_stats(
                raw_rounds,
                reference=GENERIC_ROUTE,
                candidate=FIXED_ROUTE,
            ),
            "dense_vs_generic": _paired_stats(
                raw_rounds,
                reference=GENERIC_ROUTE,
                candidate=DENSE_ROUTE,
            ),
            "dense_vs_fixed": _paired_stats(
                raw_rounds,
                reference=FIXED_ROUTE,
                candidate=DENSE_ROUTE,
            ),
        },
    }


def _identity_routes(
    *,
    identity_case: Mapping[str, Any],
    query: Any,
    key_cache: Any,
    value_cache: Any,
    sequence_lengths: Any,
    generic_entry: Any,
    fixed_entry: Any,
    dense_entry: Any,
    dense_key: Any,
    dense_value: Any,
) -> dict[str, _Route]:
    def generic() -> Any:
        return generic_entry(
            query,
            key_cache,
            value_cache,
            identity_case["block_table"],
            sequence_lengths,
            softmax_scale=SOFTMAX_SCALE,
            out=identity_case["generic_out"],
            causal=True,
        )

    def fixed() -> Any:
        return fixed_entry(
            query,
            key_cache,
            value_cache,
            identity_case["fixed_out"],
            identity_case["fixed_lse"],
            identity_case["block_table"],
            sequence_lengths,
            SOFTMAX_SCALE,
        )

    def dense() -> Any:
        return dense_entry(
            query,
            dense_key,
            dense_value,
            softmax_scale=SOFTMAX_SCALE,
            causal=True,
            out=identity_case["dense_out"],
        )

    return {
        GENERIC_ROUTE: _Route(GENERIC_ROUTE, GENERIC_RUNTIME_ENV, generic),
        FIXED_ROUTE: _Route(FIXED_ROUTE, FIXED_RUNTIME_ENV, fixed),
        DENSE_ROUTE: _Route(DENSE_ROUTE, DENSE_RUNTIME_ENV, dense),
    }


def _route_metadata() -> dict[str, Any]:
    return {
        GENERIC_ROUTE: {
            "entrypoint": "flash_attn_v100.flash_attn_prefill_paged_bhmd",
            "causal": True,
            "dispatch_environment": PAGED_BM32_ENV,
            "out_preallocated": True,
            "softmax_lse_preallocated": False,
            "softmax_lse_reason": "generic pybind API does not accept an LSE buffer",
        },
        FIXED_ROUTE: {
            "entrypoint": (
                "flash_attn_v100_cuda.prefill_paged_d256_bm32_allp_pair_scratch_fwd"
            ),
            "causal": True,
            "dispatch_environment": FIXED_RUNTIME_ENV,
            "out_preallocated": True,
            "softmax_lse_preallocated": True,
        },
        DENSE_ROUTE: {
            "entrypoint": "flash_attn_v100.flash_attn_bhmd_func",
            "causal": True,
            "dispatch_environment": DENSE_ENV,
            "out_preallocated": True,
            "softmax_lse_preallocated": False,
            "softmax_lse_reason": "dense pybind API does not accept an LSE buffer",
            "scope": "identity-page zero-copy comparator only",
        },
    }


def _result_template(args: argparse.Namespace) -> dict[str, Any]:
    page_tables = {
        name: {
            "page_ids": list(specification["page_ids"]),
            "description": specification["description"],
            "dense_zero_copy": dict(specification["dense_zero_copy"]),
        }
        for name, specification in _page_specs().items()
    }
    return {
        "schema_version": 1,
        "benchmark": "sm70_flashinfer_realshape_ab",
        "scope": (
            "CUDA-event operator microbenchmark only; no model, engine, HTTP, "
            "or end-to-end execution is performed."
        ),
        "status": "running",
        "geometry": _geometry(),
        "configuration": {
            "physical_gpu": args.physical_gpu,
            "warmup_rounds": args.warmup_rounds,
            "paired_rounds": PAIRED_ROUNDS,
            "load_monitor_interval_ms": LOAD_MONITOR_INTERVAL_MS,
            "seed": args.seed,
        },
        "routes": _route_metadata(),
        "page_tables": page_tables,
        "gpu": {
            "required_graphics_clock_mhz": REQUIRED_GRAPHICS_CLOCK_MHZ,
            "concurrent_gpu_occupancy": "fail",
            "load_monitor": {"status": "not_started"},
            "snapshots": {},
        },
    }


def _run_gpu_benchmark(args: argparse.Namespace, result: dict[str, Any]) -> None:
    _configure_cuda_visibility(args.physical_gpu)
    before = _record_gpu_snapshot(
        result,
        physical_gpu=args.physical_gpu,
        phase="before_cuda_initialization",
        allowed_pids=set(),
    )
    result["gpu"]["processes_before"] = before["gpu_processes"]
    _require_cuda_runtime()
    generic_entry, fixed_entry, dense_entry, api_metadata = _load_apis()
    result["runtime"] = _runtime_metadata(api_metadata)

    query, key_cache, value_cache, sequence_lengths = _make_inputs(args.seed)
    dense_key, dense_key_details = _make_identity_dense_view(key_cache)
    dense_value, dense_value_details = _make_identity_dense_view(value_cache)
    page_specs = _page_specs()
    cases = _make_cases(query, page_specs)
    result["page_tables"]["identity"]["dense_zero_copy"]["key_view"] = dense_key_details
    result["page_tables"]["identity"]["dense_zero_copy"]["value_view"] = (
        dense_value_details
    )
    torch.cuda.synchronize()
    _record_gpu_snapshot(
        result,
        physical_gpu=args.physical_gpu,
        phase="after_preallocation",
        allowed_pids={os.getpid()},
    )

    monitor = _LoadMonitor(
        physical_gpu=args.physical_gpu,
        allowed_pids={os.getpid()},
        interval_ms=LOAD_MONITOR_INTERVAL_MS,
    )
    with torch.inference_mode(), _restore_control_environment():
        try:
            monitor.start()
            monitor.assert_healthy()
            result["correctness"] = _run_correctness(
                cases=cases,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                sequence_lengths=sequence_lengths,
                generic_entry=generic_entry,
                fixed_entry=fixed_entry,
                dense_entry=dense_entry,
                dense_key=dense_key,
                dense_value=dense_value,
                health_check=monitor.assert_healthy,
            )
            torch.cuda.synchronize()
            monitor.assert_healthy()
            _record_gpu_snapshot(
                result,
                physical_gpu=args.physical_gpu,
                phase="after_correctness",
                allowed_pids={os.getpid()},
            )

            routes = _identity_routes(
                identity_case=cases["identity"],
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                sequence_lengths=sequence_lengths,
                generic_entry=generic_entry,
                fixed_entry=fixed_entry,
                dense_entry=dense_entry,
                dense_key=dense_key,
                dense_value=dense_value,
            )
            result["timing"] = _run_interleaved_timing(
                routes,
                warmup_rounds=args.warmup_rounds,
                health_check=monitor.assert_healthy,
            )
            torch.cuda.synchronize()
            monitor.assert_healthy()
        finally:
            monitor_record = monitor.stop()
            result["gpu"]["load_monitor"] = monitor_record
            if monitor_record["samples"]:
                result["gpu"]["processes_after"] = monitor_record["samples"][-1][
                    "gpu_processes"
                ]

        monitor.assert_healthy()
        after = _record_gpu_snapshot(
            result,
            physical_gpu=args.physical_gpu,
            phase="after_timing",
            allowed_pids={os.getpid()},
        )

    result["gpu"]["processes_after"] = after["gpu_processes"]


def _emit_result(args: argparse.Namespace, result: Mapping[str, Any]) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out is None:
        print(payload)
        return
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(payload + "\n")
    print(json.dumps({"status": result["status"], "json_out": str(args.json_out)}))


def main() -> int:
    args = _parse_args()
    result = _result_template(args)
    exit_code = 0
    try:
        result["static_contract"] = _static_contract()
        if args.static_check:
            result["status"] = "static_check_passed"
        else:
            _run_gpu_benchmark(args, result)
            result["status"] = "passed"
    except BaseException as exc:
        exit_code = 1
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    _emit_result(args, result)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
