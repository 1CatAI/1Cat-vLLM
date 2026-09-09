# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Useful model work accounting and independent NVML sampling."""

from __future__ import annotations

import contextlib
import json
import math
import statistics
import threading
import time
from pathlib import Path


def loaded_kernel_provenance():
    """Identify actual loaded native binaries, including task-owned JIT builds."""
    import hashlib
    import sys

    paths = {
        str(Path(filename).resolve())
        for name, module in list(sys.modules.items())
        if name.startswith(("vllm._h3_", "onecat_h3_"))
        and (filename := getattr(module, "__file__", None))
    }
    return {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in sorted(paths)
    }


class DenoiseWorkCounter:
    """Count actual TP-local matrix shapes, excluding structural padding.

    Hooks apply only to the DiT. Rotation, dequantization, cache preparation,
    padding and redundant output projections contribute no numerator.
    """

    def __init__(self, model, *, used_length, video_outputs, audio_outputs):
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.models.minimax_h3.attention import Attention
        from vllm.model_executor.models.minimax_h3.lora import (
            TurboLinearMethod,
            lora_scale,
        )

        self.flops = 0
        self.calls = 0
        self.blocks: dict[str, int] = {}
        self.steps: list[dict] = []
        self._step_events = []
        self.by_layer: dict[str, int] = {}
        self.handles = []

        def completed_call(module, inputs, output):
            self.calls += 1

        self.handles.append(model.register_forward_hook(completed_call))
        from vllm.model_executor.models.minimax_h3.transformer import (
            MiniMaxH3DiTBlock,
            MiniMaxH3TokenRefinerBlock,
        )

        self.blocks_per_call = sum(
            isinstance(module, (MiniMaxH3DiTBlock, MiniMaxH3TokenRefinerBlock))
            for module in model.modules()
        )
        for name, module in model.named_modules():
            if isinstance(module, (MiniMaxH3DiTBlock, MiniMaxH3TokenRefinerBlock)):

                def block_hook(layer, inputs, output, name=name):
                    self.blocks[name] = self.blocks.get(name, 0) + 1

                self.handles.append(module.register_forward_hook(block_hook))
            elif isinstance(module, LinearBase):

                def linear_hook(layer, inputs, output, name=name):
                    rows = inputs[0].numel() // inputs[0].shape[-1]
                    effective = min(rows, used_length)
                    if name == "final_layer.video_out":
                        effective = min(rows, video_outputs)
                    elif name == "final_layer.audio_out":
                        effective = min(rows, audio_outputs)
                    n, k = layer.weight.shape
                    count = 2 * effective * n * k
                    self.flops += count
                    self.by_layer[name] = self.by_layer.get(name, 0) + count
                    method = layer.quant_method
                    if isinstance(method, TurboLinearMethod) and lora_scale.get() != 0:
                        work = sum(
                            2
                            * effective
                            * (
                                getattr(layer, f"h3_lora_a_{index}").numel()
                                + getattr(layer, f"h3_lora_b_{index}").numel()
                            )
                            for index, _, _ in method.parts
                        )
                        self.flops += work
                        key = name + ".lora"
                        self.by_layer[key] = self.by_layer.get(key, 0) + work

                self.handles.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, Attention):

                def attention_hook(layer, inputs, output, name=name):
                    q, k, v, metadata = inputs
                    used = metadata.extra.get("valid_kv_length", q.shape[1])
                    count = 4 * q.shape[0] * q.shape[2] * used * used * q.shape[3]
                    self.flops += count
                    self.by_layer[name] = self.by_layer.get(name, 0) + count

                self.handles.append(module.register_forward_hook(attention_hook))

    @contextlib.contextmanager
    def step(self, index):
        """Record stream spans without introducing per-step synchronization.

        GPU events include dependent communication and stream waits. CPU enqueue
        time is separate; neither replaces complete synchronized denoise wall time.
        """
        import torch

        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        before = self.flops, self.calls, sum(self.blocks.values())
        start.record()
        started = time.perf_counter()
        yield
        end.record()
        self.steps.append(
            {
                "index": index,
                "cpu_enqueue_seconds": time.perf_counter() - started,
                "useful_flops": self.flops - before[0],
                "dit_calls": self.calls - before[1],
                "executed_blocks": sum(self.blocks.values()) - before[2],
                # This counter currently instruments dense, uncached execution.
                "sparse_blocks": 0,
                "cache_hits": 0,
            }
        )
        self._step_events.append((start, end))

    def finish_steps(self):
        """Read events only after the caller's complete-denoise synchronization."""
        for record, (start, end) in zip(self.steps, self._step_events):
            record["gpu_seconds"] = start.elapsed_time(end) / 1000
        return self.steps

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        for handle in self.handles:
            handle.remove()


class NVMLMonitor:
    def __init__(self, gpu_ids, path, interval=1.0):
        self.gpu_ids = gpu_ids
        self.path = Path(path)
        self.interval = interval
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        import pynvml as nvml

        try:
            nvml.nvmlInit()
            get_processes = nvml.nvmlDeviceGetComputeRunningProcesses
            handles = [
                (index, nvml.nvmlDeviceGetHandleByIndex(index))
                for index in self.gpu_ids
            ]
            with self.path.open("w") as stream:
                while not self.stop.is_set():
                    for index, handle in handles:
                        record = {"timestamp": time.time(), "gpu": index}
                        queries = {
                            "memory_used_bytes": lambda handle=handle: (
                                nvml.nvmlDeviceGetMemoryInfo(handle).used
                            ),
                            "gpu_util_percent": lambda handle=handle: (
                                nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                            ),
                            "memory_util_percent": lambda handle=handle: (
                                nvml.nvmlDeviceGetUtilizationRates(handle).memory
                            ),
                            "power_watts": lambda handle=handle: (
                                nvml.nvmlDeviceGetPowerUsage(handle) / 1000
                            ),
                            "sm_clock_mhz": lambda handle=handle: (
                                nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
                            ),
                            "memory_clock_mhz": lambda handle=handle: (
                                nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                            ),
                            "temperature_c": lambda handle=handle: (
                                nvml.nvmlDeviceGetTemperature(
                                    handle, nvml.NVML_TEMPERATURE_GPU
                                )
                            ),
                            "throttle_reasons": lambda handle=handle: (
                                nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                            ),
                            "compute_processes": lambda handle=handle: [
                                {
                                    "pid": process.pid,
                                    "memory_used_bytes": process.usedGpuMemory,
                                }
                                for process in get_processes(handle)
                            ],
                        }
                        for key, query in queries.items():
                            try:
                                record[key] = query()
                            except nvml.NVMLError as exc:
                                record[key] = None
                                record.setdefault("unavailable", {})[key] = str(exc)
                        stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    self.stop.wait(self.interval)
        except Exception as exc:
            self.path.with_suffix(".error.txt").write_text(str(exc))
        finally:
            with contextlib.suppress(Exception):
                nvml.nvmlShutdown()

    def __exit__(self, *_):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _validate_dense_workload(rank):
    workload = rank["denoise_workload"]
    if (
        workload["attention_algorithm"] != "dense"
        or workload["cache_algorithm"] is not None
    ):
        raise ValueError(
            "sparse/cache workflows require their own measured work accounting"
        )
    schedules = [workload[key] for key in ("video_sigmas", "audio_sigmas")]
    for schedule in schedules:
        if (
            len(schedule) < 2
            or any(not math.isfinite(s) or not 0 <= s <= 1 for s in schedule)
            or any(a <= b for a, b in zip(schedule, schedule[1:]))
            or schedule[-1] != 0
        ):
            raise ValueError("invalid measured sigma schedule")
    if len(schedules[0]) != len(schedules[1]):
        raise ValueError("video/audio schedules must have equal length")
    calls = len(schedules[0]) - 1
    blocks = workload["blocks_per_call"]
    if type(blocks) is not int or blocks <= 0 or rank["dit_calls"] != calls:
        raise ValueError("incomplete denoiser execution for the measured schedule")
    steps = rank["denoise_steps"]
    if len(steps) != calls or [step["index"] for step in steps] != list(range(calls)):
        raise ValueError("missing, duplicate or unordered denoise steps")
    for step in steps:
        if (
            step["dit_calls"] != 1
            or step["executed_blocks"] != blocks
            or step["cache_hits"] != 0
            or step["sparse_blocks"] != 0
        ):
            raise ValueError("dense step work does not match the workflow")
        if type(step["useful_flops"]) is not int or step["useful_flops"] <= 0:
            raise ValueError("useful FLOPs must be positive integer counts")
        for key in ("gpu_seconds", "cpu_enqueue_seconds"):
            if not math.isfinite(step[key]) or step[key] <= 0:
                raise ValueError("invalid measured step duration")
    if (
        sum(step["useful_flops"] for step in steps) != rank["useful_denoise_flops"]
        or sum(rank["denoise_flops_by_layer"].values()) != rank["useful_denoise_flops"]
        or len(rank["denoise_executed_blocks"]) != blocks
        or any(value != calls for value in rank["denoise_executed_blocks"].values())
    ):
        raise ValueError("step, layer and complete-denoise work counts disagree")
    return workload


def evaluate_performance(runs, *, warmup):
    """Evaluate three full requests after a completed, same-session warmup.

    The runtime descriptor supplies the actual sigma intervals. API step counts
    are deliberately not interpreted here: LightX2V and DMD2 differ. Passing a
    shape does not establish coverage of other workloads or their quality.
    """
    if len(runs) != 3:
        raise ValueError("acceptance requires three post-warmup measurements")
    baseline = runs[0]
    tp = baseline["config"]["tensor_parallel_size"]
    if tp not in (1, 2, 4):
        raise ValueError("invalid H3 tensor parallel size")
    if not baseline.get("engine_session_id"):
        raise ValueError("completed same-session warmup evidence is required")
    indices = [run["request_index"] for run in (warmup, *runs)]
    if indices != list(range(indices[0], indices[0] + 4)):
        raise ValueError(
            "warmup and measurements must be consecutive complete requests"
        )
    descriptor = baseline["ranks"][0]["denoise_workload"]
    for index, run in enumerate((warmup, *runs)):
        if sorted(rank["rank"] for rank in run["ranks"]) != list(range(tp)):
            raise ValueError("each measurement must contain every unique TP rank")
        for key in ("config", "request", "gpus", "engine_session_id"):
            if run[key] != baseline[key]:
                raise ValueError(
                    "acceptance measurements must use the same configuration"
                )
        if len(run["gpus"]) != tp or len(set(run["gpus"])) != tp:
            raise ValueError("invalid physical GPU group")
        measurement = run.get("measurement", {})
        if measurement.get("profiled") is not False:
            raise ValueError("formal timing must be explicitly recorded as unprofiled")
        if measurement.get("warmup") is not (index == 0):
            raise ValueError("warmup must be complete and excluded from measurements")
        if measurement.get("capture") is not False:
            raise ValueError("quality captures must be separate from performance runs")
        if run.get("timing_valid") is False:
            raise ValueError("run timing was excluded from performance evidence")
        if (
            not math.isfinite(run["end_to_end_seconds"])
            or run["end_to_end_seconds"] <= 0
        ):
            raise ValueError("invalid end-to-end duration")
        for rank in run["ranks"]:
            if _validate_dense_workload(rank) != descriptor:
                raise ValueError(
                    "all ranks and requests must execute the same workflow"
                )
            seconds = rank["stage_seconds"]["denoise"]
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid denoise duration")
            memory = rank["peak_allocated_bytes"]
            if type(memory) is not int or memory <= 0:
                raise ValueError("invalid peak memory measurement")
    ordered = [sorted(run["ranks"], key=lambda rank: rank["rank"]) for run in runs]
    seconds = [
        max(rank["stage_seconds"]["denoise"] for rank in ranks) for ranks in ordered
    ]
    medians = [
        statistics.median(
            ranks[rank]["useful_denoise_flops"] / duration / 1e12
            for ranks, duration in zip(ordered, seconds)
        )
        for rank in range(tp)
    ]
    cv = statistics.pstdev(seconds) / statistics.mean(seconds)
    memory_passed = all(
        rank["peak_allocated_bytes"] <= 30 * 1024**3
        for run in runs
        for rank in run["ranks"]
    )
    return {
        "workflow": descriptor,
        "sampling": baseline["request"]["sampling"],
        "tensor_parallel_size": tp,
        "rank_median_tflops": medians,
        "denoise_seconds": seconds,
        "denoise_cv": cv,
        "end_to_end_seconds": [run["end_to_end_seconds"] for run in runs],
        "peak_allocated_bytes": [
            max(ranks[rank]["peak_allocated_bytes"] for ranks in ordered)
            for rank in range(tp)
        ],
        "memory_passed": memory_passed,
        "performance_passed": all(value > 80 for value in medians) and cv <= 0.05,
        "quality_status": "requires_separate_numerical_and_human_review",
    }
