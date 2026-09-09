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
        self.by_layer: dict[str, int] = {}
        self.handles = []

        def completed_call(module, inputs, output):
            self.calls += 1

        self.handles.append(model.register_forward_hook(completed_call))
        for name, module in model.named_modules():
            if isinstance(module, LinearBase):

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


def evaluate_performance(runs):
    """Evaluate three unprofiled post-warmup runs; never use utilization as FLOPs."""
    if len(runs) != 3 or any(len(run["ranks"]) != 4 for run in runs):
        raise ValueError("acceptance requires three four-rank measurements")
    baseline = runs[0]
    for run in runs:
        if sorted(rank["rank"] for rank in run["ranks"]) != list(range(4)):
            raise ValueError("each measurement must contain four unique TP ranks")
        if any(run[key] != baseline[key] for key in ("config", "request", "gpus")):
            raise ValueError("acceptance measurements must use the same configuration")
        sampling = run["request"]["sampling"]
        expected = {
            "width": 1344,
            "height": 768,
            "num_frames": 243,
            "fps": 24,
            "seed": 42,
            "num_inference_steps": 50,
        }
        if any(sampling.get(key) != value for key, value in expected.items()):
            raise ValueError("acceptance requires the fixed primary workload")
        if run.get("measurement", {}).get("profiled") is not False:
            raise ValueError("formal timing must be explicitly recorded as unprofiled")
        if run.get("timing_valid") is False:
            raise ValueError("run timing was excluded from performance evidence")
        if any(rank["dit_calls"] != 49 for rank in run["ranks"]):
            raise ValueError("the primary schedule requires 49 completed DiT calls")
        if any(
            type(rank["useful_denoise_flops"]) is not int
            or rank["useful_denoise_flops"] <= 0
            for rank in run["ranks"]
        ):
            raise ValueError("useful FLOPs must be positive integer counts")
    ordered = [sorted(run["ranks"], key=lambda rank: rank["rank"]) for run in runs]
    seconds = [
        max(rank["stage_seconds"]["denoise"] for rank in run["ranks"]) for run in runs
    ]
    if any(not math.isfinite(value) or value <= 0 for value in seconds):
        raise ValueError("invalid denoise duration")
    medians = []
    for rank in range(4):
        values = [
            ranks[rank]["useful_denoise_flops"] / duration / 1e12
            for ranks, duration in zip(ordered, seconds)
        ]
        medians.append(statistics.median(values))
    cv = statistics.pstdev(seconds) / statistics.mean(seconds)
    memory_passed = all(
        rank["peak_allocated_bytes"] <= 30 * 1024**3
        for run in runs
        for rank in run["ranks"]
    )
    return {
        "rank_median_tflops": medians,
        "denoise_seconds": seconds,
        "denoise_cv": cv,
        "memory_passed": memory_passed,
        "performance_passed": all(value > 80 for value in medians) and cv <= 0.05,
    }
