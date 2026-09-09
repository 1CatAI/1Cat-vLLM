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
        if name.startswith(
            (
                "vllm._h3_",
                "onecat_h3_",
                "onecat_sm70_sparse_attention",
                "vllm._sm70_sparse_attention_C",
            )
        )
        and (filename := getattr(module, "__file__", None))
    }
    return {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def lora_work(layer, parts, rows, *, replicated_a):
    """Separate identical column-A replicas from useful TP partial products.

    Column-parallel A has identical weights and full input rows on every rank.
    Attribute each row once, balanced by rank, including non-divisible tails.
    Row-parallel A instead consumes distinct input shards; its B products are
    distinct partial contributions and are not identical replicas.
    """
    a_elements = sum(
        getattr(layer, f"h3_lora_a_{index}").numel() for index, _, _ in parts
    )
    b_elements = sum(
        getattr(layer, f"h3_lora_b_{index}").numel() for index, _, _ in parts
    )
    executed = 2 * rows * (a_elements + b_elements)
    redundant = 0
    if replicated_a:
        owner_rows = rows // layer.tp_size + (layer.tp_rank < rows % layer.tp_size)
        redundant = 2 * (rows - owner_rows) * a_elements
    return executed - redundant, redundant


class DenoiseWorkCounter:
    """Count actual TP-local matrix shapes, excluding structural padding.

    Hooks apply only to the DiT. Rotation, dequantization, cache preparation,
    padding and redundant output projections contribute no numerator.
    """

    def __init__(self, model, *, used_length, video_outputs, audio_outputs):
        from vllm.model_executor.layers.linear import ColumnParallelLinear, LinearBase
        from vllm.model_executor.models.minimax_h3.attention import Attention
        from vllm.model_executor.models.minimax_h3.lora import (
            TurboLinearMethod,
            lora_scale,
        )

        self.flops = 0
        self.redundant_flops = 0
        self.calls = 0
        self.sparse_blocks = 0
        self.sparse_pairs = 0
        self.sparse_compression_flops = 0
        self.attention_avoided_flops = 0
        self.sparse_by_layer: dict[str, dict] = {}
        self.blocks: dict[str, int] = {}
        self.steps: list[dict] = []
        self._step_events = []
        self.by_layer: dict[str, int] = {}
        self.redundant_by_layer: dict[str, int] = {}
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
                        work, redundant = lora_work(
                            layer,
                            method.parts,
                            effective,
                            replicated_a=isinstance(layer, ColumnParallelLinear),
                        )
                        self.flops += work
                        key = name + ".lora"
                        self.by_layer[key] = self.by_layer.get(key, 0) + work
                        self.redundant_flops += redundant
                        self.redundant_by_layer[key] = (
                            self.redundant_by_layer.get(key, 0) + redundant
                        )

                self.handles.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, Attention):

                def attention_hook(layer, inputs, output, name=name):
                    q, k, v, metadata = inputs
                    used = metadata.extra.get("valid_kv_length", q.shape[1])
                    if layer.backend == "FASTVIDEO_VSA":
                        work = metadata.extra["sparse_work"]
                        count = 4 * work["selected_token_pairs"] * q.shape[3]
                        compression = work["compression_flops"]
                        self.sparse_blocks += work["selected_blocks"]
                        self.sparse_pairs += work["selected_token_pairs"]
                        self.sparse_compression_flops += compression
                        avoided = (
                            4
                            * (work["dense_token_pairs"] - work["selected_token_pairs"])
                            * q.shape[3]
                        )
                        self.attention_avoided_flops += avoided
                        record = self.sparse_by_layer.setdefault(
                            name,
                            {
                                "head_size": q.shape[3],
                                "heads": q.shape[2],
                                "selected_blocks": 0,
                                "selected_token_pairs": 0,
                                "compression_flops": 0,
                                "dense_token_pairs": 0,
                            },
                        )
                        for key in (
                            "selected_blocks",
                            "selected_token_pairs",
                            "compression_flops",
                            "dense_token_pairs",
                        ):
                            record[key] += work[key]
                        key = name + ".compression"
                        self.by_layer[key] = self.by_layer.get(key, 0) + compression
                        self.flops += compression
                    else:
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
        before = self.flops, self.calls, sum(self.blocks.values()), self.redundant_flops
        sparse_before = self.sparse_blocks
        pairs_before = self.sparse_pairs
        compression_before = self.sparse_compression_flops
        avoided_before = self.attention_avoided_flops
        start.record()
        started = time.perf_counter()
        yield
        end.record()
        self.steps.append(
            {
                "index": index,
                "cpu_enqueue_seconds": time.perf_counter() - started,
                "useful_flops": self.flops - before[0],
                "redundant_flops": self.redundant_flops - before[3],
                "dit_calls": self.calls - before[1],
                "executed_blocks": sum(self.blocks.values()) - before[2],
                "sparse_blocks": self.sparse_blocks - sparse_before,
                "sparse_token_pairs": self.sparse_pairs - pairs_before,
                "sparse_compression_flops": self.sparse_compression_flops
                - compression_before,
                "attention_avoided_flops": self.attention_avoided_flops
                - avoided_before,
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


def _validate_sparse_work(rank, calls):
    workload = rank["denoise_workload"]
    config = workload["sparse_config"]
    for key in ("topk", "gated_blocks", "heads", "head_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("invalid sparse execution configuration")
    if (
        config["head_size"] != 128
        or config["gated_blocks"] != workload["blocks_per_call"] - 2
        or "FASTVIDEO_VSA" not in workload["actual_backends"]
    ):
        raise ValueError("sparse execution must cover the actual gated H3 blocks")
    prefix, shape = config["prefix_segments"], config["video_shape"]
    if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in (*prefix, *shape)):
        raise ValueError("invalid sparse geometry")
    if sum(prefix) + math.prod(shape) != workload["used_length"]:
        raise ValueError("sparse geometry disagrees with valid token count")
    prefix_blocks = sum((n + 63) // 64 for n in prefix)
    video_blocks = math.prod((n + 3) // 4 for n in shape)
    blocks = prefix_blocks + video_blocks
    per_layer_blocks = config["heads"] * (
        prefix_blocks * blocks
        + video_blocks * (prefix_blocks + min(config["topk"], video_blocks))
    )
    per_layer_dense_pairs = config["heads"] * workload["used_length"] ** 2
    per_layer_compression = 4 * config["heads"] * blocks**2 * config["head_size"]
    layers = rank["denoise_sparse_work_by_layer"]
    if len(layers) != config["gated_blocks"]:
        raise ValueError("missing sparse layer execution records")
    sums = dict(
        selected_blocks=0,
        selected_token_pairs=0,
        compression_flops=0,
        dense_token_pairs=0,
    )
    for name, layer in layers.items():
        if any(type(layer[key]) is not int or layer[key] <= 0 for key in sums):
            raise ValueError("sparse work must use positive integer counts")
        if (
            layer["head_size"] != config["head_size"]
            or layer["heads"] != config["heads"]
            or layer["selected_blocks"] != calls * per_layer_blocks
            or layer["dense_token_pairs"] != calls * per_layer_dense_pairs
            or layer["compression_flops"] != calls * per_layer_compression
            or not layer["selected_blocks"]
            <= layer["selected_token_pairs"]
            <= min(layer["dense_token_pairs"], layer["selected_blocks"] * 64**2)
            or rank["denoise_flops_by_layer"][name]
            != 4 * layer["selected_token_pairs"] * config["head_size"]
            or rank["denoise_flops_by_layer"][name + ".compression"]
            != layer["compression_flops"]
        ):
            raise ValueError("sparse layer pairs, blocks or compression disagree")
        for key in sums:
            sums[key] += layer[key]
    steps = rank["denoise_steps"]
    for step in steps:
        if (
            step["sparse_blocks"] != per_layer_blocks * config["gated_blocks"]
            or type(step["sparse_token_pairs"]) is not int
            or not 0
            < step["sparse_token_pairs"]
            <= per_layer_dense_pairs * config["gated_blocks"]
            or step["sparse_compression_flops"]
            != per_layer_compression * config["gated_blocks"]
            or step["attention_avoided_flops"]
            != 4
            * config["head_size"]
            * (
                per_layer_dense_pairs * config["gated_blocks"]
                - step["sparse_token_pairs"]
            )
        ):
            raise ValueError("sparse step counters disagree with executed geometry")
    if any(
        sum(step[key] for step in steps) != sums[target]
        for key, target in (
            ("sparse_blocks", "selected_blocks"),
            ("sparse_token_pairs", "selected_token_pairs"),
            ("sparse_compression_flops", "compression_flops"),
        )
    ):
        raise ValueError("sparse step and layer totals disagree")


def _validate_workload(rank):
    workload = rank["denoise_workload"]
    sparse = workload["attention_algorithm"] == "vsa"
    expected_version = "sparse_tp_v1" if sparse else "dense_tp_lora_v2"
    if workload.get("work_accounting") != expected_version:
        raise ValueError("legacy work counts may include replicated LoRA projections")
    if (
        workload["attention_algorithm"] not in ("dense", "vsa")
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
            or (not sparse and step["sparse_blocks"] != 0)
        ):
            raise ValueError("dense step work does not match the workflow")
        if type(step["useful_flops"]) is not int or step["useful_flops"] <= 0:
            raise ValueError("useful FLOPs must be positive integer counts")
        if type(step["redundant_flops"]) is not int or step["redundant_flops"] < 0:
            raise ValueError("redundant work must be a nonnegative integer count")
        for key in ("gpu_seconds", "cpu_enqueue_seconds"):
            if not math.isfinite(step[key]) or step[key] <= 0:
                raise ValueError("invalid measured step duration")
    if (
        sum(step["useful_flops"] for step in steps) != rank["useful_denoise_flops"]
        or sum(rank["denoise_flops_by_layer"].values()) != rank["useful_denoise_flops"]
        or len(rank["denoise_executed_blocks"]) != blocks
        or any(value != calls for value in rank["denoise_executed_blocks"].values())
        or sum(step["redundant_flops"] for step in steps)
        != rank["redundant_denoise_flops"]
        or sum(rank["redundant_flops_by_layer"].values())
        != rank["redundant_denoise_flops"]
    ):
        raise ValueError("step, layer and complete-denoise work counts disagree")
    if sparse:
        _validate_sparse_work(rank, calls)
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
            if _validate_workload(rank) != descriptor:
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
        "attention_avoided_flops_by_run_and_rank": [
            [
                sum(
                    step.get("attention_avoided_flops", 0)
                    for step in rank["denoise_steps"]
                )
                for rank in ranks
            ]
            for ranks in ordered
        ],
        "performance_passed": all(value > 80 for value in medians) and cv <= 0.05,
        "quality_status": "requires_separate_numerical_and_human_review",
    }
