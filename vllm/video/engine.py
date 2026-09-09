# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent, serial TP worker group shared by offline and HTTP video APIs."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import socket
import threading
import time
import traceback
from dataclasses import asdict
from multiprocessing.connection import Connection
from pathlib import Path
from typing import cast

from vllm.media.progress import DeviceProgress, ProgressCallback, reporting
from vllm.model_executor.models.minimax_h3.config import H3Config, H3Request

from .gpu import acquire_gpu_group, worker_device_mask
from .gpu import select_gpu_group as select_gpu_group


def _worker(rank, config, gpu_ids, endpoint, connection):
    os.environ["CUDA_VISIBLE_DEVICES"] = worker_device_mask(gpu_ids)
    from datetime import timedelta

    import torch

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.models.minimax_h3.pipeline import MiniMaxH3Pipeline

    torch.set_num_threads(4)
    torch.accelerator.set_device_index(rank)
    # Explicitly disallow reduced-precision GEMM reductions for FP16 islands.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    current = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=config.tensor_parallel_size)
    )
    try:
        with set_current_vllm_config(current):
            init_distributed_environment(
                config.tensor_parallel_size,
                rank,
                endpoint,
                rank,
                "nccl",
                timeout=timedelta(minutes=30),
            )
            initialize_model_parallel(config.tensor_parallel_size)
            started = time.perf_counter()
            pipeline = MiniMaxH3Pipeline(config)
            connection.send(
                {
                    "ready": True,
                    "rank": rank,
                    "startup_seconds": time.perf_counter() - started,
                }
            )
            while True:
                command = connection.recv()
                if command is None:
                    break
                request, output_dir, track_progress = command
                torch.accelerator.reset_peak_memory_stats()
                callback = (
                    (lambda event: connection.send({"event": "progress", **event}))
                    if rank == 0 and track_progress
                    else None
                )
                with (
                    DeviceProgress(callback, device=rank) as observer,
                    reporting(observer),
                ):
                    video, audio = pipeline(request)
                result = {
                    "rank": rank,
                    "stage_seconds": pipeline.stage_durations,
                    "dit_calls": pipeline.actual_dit_calls,
                    "useful_denoise_flops": pipeline.useful_denoise_flops,
                    "denoise_flops_by_layer": pipeline.denoise_flops_by_layer,
                    "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
                }
                if rank == 0:
                    from .media import export_video

                    if callback:
                        callback({"stage": "packaging"})
                    started = time.perf_counter()
                    result.update(
                        export_video(
                            video,
                            audio,
                            output_dir,
                            fps=request.sampling.fps,
                            encoder=config.video_encoder,
                            gpu_index=rank,
                        )
                    )
                    result["stage_seconds"]["packaging"] = time.perf_counter() - started
                    from vllm.model_executor.models.minimax_h3.time_request import (
                        minimax_h3_align_frame_count,
                    )

                    from .quality import inspect_video

                    started = time.perf_counter()
                    duration = request.sampling.extra_args.get("duration_seconds")
                    frames = (
                        round(duration * request.sampling.fps)
                        if duration is not None
                        else request.sampling.num_frames
                    )
                    result["quality"] = inspect_video(
                        result["video"],
                        expected_frames=minimax_h3_align_frame_count(frames),
                        expected_width=request.sampling.width,
                        expected_height=request.sampling.height,
                        expected_fps=request.sampling.fps,
                    )
                    result["stage_seconds"]["quality_validation"] = (
                        time.perf_counter() - started
                    )
                connection.send(result)
                del video, audio
    except BaseException:
        connection.send({"error": traceback.format_exc(), "rank": rank})
    finally:
        cleanup_dist_env_and_memory()
        connection.close()


class H3Engine:
    def __init__(self, config: H3Config):
        self.config = config
        self._gpu_lease = None
        self._lock = threading.Lock()
        self._closed = False
        self.workers = []
        self.connections = []
        self.startup = []
        context = mp.get_context("spawn")
        try:
            self._gpu_lease = acquire_gpu_group(config.tensor_parallel_size)
            self.gpu_ids = self._gpu_lease.gpu_ids
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            endpoint = f"tcp://127.0.0.1:{port}"
            for rank in range(config.tensor_parallel_size):
                parent, child = context.Pipe()
                worker = context.Process(
                    target=_worker, args=(rank, config, self.gpu_ids, endpoint, child)
                )
                worker.start()
                child.close()
                self.workers.append(worker)
                self.connections.append(parent)
            self.startup = self._receive_all(timeout=7200)
        except BaseException:
            self.close()
            raise

    def _receive_all(self, *, timeout, on_progress: ProgressCallback | None = None):
        from multiprocessing.connection import wait

        pending = set(self.connections)
        results = []
        deadline = time.monotonic() + timeout
        while pending:
            if time.monotonic() >= deadline:
                raise TimeoutError("H3 distributed worker timed out")
            for ready_connection in wait(pending, timeout=1):
                connection = cast(Connection, ready_connection)
                message = connection.recv()
                if "error" in message:
                    raise RuntimeError(message["error"])
                if message.get("event") == "progress":
                    if on_progress is not None:
                        on_progress({k: v for k, v in message.items() if k != "event"})
                    continue
                results.append(message)
                pending.remove(connection)
            for worker, connection in zip(self.workers, self.connections):
                if connection in pending and not worker.is_alive():
                    raise RuntimeError(
                        f"H3 worker {worker.pid} exited ({worker.exitcode})"
                    )
        return sorted(results, key=lambda result: result["rank"])

    def generate(
        self,
        request: H3Request,
        output_dir: str | Path,
        *,
        on_progress: ProgressCallback | None = None,
    ):
        import json

        from vllm.model_executor.models.minimax_h3.validation import validate_request

        from .metrics import NVMLMonitor

        validate_request(self.config, request)

        with self._lock:
            if self._closed:
                raise RuntimeError("H3 engine is closed")
            output_dir = Path(output_dir).absolute()
            output_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                for connection in self.connections:
                    connection.send((request, str(output_dir), on_progress is not None))
                with NVMLMonitor(self.gpu_ids, output_dir / "nvml.jsonl"):
                    ranks = self._receive_all(
                        timeout=24 * 3600, on_progress=on_progress
                    )
            except BaseException:
                # A failing rank invalidates the whole communicator. Release
                # only owned workers instead of reusing a partially alive group.
                self.close()
                raise
            result = {
                "config": asdict(self.config),
                "request": asdict(request),
                "gpus": self.gpu_ids,
                "ranks": ranks,
                "startup": self.startup,
                "end_to_end_seconds": time.perf_counter() - started,
            }
            (output_dir / "run.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False)
            )
            return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        for connection in self.connections:
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                connection.send(None)
        for worker in self.workers:
            worker.join(timeout=3)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=3)
            if worker.is_alive():
                worker.kill()
                worker.join()
        for connection in self.connections:
            connection.close()
        if self._gpu_lease is not None:
            self._gpu_lease.close()
            self._gpu_lease = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
