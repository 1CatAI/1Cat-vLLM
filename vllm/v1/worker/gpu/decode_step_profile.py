# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-decode-step profiling of segment timings (**diagnostic, off by default**).

Splits one decode round (execute_model + sample_tokens) into per-segment
timings, reporting two timelines: the CPU side (dispatch and preparation) and
the GPU side (actual device work).

This module **does not change inference behaviour**: when disabled every entry
point returns on its first line, and the `gpu()` context manager degrades to a
nullcontext.

Enable with:
    VLLM_DECODE_STEP_PROFILE=1

Optional:
    VLLM_DECODE_STEP_PROFILE_EVERY=<n>    sample and print once every n steps (default 20)
    VLLM_DECODE_STEP_PROFILE_WARMUP=<n>   skip the first n steps (default 5)
    VLLM_DECODE_STEP_PROFILE_PHASE=decode sample pure-decode steps only (default
                                          decode; one of all/decode/prefill)
    VLLM_DECODE_STEP_PROFILE_RANK=<n>     print on that rank only (default: all)

How the timings are taken:
  * CPU segments use ``time.perf_counter()``: wall-clock from the previous
    marker to this one.
  * GPU segments use ``torch.cuda.Event``: real device time for that segment on
    the **main stream**. ``elapsed_time`` is only valid once both ends have
    completed; this runner guarantees that in ``AsyncOutput.get_output()``
    (copy_stream.wait_stream(main_stream) + synchronize), so no extra
    synchronize is needed and the measurement does not perturb the run.
"""

from __future__ import annotations

import contextlib
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _rank_tag() -> str:
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        tp = get_tensor_model_parallel_rank()
    except Exception:
        tp = -1
    try:
        return f"tp{tp}/dev{torch.cuda.current_device()}"
    except Exception:
        return f"tp{tp}"


class DecodeStepProfiler:
    """Accumulate per-step CPU/GPU segment timings; print one line every ``every`` steps."""

    def __init__(self, tag: str = "decode") -> None:
        self.tag = tag
        self.enabled = os.getenv("VLLM_DECODE_STEP_PROFILE", "0") == "1"
        self.every = max(1, _env_int("VLLM_DECODE_STEP_PROFILE_EVERY", 20))
        self.warmup = max(0, _env_int("VLLM_DECODE_STEP_PROFILE_WARMUP", 5))
        self.phase = os.getenv("VLLM_DECODE_STEP_PROFILE_PHASE", "decode").lower()
        _rank = _env_int("VLLM_DECODE_STEP_PROFILE_RANK", -1)
        self.rank_tag = _rank_tag()
        self.silent = _rank >= 0 and not self.rank_tag.startswith(f"tp{_rank}/")

        self.step = 0
        self.sampling = False
        self.is_decode = False
        self.num_reqs = 0
        self.num_toks = 0
        self._t0 = 0.0
        self._tprev = 0.0
        self._cpu: dict[str, float] = {}
        self._gpu_ev: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        # Accumulated across the execute_model and sample_tokens RPCs.
        self._carry_cpu: dict[str, float] = {}
        self._carry_gpu: dict[str, tuple] = {}
        self._step_started = False
        if self.enabled and not self.silent:
            logger.info(
                "DECODE_STEP_PROFILE enabled tag=%s rank=%s every=%d warmup=%d "
                "phase=%s",
                self.tag,
                self.rank_tag,
                self.every,
                self.warmup,
                self.phase,
            )

    # ------------------------------------------------------------ step lifecycle

    def _want(self) -> bool:
        return (
            self.phase == "all"
            or (self.phase == "decode" and self.is_decode)
            or (self.phase == "prefill" and not self.is_decode)
        )

    def begin_step(self, num_reqs: int = 0, num_toks: int = 0) -> None:
        """Called at the execute_model entry point.

        Phase can only be estimated here: a decode step under speculative
        decoding schedules ``num_reqs * (1 + draft tokens)`` tokens, far below
        chunked prefill's 4096 budget, so a loose ratio is used first and then
        corrected by :meth:`set_decode` from ``input_batch``'s authoritative
        ``is_prefilling``. The loose first pass exists so the early segments
        are not lost.
        """
        if not self.enabled:
            return
        self.step += 1
        self.num_reqs = num_reqs
        self.num_toks = num_toks
        self.is_decode = num_reqs > 0 and num_toks <= num_reqs * 16
        self.sampling = (
            self._want()
            and self.step > self.warmup
            and self.step % self.every == 0
        )
        if not self.sampling:
            return
        now = time.perf_counter()
        self._t0 = now
        self._tprev = now
        self._cpu = {}
        self._gpu_ev = {}
        self._step_started = True

    def set_decode(self, is_decode: bool) -> None:
        """Correct the phase decision from ``input_batch.is_prefilling``."""
        if not self.sampling:
            return
        self.is_decode = is_decode
        if not self._want():
            self.sampling = False

    def begin_sample(self) -> None:
        """Called at the sample_tokens entry point; restarts continuous timing across RPCs."""
        if not self.sampling:
            return
        self._tprev = time.perf_counter()

    def end_step(self, extra: str = "") -> None:
        """Called before sample_tokens returns, when all GPU events have completed."""
        if not self.sampling:
            return
        now = time.perf_counter()
        self._cpu["tail"] = self._cpu.get("tail", 0.0) + (now - self._tprev) * 1000.0

        # 合并 execute 阶段结转的分段
        cpu = dict(self._carry_cpu)
        for k, v in self._cpu.items():
            cpu[k] = cpu.get(k, 0.0) + v
        gpu_ev = dict(self._carry_gpu)
        gpu_ev.update(self._gpu_ev)

        gpu: dict[str, float] = {}
        for name, (s, e) in gpu_ev.items():
            try:
                # Draft-propose events are recorded after AsyncOutput's
                # copy_event, so get_output()'s synchronize does not cover
                # them; synchronize each (a no-op once finished).
                e.synchronize()
                gpu[name] = s.elapsed_time(e)
            except Exception:
                gpu[name] = float("nan")

        cpu_total = sum(cpu.values())
        gpu_total = sum(v for v in gpu.values() if v == v)
        parts = " ".join(
            f"{k}={v:.3f}" for k, v in sorted(cpu.items(), key=lambda x: -x[1])
        )
        gparts = " ".join(
            f"{k}={v:.3f}" for k, v in sorted(gpu.items(), key=lambda x: -x[1])
        )
        if not self.silent:
            logger.info(
                "DECODE_STEP_PROFILE[%s] step=%d rank=%s decode=%s "
                "reqs=%d toks=%d "
                "cpu_total_ms=%.3f gpu_total_ms=%.3f | CPU %s | GPU %s%s",
                self.tag,
                self.step,
                self.rank_tag,
                self.is_decode,
                self.num_reqs,
                self.num_toks,
                cpu_total,
                gpu_total,
                parts,
                gparts,
                f" | {extra}" if extra else "",
            )

        self.sampling = False
        self._step_started = False
        self._carry_cpu = {}
        self._carry_gpu = {}

    def carry_over(self) -> None:
        """Called before execute_model returns: move this segment to carry for end_step to merge."""
        if not self.sampling:
            return
        self._carry_cpu = dict(self._cpu)
        self._carry_gpu = dict(self._gpu_ev)
        self._cpu = {}
        self._gpu_ev = {}

    # ------------------------------------------------------------------- CPU 标记

    def cpu(self, name: str) -> None:
        """Record wall-clock from the previous marker to now under ``name``."""
        if not self.sampling:
            return
        now = time.perf_counter()
        self._cpu[name] = self._cpu.get(name, 0.0) + (now - self._tprev) * 1000.0
        self._tprev = now

    # ------------------------------------------------------------------- GPU 标记

    def gpu_begin(self, name: str) -> None:
        """Start GPU timing (for segments that need to wrap an if/else)."""
        if not self.sampling:
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._gpu_ev[name] = (start, start)

    def gpu_end(self, name: str) -> None:
        if not self.sampling:
            return
        entry = self._gpu_ev.get(name)
        if entry is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._gpu_ev[name] = (entry[0], end)

    @contextlib.contextmanager
    def gpu(self, name: str):
        """测量该段在 main stream 上的设备时间（同名多段合并为总跨度）。"""
        if not self.sampling:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            s_prev, _ = self._gpu_ev.get(name, (None, None))
            self._gpu_ev[name] = (s_prev or start, end)


# One per model runner instance, created by GPUModelRunner.__init__.
_profiler: DecodeStepProfiler | None = None


def get_profiler(tag: str = "decode") -> DecodeStepProfiler:
    global _profiler
    if _profiler is None:
        _profiler = DecodeStepProfiler(tag)
    return _profiler
