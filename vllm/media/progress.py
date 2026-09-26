# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-local progress without device synchronization or tensor transfers."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]
_callback: ContextVar[ProgressCallback | None] = ContextVar(
    "media_progress_callback", default=None
)


def report_loading(completed, total, component, *, rank=0, world_size=1, detail=None):
    """Startup progress travels via the existing owned process log."""
    import json

    print(
        "ONECAT_MEDIA_PROGRESS "
        + json.dumps(
            {
                "stage": "loading_weights",
                "completed": completed,
                "total": total,
                "component": component,
                "unit": "components",
                "rank": rank,
                "world_size": world_size,
                **({"detail": detail} if detail is not None else {}),
            }
        ),
        flush=True,
    )


class LoadingProgress:
    """Bounded-rate preparation detail; does not synchronize a device."""

    def __init__(self, component, component_index, rank, world_size):
        self.component = component
        self.index = component_index
        self.rank = rank
        self.world_size = world_size
        self.phase = None
        self.started_at = 0.0
        self.last_report = 0.0

    def __call__(self, phase, completed=0, total=None, unit="items"):
        import time

        now = time.time()
        changed = phase != self.phase
        if changed:
            self.phase, self.started_at = phase, now
        if not changed and now - self.last_report < 1 and completed != total:
            return
        self.last_report = now
        report_loading(
            self.index,
            4,
            self.component,
            rank=self.rank,
            world_size=self.world_size,
            detail={
                "phase": phase,
                "completed": completed,
                "total": total,
                "unit": unit,
                "started_at": self.started_at,
                "updated_at": now,
            },
        )

    def weights(self, values, total):
        self("reading_weights", 0, total, "tensors")
        for index, value in enumerate(values, 1):
            yield value
            self("reading_weights", index, total, "tensors")


@contextmanager
def reporting(callback: ProgressCallback | None) -> Iterator[None]:
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)


def report(stage: str, *, completed: int | None = None, total: int | None = None):
    callback = _callback.get()
    if callback is None:
        return
    event: dict[str, Any] = {"stage": stage}
    if completed is not None and total is not None:
        if not 0 <= completed <= total or total <= 0:
            raise ValueError("Invalid media stage count")
        event["completed"] = completed
        event["total"] = total
    callback(event)


def update_metadata(metadata: dict, event: dict, now: float):
    """Apply events on the API event loop; never infer an overall percentage."""
    stage = event["stage"]
    if metadata.get("stage") != stage:
        metadata.update(stage=stage, stage_started_at=now)
    metadata["updated_at"] = now
    metadata["stage_progress"] = (
        {"completed": event["completed"], "total": event["total"]}
        if "completed" in event and "total" in event
        else None
    )
    if stage == "denoising":
        metadata["denoise_progress"] = metadata["stage_progress"]


class DeviceProgress:
    """Publish steps only after their stream events complete, without waiting on GPU.

    A model forward may only enqueue kernels. A small reporting thread polls
    recorded events, preserving stage order without adding a device/TP barrier.
    Disabled reporting records no events and starts no thread.
    """

    def __init__(
        self, callback: ProgressCallback | None, *, device=0, event_factory=None
    ):
        import threading
        from collections import deque

        self.callback = callback
        self.device = device
        self.event_factory = event_factory
        self.pending: deque[tuple[dict, Any]] = deque()
        self.condition = threading.Condition()
        self.closing = False
        self.stopping = False
        self.thread: threading.Thread | None = None

    def __enter__(self):
        import threading

        if self.callback is None:
            return None
        self.thread = threading.Thread(
            target=self._publish, daemon=True, name="media-progress"
        )
        self.thread.start()
        return self.emit

    def emit(self, event):
        fence = None
        if event.get("stage") == "denoising" and event.get("completed", 0) > 0:
            if self.event_factory is not None:
                fence = self.event_factory()
            else:
                import torch

                fence = torch.Event(device=f"cuda:{self.device}", enable_timing=False)
            fence.record()
        with self.condition:
            self.pending.append((dict(event), fence))
            self.condition.notify()

    def _publish(self):
        try:
            while True:
                with self.condition:
                    if self.stopping or (self.closing and not self.pending):
                        return
                    if not self.pending:
                        self.condition.wait(0.01)
                        continue
                    event, fence = self.pending[0]
                    if fence is not None and not fence.query():
                        self.condition.wait(0.01)
                        continue
                    self.pending.popleft()
                if self.callback is not None:
                    self.callback(event)
        except Exception:
            # A broken observer is not a reason to change inference or discard pixels.
            import logging

            logging.getLogger(__name__).exception("Media progress observer failed")

    def __exit__(self, error_type, *_):
        if self.thread is None:
            return
        with self.condition:
            self.closing = True
            if error_type is not None:
                self.pending.clear()
            self.condition.notify()
        # A successful pipeline has already copied the final output to host. Its
        # step fences are ready. Bound shutdown in case a failed observer blocks.
        self.thread.join(timeout=5)
        with self.condition:
            self.stopping = True
            self.condition.notify()
