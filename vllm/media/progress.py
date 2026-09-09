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
