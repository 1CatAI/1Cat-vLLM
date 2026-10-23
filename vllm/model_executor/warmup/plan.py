# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ordered warmup execution. Each task retains ownership of its resources."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class WarmupTask:
    name: str
    run: Callable[[], tuple[str, ...]]


def run_warmup_tasks(tasks: Iterable[WarmupTask]) -> tuple[str, ...]:
    warmed: list[str] = []
    for task in tasks:
        warmed.extend(task.run())
    return tuple(warmed)


def warmup_boolean(name: str, run: Callable[[], bool]) -> WarmupTask:
    def execute() -> tuple[str, ...]:
        return (name,) if run() else ()

    return WarmupTask(name, execute)


def warmup_unconditional(name: str, run: Callable[[], object]) -> WarmupTask:
    def execute() -> tuple[str, ...]:
        run()
        return (name,)

    return WarmupTask(name, execute)
