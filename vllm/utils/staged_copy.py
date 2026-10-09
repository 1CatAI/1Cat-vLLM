# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded ownership of asynchronous host-to-device copy sources."""

from contextlib import suppress

import torch


class StagedCopyOwner:
    """One owner per logical CPU/GPU buffer, preserving the 64-copy bound."""

    def __init__(self) -> None:
        self.pending: list[
            tuple[torch.cuda.Event | torch.cuda.Stream | None, torch.Tensor]
        ] = []

    def prune(self) -> None:
        if self.pending:
            self.pending = [
                (event, tensor)
                for event, tensor in self.pending
                if event is None or not event.query()
            ]

    def copy(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        if dst.device.type != "cuda":
            return dst.copy_(src, non_blocking=True)
        self.prune()
        if len(self.pending) >= 64:
            # Retain ownership even if synchronization raises.
            event, _ = self.pending[0]
            if event is None:
                with torch.accelerator.device_index(dst.device.index):
                    torch.accelerator.synchronize()
            else:
                event.synchronize()
            self.pending.pop(0)
        staging = torch.empty(
            src.shape, dtype=src.dtype, device="cpu", pin_memory=src.is_pinned()
        )
        staging.copy_(src)
        stream = None
        try:
            result = dst.copy_(staging, non_blocking=True)
            event = torch.cuda.Event()
            # Keep the lookup after enqueue: its host work overlaps the copy.
            stream = torch.cuda.current_stream(dst.device)
            event.record(stream)
        except BaseException:
            # A failed enqueue/record can still leave a copy in flight. Retain
            # its source until the stream itself confirms completion.
            if stream is None:
                with suppress(Exception):
                    stream = torch.cuda.current_stream(dst.device)
            # An unavailable stream is an unfenced lease. Capacity recovery
            # must synchronize the device before releasing that source.
            self.pending.append((stream, staging))
            raise
        self.pending.append((event, staging))
        return result
