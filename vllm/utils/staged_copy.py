# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded ownership of asynchronous host-to-device copy sources."""

import torch


class StagedCopyOwner:
    """One owner per logical CPU/GPU buffer, preserving the 64-copy bound."""

    def __init__(self) -> None:
        self.pending: list[
            tuple[torch.cuda.Event | torch.cuda.Stream, torch.Tensor]
        ] = []

    def prune(self) -> None:
        if self.pending:
            self.pending = [
                (event, tensor) for event, tensor in self.pending if not event.query()
            ]

    def copy(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        if dst.device.type != "cuda":
            return dst.copy_(src, non_blocking=True)
        self.prune()
        if len(self.pending) >= 64:
            # Retain ownership even if synchronization raises.
            event, _ = self.pending[0]
            event.synchronize()
            self.pending.pop(0)
        staging = torch.empty(
            src.shape, dtype=src.dtype, device="cpu", pin_memory=src.is_pinned()
        )
        staging.copy_(src)
        stream = torch.cuda.current_stream(dst.device)
        try:
            result = dst.copy_(staging, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        except BaseException:
            # A failed enqueue/record can still leave a copy in flight. Retain
            # its source until the stream itself confirms completion.
            self.pending.append((stream, staging))
            raise
        self.pending.append((event, staging))
        return result
