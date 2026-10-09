# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native bindings for identical execution stages with different weights."""

from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm import _sm70_ops as ops
from vllm.logger import _VllmLogger, log_once_seen
from vllm.model_executor.layers.fused_moe.sm70.declarations import native_binding


@dataclass(frozen=True)
class Sm70MoEWeightCodec:
    """Resolve native bindings through their sole module owner, never policy.

    The stage names describe execution, not formats. Native symbol names stay
    unchanged and rebinding the public operator owner remains visible.
    """

    name: Literal["AWQ", "FP8"]
    logger: _VllmLogger
    diagnostic: bool = False
    bindings: Any = None

    @property
    def operators(self):
        return self.bindings if self.bindings is not None else ops

    def prepare_weights(
        self,
        w13: tuple[torch.Tensor, ...],
        w2: tuple[torch.Tensor, ...],
        group_size: int,
        *,
        compact_metadata: bool = False,
        w13_interleaved: bool = False,
    ) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        """Prepare alternating W13/W2 expert banks in the original order.

        Layout arguments are a preparation contract. The codec never reads or
        resolves the execution policy, and never owns layer scratch tensors.
        """
        suffix = "_compact" if compact_metadata else ""
        prepare = getattr(self.operators, self.name.lower() + "_sm70_prepare" + suffix)
        result13: list[list[torch.Tensor]] = [[], [], []]
        result2: list[list[torch.Tensor]] = [[], [], []]
        for expert in range(w13[0].shape[0]):
            for weights, result, interleaved in (
                (w13, result13, w13_interleaved),
                (w2, result2, False),
            ):
                args = tuple(tensor[expert] for tensor in weights) + (group_size,)
                if self.name == "AWQ":
                    args += (interleaved,)
                prepared = prepare(*args)
                for destination, tensor in zip(result, prepared):
                    destination.append(tensor)
        return result13, result2

    def gemm_w13(self, mode: str, *args: Any) -> None:
        getattr(self.operators, native_binding(self.name, "w13", mode))(*args)

    def gemm_w2(self, mode: str, *args: Any) -> None:
        getattr(self.operators, native_binding(self.name, "w2", mode))(*args)

    def log(self, message: str, *args: Any) -> None:
        if not self.diagnostic and not torch.compiler.is_compiling():
            key = ("sm70_moe_stage", self.logger.name, self.name, message, args)
            if not log_once_seen(key):
                self.logger.info_once(
                    "SM70 " + self.name + " " + message, *args, key=key
                )
