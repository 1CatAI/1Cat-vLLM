# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Format bindings consumed by the shared SM70 MoE execution skeleton."""

from typing import Any, Protocol

import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
    LayerWorkspaceView,
)


class Sm70MoEWeightCodec(Protocol):
    """Adapt prepared weights/dequant parameters to common stage scheduling.

    Stage calls receive the layer so formats can bind their own scale/zero
    descriptors. Modes describe compute order, not a storage encoding.
    Legacy capability/logging adapters remain until typed policy migration.
    """

    @property
    def native_ops(self) -> Any: ...

    def prepare_weights(self, method: Any, layer: RoutedExperts) -> None: ...

    def gemm_w13(self, mode: str, layer: RoutedExperts, *args: Any) -> None: ...

    def gemm_w2(self, mode: str, layer: RoutedExperts, *args: Any) -> None: ...

    def policy(self, layer: RoutedExperts) -> LayerWorkspaceView: ...

    def buffer(self, layer: RoutedExperts, name: str) -> torch.Tensor: ...

    def enabled(self, capability: str) -> bool: ...

    def message(self, suffix: str) -> str: ...

    def log(self, message: str, *args: Any) -> None: ...
