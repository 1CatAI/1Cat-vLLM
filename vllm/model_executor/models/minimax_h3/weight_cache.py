# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Budgeted cache of explicitly selected, rotated FP16 H3 weights."""

import torch


class FP16WeightCache:
    def __init__(self, model, *, budget_gib, layers):
        self.layers = []
        self.bytes = 0
        modules = dict(model.named_modules())
        for name in layers:
            if name not in modules:
                raise ValueError(f"unknown H3 cache layer {name}")
            layer = modules[name]
            if layer.weight.dtype != torch.int8 or not hasattr(layer, "weight_scale"):
                raise ValueError(f"FP16 cache requires a quantized layer: {name}")
            self.bytes += layer.weight.numel() * 2
            self.layers.append(layer)
        if self.bytes > int(budget_gib * 1024**3):
            raise ValueError("fixed H3 FP16 cache list exceeds its memory budget")

    def prepare(self):
        from .cuda_ops import w8a16_extension

        try:
            for layer in self.layers:
                free, _ = torch.accelerator.get_memory_info(layer.weight.device)
                required = layer.weight.numel() * 2
                if (
                    required * 2 >= free
                    or torch.accelerator.memory_allocated(layer.weight.device)
                    + required * 2
                    > 30 * 1024**3
                ):
                    raise RuntimeError(
                        "H3 FP16 weight cache exceeds available GPU memory"
                    )
                # The weight stays in ConvRot coordinates; activations still
                # receive the original rotation on every invocation.
                weight = w8a16_extension().dequantize(layer.weight, layer.weight_scale)
                # Preserve logical [N,K], but store physical [K,N] for the
                # measured cuBLASLt path. Budget both conversion temporaries.
                layer.h3_fp16_weight = weight.t().contiguous().t()
                del weight
        except BaseException:
            self.clear()
            raise

    def clear(self):
        for layer in self.layers:
            if hasattr(layer, "h3_fp16_weight"):
                del layer.h3_fp16_weight
