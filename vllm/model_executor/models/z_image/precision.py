# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep outlier projection outputs finite before the following RMSNorm."""

import torch
from torch import nn
from torch.nn import functional as F


class FP32OutputLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # The first noise-refiner attention projection can exceed FP16's 65504
        # range on official Turbo weights. Clamping loses the following norm's
        # scale. Keep FP16 operands/tensor cores, but retain the FP32 accumulator.
        with torch.amp.autocast(value.device.type, enabled=False):
            if value.is_cuda and value.dtype == self.weight.dtype == torch.float16:
                output = torch.mm(
                    value.reshape(-1, value.shape[-1]),
                    self.weight.t(),
                    out_dtype=torch.float32,
                ).reshape(*value.shape[:-1], self.weight.shape[0])
                return output if self.bias is None else output + self.bias.float()
            return F.linear(
                value.float(),
                self.weight.float(),
                None if self.bias is None else self.bias.float(),
            )


class FP32FeedForward(nn.Module):
    def __init__(self, original: nn.Module):
        super().__init__()
        self.w1 = FP32OutputLinear(original.w1)
        self.w2 = FP32OutputLinear(original.w2)
        self.w3 = FP32OutputLinear(original.w3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Gated activations also exceed half range in the first joint block.
        # Do not clamp them: retain FP32 through gating and the down projection,
        # then the checkpoint's existing RMSNorm safely returns FP16 features.
        return self.w2(F.silu(self.w1(value)) * self.w3(value))


def preserve_projection_range(transformer: nn.Module):
    for block in transformer.modules():
        attention = getattr(block, "attention", None)
        if attention is not None and isinstance(attention.to_out[0], nn.Linear):
            attention.to_out[0] = FP32OutputLinear(attention.to_out[0])
        feed_forward = getattr(block, "feed_forward", None)
        if feed_forward is not None and isinstance(feed_forward.w2, nn.Linear):
            block.feed_forward = FP32FeedForward(feed_forward)
