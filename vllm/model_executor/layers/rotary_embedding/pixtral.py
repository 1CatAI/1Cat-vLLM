# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pixtral's flat-grid RoPE, independent of Transformers' private helpers."""

import torch
from torch import nn


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        rows, columns = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
        positions.append((rows * max_width + columns).reshape(-1))
    return torch.cat(positions)


class PixtralRotaryEmbedding(nn.Module):
    """Preserve the original FP32 H-W-H-W frequencies and flat position IDs.

    Transformers 5 uses a different public class and two-coordinate IDs.
    Keep vLLM's existing flat-grid interface and arithmetic instead of
    interpreting those IDs through the new interface.
    """

    def __init__(self, config, device=None):
        super().__init__()
        dim = config.head_dim
        rope_parameters = getattr(config, "rope_parameters", None)
        base = (
            rope_parameters["rope_theta"]
            if rope_parameters is not None
            else config.rope_theta
        )
        side = config.image_size // config.patch_size
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        h = torch.arange(side, device=freqs.device)
        w = torch.arange(side, device=freqs.device)
        freqs_h = torch.outer(h, freqs[::2]).float()
        freqs_w = torch.outer(w, freqs[1::2]).float()
        grid = torch.cat(
            [
                freqs_h[:, None, :].repeat(1, side, 1),
                freqs_w[None, :, :].repeat(side, 1, 1),
            ],
            dim=-1,
        ).reshape(-1, dim // 2)
        self.register_buffer(
            "inv_freq", torch.cat((grid, grid), dim=-1), persistent=False
        )

    @torch.no_grad()
    def forward(self, x, position_ids):
        freqs = self.inv_freq[position_ids]
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            cos = freqs.cos()
            sin = freqs.sin()
        return cos.to(x.dtype), sin.to(x.dtype)
