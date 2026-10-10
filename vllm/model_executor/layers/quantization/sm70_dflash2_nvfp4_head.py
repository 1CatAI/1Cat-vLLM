# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A private NVFP4 candidate head, independent of target verification weights."""

import torch
from torch import nn

from vllm import _sm70_ops as ops


@torch.no_grad()
def pack_draft_head(dense: torch.Tensor):
    """Quantize N x K FP16 weights with E2M1 / per-16 E4M3 scales.

    Bound temporary memory by quantizing row chunks. This happens once, before
    KV allocation and graph capture; no quantization runs during generation.
    """
    n, k = dense.shape
    global_scale = float(dense.abs().max()) / (6.0 * 448.0)
    if global_scale == 0:
        global_scale = 1.0
    packed = torch.empty(n, k // 2, device=dense.device, dtype=torch.uint8)
    scales = torch.empty(n, k // 16, device=dense.device, dtype=torch.float8_e4m3fn)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dense.device
    )
    for begin in range(0, n, 1024):
        end = min(begin + 1024, n)
        groups = dense[begin:end].float().reshape(end - begin, k // 16, 16)
        block_scale = (groups.abs().amax(-1) / (6.0 * global_scale)).to(
            torch.float8_e4m3fn
        )
        effective = (block_scale.float() * global_scale).half().float()
        normalized = groups / effective.clamp_min(2.0**-24)[..., None]
        magnitude = normalized.abs()
        code = torch.bucketize(magnitude, boundaries)
        boundary = boundaries[code.clamp_max(6)]
        code += ((magnitude == boundary) & (code < 7) & ((code & 1) != 0)).long()
        code = (code | (normalized.signbit().long() << 3)).to(torch.uint8)
        code = code.reshape(end - begin, k)
        packed[begin:end] = code[:, 0::2] | (code[:, 1::2] << 4)
        scales[begin:end] = block_scale
    codes, native_scales = ops.nvfp4_qpn2_prepare_sm70(packed, scales)
    return codes, native_scales, global_scale


class SM70DFlash2NVFP4Head(nn.Module):
    """One eight-row tile for draft7 candidates; larger batches keep their head."""

    def __init__(self, codes, scales, global_scale, vocab_width, hidden_size):
        super().__init__()
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.global_scale = global_scale
        self.vocab_width = vocab_width
        self.hidden_size = hidden_size

    @classmethod
    @torch.no_grad()
    def from_turbomind_head(cls, head):
        # Decode the original packed FP8 matrix without modifying its tensors.
        dense_kn = torch.empty(
            head.embedding_dim,
            head.num_embeddings_per_partition,
            device=head.weight.device,
            dtype=torch.float16,
        )
        ops.fp8_sm70_dequantize_out(dense_kn, head.weight, head.weight_scale_inv, 128)
        codes, scales, global_scale = pack_draft_head(dense_kn.t())
        return cls(
            codes,
            scales,
            global_scale,
            head.num_embeddings_per_partition,
            head.embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        if (
            x.ndim != 2
            or x.shape[0] not in (7, 8)
            or x.shape[1] != self.hidden_size
            or x.dtype != torch.float16
            or not x.is_contiguous()
        ):
            return None
        output = x.new_empty((x.shape[0], self.vocab_width))
        ops.nvfp4_qpn2_gemm_sm70_out(
            output, x, self.codes, self.scales, self.global_scale, 16, 1
        )
        return output
