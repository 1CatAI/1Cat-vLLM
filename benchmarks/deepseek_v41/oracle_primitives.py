# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference building blocks for the V4.1 GGUF oracle.

These are validation helpers, not a registered model or a complete M1 oracle.
Weights remain memory mapped; only requested rows are dequantized to FP32.
"""

from collections.abc import Iterable
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F


class GGUFWeights:
    def __init__(self, paths: Iterable[Path]):
        self.readers = [gguf.GGUFReader(path, mode="r") for path in paths]
        self.tensors = {}
        for reader in self.readers:
            for tensor in reader.tensors:
                if tensor.name in self.tensors:
                    raise ValueError(f"Duplicate tensor: {tensor.name}")
                self.tensors[tensor.name] = tensor

    def integer(self, name: str) -> torch.Tensor:
        tensor = self.tensors[name]
        if tensor.tensor_type not in (
            gguf.GGMLQuantizationType.I32,
            gguf.GGMLQuantizationType.I64,
        ):
            raise ValueError(f"Expected integer tensor: {name}")
        return torch.from_numpy(np.array(tensor.data, dtype=np.int64, copy=True))

    def rows(self, name: str, row_ids: torch.Tensor) -> torch.Tensor:
        """Read flattened outer rows, preserving duplicates and caller shape."""
        tensor = self.tensors[name]
        ids = row_ids.to(device="cpu", dtype=torch.int64).numpy()
        width = int(tensor.shape[0])
        row_count = int(np.prod(tensor.shape[1:]))
        if np.any(ids < 0) or np.any(ids >= row_count):
            raise ValueError(f"Row outside tensor: {name}")
        unique, inverse = np.unique(ids, return_inverse=True)
        packed = tensor.data.reshape(row_count, -1)[unique]
        values = gguf.quants.dequantize(packed, tensor.tensor_type)
        values = np.asarray(values, dtype=np.float32).reshape(len(unique), width)
        return torch.from_numpy(values[inverse].reshape(*ids.shape, width))

    @torch.inference_mode()
    def linear(
        self, name: str, x: torch.Tensor, rows_per_chunk: int = 256
    ) -> torch.Tensor:
        tensor = self.tensors[name]
        if len(tensor.shape) != 2 or x.shape[-1] != tensor.shape[0]:
            raise ValueError(f"Linear shape mismatch: {name}")
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        x = x.to(device="cpu", dtype=torch.float32)
        out_features = int(tensor.shape[1])
        result = torch.empty((*x.shape[:-1], out_features), dtype=torch.float32)
        for start in range(0, out_features, rows_per_chunk):
            end = min(start + rows_per_chunk, out_features)
            weight = self.rows(name, torch.arange(start, end))
            result[..., start:end] = F.linear(x, weight)
        return result


class EngramHashState:
    """Hash from checkpoint auxiliaries with only max_ngram_size-1 cached ids.

    A masked token blocks lookback across its position. Calls must be contiguous;
    start_pos=0 explicitly starts a new sequence, as in the official reference.
    """

    DEAD = -1

    def __init__(
        self,
        token_map: torch.Tensor,
        primes: torch.Tensor,
        offsets: torch.Tensor,
        multipliers: torch.Tensor,
        pad_token_id: int,
    ):
        self.token_map = token_map.to(device="cpu", dtype=torch.int64)
        self.primes = primes.to(device="cpu", dtype=torch.int64)
        self.offsets = offsets.to(device="cpu", dtype=torch.int64)
        self.multipliers = multipliers.to(device="cpu", dtype=torch.int64)
        self.pad_id = int(self.token_map[pad_token_id])
        self.max_ngram_size = multipliers.shape[-1]
        self.history = None
        self.next_pos = 0

    @torch.inference_mode()
    def __call__(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device="cpu", dtype=torch.int64)
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Expected a non-empty [batch, sequence] token tensor")
        if torch.any(input_ids < 0) or torch.any(input_ids >= len(self.token_map)):
            raise ValueError("Token id outside compressed token map")
        batch, length = input_ids.shape
        history_length = self.max_ngram_size - 1
        if start_pos == 0:
            self.history = torch.full((batch, history_length), self.DEAD)
            self.next_pos = 0
        if start_pos != self.next_pos or self.history is None:
            raise ValueError("Hash calls must have contiguous start_pos")
        if self.history.shape[0] != batch:
            raise ValueError("Batch size changed during a sequence")
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            if token_mask.shape != input_ids.shape:
                raise ValueError("Token mask shape mismatch")
            compressed = compressed.masked_fill(~token_mask.cpu().bool(), self.DEAD)
        context = torch.cat((self.history, compressed), dim=1)
        tokens = []
        blocked = torch.zeros_like(compressed, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            begin = history_length - shift
            source = context[:, begin : begin + length]
            blocked |= source == self.DEAD
            tokens.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(tokens, dim=-1).unsqueeze(2) * self.multipliers
        heads = self.primes.shape[-1] // history_length
        primes = self.primes.unflatten(-1, (history_length, heads))
        rolling, hashes = products[..., 0], []
        for shift in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling.unsqueeze(-1) % primes[:, shift - 1])
        self.history = context[:, -history_length:].clone()
        self.next_pos += length
        return torch.cat(hashes, dim=-1) + self.offsets


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()


def route(
    x: torch.Tensor,
    weight: torch.Tensor,
    selection_bias: torch.Tensor,
    topk: int = 6,
    scale: float = 1.5,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softplus(F.linear(x.float(), weight.float()) / temperature).sqrt()
    indices = (scores + selection_bias.float()).topk(topk, dim=-1).indices
    weights = scores.gather(-1, indices)
    if topk > 1:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return weights * scale, indices


def engram_update(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    eps: float = 1e-20,
) -> torch.Tensor:
    """FP32 signed-square-root gate, normalized separately for each HC copy."""
    copies, dim = x.shape[-2:]
    key, value = kv.float().split((copies * dim, dim), dim=-1)
    key = key.unflatten(-1, (copies, dim))
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps)
    rstd *= torch.rsqrt(key.square().mean(-1) + eps)
    weight = q_weight.float() * k_weight.float()
    dot = (h * weight * key).sum(-1)
    dot = dot * rstd * dim**-0.5
    signed_root = torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)
    gate = torch.sigmoid(signed_root)
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return h + gate.unsqueeze(-1) * value.unsqueeze(-2)
