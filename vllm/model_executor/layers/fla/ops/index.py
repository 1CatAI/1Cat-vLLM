# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

from vllm.triton_utils import tl, triton

from .utils import tensor_cache

# Fixed inner-tile width for the fused chunk-index kernel below. It is a compile-time constant so
# the kernel compiles EXACTLY ONCE regardless of sequence lengths (the data-dependent inner loop
# covers sequences whose chunk count exceeds BLOCK, e.g. a single very long prefill). A per-shape
# constexpr block would instead force a fresh Triton compile per distinct max-chunk-count.
_CHUNK_INDICES_BLOCK = 256


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@triton.jit
def _build_chunk_indices_kernel(offs_ptr, out_ptr, BLOCK: tl.constexpr):
    # One program per sequence. Program i reads its chunk-run [start, end) from the chunk_offsets
    # vector (the exclusive prefix-sum of per-sequence chunk counts, == prepare_chunk_offsets), and
    # writes, for every global chunk row t in [start, end) with local index k = t - start:
    #     out[t, 0] = i   (sequence id)      out[t, 1] = k   (local chunk index)
    # This is exactly the table prepare_chunk_indices builds with its Python loop + torch.cat, but
    # in a single kernel launch. The `for base in range(0, n_i, BLOCK)` loop lets the fixed BLOCK
    # cover an arbitrarily long sequence, keeping BLOCK compile-time-constant (one compile).
    i = tl.program_id(0)
    start = tl.load(offs_ptr + i)
    end = tl.load(offs_ptr + i + 1)
    n_i = end - start
    for base in range(0, n_i, BLOCK):
        k = base + tl.arange(0, BLOCK)
        mask = k < n_i
        t = start + k
        tl.store(out_ptr + t * 2, (i + k * 0).to(tl.int32), mask=mask)
        tl.store(out_ptr + t * 2 + 1, k.to(tl.int32), mask=mask)


def prepare_chunk_indices_fused(
    chunk_offsets_cpu: torch.Tensor, chunk_offsets_dev: torch.Tensor
) -> torch.Tensor:
    """Build the [NT, 2] chunk-index table in ONE Triton launch.

    Bit-identical to ``prepare_chunk_indices(cu_seqlens, chunk_size).to(device)`` but replaces the
    per-sequence Python loop + ``torch.cat`` + host->device copy with a single kernel launch that
    reuses the already-on-device ``chunk_offsets`` vector as its input.

    Args:
        chunk_offsets_cpu: the CPU int32 output of ``prepare_chunk_offsets`` — used only for the
            host-side grid size and ``NT`` (``int(chunk_offsets_cpu[-1])``), so there is **no
            device->host sync**.
        chunk_offsets_dev: that same vector already copied to the target device (the GDN metadata
            builder copies it there today regardless), reused directly as the kernel input.

    Returns:
        ``chunk_indices`` [NT, 2] int32 on ``chunk_offsets_dev.device``.
    """
    num_seqs = chunk_offsets_cpu.numel() - 1
    nt = int(chunk_offsets_cpu[-1])  # host int, no device sync (CPU tensor)
    out = torch.empty(nt, 2, dtype=torch.int32, device=chunk_offsets_dev.device)
    _build_chunk_indices_kernel[(num_seqs,)](
        chunk_offsets_dev, out, BLOCK=_CHUNK_INDICES_BLOCK
    )
    return out


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
