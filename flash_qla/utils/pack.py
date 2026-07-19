# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch


def unpack(
    x: torch.Tensor,  # [B, T, H]
    cu_seqlens: torch.Tensor,
):
    assert x.shape[0] == 1
    assert len(cu_seqlens.shape) == 1
    max_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    batch_size = cu_seqlens.shape[0] - 1
    y = torch.zeros((batch_size, max_len, *x.shape[2:]), dtype=x.dtype, device=x.device)
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        y[i, : end - start] = x[0, start:end]
    return y


def pack(
    x: torch.Tensor,  # [B, T, H]
    cu_seqlens: torch.Tensor,
):
    assert len(cu_seqlens.shape) == 1
    sum_len = cu_seqlens[-1].item()
    batch_size = cu_seqlens.shape[0] - 1
    y = torch.empty((1, sum_len, *x.shape[2:]), dtype=x.dtype, device=x.device)
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        y[0, start:end] = x[i, : end - start]
    return y


def pad_and_reshape(
    x: torch.Tensor,
    dim: int,
    chunk_size: int = 64,
):
    sequence_length = x.shape[dim]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    zeros = [
        0,
    ] * (2 * (len(x.shape) - 1 - dim))
    padded = torch.nn.functional.pad(x, (*zeros, 0, pad_size))
    return padded.reshape((*x.shape[:dim], -1, chunk_size, *x.shape[dim + 1 :]))


def fill_last_chunk_of_g(
    g: torch.Tensor,
    num_tokens: int,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 64,
    reverse: bool = False,
):
    if cu_seqlens is None:
        last_chunk_size = num_tokens % chunk_size
        if last_chunk_size > 0:
            if reverse:
                g[:, -1, last_chunk_size - 1] += g[:, -1, -1]
            else:
                g[:, -1, last_chunk_size:] = g[
                    :, -1, last_chunk_size - 1 : last_chunk_size
                ]
    else:
        for i in range(cu_seqlens.shape[0] - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            last_chunk_idx = (end - start) // chunk_size
            last_chunk_size = (end - start) % chunk_size
            if last_chunk_size > 0:
                if reverse:
                    g[i, last_chunk_idx, last_chunk_size - 1] += g[
                        i, last_chunk_idx, -1
                    ]
                else:
                    g[i, last_chunk_idx, last_chunk_size:] = g[
                        i, last_chunk_idx, last_chunk_size - 1 : last_chunk_size
                    ]
    return g
