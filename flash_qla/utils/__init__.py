# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from .pack import pad_and_reshape, pack, unpack, fill_last_chunk_of_g
from .math import l2norm

try:
    from .profiler import profile
    from .index import prepare_chunk_indices, prepare_chunk_offsets, tensor_cache
except ImportError:

    def _tilelang_unavailable(*args, **kwargs):
        raise RuntimeError("This FlashQLA utility requires tilelang.")

    profile = _tilelang_unavailable
    prepare_chunk_indices = _tilelang_unavailable
    prepare_chunk_offsets = _tilelang_unavailable
    tensor_cache = _tilelang_unavailable


__all__ = [
    "profile",
    "pad_and_reshape",
    "pack",
    "unpack",
    "fill_last_chunk_of_g",
    "l2norm",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "tensor_cache",
]
