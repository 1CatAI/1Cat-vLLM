# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch


@torch.compile
def l2norm_compiled(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return (x * inv_norm).to(x.dtype)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    assert dim == -1
    assert x.stride(-1) == 1
    raw_shape = x.shape
    x = x.view((-1, raw_shape[-1]))
    torch._dynamo.mark_dynamic(x, 0)
    y = l2norm_compiled(x, dim, eps)
    y = y.view(raw_shape)
    return y
