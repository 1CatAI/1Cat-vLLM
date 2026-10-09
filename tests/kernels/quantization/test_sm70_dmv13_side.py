# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0)
    or not hasattr(torch.ops._C, "sm70_dmv13_out"),
    reason="requires SM70 and sm70_dmv13_out",
)

BLOCK = {12: 144, 14: 210}


def _raw(rng, rows, k, qtype):
    size = BLOCK[qtype]
    raw = rng.integers(0, 256, (rows, k // 256, size), dtype=np.uint8)
    d = np.full((rows, k // 256), 0.002, np.float16).view(np.uint8)
    if qtype == 12:
        raw[..., 0:2] = d.reshape(rows, k // 256, 2)
        raw[..., 2:4] = d.reshape(rows, k // 256, 2)
    else:
        raw[..., 208:210] = d.reshape(rows, k // 256, 2)
    return raw.reshape(rows, -1)


class _Shard:
    def __init__(self, raw, qtype):
        self.qweight = torch.from_numpy(raw)
        self.qweight_type = type("T", (), {"weight_type": qtype})()
        self.prefix = f"test.{qtype}.{raw.shape[0]}"


class _Extra:
    def __init__(self, weight):
        self.weight = weight


class _MergedBf16Extra:
    def __init__(self, weight):
        pieces = [part.contiguous().view(torch.uint8) for part in weight.chunk(2)]
        self.qweight = pieces[0]
        self.qweight.data_container = pieces
        self.qweight.shard_id = [0, 1]
        self.qweight.shard_id_map = {0: 0, 1: 1}
        self.qweight_type = type("T", (), {"shard_weight_type": {0: 30, 1: 30}})()


@pytest.mark.parametrize("qtype", [12, 14])
@pytest.mark.parametrize("tokens", [1, 5, 8])
@pytest.mark.parametrize("merged_bf16", [False, True])
def test_side_projection_matches_dequant(qtype, tokens, merged_bf16):
    from vllm.model_executor.layers.quantization.sm70_dmv13_projection import (
        _PROJECTIONS,
        Dmv13Projection,
    )

    rng = np.random.default_rng(qtype + tokens)
    n, k, extra_n = 256, 2560, 24
    raw = _raw(rng, n, k, qtype)
    extra = (torch.randn(extra_n, k, device="cuda") * 0.02).half()
    if merged_bf16:
        extra = extra.bfloat16()
    side = _MergedBf16Extra(extra) if merged_bf16 else _Extra(extra)
    proj = Dmv13Projection(_Shard(raw, qtype), side)
    assert proj.ready
    _PROJECTIONS[proj.name] = proj
    x = (torch.randn(tokens, k, device="cuda") * 0.5).half()
    out, extra_out = proj(x)
    w = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(qtype))
    ref = x.float().cpu() @ torch.from_numpy(w).float().T
    err = (out.float().cpu() - ref).norm() / ref.norm()
    assert float(err) < 2e-3
    torch.testing.assert_close(
        extra_out.float(), (x.float() @ extra.half().float().T), atol=2e-2, rtol=2e-2
    )
