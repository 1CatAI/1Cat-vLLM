# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Late K/V scaling in the existing dense v37 QK/softmax/PV algorithm.

The ordinary attention gate is fixed at atol=0.004, rtol=0.015. This checks
arithmetic after codec encoding, separately from reconstruction and model quality.
"""

import pytest
import torch

from tests.kernels.attention.test_sm70_kv_engine_bindings import CODECS, cache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.fixture
def native():
    from vllm.v1.attention.backends.flash_v100 import ops

    ops.get_sm70_d256_gqa_architecture_op()
    namespace = torch.ops._vllm_fa2_C
    assert namespace.sm70_d256_gqa_v37_late_scale_abi() == 1
    return namespace


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize(
    "q_len,kv_len", [(64, 96), (384, 3136), (64, 16512), (1600, 8192), (8192, 8224)]
)
def test_dense_late_scales_after_noncontiguous_page_expansion(
    native, codec, q_len, kv_len
):
    from flash_attn_v100 import paged_kv_to_fp16

    torch.manual_seed(944)
    page, heads = 32, 1
    pages = (kv_len + page - 1) // page
    _, view = cache(codec, pages=pages + 2, page=page, heads=heads)
    table = (torch.randperm(pages, device="cuda") + 2).int()[None]
    length = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
    k = torch.empty((pages, page, heads, 256), device="cuda", dtype=torch.float16)
    v = torch.empty_like(k)
    scale_storage = torch.empty((pages, page, heads, 4), device="cuda")
    ks, vs = scale_storage[..., 0], scale_storage[..., 2]
    paged_kv_to_fp16(
        view.key,
        view.value,
        table,
        length,
        k,
        v,
        0.5,
        2.0,
        kv_cache_dtype=codec.name,
        **view.scale_kwargs,
        key_scale_out=ks,
        value_scale_out=vs,
    )
    k, v = k.flatten(0, 1)[None], v.flatten(0, 1)[None]
    ks, vs = ks.flatten(0, 1)[None], vs.flatten(0, 1)[None]
    q = torch.randn((1, q_len, 6, 256), device="cuda", dtype=torch.float16)
    out = torch.full_like(q, float("nan"))
    native.sm70_d256_gqa_v37_fwd(q, k, v, out, 0.0625, True, ks, vs)
    rows = (
        torch.arange(q_len, device="cuda")
        if q_len <= 384
        else torch.tensor(sorted({0, 1, q_len // 2, q_len - 1}), device="cuda")
    )
    # K scaling belongs after QK; V scaling belongs in the weighted sum.
    scores = q[0, rows].permute(1, 0, 2).double() @ k[0, :, 0].double().T
    scores *= ks[0, :, 0].double()[None, None] * 0.0625
    scores.masked_fill_(
        torch.arange(kv_len, device="cuda")[None, None]
        > (kv_len - q_len + rows)[None, :, None],
        -torch.inf,
    )
    expected = (
        (scores.softmax(-1) * vs[0, :, 0].double()[None, None]) @ v[0, :, 0].double()
    ).permute(1, 0, 2)
    torch.testing.assert_close(out[0, rows].double(), expected, atol=0.004, rtol=0.015)
    assert torch.isfinite(out).all()


def test_dense_scale_binding_rejects_incomplete_or_wrong_layout(native):
    q = torch.empty((1, 64, 6, 256), device="cuda", dtype=torch.float16)
    k = torch.empty((1, 96, 1, 256), device="cuda", dtype=torch.float16)
    scales = torch.ones((1, 96, 1), device="cuda")
    with pytest.raises(RuntimeError, match="both scale planes"):
        native.sm70_d256_gqa_v37_fwd(q, k, k, q, 0.0625, True, scales, None)
    with pytest.raises(RuntimeError, match="scale shape"):
        native.sm70_d256_gqa_v37_fwd(q, k, k, q, 0.0625, True, scales[:, :64], scales)
