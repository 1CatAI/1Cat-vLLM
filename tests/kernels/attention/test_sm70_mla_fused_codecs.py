# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused compression and Q/RoPE writers publish the common cache contract."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import mixed_mla_cache_view
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.sm70.qnorm_rope_kv_fp8_insert import (
    sm70_qnorm_rope_kv_insert,
)
from vllm.v1.attention.kv_cache_view import KVCacheView, MLAPageLayout
from vllm.v1.attention.kv_codecs import kv_codecs_for_representation

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Fused grouped cache qualification on SM70",
)
CODECS = (
    *kv_codecs_for_representation("mla_mixed"),
    *kv_codecs_for_representation("indexer"),
)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: f"{c.representation}-{c.name}")
@pytest.mark.parametrize("ratio", [4, 128])
def test_fused_compression_publishes_shared_payload_scale_and_rope(codec, ratio):
    head = 512 if codec.representation == "mla_mixed" else 128
    page = 64
    slot_bytes = (
        MLAPageLayout(448, 64, scale_bytes=codec.scale_dtype.itemsize).slot_bytes
        if head == 512
        else 132
    )
    stride = page * slot_bytes + 576
    storage = torch.full((4 * stride,), 47, dtype=codec.storage_dtype, device="cuda")
    carrier = storage.as_strided((4, page, slot_bytes), (stride, slot_bytes, 1))
    view = (
        mixed_mla_cache_view(carrier, codec.name)
        if head == 512
        else KVCacheView.bind_indexer(carrier, codec, 128)
    )
    # Each window is constant, avoiding reduction-order ambiguity in the exact
    # storage comparison; independent group amplitudes exercise scale addressing.
    vector = (
        torch.arange(1, head // 64 + 1, device="cuda").repeat_interleave(64).float()
    )
    vector[1::2] *= -1
    state_page = 2 * ratio
    state = torch.zeros((2, state_page, 2 * head), device="cuda")
    state[..., :head] = vector
    table = torch.tensor([[1], [0]], dtype=torch.int32, device="cuda")
    requests = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")
    positions = torch.tensor(
        [ratio - 1, ratio - 1, 2 * ratio - 1, 2 * ratio - 1], device="cuda"
    )
    state_slots = torch.zeros(4, dtype=torch.int64, device="cuda")
    output_slots = torch.tensor([page - 1, page, -1, 4 * page], device="cuda")
    rope = torch.cat(
        (
            torch.ones((2 * ratio, 32), device="cuda"),
            torch.zeros((2 * ratio, 32), device="cuda"),
        ),
        dim=-1,
    )
    current = torch.zeros((4, head), device="cuda")
    ape = torch.zeros((ratio, head), device="cuda")
    weights = torch.ones(head, device="cuda")

    def launch():
        compress_norm_rope_store_triton(
            state_cache=state,
            current_kv=current,
            current_score=current,
            ape=ape,
            num_actual=4,
            token_to_req_indices=requests,
            positions=positions,
            slot_mapping=state_slots,
            block_table=table,
            block_size=state_page,
            state_width=head,
            cos_sin_cache=rope,
            kv_cache=carrier,
            k_cache_metadata=SimpleNamespace(slot_mapping=output_slots),
            pdl_kwargs={},
            head_dim=head,
            rope_head_dim=64,
            compress_ratio=ratio,
            overlap=False,
            use_fp4_cache=False,
            rms_norm_weight=weights,
            rms_norm_eps=1e-6,
            quant_block=64 if head == 512 else 128,
            token_stride=576 if head == 512 else 128,
            scale_dim=8 if head == 512 else 4,
            cache_dtype=codec.name,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for repeat in range(2):
        state[..., :head].mul_(0.9)
        output_slots.copy_(output_slots.flip(0))
        before = storage.clone()
        graph.replay()
        normalized = (
            (
                state[0, 0, :head]
                * torch.rsqrt(state[0, 0, :head].square().mean() + 1e-6)
            )
            .bfloat16()
            .float()
        )
        latent = normalized[:448] if head == 512 else normalized
        if codec.native_id == 3:
            raw, scales = codec.encode(latent)
        else:
            groups = latent.reshape(-1, codec.group_size)
            exponent = torch.ceil(
                torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448)
            )
            scale = torch.exp2(exponent)
            raw = (
                (groups / scale[:, None])
                .clamp(-448, 448)
                .to(codec.element_dtype)
                .view(codec.storage_dtype)
                .reshape(-1)
            )
            scales = (exponent + 127).byte() if head == 512 else scale
        touched = set()
        for slot in output_slots.cpu().tolist():
            if not 0 <= slot < 4 * page:
                continue
            block, offset = divmod(slot, page)
            touched.add(slot)
            assert torch.equal(view.key[block, offset, 0], raw)
            assert torch.equal(view.key_scale[block, offset], scales)
            if head == 512:
                assert torch.equal(
                    view.unquantized_field("rope")[block, offset, 0],
                    normalized[448:].bfloat16(),
                )
        old_carrier = before.as_strided(carrier.shape, carrier.stride())
        old = (
            mixed_mla_cache_view(old_carrier, codec.name)
            if head == 512
            else KVCacheView.bind_indexer(old_carrier, codec, 128)
        )
        untouched = torch.ones((4, page), dtype=torch.bool, device="cuda")
        untouched.reshape(-1)[list(touched)] = False
        assert torch.equal(view.key[untouched], old.key[untouched])
        assert torch.equal(view.key_scale[untouched], old.key_scale[untouched])
        if head == 512:
            assert torch.equal(
                view.unquantized_field("rope")[untouched],
                old.unquantized_field("rope")[untouched],
            )
        for block in range(4):
            start = block * stride + view.page_size_bytes
            assert torch.equal(
                storage[start : (block + 1) * stride],
                before[start : (block + 1) * stride],
            )


@pytest.mark.parametrize(
    "codec", kv_codecs_for_representation("mla_mixed"), ids=lambda c: c.name
)
def test_fused_qnorm_rope_publishes_same_cache_as_shared_writer(codec, monkeypatch):
    torch.manual_seed(20261012)
    page = 64
    layout = MLAPageLayout(448, 64, scale_bytes=codec.scale_dtype.itemsize)
    query = torch.randn((1, 16, 512), dtype=torch.float16, device="cuda")
    values = torch.randn((1, 512), dtype=torch.float16, device="cuda")
    positions = torch.tensor([35], device="cuda")
    slots = torch.tensor([page + 3], device="cuda")
    angle = torch.randn((64, 32), device="cuda")
    rope = torch.cat((angle.cos(), angle.sin()), dim=-1)
    results = []
    for fused in (False, True):
        monkeypatch.setenv("VLLM_SM70_DSV4_QNORM_KV_FUSED_TP4", "1" if fused else "0")
        storage = torch.full(
            (3, page, layout.slot_bytes), 43, dtype=codec.storage_dtype, device="cuda"
        )
        out = query.clone()
        sm70_qnorm_rope_kv_insert(
            out,
            values,
            storage,
            slots,
            positions,
            rope,
            1e-6,
            page,
            kv_cache_dtype=codec.name,
        )
        results.append((out, storage))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
