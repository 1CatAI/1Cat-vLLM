# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.video.metrics import DenoiseWorkCounter, lora_work


@pytest.mark.parametrize("rows", [1, 97, 34551])
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_column_lora_a_is_counted_once_across_ranks_including_tails(rows, tp):
    useful = redundant = 0
    for rank in range(tp):
        layer = SimpleNamespace(
            tp_size=tp,
            tp_rank=rank,
            h3_lora_a_0=torch.empty(3, 7),
            h3_lora_b_0=torch.empty(8 // tp, 3),
        )
        work, repeated = lora_work(layer, [(0, 0, 8 // tp)], rows, replicated_a=True)
        useful += work
        redundant += repeated
    # The logical adapter is one 7->3->8 projection regardless of TP size.
    assert useful == 2 * rows * (7 * 3 + 3 * 8)
    assert redundant == 2 * rows * (7 * 3) * (tp - 1)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_row_lora_partial_products_are_distinct_work(tp):
    layer = SimpleNamespace(
        tp_size=tp,
        tp_rank=0,
        h3_lora_a_0=torch.empty(3, 8 // tp),
        h3_lora_b_0=torch.empty(11, 3),
    )
    work, redundant = lora_work(layer, [(0, 0, 11)], 13, replicated_a=False)
    assert work == 2 * 13 * (3 * (8 // tp) + 11 * 3)
    assert redundant == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a leased GPU")
@torch.inference_mode()
def test_real_block_work_excludes_padding_and_preserves_outputs(
    dist_init, default_vllm_config
):
    from vllm.model_executor.models.minimax_h3.attention import attention_backend
    from vllm.model_executor.models.minimax_h3.transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTBlock,
    )

    arch = MiniMaxH3DiTArchConfig(
        hidden_size=512,
        num_attention_heads=4,
        ffn_hidden_size=1024,
        adaln_curve_grid=2,
        adaln_out_features=18 * 512,
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = MiniMaxH3DiTBlock(arch, None, prefix="block")

        def forward(self, x, **kwargs):
            return self.block(x, **kwargs)

    token = attention_backend.set("FLASH_ATTN_V100")
    try:
        model = Model().cuda().eval()
    finally:
        attention_backend.reset(token)
    torch.manual_seed(412)
    for name, parameter in model.named_parameters():
        if "norm" in name:
            parameter.fill_(1)
        else:
            parameter.normal_(0, 0.02)
    valid, padded = 33, 64
    x = torch.randn(padded, 512, device="cuda")
    kwargs = dict(
        t_emb=torch.randn(1, 8, device="cuda"),
        combined_indices=torch.arange(padded, device="cuda") % 3,
        rope_table=torch.randn(padded, 96, device="cuda"),
        cu_seqlens=torch.tensor([0, valid], device="cuda", dtype=torch.int32),
        max_seqlen=valid,
        packed_total=padded,
    )
    expected = model(x, **kwargs)
    # Independent geometry: QKV + output + gate/up + down + AdaLN + QK/PV.
    flops = (
        2 * valid * (3 * 512 * 512 + 512 * 512 + 2 * 1024 * 512 + 512 * 1024)
        + 2 * 8 * (18 * 512)
        + 4 * 4 * valid * valid * 128
    )
    with DenoiseWorkCounter(
        model, used_length=valid, video_outputs=valid, audio_outputs=1
    ) as counter:
        for index in range(2):
            with counter.step(index):
                actual = model(x, **kwargs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.accelerator.synchronize()
        steps = counter.finish_steps()
        assert counter.calls == 2
        assert counter.blocks == {"block": 2}
        assert counter.flops == 2 * flops
        assert sum(counter.by_layer.values()) == counter.flops
        assert [step["useful_flops"] for step in steps] == [flops, flops]
        assert all(step["gpu_seconds"] > 0 for step in steps)
    model(x, **kwargs)
    assert counter.calls == 2  # Hooks must not leak into the following request.
