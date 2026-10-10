# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_original_embedding_adapter_preserves_packed_rows():
    from vllm.model_executor.model_loader.gguf_adapters.qwen4exp import Qwen4ExpAdapter

    adapter = Qwen4ExpAdapter(
        SimpleNamespace(
            num_hidden_layers=0,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
        )
    )
    adapter.packed_embeddings = True
    data = np.arange(4 * 136, dtype=np.uint8).reshape(4, 136)
    tensor = SimpleNamespace(shape=np.array([256, 4]), tensor_type=23, data=data)
    tensors = {"token_embd.weight": tensor}
    entries = dict(
        adapter.weights(tensors, adapter.build_name_map(tensors), torch.float16)
    )
    assert entries["model.embed_tokens.qweight_type"].item() == 23
    packed = entries["model.embed_tokens.qweight"]
    assert packed.dtype == torch.uint8
    torch.testing.assert_close(packed, torch.from_numpy(data), rtol=0, atol=0)


def test_hc_shards_recover_exact_checkpoint_matrices():
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import (
        _pack_hc_batch_weight,
        _unpack_hc_storage,
    )

    generator = torch.Generator().manual_seed(1020)
    down = torch.randn((336, 10240), generator=generator).half()
    up = torch.randn((10240, 320), generator=generator).half()
    rows, columns = [], []
    for rank in range(4):
        packed_down = _pack_hc_batch_weight(down, "down", rank)
        packed_up = _pack_hc_batch_weight(up, "up", rank)
        local_down, local_up = _unpack_hc_storage(packed_down, packed_up)
        torch.testing.assert_close(
            local_down, down[rank * 80 : rank * 80 + 88], rtol=0, atol=0
        )
        torch.testing.assert_close(
            local_up.reshape(4, 640, 320),
            up.reshape(4, 2560, 320)[:, rank * 640 : (rank + 1) * 640],
            rtol=0,
            atol=0,
        )
        rows.append(local_down[:80])
        columns.append(local_up.reshape(4, 640, 320))
        assert (
            packed_down.numel() + packed_up.numel() < (down.numel() + up.numel()) * 0.28
        )
    torch.testing.assert_close(torch.cat(rows), down[:320], rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat(columns, dim=1).reshape(10240, 320), up, rtol=0, atol=0
    )


def test_hcx_shards_recover_checkpoint_without_replicated_storage():
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import _unpack_hc_storage
    from vllm.models.qwen4_exp.nvidia.sm70_hcx import pack_down, pack_up

    generator = torch.Generator().manual_seed(1165)
    down = torch.randn((336, 10240), generator=generator).half()
    up = torch.randn((10240, 320), generator=generator).half()
    for rank in range(4):
        local_down, local_up = _unpack_hc_storage(
            pack_down(down, rank), pack_up(up, rank)
        )
        torch.testing.assert_close(
            local_down[:80], down[rank * 80 : (rank + 1) * 80], rtol=0, atol=0
        )
        torch.testing.assert_close(
            local_up.reshape(4, 640, 320),
            up.reshape(4, 2560, 320)[:, rank * 640 : (rank + 1) * 640],
            rtol=0,
            atol=0,
        )
        if rank == 3:
            torch.testing.assert_close(local_down[80:84], down[320:324], rtol=0, atol=0)
        else:
            assert not local_down[80:].count_nonzero()


@pytest.mark.parametrize("m", [1, 5, 8, 20])
def test_original_experts_keep_fast_route_for_graph_padding(monkeypatch, m):
    from vllm.model_executor.layers.fused_moe import MoEActivation
    from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod

    method = object.__new__(GGUFNativeMoEMethod)
    method.dp4a_admitted = True
    method.weight_types = {"w1": 21, "w3": 21, "w2": 42}
    method.input_padding = {"w2": (32, 192)}
    method.q8_intermediate = True
    bank = torch.empty(0, dtype=torch.uint8)
    layer = SimpleNamespace(
        apply_router_weight_on_input=False,
        activation=MoEActivation.SILU,
        expert_map=None,
        gguf_w1=bank,
        gguf_w3=bank,
        gguf_w2=bank,
    )
    calls = []

    def native(
        x, ids, probabilities, gate, up, down, source, down_type, left, q8, bank_aware
    ):
        calls.append((x.shape[0], source, down_type, left, q8, bank_aware))
        return x

    monkeypatch.setattr(torch.ops.vllm, "gguf_original_expert_dp4a", native)
    x = torch.zeros((m, 2560), dtype=torch.float16)
    result = method.apply(
        layer,
        x,
        torch.ones((m, 10)),
        torch.zeros((m, 10), dtype=torch.int32),
        None,
        None,
    )
    assert result is x
    assert calls == [(m, 21, 42, 32, True, True)]
