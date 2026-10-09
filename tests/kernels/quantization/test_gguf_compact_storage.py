# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch


def test_original_embedding_adapter_preserves_packed_rows():
    from vllm.model_executor.model_loader.gguf_adapters.qwen35 import Qwen35Adapter

    adapter = Qwen35Adapter(
        SimpleNamespace(
            num_hidden_layers=0,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
        )
    )
    adapter.packed_token_embeddings = True
    data = np.arange(4 * 136, dtype=np.uint8).reshape(4, 136)
    tensor = SimpleNamespace(shape=np.array([256, 4]), tensor_type=23, data=data)
    tensors = {"token_embd.weight": tensor}
    entries = dict(
        adapter.weights(tensors, adapter.build_name_map(tensors), torch.float16)
    )
    assert entries["model.embed_tokens.qweight_type"].item() == 23
    packed = entries["model.embed_tokens.qweight"]
    assert packed.dtype == torch.uint8
    assert packed.data_ptr() == torch.from_numpy(data).data_ptr()
    torch.testing.assert_close(packed, torch.from_numpy(data), rtol=0, atol=0)


def test_hc_shards_recover_exact_checkpoint_matrices():
    from vllm.models.qwen4_exp.nvidia.sm70_hc_storage import (
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
