# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [5, 20, 1025])
def test_packed_iq4_xs_lookup_matches_official_fp16(tokens):
    import gguf

    from vllm.model_executor.layers.quantization.gguf import _apply_gguf_embedding

    rng = np.random.default_rng(1020)
    packed = rng.integers(0, 256, (37, 136), dtype=np.uint8)
    packed[:, :2] = np.array([0.03125], dtype=np.float16).view(np.uint8)
    reference = torch.from_numpy(
        gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.IQ4_XS)
    ).half()
    ids = (torch.arange(tokens, device="cuda") * 7 % 37).reshape(-1, 1)
    actual = _apply_gguf_embedding(
        ids, torch.from_numpy(packed).cuda(), 23, 256, torch.float16
    )
    torch.testing.assert_close(actual.cpu(), reference[ids.cpu()], rtol=0, atol=0)
