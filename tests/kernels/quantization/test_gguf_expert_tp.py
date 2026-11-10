# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.quantization.gguf import GGUFConfig
from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
from vllm.model_executor.layers.quantization.gguf_repack import q2_0_to_q4_1
from vllm.transformers_utils.gguf_tensor_reader import dequantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def weights():
    rng = np.random.default_rng(20261003)
    packed = {}
    for shard, k, rows, value in (
        ("w1", 256, 640, 2),
        ("w3", 256, 640, 8),
        ("w2", 640, 256, 42),
    ):
        block, size = (64, 18) if value == 42 else gguf.GGML_QUANT_SIZES[value]
        data = rng.integers(0, 256, (2, rows * k // block, size), dtype=np.uint8)
        data[:, :, :2] = np.frombuffer(np.float16(0.015625).tobytes(), dtype=np.uint8)
        # Q8_0 stores signed values in the remainder of the block.
        packed[shard] = (value, data.reshape(2, rows, k // block * size))
    return packed


@pytest.mark.parametrize("m", [1, 5, 20, 64])
@pytest.mark.parametrize("original", [False, True])
def test_independent_expert_types_and_q2_conversion_preserve_tp4_ffn(m, original):
    packed = weights()
    decoded = {
        shard: torch.from_numpy(dequantize(data, value)).half().cuda()
        for shard, (value, data) in packed.items()
    }
    torch.manual_seed(20261003 + m)
    x = (torch.randn((m, 256), device="cuda") * 0.125).half()
    ids = (
        torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        .expand(m, 2)
        .contiguous()
    )
    topk = torch.tensor([[0.25, 0.75]], device="cuda").expand(m, 2)
    expected = torch.zeros((m, 256), dtype=torch.float32, device="cuda")
    for expert in range(2):
        gate = (x.float() @ decoded["w1"][expert].float().T).half()
        up = (x.float() @ decoded["w3"][expert].float().T).half()
        hidden = torch.nn.functional.silu(gate) * up
        expected += (hidden.float() @ decoded["w2"][expert].float().T) * topk[
            :, expert, None
        ]
    partials = []
    for rank in range(4):
        layer = torch.nn.Module()
        layer.tp_size, layer.tp_rank, layer.ep_size = 4, rank, 1
        layer.expert_map = None
        layer.apply_router_weight_on_input = False
        layer.activation = MoEActivation.SILU
        method = GGUFNativeMoEMethod(
            GGUFConfig(), SimpleNamespace(dp_size=1, pcp_size=1, ep_size=1)
        )
        with torch.device("cuda"):
            method.create_weights(layer, 2, 256, 160, torch.float16)
        for shard, (value, data) in packed.items():
            if value == 42 and not original:
                value, data = 3, q2_0_to_q4_1(data)
            prefix = "w2" if shard == "w2" else "w13"
            method.load_expert(
                layer,
                getattr(layer, prefix + "_qweight_type"),
                torch.tensor(value),
                shard,
                0,
            )
            for expert in range(2):
                method.load_expert(
                    layer,
                    getattr(layer, prefix + "_qweight"),
                    torch.from_numpy(data[expert]),
                    shard,
                    expert,
                )
        method.process_weights_after_loading(layer)
        partials.append(method.apply(layer, x, topk, ids, None, None))
    actual = sum(p.float() for p in partials)
    relative_error = (actual - expected).norm() / expected.norm()
    assert relative_error.item() < 0.035
    assert torch.isfinite(actual).all()
