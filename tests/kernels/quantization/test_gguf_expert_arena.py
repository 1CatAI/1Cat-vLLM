# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_expert_storage import (
    original_expert_bank,
    prepare_original_expert_arena,
)
from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
from vllm.model_executor.layers.quantization.gguf_native import (
    empty_guarded_weight,
    native_dense,
    pad_weight_tail,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_size


@pytest.mark.parametrize("rank", range(4))
def test_arena_keeps_q2_boundary_blocks_and_tail(rank):
    shape, tail = original_expert_bank((640, 2560, 512), 42, "w2", 2560, 160, 4, rank)
    assert shape == (512, 2560, 54)
    assert tail == 90


def _arena(device):
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    layer = model.experts
    method = object.__new__(GGUFNativeMoEMethod)
    method.hidden_size, method.intermediate_size, method.num_experts = 2560, 160, 2
    method.guarded_shards = set()
    layer.quant_method, layer.tp_size, layer.tp_rank = method, 4, 1
    mapping, tensors = {}, {}
    for projection, shape, kind in (
        ("gate_proj", (2560, 640, 2), 21),
        ("up_proj", (2560, 640, 2), 21),
        ("down_proj", (640, 2560, 2), 42),
    ):
        raw = projection + ".weight"
        mapping[raw] = "experts." + raw
        tensors[raw] = SimpleNamespace(shape=shape, tensor_type=kind)
    # Match the loader's model.layers.N.mlp.experts suffix.
    root = torch.nn.Module()
    root.model = model
    mapping = {raw: "model." + name for raw, name in mapping.items()}
    prepare_original_expert_arena(root, mapping, tensors, device)
    return layer


def test_arena_views_are_disjoint_and_keep_zero_guards():
    layer = _arena("cpu")
    banks = [layer.gguf_w1, layer.gguf_w3, layer.gguf_w2]
    assert len({bank.untyped_storage().data_ptr() for bank in banks}) == 1
    assert layer.quant_method.guarded_shards == {"w1", "w3", "w2"}
    for bank, kind in zip(banks, (21, 21, 42)):
        assert bank.is_contiguous() and bank.storage_offset() % 256 == 0
        bank.fill_(255)
        assert pad_weight_tail(bank, kind, storage_has_zero_tail=True) is bank
        block, size = quant_size(kind)
        tail = ((-(bank.shape[-1] // size * block)) % 512) // block * size
        backing = torch.empty(0, dtype=torch.uint8).set_(
            bank.untyped_storage(), bank.storage_offset() + bank.numel(), (tail,), (1,)
        )
        assert backing.count_nonzero() == 0
    assert all(bank.min() == 255 for bank in banks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [5, 20])
def test_native_dense_reads_arena_offsets_exactly_and_replays(m):
    import vllm._C_gguf  # noqa: F401

    layer = _arena("cuda:0")
    assert layer.gguf_w3.data_ptr() % 256 == 0
    # Poison the preceding bank: the selected bank starts at a nonzero offset.
    layer.gguf_w1.fill_(255)
    bank = layer.gguf_w3
    kind = 21
    block, size = quant_size(kind)
    payload = torch.randint(0, 256, bank.shape, dtype=torch.uint8, device=bank.device)
    blocks = payload.view(-1, size)
    # Finite small FP16 block scales; leave all codebook/sign bits untouched.
    blocks[:, :2].copy_(torch.tensor([0, 32], dtype=torch.uint8, device=bank.device))
    bank.copy_(payload)
    control = empty_guarded_weight(bank.shape, kind, bank.device)
    control.copy_(bank)
    x = torch.randn(m, 2560, dtype=torch.float16, device=bank.device)
    weights = bank.view(-1, bank.shape[-1])
    control_weights = control.view_as(weights)

    def run(weight):
        return native_dense(x, weight, kind, prefill_min_m=32)

    expected, actual = run(control_weights), run(weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(weights)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(captured, run(control_weights), rtol=0, atol=0)
