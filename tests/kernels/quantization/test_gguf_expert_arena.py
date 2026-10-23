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


def _pipeline_arena(rank, *, wrapped=False, count=7):
    from vllm.model_executor.models.utils import PPMissingLayer

    local = torch.nn.Module()
    local.mlp = torch.nn.Module()
    local.mlp.experts = torch.nn.Module()
    experts = local.mlp.experts
    method = object.__new__(GGUFNativeMoEMethod)
    method.hidden_size = method.intermediate_size = 256
    method.num_experts = 2
    method.guarded_shards = set()
    experts.quant_method, experts.tp_size, experts.tp_rank = method, 1, 0
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList(
        local if index == rank else PPMissingLayer() for index in range(count)
    )
    prefix = "model.layers"
    if wrapped:
        wrapper = torch.nn.Module()
        wrapper.language_model = root
        root = wrapper
        prefix = "language_model." + prefix
    mapping, tensors = {}, {}
    for index in range(count):
        for projection, kind in (("gate_proj", 21), ("up_proj", 21), ("down_proj", 23)):
            raw = f"blk.{index}.{projection}.weight"
            mapping[raw] = f"{prefix}.{index}.mlp.experts.{projection}.weight"
            # An absent stage must be filtered before its tensor is inspected,
            # even if no local tensor handle exists for that checkpoint entry.
            if index == rank:
                tensors[raw] = SimpleNamespace(shape=(256, 256, 2), tensor_type=kind)
    return root, experts, mapping, tensors


@pytest.fixture
def isolated_pp_arena(monkeypatch):
    from vllm.model_executor.layers.quantization import gguf_expert_storage as storage
    from vllm.model_executor.models import utils

    # Each test constructs a distinct model; isolate the existing ID-keyed PP
    # cache so CPython object ID reuse between cases cannot affect ownership.
    monkeypatch.setattr(utils, "_model_to_pp_missing_layer_names", {})
    monkeypatch.setattr(storage, "_expert_cache_config", lambda: None)
    return storage


@pytest.mark.cpu_test
@pytest.mark.parametrize("rank", range(7))
@pytest.mark.parametrize("wrapped", [False, True])
def test_arena_preallocates_and_prefetches_only_local_pp_experts(
    isolated_pp_arena, rank, wrapped
):
    root, layer, mapping, tensors = _pipeline_arena(rank, wrapped=wrapped)
    targets = prepare_original_expert_arena(root, mapping, tensors, "cpu")
    assert set(targets) == set(tensors)
    assert {entry[2] for entry in targets.values()} == {"w1", "w3", "w2"}
    assert all(entry[:2] == (layer, layer.quant_method) for entry in targets.values())
    banks = [layer.gguf_w1, layer.gguf_w3, layer.gguf_w2]
    assert len({bank.untyped_storage().data_ptr() for bank in banks}) == 1
    assert all(
        bank.dtype == torch.uint8 and bank.shape[:2] == (2, 256) for bank in banks
    )
    assert layer.quant_method.guarded_shards == {"w1", "w3", "w2"}


@pytest.mark.cpu_test
def test_arena_pp_layer_prefix_has_component_boundary(isolated_pp_arena):
    root, layer, mapping, tensors = _pipeline_arena(10, count=11)
    targets = prepare_original_expert_arena(root, mapping, tensors, "cpu")
    assert set(targets) == set(tensors)
    assert layer.gguf_w1.shape[0] == 2


@pytest.mark.cpu_test
def test_arena_cache_receives_only_admitted_local_banks(isolated_pp_arena, monkeypatch):
    root, layer, mapping, tensors = _pipeline_arena(3, wrapped=True)
    calls = []
    config = object()
    monkeypatch.setattr(isolated_pp_arena, "_expert_cache_config", lambda: config)
    monkeypatch.setattr(
        isolated_pp_arena,
        "_prepare_expert_cache",
        lambda banks, settings, device: calls.append((banks, settings, device)),
    )
    targets = prepare_original_expert_arena(root, mapping, tensors, "cpu")
    assert set(targets) == set(tensors)
    assert len(calls) == 1
    banks, settings, device = calls[0]
    assert len(banks) == 3 and all(bank[0] is layer for bank in banks)
    assert settings is config and device == "cpu"


@pytest.mark.cpu_test
@pytest.mark.parametrize("stage_missing", [False, True])
def test_arena_skips_an_entire_missing_model(
    isolated_pp_arena, monkeypatch, stage_missing
):
    from vllm.model_executor.models.utils import PPMissingLayer, StageMissingLayer

    root = StageMissingLayer("language_model") if stage_missing else PPMissingLayer()
    monkeypatch.setattr(
        isolated_pp_arena.torch,
        "empty",
        lambda *args, **kwargs: pytest.fail("missing stage allocated an arena"),
    )
    mapping = {"raw": "model.layers.0.mlp.experts.gate_proj.weight"}
    assert prepare_original_expert_arena(root, mapping, {}, "cpu") == {}


@pytest.mark.cpu_test
def test_arena_does_not_hide_invalid_local_expert_path(isolated_pp_arena):
    root, _, mapping, tensors = _pipeline_arena(0)
    mapping["blk.0.gate_proj.weight"] = (
        "model.layers.0.missing.experts.gate_proj.weight"
    )
    with pytest.raises(AttributeError, match="missing"):
        prepare_original_expert_arena(root, mapping, tensors, "cpu")


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
