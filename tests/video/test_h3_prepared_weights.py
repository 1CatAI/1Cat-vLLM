# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import json

import pytest
import torch
from torch import nn

from vllm.model_executor.models.minimax_h3.prepared_weights import (
    PreparedWeights,
    checkpoint_identity,
)
from vllm.model_executor.models.minimax_h3.quantization import FP16LinearMethod
from vllm.model_executor.models.minimax_h3.residency import PinnedModuleStager


def model():
    module = nn.Module()
    module.register_parameter(
        "weight", nn.Parameter(torch.empty_strided((3, 4), (1, 3)))
    )
    module.register_buffer("scale", torch.ones(3))
    return module


def build(root, key="a" * 64, *, limit=4096):
    module = model()
    cache = PreparedWeights(root, key, module, limit_bytes=limit, reserve_bytes=0)
    assert not cache.restore(module)
    PinnedModuleStager.map_cpu_weights(module, cache, preserve_parameters=False)
    with torch.no_grad():
        module.weight.copy_(torch.arange(12).reshape(3, 4))
    cache.publish(module)
    return module, cache


def test_reuses_final_strides_and_protects_cache_from_adapter_writes(tmp_path):
    module, cache = build(tmp_path)
    expected = module.weight.detach().clone()
    assert module.weight.stride() == (1, 3)
    with torch.no_grad():
        module.weight.add_(100)
    cache.close()
    restored = model()
    reader = PreparedWeights(tmp_path, "a" * 64, restored, reserve_bytes=0)
    assert reader.restore(restored)
    torch.testing.assert_close(restored.weight, expected, rtol=0, atol=0)
    assert restored.weight.stride() == (1, 3)
    pointer = restored.weight.data_ptr()
    PinnedModuleStager.map_cpu_weights(restored, reader)
    assert restored.weight.data_ptr() == pointer
    assert not list(reader.fallback.directory.iterdir())
    reader.close()


@pytest.mark.parametrize(
    "damage", ["data", "metadata", "missing", "truncated", "unfinished"]
)
def test_invalid_entries_rebuild_without_partially_binding(tmp_path, damage):
    _, cache = build(tmp_path)
    cache.close()
    manifest = json.loads((cache.entry / "manifest.json").read_bytes())
    path = cache.entry / manifest["groups"][0]["file"]
    if damage == "data":
        data = bytearray(path.read_bytes())
        data[0] ^= 0x80
        path.write_bytes(data)
    elif damage == "metadata":
        (cache.entry / "manifest.json").write_text("{}")
    elif damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(b"")
    else:
        (cache.entry / "ready.json").unlink()
    target = model()
    target.weight.data.fill_(7)
    before = target.weight.clone()
    reader = PreparedWeights(tmp_path, "a" * 64, target, reserve_bytes=0)
    assert not reader.restore(target)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)
    assert not (reader.entry / "ready.json").exists()
    reader.close()


def test_active_cache_is_not_evicted_and_inactive_cache_can_be_reclaimed(tmp_path):
    _, first = build(tmp_path, limit=80)
    target = model()
    second = PreparedWeights(
        tmp_path, "b" * 64, target, limit_bytes=80, reserve_bytes=0
    )
    assert not second.restore(target)
    with pytest.raises(RuntimeError, match="active caches"):
        PinnedModuleStager.map_cpu_weights(target, second, preserve_parameters=False)
    assert (first.entry / "ready.json").exists()
    first.close()
    PinnedModuleStager.map_cpu_weights(target, second, preserve_parameters=False)
    assert not first.entry.exists()
    second.close()


def test_checkpoint_replacement_changes_identity(tmp_path):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"original")
    before = checkpoint_identity([path])
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"original")
    replacement.replace(path)
    assert checkpoint_identity([path]) != before


def test_corrupt_binding_cannot_escape_storage_even_with_valid_manifest_hash(tmp_path):
    _, cache = build(tmp_path)
    cache.close()
    path = cache.entry / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["groups"][0]["bindings"][0]["offset"] = 2**60
    data = json.dumps(manifest).encode()
    path.write_bytes(data)
    (cache.entry / "ready.json").write_text(
        json.dumps({"sha256": hashlib.sha256(data).hexdigest()})
    )
    reader = PreparedWeights(tmp_path, "a" * 64, model(), reserve_bytes=0)
    assert not reader.restore(model())
    reader.close()


def test_final_layout_can_be_filled_directly_without_second_allocation():
    module = nn.Linear(7, 5, bias=False, dtype=torch.float16)
    module.h3_fp16_weight_layout = "column"
    method = FP16LinearMethod()
    method.prepare_weights_before_loading(module)
    source = torch.arange(35, dtype=torch.bfloat16).reshape(5, 7)
    with torch.no_grad():
        module.weight.copy_(source)
    pointer = module.weight.data_ptr()
    method.process_weights_after_loading(module)
    assert module.weight.data_ptr() == pointer
    assert module.weight.stride() == (1, 5)
    torch.testing.assert_close(module.weight, source.to(torch.float16), rtol=0, atol=0)


def test_identity_separates_rank_topology_and_processing_recipe(tmp_path):
    from vllm.model_executor.models.minimax_h3.prepared_weights import preparation_key

    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"test checkpoint identity")
    kwargs = dict(
        component="transformer", rank=0, world_size=4, options={"layout": "column"}
    )
    baseline = preparation_key([path], **kwargs)
    assert baseline == preparation_key([path], **kwargs)
    for change in (
        {"rank": 1},
        {"world_size": 2},
        {"component": "text_encoder"},
        {"options": {"layout": "row"}},
    ):
        assert preparation_key([path], **(kwargs | change)) != baseline


def test_mixed_precision_shared_storage_and_offset_survive_restore(tmp_path):
    def mixed():
        module = nn.Module()
        raw = torch.arange(24, dtype=torch.float32)
        module.register_buffer("first", raw[:12].reshape(3, 4))
        module.register_buffer("second", raw[12:].reshape(3, 4))
        module.register_buffer("quantized", torch.arange(-12, 12, dtype=torch.int8))
        module.register_buffer("half_values", torch.arange(7, dtype=torch.float16))
        return module

    source = mixed()
    cache = PreparedWeights(tmp_path, "c" * 64, source, reserve_bytes=0)
    assert not cache.restore(source)
    cache.publish(source)
    cache.close()
    target = mixed()
    reader = PreparedWeights(tmp_path, "c" * 64, target, reserve_bytes=0)
    assert reader.restore(target)
    for name, tensor in source.named_buffers():
        torch.testing.assert_close(getattr(target, name), tensor, rtol=0, atol=0)
    assert (
        target.first.untyped_storage().data_ptr()
        == target.second.untyped_storage().data_ptr()
    )
    assert target.second.storage_offset() == 12
    reader.close()


def test_int8_final_layout_preserves_signed_bytes_and_scale_without_copy():
    from types import SimpleNamespace

    from vllm.model_executor.models.minimax_h3.quantization import (
        Int8ConvRotLinearMethod,
    )

    module = nn.Module()
    module.register_parameter(
        "weight", nn.Parameter(torch.empty(5, 7, dtype=torch.int8), requires_grad=False)
    )
    module.register_parameter("weight_scale", nn.Parameter(torch.ones(5, 1)))
    method = Int8ConvRotLinearMethod(
        SimpleNamespace(weight_layout="column"), None, prefix="blocks.0.proj"
    )
    method.prepare_weights_before_loading(module)
    source = torch.arange(-17, 18, dtype=torch.int8).reshape(5, 7)
    module.weight.copy_(source)
    pointer = module.weight.data_ptr()
    method.process_weights_after_loading(module)
    assert module.weight.data_ptr() == pointer
    assert module.weight.stride() == (1, 5)
    assert torch.equal(module.weight, source)
    assert torch.equal(module.weight_scale, torch.ones(5))
