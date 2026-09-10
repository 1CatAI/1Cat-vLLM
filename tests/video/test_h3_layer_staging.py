# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded layer ownership, failure cleanup and exact device transitions."""

import argparse

import pytest
import torch
from torch import nn

from vllm.model_executor.models.minimax_h3.config import H3Config, H3InputError
from vllm.model_executor.models.minimax_h3.residency import (
    BoundedAllocatorCache,
    LayerwiseModuleStager,
    MMapHostWeights,
    PinnedModuleStager,
)


class Block(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size, size) / size)
        self.register_buffer("adapter_a", torch.randn(2, size) / size)
        self.register_buffer("adapter_b", torch.randn(size, 2) / size)
        self.fail = False

    def forward(self, x):
        if self.fail:
            raise RuntimeError("block failure")
        return nn.functional.linear(x, self.weight) + 0.25 * nn.functional.linear(
            nn.functional.linear(x, self.adapter_a), self.adapter_b
        )


class Model(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.blocks = nn.ModuleList([Block(size), Block(size)])
        # Aliased storage spanning a block and the outer model stays resident.
        self.register_buffer("shared", self.blocks[0].adapter_a[0])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x + self.shared


def cpu_snapshot(model):
    snapshot = PinnedModuleStager.__new__(PinnedModuleStager)
    snapshot._groups = snapshot._snapshot_groups((model,), pin_memory=False)
    snapshot.loaded = False
    snapshot._device_storages = []
    snapshot.device = torch.device("cpu")
    snapshot.cache_retention = None
    return snapshot


def test_layer_storage_partition_and_rejected_overlap():
    model = Model()
    snapshot = cpu_snapshot(model)
    plan = LayerwiseModuleStager(snapshot, model.blocks)
    groups = [g for s in (*plan.stagers, plan.resident) for g in s._groups]
    assert len(groups) == len({id(g) for g in groups}) == len(snapshot._groups)
    shared = [g for g in plan.resident._groups if len(g.bindings) == 2]
    assert len(shared) == 1
    assert {id(b.target) for b in shared[0].bindings} == {
        id(model.shared),
        id(model.blocks[0].adapter_a),
    }
    with pytest.raises(ValueError, match="unique"):
        LayerwiseModuleStager(snapshot, [model.blocks[0]] * 2)
    with pytest.raises(ValueError, match="nested"):
        LayerwiseModuleStager(snapshot, [model, model.blocks[0]])
    protected = LayerwiseModuleStager(
        snapshot, model.blocks, resident_modules=(model.blocks[1],)
    )
    assert not protected.stagers[1]._groups
    assert {id(b.target) for g in protected.resident._groups for b in g.bindings} >= {
        id(t) for t in model.blocks[1].parameters()
    }


def test_layer_forward_lifetime_failure_and_reentry(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(
        BoundedAllocatorCache, "release_if_needed", lambda *a, **k: False
    )
    monkeypatch.setattr(
        PinnedModuleStager, "_load_once", lambda self: setattr(self, "loaded", True)
    )
    model = Model().eval()
    snapshot = cpu_snapshot(model)
    plan = LayerwiseModuleStager(snapshot, model.blocks)
    seen = []
    for index, block in enumerate(model.blocks):
        # Observe after the plan's pre-hook runs, inside the actual forward.
        original = block.forward

        def observe(x, index=index, original=original):
            assert plan.resident.loaded
            assert [s.loaded for s in plan.stagers] == [i == index for i in range(2)]
            seen.append(index)
            return original(x)

        block.forward = observe
    for fail in (False, True, False):
        model.blocks[1].fail = fail
        with plan.on_device():
            with pytest.raises(RuntimeError, match="idle host"), plan.on_device():
                pass
            with pytest.raises(RuntimeError, match="overlaps"):
                snapshot.load()
            if fail:
                with pytest.raises(RuntimeError, match="block failure"):
                    model(torch.ones(3, 8))
            else:
                assert torch.isfinite(model(torch.ones(3, 8))).all()
        assert not snapshot._layerwise_active
        assert all(not s.loaded for s in (*plan.stagers, plan.resident))
        assert all(
            not b._forward_pre_hooks and not b._forward_hooks for b in model.blocks
        )
    assert seen == [0, 1] * 3
    assert plan.loaded_bytes > 0


def test_layer_policy_rejects_incompatible_fixed_cache():
    assert H3Config().weight_offload == "component"
    assert H3Config(weight_offload="layer").weight_offload == "layer"
    with pytest.raises(H3InputError, match="offload"):
        H3Config(weight_offload="automatic")
    with pytest.raises(H3InputError, match="fixed GPU weight cache"):
        H3Config(weight_offload="layer", fp16_cache_layers=("blocks.0",))


@pytest.mark.parametrize("mode", ["generate", "serve"])
def test_layer_policy_cli(mode):
    from vllm.entrypoints.cli.video import VideoSubcommand

    parser = argparse.ArgumentParser()
    VideoSubcommand().subparser_init(parser.add_subparsers())
    assert parser.parse_args(["video", mode]).weight_offload == "component"
    assert (
        parser.parse_args(["video", mode, "--weight-offload", "layer"]).weight_offload
        == "layer"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("mapped", [False, True])
def test_gpu_layerwise_adapter_and_alias_roundtrips(dtype, mapped, tmp_path):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(96)
    model = Model(128).to(device="cuda", dtype=dtype).eval()
    # Module.to can split aliases; establish the actual mixed-owner alias on GPU.
    model.shared = model.blocks[0].adapter_a[0]
    for block in model.blocks:
        block.weight.data = block.weight.t().contiguous().t()
    inputs = [torch.randn(65, 128, device="cuda", dtype=dtype) for _ in range(3)]
    with torch.inference_mode():
        expected = [model(x).clone() for x in inputs]
        snapshot = PinnedModuleStager(
            model,
            torch.device("cuda"),
            pin_memory=False,
            host_backing=MMapHostWeights(tmp_path) if mapped else None,
        )
        plan = LayerwiseModuleStager(snapshot, model.blocks)
        for x, reference in zip(inputs, expected):
            with plan.on_device():
                actual = model(x)
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            assert all(p.device.type == "cpu" for p in model.parameters())
            if mapped:
                assert all(p.untyped_storage().filename for p in model.parameters())
            assert (
                model.shared.untyped_storage().data_ptr()
                == model.blocks[0].adapter_a.untyped_storage().data_ptr()
            )
