# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP2 stage-local shared experts: cold set of zero, devices 4-7, stage loading.

CPU only: no CUDA allocation, no native extension.
"""

import fcntl
import struct
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config.kernel import Sm70GgufConfig
from vllm.model_executor.layers.quantization.gguf_shared_layout import (
    LOAD_BATCH,
    SharedExpertBudget,
    SharedExpertLayout,
)
from vllm.model_executor.layers.quantization.gguf_shared_runtime import (
    ResidentExpertOwner,
    SharedExpertLayerSpec,
    _spec_fingerprint,
    resident_tables,
)
from vllm.model_executor.layers.quantization.gguf_shared_storage import (
    stage_device_problem,
    stage_truncation_problem,
)
from vllm.model_executor.models.utils import PPMissingLayer

# One expert's three projections at IQ2_XS (2.3125 bpw), the largest
# per-layer type in the #1277 checkpoint (5120 x 2304 x 3 x 2.3125 / 8).
IQ2_XS_EXPERT = 10_229_760
MIB = 1024**2


def _resident_budget(**updates):
    values: dict = dict(
        num_experts=384,
        offload_fraction=0.0,
        expert_bytes=(IQ2_XS_EXPERT,) * 20,
        spare_slots=0,
    )
    values.update(updates)
    return SharedExpertBudget(**values)


def test_cold_set_of_zero_keeps_every_expert_in_its_own_slot():
    budget = _resident_budget()
    assert budget.all_resident
    assert budget.cold_experts == 0 and budget.cold_bytes == 0
    assert budget.hot_experts == budget.gpu_slots == 384
    # No host pool: the arena is the exchange reserve alone.
    assert budget.arena_bytes == budget.exchange_bytes == 512 * MIB
    assert budget.required_exchange_bytes == LOAD_BATCH * IQ2_XS_EXPERT


def test_cold_zero_exchange_covers_only_the_load_batch():
    # M <= 18 (three users x six rows x top-6 = 108 routes) needs no larger
    # exchange at cold 0: there is no decode plan.
    assert _resident_budget(max_routes=108).required_exchange_bytes < 128 * MIB
    assert _resident_budget(exchange_bytes=LOAD_BATCH * IQ2_XS_EXPERT).all_resident
    with pytest.raises(ValueError):
        _resident_budget(exchange_bytes=LOAD_BATCH * IQ2_XS_EXPERT - 1)


def test_offloaded_m18_plan_needs_a_larger_exchange():
    # At cold > 0 the coordinator's plan grows with max_routes: 116 experts.
    with pytest.raises(ValueError):
        SharedExpertBudget(384, 0.30, (IQ2_XS_EXPERT,) * 20, max_routes=108)
    budget = SharedExpertBudget(
        384,
        0.30,
        (IQ2_XS_EXPERT,) * 20,
        max_routes=108,
        exchange_bytes=116 * IQ2_XS_EXPERT,
    )
    assert budget.required_exchange_bytes == 116 * IQ2_XS_EXPERT


@pytest.mark.parametrize(
    "updates",
    [
        {"spare_slots": 8},  # cold 0 with spares
        {"offload_fraction": 0.001},  # positive offload rounding to cold 0
        {"offload_fraction": 0.30},  # nonzero cold set without spares
        {"offload_fraction": -0.01},
        {"spare_slots": -1},
        {"max_routes": 385},
    ],
)
def test_ambiguous_or_invalid_cold_zero_is_rejected(updates):
    with pytest.raises(ValueError):
        _resident_budget(**updates)


def _shared(**overrides):
    values: dict = dict(
        expert_cache_host_storage="shared_original",
        expert_storage="original",
        expert_arena=True,
        expert_cache_host_slack=0,
    )
    values.update(overrides)
    return Sm70GgufConfig(**values)


def test_config_admits_cold_zero_only_with_zero_spares():
    policy = _shared(expert_cache_offload_fraction=0, expert_cache_spare_slots=0)
    assert policy.expert_cache_offload_fraction == 0
    for overrides in (
        {"expert_cache_offload_fraction": 0},
        {"expert_cache_offload_fraction": 0.3, "expert_cache_spare_slots": 0},
    ):
        with pytest.raises(ValueError):
            _shared(**overrides)


def test_rank_local_cache_keeps_its_spare_slots():
    with pytest.raises(ValueError):
        Sm70GgufConfig(expert_cache_fraction=0.7, expert_cache_spare_slots=0)
    # The plain arena smoke path (cache off) may carry spare 0.
    assert Sm70GgufConfig(expert_cache_spare_slots=0).expert_cache_fraction == 0


@pytest.mark.parametrize("pp_rank", [0, 1])
@pytest.mark.parametrize("tp_rank", range(4))
def test_stage_devices_are_the_group_device_not_the_tp_rank(pp_rank, tp_rank):
    device = torch.device("cuda", 4 * pp_rank + tp_rank)
    assert stage_device_problem(4, device, device, device.index, 8, 2) is None


def test_single_stage_tp4_is_unchanged():
    device = torch.device("cuda", 2)
    assert stage_device_problem(4, device, device, 2, 4, 1) is None
    assert stage_device_problem(4, None, device, 2, None, 1) is None


@pytest.mark.parametrize(
    "args,reason",
    [
        ((8, "cuda:4", "cuda:4", 4, 8, 1), "four ranks"),
        ((4, "cuda:4", "cuda:5", 5, 8, 2), "TP group's device"),
        ((4, "cuda:5", "cuda:5", 1, 8, 2), "current device"),
        ((4, "cuda:5", "cuda:5", 5, 8, 1), "pipeline stages"),
        ((4, None, "cuda", 0, 4, 1), "indexed CUDA"),
        ((4, None, "cpu", 0, 4, 1), "indexed CUDA"),
    ],
)
def test_stage_device_mismatches_name_their_reason(args, reason):
    tp_world, tp_device, device, current, world, pp = args
    problem = stage_device_problem(
        tp_world,
        None if tp_device is None else torch.device(tp_device),
        torch.device(device),
        current,
        world,
        pp,
    )
    assert problem is not None and reason in problem


@pytest.mark.parametrize(
    "pp,resident,swap_free,cutoff",
    [
        (1, False, "1", "0"),  # one process sees every layer's truncation
        (1, False, "1", "3"),
        (2, True, "1", "3"),  # cold 0: no misses, no planner
        (2, False, "0", "0"),  # compaction exchange never truncates
        (2, False, None, None),
    ],
)
def test_truncation_is_safe_without_cross_stage_limits(pp, resident, swap_free, cutoff):
    assert stage_truncation_problem(pp, resident, swap_free, cutoff) is None


@pytest.mark.parametrize(
    "swap_free,cutoff,knob",
    [
        ("1", "0", "VLLM_V41_SWAP_FREE=0"),  # the runtime's default
        ("1x", "0", "VLLM_V41_SWAP_FREE=0"),  # the planner reads only v[0]
        ("0", "2", "VLLM_V41_TRUNC_CUTOFF=0"),
    ],
)
def test_staged_cold_set_refuses_truncating_planners(swap_free, cutoff, knob):
    # The last stage's sampler reads only its own ranks' truncation limit.
    problem = stage_truncation_problem(2, False, swap_free, cutoff)
    assert problem is not None and knob in problem


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_stage_two_ranks_use_the_same_carriers_as_stage_one(kind):
    # Layouts depend on the TP rank only; cuda:4-7 hold TP ranks 0-3.
    spec = SharedExpertLayerSpec(5120, 2304, 384, 384, (kind, kind, kind))
    for rank in range(4):
        for layout, role in zip(spec.layouts(rank), ("w1", "w3", "w2")):
            assert layout == SharedExpertLayout.for_type(
                5120, 2304, 4, rank, role, kind
            )
            if role == "w2":  # VLLM_V41_DOWN_ROWS=1 is the #1277 profile
                assert layout.device_rows in (5120, 1280)


def test_resident_spec_and_identity_tables():
    specs = (
        SharedExpertLayerSpec(5120, 2304, 384, 384, (16, 16, 29)),
        SharedExpertLayerSpec(5120, 2304, 384, 384, (17, 17, 16)),
    )
    table, cold = resident_tables(specs)
    assert table.dtype == torch.int32 and cold.dtype == torch.int64
    expected = torch.arange(384, dtype=torch.int32).expand(2, 384)
    assert torch.equal(table, expected)  # slot = expert id, as the planner starts
    assert bool((cold == -1).all())
    for resident in (0, 385):
        with pytest.raises(ValueError):
            SharedExpertLayerSpec(5120, 2304, 384, resident, (16, 16, 16))
    with pytest.raises(ValueError):
        resident_tables((SharedExpertLayerSpec(5120, 2304, 384, 383, (16,) * 3),))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="memfd seals")
def test_resident_owner_shares_only_the_exchange_and_fingerprint(monkeypatch):
    # Some standalone CPython builds omit the Linux seal constants.
    for name, value in (
        ("F_ADD_SEALS", 1033),
        ("F_GET_SEALS", 1034),
        ("F_SEAL_SEAL", 1),
        ("F_SEAL_SHRINK", 2),
        ("F_SEAL_GROW", 4),
    ):
        if not hasattr(fcntl, name):
            monkeypatch.setattr(fcntl, name, value, raising=False)
    specs = (SharedExpertLayerSpec(5120, 2304, 384, 384, (16, 16, 29)),)
    owner = ResidentExpertOwner(specs, exchange_bytes=LOAD_BATCH * IQ2_XS_EXPERT)
    try:
        assert owner.data.size >= LOAD_BATCH * IQ2_XS_EXPERT
        assert owner.data.size < LOAD_BATCH * IQ2_XS_EXPERT + 4096
        stored = struct.unpack_from("=q", owner.control.buffer, 56)[0]
        assert stored == _spec_fingerprint(specs)
    finally:
        owner.close()
    with pytest.raises(ValueError):
        ResidentExpertOwner((SharedExpertLayerSpec(5120, 2304, 384, 268, (16,) * 3),))


class _Stage(nn.Module):
    """layers 0-3 with a 2/2 split; this process is the second stage."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [PPMissingLayer(), PPMissingLayer(), nn.Linear(1, 1), nn.Linear(1, 1)]
        )
        self.model.start_layer, self.model.end_layer = 2, 4
        self.model.engram_token_map = nn.Parameter(torch.zeros(1))
        self.lm_head = nn.Linear(1, 1)


def _name_map():
    return {
        "blk.1.ffn_gate_exps.weight": "model.layers.1.ffn.experts.gate_proj.weight",
        "blk.1.engram_wkv.weight": "model.layers.1.engram.wkv.weight",
        "blk.2.ffn_gate_exps.weight": "model.layers.2.ffn.experts.gate_proj.weight",
        "blk.3.attn_kv.weight": "model.layers.3.attn.wkv.weight",
        "engram_token_map.weight": "model.engram_token_map",
        "output.weight": "lm_head.weight",
    }


@pytest.mark.parametrize("first", [False, True])
def test_gguf_stage_filter_reads_only_local_tensors(monkeypatch, first):
    import vllm.distributed
    from vllm.model_executor.model_loader.gguf_adapters.deepseek_v41 import (
        DeepseekV41Adapter,
    )
    from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader

    pp = SimpleNamespace(
        world_size=2, rank_in_group=0 if first else 1, is_first_rank=first
    )
    pp.is_last_rank = not first
    monkeypatch.setattr(vllm.distributed, "get_pp_group", lambda: pp)
    loader = SimpleNamespace()
    model = _Stage()
    result = GGUFModelLoader._stage_weights_map(
        loader, DeepseekV41Adapter, model, _name_map()
    )
    expected = {
        "blk.2.ffn_gate_exps.weight",
        "blk.3.attn_kv.weight",
        "output.weight",
    }
    if first:  # the hash tensors stay on the first stage
        expected.add("engram_token_map.weight")
    assert set(result) == expected
    assert loader._gguf_prepared_weights_map is result


def test_arena_skips_other_stage_experts_before_module_lookup():
    from vllm.model_executor.layers.quantization.gguf_expert_storage import (
        prepare_original_expert_arena,
    )

    model = _Stage()
    names = {
        "blk.1.ffn_gate_exps.weight": "model.layers.1.ffn.experts.gate_proj.weight"
    }
    # Previously get_submodule raised on the PPMissingLayer placeholder.
    assert prepare_original_expert_arena(model, names, {}, torch.device("cpu")) == {}


def test_dense_postload_visits_only_local_blocks():
    from vllm.models.deepseek_v41.dense_tm import _blocks

    model = _Stage().model
    assert _blocks(model) == [model.layers[2], model.layers[3]]
