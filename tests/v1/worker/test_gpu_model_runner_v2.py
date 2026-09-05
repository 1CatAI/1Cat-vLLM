# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.parallel_state as parallel_state
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu import model_runner as mrv2


def test_sm70_v2_mtp_profile_gate(monkeypatch):
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.speculative_config = SimpleNamespace(method="mtp")
    runner.is_last_pp_rank = True
    runner.device = torch.device("cuda")

    monkeypatch.setenv("VLLM_SM70_MTP_PROFILE", "1")
    assert runner._sm70_v2_mtp_profile_enabled()

    runner.speculative_config = SimpleNamespace(method="dflash")
    assert not runner._sm70_v2_mtp_profile_enabled()
    runner.speculative_config = SimpleNamespace(method="mtp")
    runner.is_last_pp_rank = False
    assert not runner._sm70_v2_mtp_profile_enabled()


def test_sm70_v2_mtp_profile_composes_target_verifier(monkeypatch):
    class FakeEvent:
        def __init__(self, elapsed_ms: float = 0.0):
            self.elapsed_ms = elapsed_ms

        def elapsed_time(self, end: "FakeEvent") -> float:
            del end
            return self.elapsed_ms

        def synchronize(self) -> None:
            pass

    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    monkeypatch.setenv("VLLM_SM70_MTP_PROFILE_INTERVAL", "1")
    monkeypatch.setattr(parallel_state, "_PP", None)
    monkeypatch.setattr(parallel_state, "_TP", None)
    ctx = {
        "events": [
            ("target_forward", FakeEvent(10.0), FakeEvent()),
            ("target_sample", FakeEvent(2.0), FakeEvent()),
            ("target_state_update", FakeEvent(1.0), FakeEvent()),
            ("draft_total", FakeEvent(5.0), FakeEvent()),
            ("total_gpu", FakeEvent(20.0), FakeEvent()),
        ],
        "num_tokens": 5,
        "num_draft_tokens": 4,
        "target_verifier_wall_cpu": 14.0,
        "total_wall_start": time.perf_counter(),
    }

    runner._sm70_v2_mtp_profile_report(ctx)

    assert runner._sm70_v2_mtp_profile_totals["target_verifier_gpu"] == 13.0
    assert runner._sm70_v2_mtp_profile_totals["target_verifier_wall_cpu"] == 14.0
    assert runner._sm70_v2_mtp_profile_totals["draft_total"] == 5.0
    assert runner._sm70_v2_mtp_profile_totals["total_gpu"] == 20.0


@pytest.mark.parametrize(
    ("global_first", "pp_last", "tp_rank", "expected"),
    [
        pytest.param(False, True, 0, True, id="pp2-last-stage-leader"),
        pytest.param(False, True, 1, False, id="pp2-last-stage-tp1"),
        pytest.param(True, False, 0, False, id="pp2-first-stage"),
    ],
)
def test_sm70_v2_mtp_profile_reports_from_last_pp_stage(
    monkeypatch, global_first, pp_last, tp_rank, expected
):
    """Regression for #414: profiling is enabled on the last PP stage only,
    so the report must not be gated on the global first rank."""

    class FakeEvent:
        def elapsed_time(self, end: "FakeEvent") -> float:
            del end
            return 1.0

        def synchronize(self) -> None:
            pass

    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    monkeypatch.setenv("VLLM_SM70_MTP_PROFILE_INTERVAL", "1")
    monkeypatch.setattr(
        parallel_state, "_WORLD", SimpleNamespace(is_first_rank=global_first)
    )
    monkeypatch.setattr(parallel_state, "_PP", SimpleNamespace(is_last_rank=pp_last))
    monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(rank_in_group=tp_rank))
    messages: list[str] = []
    monkeypatch.setattr(
        mrv2.logger, "info", lambda msg, *args, **kwargs: messages.append(msg)
    )
    ctx = {
        "events": [("target_forward", FakeEvent(), FakeEvent())],
        "num_tokens": 1,
        "num_draft_tokens": 0,
        "target_verifier_wall_cpu": 1.0,
        "total_wall_start": time.perf_counter(),
    }

    runner._sm70_v2_mtp_profile_report(ctx)

    assert runner._sm70_v2_mtp_profile_totals["target_forward"] == 1.0
    assert bool(messages) is expected


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    runner.vllm_config = SimpleNamespace()
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        mrv2,
        "init_attn_backend",
        lambda *args: ([], attn_cg_support, [8, 262144]),
    )
    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(mrv2, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]
