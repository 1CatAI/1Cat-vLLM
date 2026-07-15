# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

import sys
import types
from types import SimpleNamespace

import pytest

for module_name in ("vllm._C", "vllm._C_stable_libtorch"):
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn_v100 import (
    FlashAttnV100Impl,
    FlashAttnV100MetadataBuilder,
)
from vllm.v1.attention.backends.flashinfer_sm70 import (
    FlashInferSM70Backend,
    FlashInferSM70Impl,
    FlashInferSM70MetadataBuilder,
)
from vllm.v1.attention.backends.flashinfer_sm70_planner import (
    FlashInferSM70Plan,
    make_delegate_decision,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_flashinfer_sm70_registry_is_independent() -> None:
    assert AttentionBackendEnum.FLASHINFER_SM70.get_class() is FlashInferSM70Backend
    assert FlashInferSM70Backend.get_name() == "FLASHINFER_SM70"
    assert (
        AttentionBackendEnum.FLASHINFER_SM70.get_path()
        != AttentionBackendEnum.FLASH_ATTN_V100.get_path()
    )


def test_flashinfer_sm70_only_supports_volta() -> None:
    assert FlashInferSM70Backend.supports_compute_capability(DeviceCapability(7, 0))
    assert not FlashInferSM70Backend.supports_compute_capability(DeviceCapability(7, 5))
    assert not FlashInferSM70Backend.supports_compute_capability(DeviceCapability(8, 0))


def test_generic_prefill_shape_does_not_claim_bm32_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = SimpleNamespace(flash_v100_cudagraph_capture=False)
    common = SimpleNamespace(
        max_query_len=16,
        num_actual_tokens=64,
        max_seq_len=1024,
        num_reqs=1,
    )
    builder = object.__new__(FlashInferSM70MetadataBuilder)
    builder.num_heads_q = 6
    builder.num_heads_kv = 1

    monkeypatch.setattr(
        FlashAttnV100MetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )

    result = builder.build(0, common)
    decision = result.flashinfer_sm70_planner_decision

    assert decision.stage == "build"
    assert decision.workload == "prefill"
    assert decision.plan.kv_chunk_size == 128
    assert decision.planner_candidate == "accepted_bm32_resource_envelope"
    assert decision.delegate_dispatch == "flash_v100_runtime_dispatch"
    assert decision.delegate_dispatch != "accepted_bm32"
    assert result.flashinfer_sm70_route_proof == decision.route_proof
    assert "executed_kernel" not in result.flashinfer_sm70_route_proof
    assert result.flashinfer_sm70_dispatch_observed is None
    assert result.flashinfer_sm70_prefill_paged_routes_observed == ()
    assert result.flashinfer_sm70_runtime_route_proof is None


def test_capture_metadata_records_quality_locked_decode_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = SimpleNamespace(
        flash_v100_cudagraph_capture=True,
        seq_lens_cpu=[1025],
    )
    common = SimpleNamespace(
        max_query_len=1,
        num_actual_tokens=1,
        max_seq_len=1025,
        num_reqs=1,
        _seq_lens_cpu=[1025],
        seq_lens_cpu_upper_bound=None,
    )
    builder = object.__new__(FlashInferSM70MetadataBuilder)
    builder.num_heads_q = 6
    builder.num_heads_kv = 1

    monkeypatch.setattr(
        FlashAttnV100MetadataBuilder,
        "build_for_cudagraph_capture",
        lambda *args, **kwargs: metadata,
    )

    result = builder.build_for_cudagraph_capture(common)
    decision = result.flashinfer_sm70_planner_decision

    assert decision.stage == "capture"
    assert decision.workload == "decode"
    assert decision.plan.kv_chunk_size == 256
    assert decision.plan.logical_ctas == 5
    assert decision.plan.padded_ctas == 72
    assert decision.planner_candidate == "accepted_decode_resource_envelope"
    assert decision.delegate_dispatch == "flash_v100_runtime_dispatch"
    assert not decision.kernel_promoted


def test_impl_records_parent_dispatch_and_observable_paged_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = FlashInferSM70Plan(
        split_kv=False,
        kv_chunk_size=1024,
        logical_ctas=12,
        padded_ctas=12,
        resident_ctas_per_sm=2,
        grid_capacity=144,
    )
    decision = make_delegate_decision(
        stage="build",
        workload="prefill",
        plan=plan,
    )
    metadata = SimpleNamespace(
        flashinfer_sm70_planner_decision=decision,
        flashinfer_sm70_route_proof=decision.route_proof,
        flashinfer_sm70_dispatch_observed=None,
        flashinfer_sm70_prefill_paged_routes_observed=(),
        flashinfer_sm70_runtime_route_proof=None,
    )
    calls = []
    expected = object()

    def accepted_forward(self, *args, **kwargs):
        calls.append(args[5])
        return self._run_prefill_paged_call(
            route="prefill_prefix_paged",
            q_len=16,
            seq_len=1024,
            heads_q=6,
            heads_kv=1,
            head_dim=128,
            block_size=16,
            fn=lambda: expected,
        )

    def accepted_paged_call(self, **kwargs):
        return kwargs["fn"]()

    monkeypatch.setattr(FlashAttnV100Impl, "forward", accepted_forward)
    monkeypatch.setattr(
        FlashAttnV100Impl,
        "_run_prefill_paged_call",
        accepted_paged_call,
    )
    impl = object.__new__(FlashInferSM70Impl)

    result = impl.forward(
        object(),
        object(),
        object(),
        object(),
        object(),
        metadata,
    )

    assert result is expected
    assert calls == [metadata]
    assert metadata.flashinfer_sm70_dispatch_observed == "flash_v100_runtime_dispatch"
    assert metadata.flashinfer_sm70_prefill_paged_routes_observed == (
        "prefill_prefix_paged",
    )
    assert impl.last_route_proof["selected_backend"] == "FLASHINFER_SM70"
    assert impl.last_route_proof["delegate_backend"] == "FLASH_ATTN_V100"
    assert impl.last_route_proof["observed_dispatch"] == "flash_v100_runtime_dispatch"
    assert impl.last_route_proof["observed_prefill_paged_routes"] == [
        "prefill_prefix_paged"
    ]
    assert "executed_kernel" not in impl.last_route_proof
    assert impl.last_route_proof["kernel_promoted"] is False
    assert metadata.flashinfer_sm70_runtime_route_proof == impl.last_route_proof
