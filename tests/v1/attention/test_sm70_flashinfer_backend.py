# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

import sys
import types
from types import SimpleNamespace

import pytest
import torch

for module_name in ("vllm._C", "vllm._C_stable_libtorch"):
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends import flashinfer_sm70 as flashinfer_sm70_backend
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
    SM70KernelResources,
    make_delegate_decision,
    plan_prefill_requests,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _make_fixed_prefill_case(
    *,
    query_len: int = 32,
    capture: bool = False,
):
    work = plan_prefill_requests(
        qo_lens=[query_len],
        kv_lens=[913],
        num_qo_heads=6,
        num_kv_heads=2,
        head_dim_vo=256,
        cta_tile_q=32,
        kv_tile_size=128,
        resources=SM70KernelResources(512, 64, 41_936),
        enable_cuda_graph=capture,
        fixed_split_size=1024,
    )
    decision = make_delegate_decision(
        stage="capture" if capture else "build",
        workload="prefill",
        plan=work.plan,
    )
    metadata = SimpleNamespace(
        flashinfer_sm70_planner_decision=decision,
        flashinfer_sm70_prefill_work_description=work,
        flashinfer_sm70_route_proof=decision.route_proof,
        flashinfer_sm70_dispatch_observed=None,
        flashinfer_sm70_prefill_paged_routes_observed=(),
        flashinfer_sm70_runtime_route_proof=None,
        flash_v100_cudagraph_capture=capture,
        causal=True,
        ddtree_parent_ids=None,
        ddtree_num_tree_tokens_cpu=None,
        num_actual_tokens=query_len,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_lens=torch.tensor([913], dtype=torch.int32),
    )
    query = torch.zeros((query_len, 6, 256), dtype=torch.float16)
    output = torch.empty_like(query)
    key_cache = torch.zeros((2, 784, 2, 256), dtype=torch.float16)
    value_cache = torch.zeros_like(key_cache)
    return decision, metadata, query, (key_cache, value_cache), output


def _make_fixed_prefill_impl(fixed_entry, splitkv3_entry=None):
    impl = object.__new__(FlashInferSM70Impl)
    impl.num_heads = 6
    impl.num_kv_heads = 2
    impl.head_size = 256
    impl.scale = 0.0625
    impl.kv_cache_dtype = "auto"
    impl.attn_type = AttentionType.DECODER
    impl.sliding_window = (-1, -1)
    impl.use_triton_prefill = False
    impl.use_flash_v100_prefill_paged = True
    impl.use_flash_v100_prefill_splitkv = False
    impl.use_flash_v100_prefill_bfla = False
    impl.use_flash_v100_prefill_contig_dense = True
    impl.alibi_slopes = None
    impl.logits_soft_cap = 0.0
    impl.sinks = None
    impl.flash_attn_prefill_paged_d256_bm32_allp_pair_scratch = fixed_entry
    impl.flash_attn_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3 = (
        splitkv3_entry
    )
    impl.last_route_proof = None
    impl._flashinfer_sm70_active_metadata = None
    return impl


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
        query_start_loc_cpu=[0, 64],
        _seq_lens_cpu=[1024],
        seq_lens_cpu_upper_bound=None,
    )
    builder = object.__new__(FlashInferSM70MetadataBuilder)
    builder.num_heads_q = 6
    builder.num_heads_kv = 1
    builder.headdim = 256

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
    work = result.flashinfer_sm70_prefill_work_description
    assert work.is_exact
    assert work.qo_lens == (64,)
    assert work.packed_qo_lens == (384,)
    assert work.kv_lens == (1024,)
    assert decision.plan is work.plan
    assert decision.planner_candidate == "accepted_bm32_resource_envelope"
    assert decision.delegate_dispatch == "flash_v100_runtime_dispatch"
    assert decision.delegate_dispatch != "accepted_bm32"
    assert result.flashinfer_sm70_route_proof == decision.route_proof
    assert "executed_kernel" not in result.flashinfer_sm70_route_proof
    assert result.flashinfer_sm70_dispatch_observed is None
    assert result.flashinfer_sm70_prefill_paged_routes_observed == ()
    assert result.flashinfer_sm70_runtime_route_proof is None


def test_prefill_metadata_marks_missing_exact_seq_lens_as_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = SimpleNamespace(
        flash_v100_cudagraph_capture=False,
        seq_lens_cpu=[128, 512],
    )
    common = SimpleNamespace(
        max_query_len=5,
        num_actual_tokens=7,
        max_seq_len=512,
        num_reqs=2,
        query_start_loc_cpu=[0, 2, 7],
        _seq_lens_cpu=None,
        seq_lens_cpu_upper_bound=[128, 512],
    )
    builder = object.__new__(FlashInferSM70MetadataBuilder)
    builder.num_heads_q = 2
    builder.num_heads_kv = 1
    builder.headdim = 8

    monkeypatch.setattr(
        FlashAttnV100MetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )

    result = builder.build(0, common)
    work = result.flashinfer_sm70_prefill_work_description

    assert work.planning_mode == "upper_bound"
    assert not work.is_exact
    assert work.qo_lens == (2, 5)
    assert work.packed_qo_lens == (4, 10)
    assert work.kv_lens == (128, 512)
    assert work.request_indices == (0, 1, 1, 1, 1)
    assert result.flashinfer_sm70_planner_decision.plan is work.plan


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
    assert result.flashinfer_sm70_prefill_work_description is None


def test_eligible_fixed_prefill_bypasses_parent_and_records_runtime_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision, metadata, query, kv_cache, output = _make_fixed_prefill_case()
    calls = []
    route_calls = []

    def fixed_entry(
        query_bhmd,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        *,
        softmax_scale,
    ):
        calls.append(
            (
                query_bhmd.shape,
                query_bhmd.is_contiguous(),
                key_cache.shape,
                value_cache.shape,
                block_table.shape,
                seq_lens.tolist(),
                softmax_scale,
            )
        )
        return torch.full_like(query_bhmd, 7), torch.zeros(
            query_bhmd.shape[:-1], dtype=torch.float32
        )

    def unexpected_parent(*args, **kwargs):
        raise AssertionError("eligible fixed prefill must bypass parent forward")

    monkeypatch.setattr(FlashAttnV100Impl, "forward", unexpected_parent)
    monkeypatch.setattr(flashinfer_sm70_backend, "_record_route", route_calls.append)
    impl = _make_fixed_prefill_impl(fixed_entry)

    result = impl.forward(
        object(),
        query,
        query[:, :2],
        query[:, :2],
        kv_cache,
        metadata,
        output,
    )

    assert result is output
    assert torch.equal(output, torch.full_like(output, 7))
    assert calls == [
        (
            torch.Size([1, 6, 32, 256]),
            True,
            torch.Size([2, 784, 2, 256]),
            torch.Size([2, 784, 2, 256]),
            torch.Size([1, 2]),
            [913],
            0.0625,
        )
    ]
    assert metadata.flashinfer_sm70_route_proof == decision.route_proof
    assert decision.kernel_promoted is False
    runtime_proof = metadata.flashinfer_sm70_runtime_route_proof
    assert runtime_proof["planner_proof"] == decision.route_proof
    assert runtime_proof["observed_dispatch"] == "flashinfer_sm70_fixed_entry"
    assert runtime_proof["executed_backend"] == "FLASHINFER_SM70"
    assert runtime_proof["executed_kernel"] == (
        "flash_attn_prefill_paged_d256_bm32_allp_pair_scratch<true,true,true>"
    )
    assert runtime_proof["kernel_promoted"] is True
    assert impl.last_route_proof == runtime_proof
    assert route_calls == ["flashinfer_sm70_fixed_entry"]


def test_splitkv3_promotion_policy_uses_validated_chunk_lengths() -> None:
    assert flashinfer_sm70_backend._splitkv3_has_visible_anchor(8096, 121440)
    assert not flashinfer_sm70_backend._splitkv3_has_visible_anchor(512, 513)
    validated_kv_lens = (80960, 89056, 97152, 105248, 113344, 121440)
    for kv_len in validated_kv_lens:
        assert flashinfer_sm70_backend._is_promoted_splitkv3_shape(
            query_len=8096,
            kv_len=kv_len,
            num_heads=6,
            num_kv_heads=1,
        )
    for changed in (
        {"query_len": 8095},
        {"kv_len": 72864},
        {"kv_len": 121439},
        {"kv_len": 129536},
        {"num_heads": 5},
        {"num_kv_heads": 2},
    ):
        shape = {
            "query_len": 8096,
            "kv_len": 121440,
            "num_heads": 6,
            "num_kv_heads": 1,
            **changed,
        }
        assert not flashinfer_sm70_backend._is_promoted_splitkv3_shape(**shape)


def test_promoted_splitkv3_bypasses_fixed_entry_and_records_actual_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision, metadata, query, kv_cache, output = _make_fixed_prefill_case()
    calls = []
    route_calls = []

    def splitkv3_entry(
        query_bhmd,
        key_cache,
        value_cache,
        block_table,
        actual_n,
        *,
        softmax_scale,
    ):
        calls.append(
            (
                query_bhmd.shape,
                key_cache.shape,
                value_cache.shape,
                block_table.shape,
                actual_n,
                softmax_scale,
            )
        )
        return torch.full_like(query_bhmd, 9), torch.zeros(
            query_bhmd.shape[:-1], dtype=torch.float32
        )

    def unexpected_fixed_entry(*args, **kwargs):
        raise AssertionError("promoted split-KV must bypass fixed unsplit")

    def unexpected_parent(*args, **kwargs):
        raise AssertionError("promoted split-KV must bypass parent forward")

    monkeypatch.setattr(
        flashinfer_sm70_backend,
        "_is_promoted_splitkv3_shape",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(FlashAttnV100Impl, "forward", unexpected_parent)
    monkeypatch.setattr(flashinfer_sm70_backend, "_record_route", route_calls.append)
    impl = _make_fixed_prefill_impl(unexpected_fixed_entry, splitkv3_entry)

    result = impl.forward(
        object(),
        query,
        query[:, :2],
        query[:, :2],
        kv_cache,
        metadata,
        output,
    )

    assert result is output
    assert torch.equal(output, torch.full_like(output, 9))
    assert calls == [
        (
            torch.Size([1, 6, 32, 256]),
            torch.Size([2, 784, 2, 256]),
            torch.Size([2, 784, 2, 256]),
            torch.Size([1, 2]),
            913,
            0.0625,
        )
    ]
    runtime_proof = metadata.flashinfer_sm70_runtime_route_proof
    assert runtime_proof["observed_dispatch"] == (
        "flashinfer_sm70_splitkv3_fast_visible"
    )
    assert runtime_proof["executed_kernel"] == (
        "flash_attn_prefill_paged_d256_bm32_splitkv3_partial<false>+merge"
    )
    assert runtime_proof["actual_n"] == 913
    assert runtime_proof["splitkv3_fast_visible"] is True
    assert runtime_proof["kernel_promoted"] is True
    assert route_calls == ["flashinfer_sm70_splitkv3_fast_visible"]


def test_promoted_splitkv3_missing_entry_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, metadata, query, kv_cache, output = _make_fixed_prefill_case()

    def unexpected_parent(*args, **kwargs):
        raise AssertionError("missing split entry must not delegate")

    monkeypatch.setattr(
        flashinfer_sm70_backend,
        "_is_promoted_splitkv3_shape",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(FlashAttnV100Impl, "forward", unexpected_parent)
    impl = _make_fixed_prefill_impl(lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="split-KV prefill entry is unavailable"):
        impl.forward(
            object(),
            query,
            query[:, :2],
            query[:, :2],
            kv_cache,
            metadata,
            output,
        )

    assert metadata.flashinfer_sm70_dispatch_observed is None
    assert metadata.flashinfer_sm70_runtime_route_proof is None
    assert impl.last_route_proof is None


def test_ineligible_fixed_prefill_delegates_to_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, metadata, query, kv_cache, output = _make_fixed_prefill_case(query_len=16)
    expected = object()
    parent_calls = []

    def parent_forward(self, *args, **kwargs):
        parent_calls.append(args[5])
        return expected

    def unexpected_fixed_entry(*args, **kwargs):
        raise AssertionError("M < 32 must not call the fixed entry")

    monkeypatch.setattr(FlashAttnV100Impl, "forward", parent_forward)
    impl = _make_fixed_prefill_impl(unexpected_fixed_entry)

    result = impl.forward(
        object(),
        query,
        query[:, :2],
        query[:, :2],
        kv_cache,
        metadata,
        output,
    )

    assert result is expected
    assert parent_calls == [metadata]
    assert metadata.flashinfer_sm70_dispatch_observed == ("flash_v100_runtime_dispatch")
    assert metadata.flashinfer_sm70_runtime_route_proof["kernel_promoted"] is False
    assert "executed_kernel" not in metadata.flashinfer_sm70_runtime_route_proof


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("attn_type", AttentionType.ENCODER),
        ("use_triton_prefill", True),
        ("use_flash_v100_prefill_paged", False),
        ("use_flash_v100_prefill_splitkv", True),
    ],
)
def test_fixed_prefill_policy_ineligible_shapes_delegate(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: object,
) -> None:
    _, metadata, query, kv_cache, output = _make_fixed_prefill_case()
    expected = object()
    parent_calls = []

    def parent_forward(self, *args, **kwargs):
        parent_calls.append(args[5])
        return expected

    def unexpected_fixed_entry(*args, **kwargs):
        raise AssertionError(f"{attribute}={value!r} must delegate")

    monkeypatch.setattr(FlashAttnV100Impl, "forward", parent_forward)
    impl = _make_fixed_prefill_impl(unexpected_fixed_entry)
    setattr(impl, attribute, value)

    result = impl.forward(
        object(),
        query,
        query[:, :2],
        query[:, :2],
        kv_cache,
        metadata,
        output,
    )

    assert result is expected
    assert parent_calls == [metadata]
    assert metadata.flashinfer_sm70_dispatch_observed == ("flash_v100_runtime_dispatch")
    assert metadata.flashinfer_sm70_runtime_route_proof["kernel_promoted"] is False


def test_fixed_prefill_entry_error_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, metadata, query, kv_cache, output = _make_fixed_prefill_case()
    route_calls = []

    def failed_fixed_entry(*args, **kwargs):
        raise RuntimeError("fixed entry failed")

    def unexpected_parent(*args, **kwargs):
        raise AssertionError("fixed entry errors must not delegate")

    monkeypatch.setattr(FlashAttnV100Impl, "forward", unexpected_parent)
    monkeypatch.setattr(flashinfer_sm70_backend, "_record_route", route_calls.append)
    impl = _make_fixed_prefill_impl(failed_fixed_entry)

    with pytest.raises(RuntimeError, match="fixed entry failed"):
        impl.forward(
            object(),
            query,
            query[:, :2],
            query[:, :2],
            kv_cache,
            metadata,
            output,
        )

    assert metadata.flashinfer_sm70_dispatch_observed is None
    assert metadata.flashinfer_sm70_runtime_route_proof is None
    assert impl.last_route_proof is None
    assert route_calls == []


def test_capture_shape_delegates_without_calling_fixed_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, metadata, query, kv_cache, output = _make_fixed_prefill_case(capture=True)
    expected = object()
    parent_calls = []

    def parent_forward(self, *args, **kwargs):
        parent_calls.append(args[5])
        return expected

    def unexpected_fixed_entry(*args, **kwargs):
        raise AssertionError("CUDA graph capture must reject the fixed entry")

    monkeypatch.setattr(FlashAttnV100Impl, "forward", parent_forward)
    impl = _make_fixed_prefill_impl(unexpected_fixed_entry)

    result = impl.forward(
        object(),
        query,
        query[:, :2],
        query[:, :2],
        kv_cache,
        metadata,
        output,
    )

    assert result is expected
    assert parent_calls == [metadata]
    assert metadata.flashinfer_sm70_dispatch_observed == ("flash_v100_runtime_dispatch")
    assert metadata.flashinfer_sm70_runtime_route_proof["kernel_promoted"] is False


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
