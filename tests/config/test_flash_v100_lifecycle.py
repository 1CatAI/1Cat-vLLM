# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import threading
from types import SimpleNamespace

import pytest

from vllm.config.execution_policy import FlashV100Policy, GraphPolicy
from vllm.config.flash_v100 import FlashV100Diagnostics, FlashV100Options
from vllm.config.sm70_runtime import RuntimeTraceConfig
from vllm.diagnostics import diagnostics_for

pytestmark = pytest.mark.cpu_test


def prepare(options=None, graph=None, trace=None, dtype="fp8_e4m3"):
    policy = FlashV100Policy(options=options or FlashV100Options())
    policy.resolve()
    graph = graph or GraphPolicy()
    trace = trace or FlashV100Diagnostics()
    policy.options.finalize(graph, trace, dtype)
    return policy, graph, trace


@pytest.mark.parametrize("order", [(False, True), (True, False)])
def test_worker_transfer_keeps_two_engine_policies_and_sources(monkeypatch, order):
    configs = []
    for value in order:
        monkeypatch.setenv("VLLM_FLASH_V100_XQA_G6_QK_PIPELINE", str(int(value)))
        configs.append(prepare()[0])
    receiver, sender = multiprocessing.Pipe(duplex=False)
    thread = threading.Thread(target=sender.send, args=(configs,), daemon=True)
    thread.start()
    try:
        assert receiver.poll(10)
        transferred = receiver.recv()
        thread.join(10)
        assert not thread.is_alive()
    finally:
        receiver.close()
        sender.close()
    monkeypatch.setattr(
        "os.getenv", lambda *a: pytest.fail("worker reread environment")
    )
    for expected, original, actual in zip(order, configs, transferred):
        actual.resolve()
        assert actual.options.xqa_g6_qk_pipeline is expected
        assert actual.compute_hash() == original.compute_hash()
        assert (
            actual.options.sources["xqa_g6_qk_pipeline"]
            == "VLLM_FLASH_V100_XQA_G6_QK_PIPELINE"
        )


@pytest.mark.parametrize("raw", ["", "0", "1", "01", "true", "-1"])
def test_graph_and_native_keep_their_distinct_legacy_parsers(monkeypatch, raw):
    name = "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO"
    monkeypatch.setenv(name, raw)
    policy, graph, _ = prepare()
    from vllm.config.flash_v100 import NATIVE_FIELDS

    index = next(i for i, entry in enumerate(NATIVE_FIELDS) if entry[1] == name)
    assert graph.e4m3_p64_p256_auto is (raw != "0")
    assert policy.options.native_effective[index] == (raw[:1] != "0")
    typed, _, _ = prepare(graph=GraphPolicy(e4m3_p64_p256_auto=False))
    assert typed.options.native_effective[index] == 0
    assert typed.options.native_inputs[index] == "0"


def test_effective_native_projection_participates_in_hash(monkeypatch):
    name = "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO"
    monkeypatch.setenv(name, "01")
    first, first_graph, _ = prepare()
    monkeypatch.setenv(name, "1")
    second, second_graph, _ = prepare()
    assert first_graph.compute_hash() == second_graph.compute_hash()
    assert first.compute_hash() != second.compute_hash()


def test_effective_package_projection_participates_in_hash(monkeypatch):
    name = "VLLM_FLASH_V100_XQA_G6_DUAL_CTA"
    monkeypatch.setenv(name, "1suffix")
    first = prepare()[0]
    monkeypatch.setenv(name, "1")
    second = prepare()[0]
    assert first.options.native_effective == second.options.native_effective
    assert not first.options.python_policy["dual_cta"]
    assert second.options.python_policy["dual_cta"]
    assert first.compute_hash() != second.compute_hash()


def test_trace_unused_format_and_resource_capacity_do_not_change_hash(monkeypatch):
    baseline = prepare(dtype="auto")[0].compute_hash()
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_E4M3_G6_P256_BEGIN", "9999")
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_TRACE", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE", "0")
    assert prepare(dtype="auto")[0].compute_hash() == baseline


def test_old_scalar_alias_has_one_canonical_field(monkeypatch):
    monkeypatch.delenv("VLLM_FLASH_V100_E4M3_SCALAR_FAST", raising=False)
    monkeypatch.setenv("VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST", "0")
    old = prepare()[0]
    assert not old.options.python_policy["scalar_fast"]
    monkeypatch.setenv("VLLM_FLASH_V100_E4M3_SCALAR_FAST", "1")
    main = prepare()[0]
    assert main.options.python_policy["scalar_fast"]
    typed = prepare(FlashV100Options(e4m3_scalar_fast=False))[0]
    assert not typed.options.python_policy["scalar_fast"]


def test_qualified_error_is_bound_without_changing_short_circuit(monkeypatch):
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "bad")
    policy, _, _ = prepare(FlashV100Options(xqa_mtp5_dual_cta=False))
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "512")
    assert policy.options.xqa_mtp5_dual_cta is False
    with pytest.raises(ValueError, match="got 'bad'") as error:
        policy.options.value("xqa_mtp5_partition_size")
    assert isinstance(error.value.__cause__, ValueError)


def test_trace_alias_priority_and_typed_override(monkeypatch):
    monkeypatch.setenv("VLLM_FLASH_V100_ROUTE_SUMMARY", "1")
    monkeypatch.setenv("VLLM_SM70_DEBUG", "")
    trace = FlashV100Diagnostics()
    trace.resolve()
    assert not trace.route_summary
    assert trace.sources["route_summary"] == "VLLM_SM70_DEBUG"
    typed = FlashV100Diagnostics(route_summary=True)
    typed.resolve()
    assert typed.route_summary
    assert typed.sources["route_summary"] == "typed"


def test_attention_observations_are_owned_by_each_engine():
    def engine():
        return SimpleNamespace(
            observability_config=SimpleNamespace(runtime_trace=RuntimeTraceConfig())
        )

    first, second = diagnostics_for(engine()), diagnostics_for(engine())
    first.histories.setdefault("flash_v100_routes", {})["decode"] = 7
    assert second.histories.get("flash_v100_routes", {}) == {}


def test_prefill_audit_reads_resolved_engine_owner_without_mutation(monkeypatch):
    from benchmarks.sm70_dflash2_prefill_route_audit import prefill_observations

    engines = []
    for v37 in (False, True):
        policy, _, trace = prepare(
            FlashV100Options(prefill_d256_gqa_v37=v37),
            trace=FlashV100Diagnostics(route_summary=True),
        )
        config = SimpleNamespace(
            attention_config=SimpleNamespace(flash_v100=policy),
            observability_config=SimpleNamespace(
                runtime_trace=RuntimeTraceConfig(flash_v100=trace)
            ),
        )
        owner = diagnostics_for(config)
        owner.histories["flash_v100_routes"] = {"prefill": int(v37) + 7}
        engines.append((config, owner))
    # Neither changed ambient flags nor the legacy global can replace the
    # worker's selected policy and counters outside a forward context.
    monkeypatch.setattr("os.getenv", lambda *a: pytest.fail("audit read environment"))
    for config, owner in engines:
        before = {name: dict(counts) for name, counts in owner.histories.items()}
        snapshot = prefill_observations(config)
        assert snapshot["routes"] == owner.histories["flash_v100_routes"]
        assert snapshot["route_counting_enabled"]
        assert snapshot["flags"]["VLLM_FLASH_V100_PREFILL_D256_GQA_V37"] == str(
            int(config.attention_config.flash_v100.options.prefill_d256_gqa_v37)
        )
        snapshot["routes"]["prefill"] = -1
        assert owner.histories == before
    config, owner = engines[0]
    owner.histories.clear()
    assert prefill_observations(config)["routes"] == {}
    assert owner.histories == {}


def test_bound_report_never_evaluates_legacy_getters(monkeypatch):
    policy, _, _ = prepare()
    expected = policy.explain()
    monkeypatch.setattr("os.getenv", lambda *a: pytest.fail("report read environment"))
    assert policy.explain() == expected


def test_smallq_metadata_uses_initialized_policy_after_environment_change(monkeypatch):
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.flash_v100.spec import verify_metadata

    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "5")
    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN", "8192")
    first = prepare()[0]
    second = FlashV100Policy(
        smallq_max_q=16, options=FlashV100Options(smallq_decode_max_model_len=0)
    )
    second.resolve()
    monkeypatch.setattr(
        "os.getenv", lambda *a: pytest.fail("metadata read environment")
    )
    for policy, expected in ((first, (5, 8192)), (second, (16, 0))):
        cfg = SimpleNamespace(attention_config=SimpleNamespace(flash_v100=policy))
        with set_current_vllm_config(cfg):
            assert (
                verify_metadata.configured_smallq_max_query_len(None),
                verify_metadata.configured_smallq_max_model_len(None),
            ) == expected


def test_attention_workspace_shutdown_keeps_other_engine_and_legacy(monkeypatch):
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.runtime_resources import release_runtime_resources
    from vllm.v1.attention.backends.flash_v100 import dense_prefill
    from vllm.v1.attention.ops.sm70_workspaces import workspace_cache

    first, second = SimpleNamespace(), SimpleNamespace()
    key = torch.zeros((1, 256, 1, 8), dtype=torch.uint8)
    legacy: dict = {}
    with set_current_vllm_config(first):
        a = dense_prefill.get_fp8_prefill_bridge_workspace(key, 1)
        workspace_cache("grouped_fp16", legacy)["tensor"] = torch.ones(1)
    with set_current_vllm_config(second):
        b = dense_prefill.get_fp8_prefill_bridge_workspace(key, 1)
        bank = workspace_cache("grouped_fp16", legacy)
        assert bank == {}
        bank["tensor"] = torch.zeros(1)
    assert a[0].data_ptr() != b[0].data_ptr()
    with set_current_vllm_config(first):
        reused = dense_prefill.get_fp8_prefill_bridge_workspace(key, 1)
        assert reused[0].data_ptr() == a[0].data_ptr()
        grown = dense_prefill.get_fp8_prefill_bridge_workspace(key, 3)
        assert grown[0].shape[0] == 3
    dense_prefill.clear_flash_attn_v100_workspaces(first)
    release_runtime_resources(first)
    with set_current_vllm_config(second):
        reused = dense_prefill.get_fp8_prefill_bridge_workspace(key, 1)
        assert reused[0].data_ptr() == b[0].data_ptr()
        assert workspace_cache("grouped_fp16", legacy) is bank
    assert legacy == {}


def test_bridge_scale_planes_share_payload_capacity_and_engine_owner():
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.runtime_resources import release_runtime_resources
    from vllm.v1.attention.backends.flash_v100 import dense_prefill

    first, second = SimpleNamespace(), SimpleNamespace()
    key = torch.zeros((1, 256, 2, 8), dtype=torch.uint8)
    with set_current_vllm_config(first):
        payload = dense_prefill.get_fp8_prefill_bridge_workspace(key, 2)
        scaled = dense_prefill.get_fp8_prefill_bridge_workspace(
            key.to(torch.int8), 2, preserve_scales=True
        )
        assert len(payload) == 3 and len(scaled) == 5
        assert payload[0].data_ptr() == scaled[0].data_ptr()
        assert all(t.dtype == torch.float32 for t in scaled[3:])
        assert all(t.shape == scaled[0].shape[:-1] for t in scaled[3:])
        scaled[3].fill_(0.125)
        reused = dense_prefill.get_fp8_prefill_bridge_workspace(
            key, 1, preserve_scales=True
        )
        assert reused[3].data_ptr() == scaled[3].data_ptr()
        assert torch.equal(reused[3], torch.full_like(reused[3], 0.125))
        grown = dense_prefill.get_fp8_prefill_bridge_workspace(
            key, 3, preserve_scales=True
        )
        assert all(t.shape[0] == 3 for t in (grown[0], *grown[3:]))
    with set_current_vllm_config(second):
        other = dense_prefill.get_fp8_prefill_bridge_workspace(
            key, 1, preserve_scales=True
        )
        assert other[3].data_ptr() != grown[3].data_ptr()
    release_runtime_resources(first)
    with set_current_vllm_config(second):
        reused = dense_prefill.get_fp8_prefill_bridge_workspace(
            key, 1, preserve_scales=True
        )
        assert reused[3].data_ptr() == other[3].data_ptr()
    release_runtime_resources(second)


def test_bridge_scale_allocation_failure_or_capture_keeps_existing_payload(monkeypatch):
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.runtime_resources import release_runtime_resources
    from vllm.v1.attention.backends.flash_v100 import dense_prefill

    owner = SimpleNamespace()
    key = torch.zeros((1, 256, 1, 8), dtype=torch.uint8)
    with set_current_vllm_config(owner):
        payload = dense_prefill.get_fp8_prefill_bridge_workspace(key, 1)
        monkeypatch.setattr(
            dense_prefill._workspace, "allocate_growing_workspace", lambda *a, **k: None
        )
        assert (
            dense_prefill.get_fp8_prefill_bridge_workspace(key, 1, preserve_scales=True)
            is None
        )
        monkeypatch.setattr(
            dense_prefill._routing, "is_cuda_graph_capturing", lambda t: True
        )
        assert (
            dense_prefill.get_fp8_prefill_bridge_workspace(key, 1, preserve_scales=True)
            is None
        )
        reused = dense_prefill.get_fp8_prefill_bridge_workspace(key, 1)
        assert payload[0].data_ptr() == reused[0].data_ptr()
    release_runtime_resources(owner)


def test_attention_diagnostic_budgets_and_output_do_not_cross_engines(tmp_path):
    import json

    from vllm.config import set_current_vllm_config
    from vllm.diagnostics import diagnostic_engine_tag, write_json_payload
    from vllm.v1.attention.backends.flash_v100 import config

    paths = []
    for _ in range(2):
        engine = SimpleNamespace(
            observability_config=SimpleNamespace(runtime_trace=RuntimeTraceConfig())
        )
        with set_current_vllm_config(engine):
            assert not config.diagnostic_seen("flash_v100.prefix_dump")
            config.mark_diagnostic("flash_v100.prefix_dump", True)
            assert config.diagnostic_seen("flash_v100.prefix_dump")
            paths.append(
                write_json_payload(
                    str(tmp_path),
                    "compare.json",
                    {"equal": True},
                    diagnostic_engine_tag(),
                )
            )
    assert paths[0] != paths[1]
    for path in paths:
        with open(path) as stream:
            assert json.load(stream) == {"equal": True}
