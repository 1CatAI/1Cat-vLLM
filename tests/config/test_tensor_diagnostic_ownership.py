# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.diagnostic_dump import TensorDiagnosticsConfig, TensorDumpConfig
from vllm.config.sm70_runtime import RuntimeTraceConfig
from vllm.diagnostics import diagnostic_channel, diagnostics_for


def config_for(dumps):
    return SimpleNamespace(
        observability_config=SimpleNamespace(
            runtime_trace=RuntimeTraceConfig(dumps=dumps)
        )
    )


@pytest.mark.parametrize("order", [(1, 3), (3, 1)])
def test_budget_source_worker_transfer_and_environment_isolation(
    monkeypatch, tmp_path, order
):
    configs = []
    for budget in order:
        monkeypatch.setenv("VLLM_SM70_DUMP_QWEN_LAYER_MAX_DUMPS", str(budget))
        monkeypatch.setenv("VLLM_SM70_DUMP_QWEN_LAYER_DIR", str(tmp_path))
        cfg = config_for(TensorDiagnosticsConfig())
        receiver, sender = multiprocessing.Pipe(duplex=False)
        try:
            thread = threading.Thread(target=sender.send, args=(cfg,), daemon=True)
            thread.start()
            assert receiver.poll(10), "configuration transfer timed out"
            configs.append(receiver.recv())
            thread.join(10)
            assert not thread.is_alive()
        finally:
            sender.close()
            receiver.close()
    monkeypatch.setenv("VLLM_SM70_DUMP_QWEN_LAYER_MAX_DUMPS", "invalid-after-init")
    owners = [diagnostics_for(cfg) for cfg in configs]
    for owner, budget in zip(owners, order):
        channel = diagnostic_channel("qwen_layer", owner=owner)
        assert channel.policy.max_dumps == budget
        assert (
            channel.policy.sources["max_dumps"] == "VLLM_SM70_DUMP_QWEN_LAYER_MAX_DUMPS"
        )
        for _ in range(budget):
            assert channel.take("same_layer", budget)
        assert not channel.take("same_layer", budget)
    assert (
        owners[0].channels["qwen_layer"].saves
        is not owners[1].channels["qwen_layer"].saves
    )


def test_trigger_exists_check_remains_dynamic_but_path_is_fixed(monkeypatch, tmp_path):
    trigger = tmp_path / "enable"
    policy = TensorDumpConfig(directory=str(tmp_path), enable_file=str(trigger))
    cfg = config_for(TensorDiagnosticsConfig(qwen_layer=policy))
    channel = diagnostics_for(cfg).channels["qwen_layer"]
    assert not channel.policy.can_save()
    trigger.touch()
    assert channel.policy.can_save()
    monkeypatch.setenv(
        "VLLM_SM70_DUMP_QWEN_LAYER_ENABLE_FILE", str(tmp_path / "missing")
    )
    assert channel.policy.can_save()
    trigger.unlink()
    assert not channel.policy.can_save()


@pytest.mark.parametrize(
    "channel,raw,selected",
    [
        ("qwen_layer", "3-1", {1, 2, 3}),
        ("qwen_layer", "bad", {0, 1}),
        ("qwen_layer", "all", None),
        ("gdn_projection", "3-1", {0, 1}),
        ("gdn_projection", "", set()),
        ("gdn_graph", "bad", set()),
        ("gdn_core", "1-3", set()),
        ("gdn_compare", "bad", {0}),
        ("gdn_compare", ",", set()),
        ("gdn_compare", " ", {0}),
    ],
)
def test_original_filter_dialects_share_parser_without_changing_semantics(
    channel, raw, selected
):
    policy = TensorDumpConfig(layers=raw)
    policy.resolve(channel)
    assert policy.filters["layers"] == selected


def test_capture_owner_retains_old_addresses_and_updates_metadata(tmp_path):
    cfg = config_for(TensorDiagnosticsConfig())
    first = diagnostics_for(cfg).channels["gdn_graph"]
    second = diagnostics_for(config_for(TensorDiagnosticsConfig())).channels[
        "gdn_graph"
    ]
    x = torch.arange(8)
    first.capture("same_key", x, {"step": 1}, refresh=True)
    old = first.buffers["same_key"]
    first.capture("same_key", x + 2, {"step": 2}, refresh=True)
    assert first.buffers["same_key"] is old
    assert torch.equal(old, x + 2)
    assert first.metadata["same_key"]["step"] == 2
    first.capture("same_key", torch.ones(16), {"step": 3})
    assert any(tensor is old for tensor in first.retired_buffers)
    assert not second.buffers


def test_disabled_gdn_diagnostics_do_not_query_cuda_or_allocate(monkeypatch):
    from vllm.model_executor.layers.fla.ops import gdn_diagnostics

    owner = diagnostics_for(config_for(TensorDiagnosticsConfig()))
    monkeypatch.setattr(
        gdn_diagnostics, "diagnostic_channel", lambda name: owner.channels[name]
    )
    fail = Mock(side_effect=AssertionError("disabled diagnostics touched CUDA"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", fail)
    monkeypatch.setattr(torch, "empty_like", fail)
    x = torch.empty(4)
    gdn_diagnostics.dump_core("test", "model.layers.0", x)
    gdn_diagnostics.capture_tensor("test", "model.layers.0", x, "core")
    assert not gdn_diagnostics.projection_requested("model.layers.0")
    fail.assert_not_called()


def test_two_engines_using_same_directory_do_not_overwrite(tmp_path):
    owners = [
        diagnostics_for(
            config_for(
                TensorDiagnosticsConfig(
                    qwen_layer=TensorDumpConfig(directory=str(tmp_path))
                )
            )
        )
        for _ in range(2)
    ]
    paths = [
        owner.channels["qwen_layer"].write("same_001.pt", {"value": i})
        for i, owner in enumerate(owners)
    ]
    assert paths[0] != paths[1]
    assert [torch.load(path, weights_only=True)["value"] for path in paths] == [0, 1]


def test_family_override_keeps_legacy_parser_short_circuit(monkeypatch):
    from vllm.config.sm70_dflash2 import proposer_diagnostic_flags

    monkeypatch.setenv("VLLM_DFLASH_DEBUG_CORRUPTION", "1")
    monkeypatch.setenv("VLLM_SPEC_DEBUG_CORRUPTION", "invalid-unused")
    trace = RuntimeTraceConfig()
    assert proposer_diagnostic_flags("dflash", trace)[0]
    with pytest.raises(ValueError, match="invalid literal"):
        proposer_diagnostic_flags("mtp", trace)


def test_disabled_family_profile_does_not_parse_an_unused_interval(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH_PROFILE", "0")
    monkeypatch.setenv("VLLM_DFLASH_PROFILE_LOG_INTERVAL", "invalid-unused")
    trace = RuntimeTraceConfig()
    assert not trace.dflash.value("profile")
    with pytest.raises(ValueError, match="invalid literal"):
        trace.dflash.value("interval")


def test_moe_legacy_typed_bridge_and_override_precedence(monkeypatch, tmp_path):
    from vllm.config.kernel import KernelConfig
    from vllm.config.sm70_moe import bind_moe_diagnostics

    monkeypatch.setenv("VLLM_SM70_DUMP_AWQ_MOE_BUFFERS", "1")
    monkeypatch.setenv("VLLM_SM70_DUMP_QWEN_LAYER_IDS", "4-2")
    kernel = KernelConfig()
    kernel.sm70_moe.awq.diagnostics.dump_dir = str(tmp_path / "legacy")
    kernel.sm70_moe.awq.diagnostics.dump_layers = "7"
    trace = RuntimeTraceConfig(
        dumps=TensorDiagnosticsConfig(
            qwen_layer=TensorDumpConfig(directory=str(tmp_path / "explicit"))
        )
    )
    bind_moe_diagnostics(kernel, trace)
    assert trace.dumps.qwen_layer.directory == str(tmp_path / "explicit")
    assert trace.dumps.awq_buffers.directory == str(tmp_path / "explicit")
    assert trace.dumps.awq_buffers.filters["layers"] == {7}
    assert trace.dumps.awq_buffers.enabled
    assert kernel.sm70_moe.awq.diagnostics.compare_policy is trace.dumps.awq_compare
    monkeypatch.setenv("VLLM_SM70_DUMP_QWEN_LAYER_IDS", "invalid")
    kernel.sm70_moe.awq.resolve("awq")
    assert kernel.sm70_moe.awq.diagnostics.dump_policy.filters["layers"] == {7}


def test_moe_unused_compare_parse_failure_stays_qualified(monkeypatch):
    from vllm.config.kernel import KernelConfig
    from vllm.config.sm70_moe import bind_moe_diagnostics

    monkeypatch.setenv("VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_COMPARE", "bad")
    kernel, trace = KernelConfig(), RuntimeTraceConfig()
    bind_moe_diagnostics(kernel, trace)
    kernel.sm70_moe.awq.resolve("awq")
    with pytest.raises(ValueError, match="invalid literal"):
        kernel.sm70_moe.fp8.resolve("fp8")
    other = KernelConfig()
    other.sm70_moe.fp8.diagnostics.compact_compare = False
    bind_moe_diagnostics(other, RuntimeTraceConfig())
    other.sm70_moe.fp8.resolve("fp8")


@pytest.mark.parametrize(
    "raw, expected", [(None, None), ("", set()), ("1-3", {1, 2, 3})]
)
def test_awq_compare_preserves_empty_filter(raw, expected):
    policy = TensorDumpConfig()
    policy.resolve("awq_compare")
    policy.layers = policy.steps = raw
    policy.parse_filters("awq_compare")
    assert policy.filters["layers"] == expected
    assert policy.filters["steps"] == expected


def test_sampler_dump_budgets_are_independent(monkeypatch, tmp_path):
    from vllm.v1.sample.sampler import _maybe_dump_sm70_sampler_logits

    cfgs = [
        config_for(
            TensorDiagnosticsConfig(
                sampler_logits=TensorDumpConfig(
                    directory=str(tmp_path), max_dumps=limit
                )
            )
        )
        for limit in (1, 2)
    ]
    owners = [diagnostics_for(cfg) for cfg in cfgs]
    metadata = SimpleNamespace(
        temperature=None,
        top_k=None,
        top_p=None,
        all_greedy=True,
        all_random=False,
        max_num_logprobs=None,
        output_token_ids=[],
    )
    monkeypatch.setenv("VLLM_SM70_DUMP_SAMPLER_LOGITS_MAX_STEPS", "bad-after-init")
    for _ in range(3):
        for owner in owners:
            _maybe_dump_sm70_sampler_logits(
                torch.ones(1, 8), metadata, "pre_process", diagnostics=owner
            )
    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 3
    assert sorted(torch.load(path, weights_only=True)["step"] for path in files) == [
        1,
        1,
        2,
    ]


def test_gdn_empty_step_filter_and_shape_selection_keep_distinct_semantics():
    compare = TensorDumpConfig(steps=",")
    compare.resolve("gdn_compare")
    assert compare.allows("steps", 1)
    graph = TensorDumpConfig(shapes=",")
    graph.resolve("gdn_graph")
    assert not graph.allows("shapes", "8x128")


@pytest.mark.parametrize("order", [(1, 2), (2, 1)])
def test_logits_diagnostic_budget_and_probe_order_are_engine_local(tmp_path, order):
    from vllm.model_executor.layers.logits_processor import _top_token_margin_dump_step

    modules = []
    for budget in order:
        dumps = TensorDiagnosticsConfig(
            top_token_margin=TensorDumpConfig(
                directory=str(tmp_path), steps="0-4", max_dumps=budget, probes="4,2,4"
            )
        )
        owner = diagnostics_for(config_for(dumps))
        modules.append(SimpleNamespace(_diagnostics=owner))
        assert owner.channels["top_token_margin"].policy.parsed("probes") == (4, 2, 4)
    for module, budget in zip(modules, order):
        assert [_top_token_margin_dump_step(module) for _ in range(3)] == (
            list(range(budget)) + [None] * (3 - budget)
        )


def test_logits_trigger_and_strict_error_checkpoint(monkeypatch, tmp_path):
    from vllm.model_executor.layers.logits_processor import _top_token_margin_dump_step

    trigger = tmp_path / "enabled"
    policy = TensorDumpConfig(
        directory=str(tmp_path), enable_file=str(trigger), steps="3-1"
    )
    owner = diagnostics_for(
        config_for(TensorDiagnosticsConfig(top_token_margin=policy))
    )
    module = SimpleNamespace(_diagnostics=owner)
    assert _top_token_margin_dump_step(module) is None
    monkeypatch.setenv("VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_STEPS", "0-9")
    trigger.touch()
    with pytest.raises(ValueError, match="invalid top-token margin step range: 3-1"):
        _top_token_margin_dump_step(module)


@pytest.mark.parametrize(
    "raw,integer,exact",
    [("0", False, False), ("1", True, True), ("2", True, False), ("", None, False)],
)
def test_coordinate_trace_keeps_consumer_dialects(monkeypatch, raw, integer, exact):
    monkeypatch.setenv("VLLM_DFLASH_DEBUG_COORD_TRACE", raw)
    policy = RuntimeTraceConfig().dflash
    monkeypatch.setenv("VLLM_DFLASH_DEBUG_COORD_TRACE", "wrong-after-init")
    assert policy.coord_exact_one is exact
    if integer is None:
        with pytest.raises(ValueError):
            policy.value("coord_integer")
    else:
        assert policy.value("coord_integer") is integer


def test_shared_dflash_dump_directory_and_separate_budgets(tmp_path):
    from vllm.config.sm70_dflash2 import DFlashDiagnosticsConfig

    cfg = DFlashDiagnosticsConfig(
        tensors=TensorDumpConfig(directory=str(tmp_path), max_dumps=1),
        pp_aux=TensorDumpConfig(max_dumps=2),
    )
    assert cfg.pp_aux.directory == str(tmp_path)
    assert cfg.pp_aux.sources["directory"] == "typed"
    engine = SimpleNamespace(
        observability_config=SimpleNamespace(
            runtime_trace=RuntimeTraceConfig(dflash=cfg)
        )
    )
    owner = diagnostics_for(engine)
    owner.channels["dflash_tensor"].reports = 1
    assert owner.channels["dflash_pp_aux"].reports == 0
