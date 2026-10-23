# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
from unittest.mock import Mock

import pytest

from tests.config.runtime_policy_utils import make_policy_defaults
from vllm import envs
from vllm.config import SchedulerConfig, set_current_vllm_config
from vllm.config.execution_policy import communication_policy, layer_policy
from vllm.config.sm70_sparse import sparse_policy
from vllm.models.qwen4_exp.nvidia.model import _validate_qsa_e4m3_scale_load


@pytest.mark.parametrize("order", [(False, True), (True, False)])
def test_remaining_provider_gates_keep_their_engine_inputs(monkeypatch, order):
    owners = []
    for enabled in order:
        defaults = make_policy_defaults()
        cfg = defaults.cfg
        cfg.kernel_config.layer_execution.mhc_fp32_stage = enabled
        cfg.kernel_config.layer_execution.ple_spec_conv = enabled
        cfg.kernel_config.sm70_sparse.qsa_strict_scales = enabled
        cfg.kernel_config.sm70_sparse.private_compressor_state = enabled
        cfg.kernel_config.sm70_sparse.qnorm_kv_fused_tp4 = enabled
        cfg.parallel_config.communication.pp_static_hidden_transfer = enabled
        cfg.parallel_config.communication.moe_sum2_q8 = enabled
        defaults.finish()
        owners.append(cfg)
    for alias in (
        "VLLM_SM70_DSV4_MHC_FP32_STAGE",
        "VLLM_SM70_MTP_PLE_CONV",
        "VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES",
        "VLLM_SM70_DSV4_PRIVATE_COMPRESSOR_STATE",
        "VLLM_SM70_DSV4_QNORM_KV_FUSED_TP4",
        "VLLM_SM70_PP_STATIC_HIDDEN_TRANSFER",
        "VLLM_SM70_GLM53_MOE_SUM2_ALLREDUCE_Q8",
    ):
        monkeypatch.setenv(alias, "invalid-after-initialization")
        monkeypatch.setitem(
            envs.environment_variables, alias, Mock(side_effect=AssertionError(alias))
        )
    for cfg, enabled in list(zip(owners, order)) * 2:
        with set_current_vllm_config(cfg):
            assert layer_policy().value("mhc_fp32_stage") is enabled
            assert layer_policy().value("ple_spec_conv") is enabled
            assert sparse_policy().value("private_compressor_state") is enabled
            assert sparse_policy().value("qnorm_kv_fused_tp4") is enabled
            assert communication_policy().value("pp_static_hidden_transfer") is enabled
            assert communication_policy().value("moe_sum2_q8") is enabled
            if enabled:
                with pytest.raises(ValueError, match="refusing to start"):
                    _validate_qsa_e4m3_scale_load({"q.k_scale"}, set(), "fp8")
            else:
                assert _validate_qsa_e4m3_scale_load({"q.k_scale"}, set(), "fp8") == {
                    "q.k_scale"
                }
            assert _validate_qsa_e4m3_scale_load({"q.k_scale"}, set(), "auto") == set()


@pytest.mark.parametrize("raw,expected", [(None, 0), ("0", 0), ("-1", 0), ("4", 4)])
def test_queue_snapshot_worker_transfer_and_hash(monkeypatch, raw, expected):
    alias = "VLLM_SM70_ASYNC_SCHEDULING_QUEUE_DEPTH"
    if raw is None:
        monkeypatch.delenv(alias, raising=False)
    else:
        monkeypatch.setenv(alias, raw)
    scheduler = SchedulerConfig.default_factory()
    receiver, sender = multiprocessing.Pipe(duplex=False)
    try:
        sender.send(scheduler)
        worker = receiver.recv()
    finally:
        sender.close()
        receiver.close()
    monkeypatch.setenv(alias, "not-an-integer")
    assert worker.sm70_queue_depth() == expected
    typed = SchedulerConfig.default_factory(sm70_async_queue_depth=7)
    assert typed.sm70_queue_depth() == 7
    assert typed.compute_hash() == worker.compute_hash()
    invalid = SchedulerConfig.default_factory()
    with pytest.raises(ValueError):
        invalid.sm70_queue_depth()


def test_output_repair_freezes_errors_and_typed_precedence(monkeypatch):
    from vllm.config.sm70_runtime import Sm70RuntimeConfig, bind_output_token_repair

    alias = "VLLM_SM70_MTP_LEGACY_OUTPUT_TOKEN_REPAIR"
    monkeypatch.setenv(alias, "1")
    captured = bind_output_token_repair(Sm70RuntimeConfig())
    standalone = bind_output_token_repair()
    monkeypatch.setenv(alias, "bad")
    invalid = bind_output_token_repair(Sm70RuntimeConfig())
    typed = bind_output_token_repair(
        Sm70RuntimeConfig(legacy_output_token_repair=False)
    )
    assert captured() and standalone()
    assert not typed()
    with pytest.raises(ValueError):
        invalid()


def test_retained_gdn_notices_are_diagnostic_snapshots(monkeypatch):
    from vllm.config.sm70_runtime import RuntimeTraceConfig

    monkeypatch.setenv("VLLM_QWEN3_NEXT_FUSED_SIGMOID_GATING", "bad-but-presence-only")
    monkeypatch.setenv("VLLM_SM70_GDN_EMPTY_CORE_OUT", "1")
    first = RuntimeTraceConfig()
    monkeypatch.delenv("VLLM_QWEN3_NEXT_FUSED_SIGMOID_GATING")
    monkeypatch.setenv("VLLM_SM70_GDN_EMPTY_CORE_OUT", "bad")
    second = RuntimeTraceConfig()
    assert first.gdn_legacy_fused_notice and first.value("gdn_empty_output_notice")
    assert not second.gdn_legacy_fused_notice
    with pytest.raises(ValueError):
        second.value("gdn_empty_output_notice")
