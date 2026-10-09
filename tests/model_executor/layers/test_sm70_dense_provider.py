# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.config.execution_policy import LayerExecutionPolicy
from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.config.sm70_native import NATIVE_FIELDS, Sm70NativeConfig
from vllm.config.sm70_runtime import RuntimeTraceConfig
from vllm.model_executor.kernels.linear import sm70_dense as provider

pytestmark = pytest.mark.skip_global_cleanup


def state(weight, **kwargs):
    return provider.DenseLinearState(
        weight,
        prefix="model.down_proj",
        policy=LayerExecutionPolicy(),
        trace=RuntimeTraceConfig(),
        **kwargs,
    )


@pytest.mark.parametrize("rows,expected", [(0, True), (1, True), (8, True), (9, False)])
@pytest.mark.parametrize("compiled", [False, True])
def test_dense_provider_keeps_row_bound_and_compiler_operator(
    monkeypatch, rows, expected, compiled
):
    weight = torch.ones((32, 16), dtype=torch.float16)
    prepared = state(weight, max_m=8)
    prepared._sm70_f16_prepared = True
    prepared.register_buffer("_sm70_f16_tm_weight", weight.clone(), persistent=False)
    prepared._sm70_f16_k_ld = 16
    monkeypatch.setattr(torch.ops._C, "sm70_f16_gemm", object(), raising=False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiled)

    def gemm_out(out, x, packed, ld, gated):
        assert ld == 16 and not gated
        out.copy_(x @ weight.t())

    packed = Mock(side_effect=gemm_out)
    original = Mock(side_effect=lambda x, w: x @ w.t())
    monkeypatch.setattr(provider.sm70_ops, "sm70_f16_gemm_out", packed)
    monkeypatch.setattr(provider.sm70_ops, "sm70_f16_gemm", original)
    x = torch.ones((rows, 16), dtype=torch.float16)
    bias = torch.arange(32, dtype=torch.float32)
    result = provider._maybe_sm70_dense_forward(prepared, x, bias)
    if not expected:
        assert result is None
        packed.assert_not_called()
        original.assert_not_called()
        return
    assert torch.equal(result, x @ weight.t() + bias)
    assert result.dtype == torch.float32
    assert packed.call_count == (not compiled)
    assert original.call_count == compiled


def test_glm_missing_native_retains_error_and_bias_fallback(monkeypatch):
    prepared = state(
        torch.empty((3336, 4096), dtype=torch.float16, device="meta"), glm_cublaslt=True
    )
    x = torch.empty((8, 4096), dtype=torch.float16, device="meta")
    # Missing op is an error for the admitted geometry, not a silent generic GEMM.
    monkeypatch.delattr(torch.ops._C, "sm70_glm53_tp8_cublaslt_out", raising=False)
    with pytest.raises(RuntimeError, match="requires its native op"):
        provider._maybe_sm70_dense_forward(prepared, x, None)
    assert (
        provider._maybe_sm70_dense_forward(
            prepared, x, torch.empty(3336, device="meta")
        )
        is None
    )


def native_value(policy, name):
    index = next(i for i, (field, *_rest) in enumerate(NATIVE_FIELDS) if field == name)
    return policy.native.values[index]


def test_native_dense_limit_has_one_effective_value(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_F16_DENSE_MAX_M", "19suffix")
    legacy = LayerExecutionPolicy()
    legacy.resolve()
    assert legacy.dense_max_m == 19
    assert native_value(legacy, "f16_dense_max_m") == "19"
    explicit = LayerExecutionPolicy(native=Sm70NativeConfig(f16_dense_max_m=7))
    explicit.resolve()
    assert explicit.dense_max_m == 7
    assert native_value(explicit, "f16_dense_max_m") == "7"
    conflict = LayerExecutionPolicy(
        dense_max_m=8, native=Sm70NativeConfig(f16_dense_max_m=7)
    )
    with pytest.raises(ValueError, match="Conflicting typed requests"):
        conflict.resolve()


def test_explicit_dflash_settings_reach_native_fp16_policy(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_DFLASH2_QPN8_RERANK", "0")
    dflash = Sm70DFlash2Config(qpn8_rerank=True)
    dflash.resolve(qualified=False)
    policy = LayerExecutionPolicy()
    policy.resolve(dflash=dflash)
    assert native_value(policy, "dflash2_qpn8_rerank") == "1"
    assert dflash.sources["qpn8_rerank"] == "typed"


def test_resolved_legacy_logger_cannot_change_with_environment(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_ENABLE_LM_HEAD_FASTPATH", "1")
    policy = LayerExecutionPolicy()
    policy.resolve()
    monkeypatch.setenv("VLLM_SM70_ENABLE_LM_HEAD_FASTPATH", "invalid")
    policy.resolve()
    assert policy.dense_log_enabled
    assert policy.dense_log_error is None


def test_legacy_logger_error_stays_deferred(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_ENABLE_LM_HEAD_FASTPATH", "true")
    policy = LayerExecutionPolicy()
    policy.resolve()
    assert policy.lm_head_dense
    assert policy.dense_log_error is not None
