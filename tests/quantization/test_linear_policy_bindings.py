# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers import linear


def test_non_linear_base_binds_policy_before_provider_return(monkeypatch):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(2, 2))
    policy, trace = object(), object()
    monkeypatch.setattr(linear, "layer_policy", lambda: policy)
    monkeypatch.setattr(linear, "capture_runtime_trace", lambda: trace)
    monkeypatch.setattr(linear.current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(linear.current_platform, "is_cuda_alike", lambda: False)
    method = linear.UnquantizedLinearMethod()
    method.process_weights_after_loading(layer)
    assert layer._layer_execution_policy is policy
    assert layer._layer_trace is trace

    # A later preparation context must not rebind an existing layer's owner.
    monkeypatch.setattr(linear, "layer_policy", lambda: object())
    monkeypatch.setattr(linear, "capture_runtime_trace", lambda: object())
    method.process_weights_after_loading(layer)
    assert layer._layer_execution_policy is policy
    assert layer._layer_trace is trace
