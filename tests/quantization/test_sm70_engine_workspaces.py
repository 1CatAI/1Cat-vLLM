# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
from types import SimpleNamespace

import pytest
import torch

from vllm import _sm70_ops
from vllm._sm70 import policy as bindings
from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.quantization.utils import sm70_layer_workspaces as ws

pytestmark = pytest.mark.cpu_test
_PREFIX = "model.layers.0.mlp.down_proj"


def _config():
    return SimpleNamespace(
        kernel_config=KernelConfig(),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def _forward(config):
    return override_forward_context(
        ForwardContext(config.compilation_config.static_forward_context, {}, {})
    )


def test_engine_pool_and_binding_ownership(monkeypatch):
    configs = [_config(), _config()]
    workspaces = [torch.ones(8), torch.zeros(16)]
    legacy: dict = {}
    pools = []
    for cfg, tensor in zip(configs, workspaces):
        with set_current_vllm_config(cfg):
            pool = ws.workspace_pool("unit", legacy)
            pool["same-device-and-shape"] = tensor
            pools.append(pool)
            ws.register_layer_workspace(SimpleNamespace(prefix=_PREFIX), tensor)
            ws.register_layer_workspace(SimpleNamespace(prefix=_PREFIX), tensor)
            with pytest.raises(RuntimeError, match="already bound"):
                ws.register_layer_workspace(
                    SimpleNamespace(prefix=_PREFIX), tensor.clone()
                )
    assert pools[0] is not pools[1]
    assert legacy == {}
    # The forward owner wins even if initialization's current config differs.
    with set_current_vllm_config(configs[0]), _forward(configs[1]):
        assert ws._workspace_binding(_PREFIX).workspace is workspaces[1]
        assert ws.workspace_pool("unit", legacy) is pools[1]
        ws.clear_layer_workspaces()
    with _forward(configs[0]):
        assert ws._workspace_binding(_PREFIX).workspace is workspaces[0]
    with _forward(configs[1]), pytest.raises(KeyError):
        ws._workspace_binding(_PREFIX)


def test_export_reload_uses_execution_engine_workspace_and_policy(monkeypatch):
    monkeypatch.setattr(bindings, "native_policy_abi_available", lambda: True)
    monkeypatch.setattr(torch.ops, "_C_qwen38", SimpleNamespace())
    configs = [_config(), _config()]
    workspaces = [torch.full((8,), 1.0), torch.full((8,), 2.0)]
    policies = []
    for i, (cfg, tensor) in enumerate(zip(configs, workspaces)):
        cfg.kernel_config.sm70_fp8.native.fp8_dense_tune_max_m = 8 * (i + 1)
        with set_current_vllm_config(cfg):
            ws.register_layer_workspace(
                SimpleNamespace(prefix=_PREFIX), tensor, family="fp8"
            )
            policies.append(ws._workspace_binding(_PREFIX).native.arguments)
    observed = []

    def native(out, address, *args, native_policy):
        current = ws._workspace_binding(_PREFIX)
        assert address == current.workspace.data_ptr()
        observed.append(native_policy)
        out.fill_(current.workspace[0].item())

    monkeypatch.setattr(_sm70_ops, "fp8_qpn8_dispatch_sm70_out", native)

    class Projection(torch.nn.Module):
        def forward(self, x):
            out = torch.empty_like(x)
            torch.ops.vllm.sm70_fp8_qpn8_dispatch(
                out, _PREFIX, x, x, x, 16, 2, False, False
            )
            return out

    lib = None
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "vllm::sm70_fp8_qpn8_dispatch", "CPU"
    ):
        lib = torch.library.Library("vllm", "IMPL", "CPU")
        lib.impl("sm70_fp8_qpn8_dispatch", ws._sm70_fp8_qpn8_dispatch)
    try:
        artifact = io.BytesIO()
        x = torch.zeros(2, 4)
        with _forward(configs[0]):
            exported = torch.export.export(Projection(), (x,))
            assert torch.equal(exported.module()(x), torch.ones_like(x))
            torch.export.save(exported, artifact)
        artifact.seek(0)
        with _forward(configs[1]):
            reloaded = torch.export.load(artifact).module()
            assert torch.equal(reloaded(x), torch.full_like(x, 2))
        assert observed == policies
    finally:
        if lib is not None:
            lib._destroy()


def test_raw_scale_expansion_is_shared_only_within_its_engine(monkeypatch):
    from vllm.model_executor.layers.fused_moe.sm70 import fp4_workspace

    monkeypatch.setattr(fp4_workspace, "_QWEN38_RAW_SCALE_WORKSPACE_ELEMENTS", 8)
    configs = [_config(), _config()]
    for cfg in configs:
        cfg.parallel_config = SimpleNamespace(use_ubatching=False)
    tensors = []
    for cfg in configs:
        with set_current_vllm_config(cfg):
            value = fp4_workspace._get_qwen38_raw_scale_workspace(torch.device("cpu:0"))
            tensors.append(value)
            assert (
                fp4_workspace._get_qwen38_raw_scale_workspace(torch.device("cpu:0"))
                is value
            )
    assert tensors[0] is not tensors[1]
    with set_current_vllm_config(configs[0]):
        fp4_workspace.clear_sm70_nvfp4_moe_workspaces()
    with set_current_vllm_config(configs[1]):
        assert (
            fp4_workspace._get_qwen38_raw_scale_workspace(torch.device("cpu:0"))
            is tensors[1]
        )
