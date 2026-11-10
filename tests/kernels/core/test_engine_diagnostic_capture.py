# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.diagnostic_dump import TensorDiagnosticsConfig, TensorDumpConfig
from vllm.config.sm70_runtime import RuntimeTraceConfig
from vllm.diagnostics import diagnostics_for
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.runtime_resources import runtime_resources_for

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_context(tmp_path):
    cfg = SimpleNamespace(
        observability_config=SimpleNamespace(
            runtime_trace=RuntimeTraceConfig(
                dumps=TensorDiagnosticsConfig(
                    qwen_layer=TensorDumpConfig(
                        directory=str(tmp_path), capture=True, direct_save=False
                    ),
                    gdn_graph=TensorDumpConfig(directory=str(tmp_path), capture=True),
                )
            )
        )
    )
    owner = diagnostics_for(cfg)
    return owner, ForwardContext(
        {}, {}, {}, runtime_resources=runtime_resources_for(cfg)
    )


def test_two_engine_capture_replay_storage_and_growth(monkeypatch, tmp_path):
    from vllm.model_executor.layers.fla.ops.gdn_diagnostics import capture_tensor
    from vllm.model_executor.models import qwen3_next  # noqa: F401

    owners_contexts = [make_context(tmp_path) for _ in range(2)]
    inputs, outputs, graphs = [], [], []
    for index, (owner, context) in enumerate(owners_contexts):
        x = torch.full((8, 32), float(index + 1), device="cuda", dtype=torch.float16)
        inputs.append(x)
        with override_forward_context(context):
            torch.ops.vllm.sm70_qwen_layer_dump(x, "synthetic", 0, "linear_attention")
            capture_tensor("synthetic", "model.layers.0.linear_attn", x, "core")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = torch.ops.vllm.sm70_qwen_layer_dump(
                    x, "synthetic", 0, "linear_attention"
                )
                capture_tensor("synthetic", "model.layers.0.linear_attn", x, "core")
        graphs.append(graph)
        outputs.append(out)
    monkeypatch.setenv("VLLM_SM70_DUMP_QWEN_LAYER_GRAPH_BUFFERS", "invalid-after-init")
    monkeypatch.setenv("VLLM_SM70_DUMP_GDN_GRAPH_DIR", "/unusable-after-init")
    for value in (3.0, -2.0, 0.0):
        for index, (owner, _) in enumerate(owners_contexts):
            inputs[index].fill_(value + index)
            graphs[index].replay()
            assert torch.equal(outputs[index], inputs[index])
            for channel in ("qwen_layer", "gdn_graph"):
                buffer = next(iter(owner.channels[channel].buffers.values()))
                assert torch.equal(buffer, inputs[index])
    for channel in ("qwen_layer", "gdn_graph"):
        first = next(iter(owners_contexts[0][0].channels[channel].buffers.values()))
        second = next(iter(owners_contexts[1][0].channels[channel].buffers.values()))
        assert first.data_ptr() != second.data_ptr()
    # Replacing an observation allocation must retain earlier captured addresses.
    channel = owners_contexts[0][0].channels["gdn_graph"]
    key = next(iter(channel.buffers))
    original = channel.buffers[key]
    channel.capture(key, torch.ones(16, 32, device="cuda", dtype=torch.float16), {})
    assert channel.retired_buffers[0] is original
    inputs[0].fill_(9.0)
    graphs[0].replay()
    assert torch.equal(original, inputs[0])
    # Same output directory and observation names still produce separate files.
    for owner, _ in owners_contexts:
        owner.channels["qwen_layer"].flush_graph(1, "synthetic")
    assert len(list(tmp_path.glob("*.pt"))) == 2


def test_engine_owned_dump_preserves_aot_values(tmp_path):
    from vllm.model_executor.models import qwen3_next  # noqa: F401

    _, context = make_context(tmp_path)

    def forward(x):
        y = torch.ops.vllm.sm70_qwen_layer_dump(x, "aot", 0, "linear_attention")
        return y + x

    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
    with override_forward_context(context):
        for size in (8, 0, 16):
            x = torch.randn(size, 32, device="cuda", dtype=torch.float16)
            assert torch.equal(compiled(x), x + x)
