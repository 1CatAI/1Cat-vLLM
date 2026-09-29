# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle

import pytest
import torch
from torch._higher_order_ops import triton_kernel_wrap as wrap

import vllm.envs as envs
from vllm.compilation.backends import _compute_backend_code_hash
from vllm.compilation.caching import VllmSerializableFunction
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.triton_utils import tl, triton

pytest.importorskip("triton")


@triton.jit
def _copy_kernel(x, y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(y + offsets, tl.load(x + offsets))


@triton.jit
def _other_kernel(x, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(x + offsets, 0)


def _graph(wrapped=False):
    kernel = _copy_kernel
    if wrapped:
        kernel = triton.autotune(
            configs=[triton.Config({}, num_warps=8, num_stages=3)], key=[]
        )(kernel)
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.call_function(
        wrap.triton_kernel_wrapper_mutation,
        kwargs={
            "kernel_idx": wrap.kernel_side_table.add_kernel(kernel),
            "constant_args_idx": wrap.kernel_side_table.add_constant_args(
                {"BLOCK": 16}
            ),
            "grid": [(1, 1, 1)],
            "kwargs": {"x": x, "y": x},
            "tma_descriptor_metadata": {},
        },
    )
    graph.output(x)
    return torch.fx.GraphModule({}, graph)


@pytest.fixture
def side_table(monkeypatch):
    table = wrap.KernelSideTable()
    table.reset_table()
    monkeypatch.setattr(wrap, "kernel_side_table", table)
    monkeypatch.setattr(envs, "VLLM_USE_MEGA_AOT_ARTIFACT", False)
    return table


def _save(graph):
    fn = VllmSerializableFunction(graph, [torch.zeros(16)], "", lambda x: x)
    return fn.serialize_compile_artifacts(fn)


def _load(data):
    with set_current_vllm_config(VllmConfig()):
        return VllmSerializableFunction.deserialize_compile_artifacts(data)


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_triton_graph_reload(side_table, populated, wrapped):
    graph = _graph(wrapped)
    # An unrelated, unimportable entry must not prevent saving this graph.
    side_table.add_kernel(object())
    data = _save(graph)
    side_table.reset_table()
    if populated:
        other_id = side_table.add_kernel(_other_kernel)
        other_args = side_table.add_constant_args({"BLOCK": 32})
    loaded = _load(data)
    node = next(n for n in loaded.graph_module.graph.nodes if n.op == "call_function")
    kernel = side_table.get_kernel(node.kwargs["kernel_idx"])
    if wrapped:
        assert kernel.fn is _copy_kernel
        assert kernel.configs[0].num_warps == 8
        assert kernel.configs[0].num_stages == 3
    else:
        assert kernel is _copy_kernel
    assert side_table.get_constant_args(node.kwargs["constant_args_idx"]) == {
        "BLOCK": 16
    }
    if populated:
        assert side_table.get_kernel(other_id) is _other_kernel
        assert side_table.get_constant_args(other_args) == {"BLOCK": 32}
        assert node.kwargs["kernel_idx"] != other_id
        assert node.kwargs["constant_args_idx"] != other_args
    # Saving again must record the remapped IDs, not the original process IDs.
    again = loaded.serialize_compile_artifacts(loaded)
    side_table.reset_table()
    reloaded = _load(again)
    node = next(n for n in reloaded.graph_module.graph.nodes if n.op == "call_function")
    kernel = side_table.get_kernel(node.kwargs["kernel_idx"])
    if wrapped:
        assert kernel.fn is _copy_kernel
        assert kernel.configs[0].num_warps == 8
    else:
        assert kernel is _copy_kernel


def test_legacy_triton_artifact_requires_regeneration(side_table):
    state = pickle.loads(_save(_graph()))
    del state["triton_side_table"]
    side_table.reset_table()
    with pytest.raises(RuntimeError, match="no Triton side table"):
        _load(pickle.dumps(state))


def test_legacy_graph_without_triton_still_loads(side_table):
    graph = torch.fx.symbolic_trace(torch.nn.ReLU())
    state = pickle.loads(_save(graph))
    del state["triton_side_table"]
    assert isinstance(_load(pickle.dumps(state)).graph_module, torch.fx.GraphModule)


def test_frozen_sources_keep_backend_cache_key_stable(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("def forward(x): return x + 1\n")
    key = _compute_backend_code_hash([str(source)])
    assert key == _compute_backend_code_hash(["<frozen os>", str(source), "<string>"])
    source.write_text("def forward(x): return x + 2\n")
    assert key != _compute_backend_code_hash([str(source)])
