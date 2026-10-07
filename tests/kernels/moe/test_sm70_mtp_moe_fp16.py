# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _sm70_mtp_moe_fp16_shape_supported,
    dispatch_fused_moe_kernel,
    invoke_fused_moe_triton_kernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl

CONFIG = dict(
    BLOCK_SIZE_M=2,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=64,
    GROUP_SIZE_M=1,
    SPLIT_K=1,
    num_warps=4,
    num_stages=3,
)


def test_default_enabled(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_MTP_MOE_FP16_EXACT", raising=False)
    envs.disable_envs_cache()
    assert envs.VLLM_SM70_MTP_MOE_FP16_EXACT


@pytest.mark.parametrize("m", [1, 2, 5, 10])
@pytest.mark.parametrize("down", [False, True])
def test_shape_admission(m, down):
    n, k = (2560, 160) if down else (320, 2560)
    x = torch.empty(m * 10 if down else m, k, device="meta", dtype=torch.float16)
    w = torch.empty(512, n, k, device="meta", dtype=torch.float16)
    y = torch.empty(m, 10, n, device="meta", dtype=torch.float16)
    ids = torch.empty(m * 10, device="meta", dtype=torch.int32)
    weights = torch.empty(m, 10, device="meta", dtype=torch.float32)
    args = (x, w, y, weights, ids, 1 if down else 10, down)
    assert _sm70_mtp_moe_fp16_shape_supported(*args) == (m in (1, 5))
    assert not _sm70_mtp_moe_fp16_shape_supported(x.bfloat16(), *args[1:])
    assert not _sm70_mtp_moe_fp16_shape_supported(
        x, w, y, weights, ids.long(), *args[5:]
    )
    assert not _sm70_mtp_moe_fp16_shape_supported(x, w, y, None, *args[4:])


@pytest.fixture(scope="module")
def cuda_weights():
    if not current_platform.is_device_capability(70):
        pytest.skip("Requires SM70")
    if not hasattr(torch.ops._C, "sm70_mtp_moe_fp16_out"):
        pytest.skip("Build the native extension first")
    torch.manual_seed(20260927)
    return tuple(
        torch.randn(512, n, k, device="cuda", dtype=torch.float16) * 0.03
        for n, k in ((320, 2560), (2560, 160))
    )


@pytest.mark.parametrize("m", [1, 5])
@pytest.mark.parametrize("down", [False, True])
def test_graph_changed_inputs_routes_and_canaries(cuda_weights, monkeypatch, m, down):
    monkeypatch.setenv("VLLM_SM70_MTP_MOE_FP16_EXACT", "1")
    monkeypatch.setenv("VLLM_SM70_MTP_MOE_TUNED_CONFIG", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    envs.disable_envs_cache()
    w = cuda_weights[int(down)]
    n, k = w.shape[-2:]
    x = torch.randn(m * 10 if down else m, k, device="cuda", dtype=torch.float16)
    ids = torch.arange(m * 10, device="cuda", dtype=torch.int32)
    weights = torch.randn(m, 10, device="cuda")
    padded = torch.tensor([m * 20], device="cuda", dtype=torch.int32)
    storage = torch.full((m * 10 * n + 16,), 19.0, device="cuda", dtype=x.dtype)
    actual = storage[8:-8].view(m, 10, n)
    expected = torch.empty_like(actual)

    native = torch.ops._C.sm70_mtp_moe_fp16_out
    hits = []

    def tracked(*args):
        hits.append(True)
        native(*args)

    monkeypatch.setattr(torch.ops._C, "sm70_mtp_moe_fp16_out", tracked)

    def candidate():
        dispatch_fused_moe_kernel(
            x,
            w,
            actual,
            None,
            None,
            None,
            weights,
            None,
            ids,
            padded,
            down,
            1 if down else 10,
            CONFIG,
            tl.float16,
            False,
            False,
            False,
            False,
            False,
        )

    def reference():
        invoke_fused_moe_triton_kernel(
            x,
            w,
            expected,
            None,
            None,
            weights,
            None,
            ids,
            padded,
            down,
            1 if down else 10,
            CONFIG,
            tl.float16,
            False,
            False,
            False,
            False,
            False,
        )

    candidate()
    assert hits, "The qualified native projection must actually be selected"
    reference()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    for scale in (0.0, 0.001, 0.1, 1.0, 3.0, 30.0):
        x.normal_(0, scale)
        ids.random_(0, 512)
        ids[0] = -1
        weights.normal_()
        actual.fill_(float("nan"))
        expected.fill_(float("nan"))
        graph.replay()
        reference()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert torch.all(storage[:8] == 19) and torch.all(storage[-8:] == 19)

    # Match the reference's early return for inactive blocks, including
    # graph replay after the padded-token count changes.
    padded.fill_(m * 20 - 2)
    actual.fill_(19)
    expected.fill_(19)
    graph.replay()
    reference()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_native_rejects_unaligned_weight(cuda_weights):
    w = cuda_weights[1]
    backing = w.new_empty(w.numel() + 1)
    unaligned = backing[1:].view_as(w)
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        torch.ops._C.sm70_mtp_moe_fp16_out(
            w.new_empty(1, 10, 2560),
            w.new_empty(10, 160),
            unaligned,
            torch.zeros(10, device="cuda", dtype=torch.int32),
            torch.ones(1, 10, device="cuda"),
            torch.tensor([20], device="cuda", dtype=torch.int32),
            True,
        )


@pytest.mark.parametrize("m", [1, 5])
@torch.inference_mode()
def test_modular_experts_route_and_graph(cuda_weights, monkeypatch, m):
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    monkeypatch.setenv("VLLM_SM70_MTP_MOE_TUNED_CONFIG", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    envs.disable_envs_cache()
    experts = TritonExperts(make_dummy_moe_config(), FUSED_MOE_UNQUANTIZED_CONFIG)
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    ids = torch.zeros(m, 10, device="cuda", dtype=torch.int32)
    weights = torch.softmax(torch.randn(m, 10, device="cuda"), -1)
    shapes = experts.workspace_shapes(
        m, 320, 2560, 10, 512, 512, None, MoEActivation.SILU
    )
    workspaces = [x.new_empty(shape) for shape in shapes[:2]]
    outputs = [torch.empty_like(x) for _ in range(2)]
    native = torch.ops._C.sm70_mtp_moe_fp16_out
    hits = []

    def tracked(*args):
        hits.append(args[-1])
        native(*args)

    monkeypatch.setattr(torch.ops._C, "sm70_mtp_moe_fp16_out", tracked)

    def run(arm):
        experts.apply(
            outputs[arm],
            x,
            *cuda_weights,
            weights,
            ids,
            MoEActivation.SILU,
            512,
            None,
            None,
            None,
            *workspaces,
            None,
            False,
        )

    graphs = []
    for arm in range(2):
        monkeypatch.setenv("VLLM_SM70_MTP_MOE_FP16_EXACT", str(arm))
        envs.disable_envs_cache()
        hits.clear()
        run(arm)
        assert hits == ([False, True] if arm else [])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(arm)
        graphs.append(graph)
    for scale in (0.0, 0.001, 0.1, 1.0, 3.0):
        x.normal_(0, scale)
        ids.random_(0, 512)
        ids[0, 0] = -1
        weights.copy_(torch.softmax(torch.randn_like(weights), -1))
        for workspace in workspaces:
            workspace.fill_(float("nan"))
        for output in outputs:
            output.fill_(float("nan"))
        for graph in graphs:
            graph.replay()
        assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))


@torch.inference_mode()
def test_narrow_prefill_activation_workspace_graph(cuda_weights, monkeypatch):
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    monkeypatch.setenv("VLLM_SM70_MTP_MOE_TUNED_CONFIG", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    envs.disable_envs_cache()
    experts = TritonExperts(make_dummy_moe_config(), FUSED_MOE_UNQUANTIZED_CONFIG)
    m = 33
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, 512, (m, 10), device="cuda", dtype=torch.int32)
    weights = torch.softmax(torch.randn(m, 10, device="cuda"), -1)
    shapes = experts.workspace_shapes(
        m, 320, 2560, 10, 512, 512, None, MoEActivation.SILU
    )
    old = [x.new_empty((m, 10, 2560)) for _ in range(2)]
    new = [x.new_empty(shape) for shape in shapes[:2]]
    outputs = [torch.empty_like(x) for _ in range(2)]
    common = x.new_empty(max(new[0].numel(), x.numel()))
    new[0] = common[: new[0].numel()].view(shapes[0])
    outputs[1] = common[: x.numel()].view_as(x)
    assert (
        new[0].untyped_storage().data_ptr() == outputs[1].untyped_storage().data_ptr()
    )

    def run(arm):
        experts.apply(
            outputs[arm],
            x,
            *cuda_weights,
            weights,
            ids,
            MoEActivation.SILU,
            512,
            None,
            None,
            None,
            *(new if arm else old),
            None,
            False,
        )

    graphs = []
    for arm in (0, 1):
        run(arm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(arm)
        graphs.append(graph)
    for _ in range(3):
        x.normal_(std=0.125)
        ids.random_(0, 512)
        for graph in graphs:
            graph.replay()
        assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
