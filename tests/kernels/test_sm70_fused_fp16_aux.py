# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm import envs
from vllm.config.kernel import KernelConfig
from vllm.models.deepseek_v4.sm70 import gemv
from vllm.sm70_profiles.acceleration import loaded_sm70_preparations


@pytest.fixture(autouse=True)
def reset_cache():
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def make_weights(k=4096, rows=(2048, 512, 64), **kwargs):
    return tuple(torch.empty((n, k), dtype=torch.float16, **kwargs) for n in rows)


@pytest.mark.parametrize(
    "k,rows",
    [
        (1024, (3, 5, 7)),
        (3072, (17, 31, 63)),
        (4096, (2048, 512, 64)),
        (5120, (256, 1024, 32)),
    ],
)
def test_capability_uses_tensor_geometry(k, rows):
    assert gemv.has_sm70_dsv4_fused_fp16_aux_weight_contract(
        make_weights(k, rows, device="meta")
    )


@pytest.mark.parametrize(
    "weights",
    [
        make_weights(1000, device="meta"),
        make_weights(0, device="meta"),
        make_weights(rows=(0, 2, 3), device="meta"),
        (torch.empty(3, device="meta"), *make_weights(device="meta")[1:]),
        (torch.empty((3, 4096), device="meta"), *make_weights(device="meta")[1:]),
        (make_weights(device="cpu")[0], *make_weights(device="meta")[1:]),
        (make_weights(2048, device="meta")[0], *make_weights(device="meta")[1:]),
        (
            torch.empty((4096, 8), dtype=torch.float16, device="meta").t(),
            *make_weights(device="meta")[1:],
        ),
        make_weights(device="meta")[:2],
    ],
)
def test_reject_invalid_weight_layout(weights):
    assert not gemv.has_sm70_dsv4_fused_fp16_aux_weight_contract(weights)


def test_existing_numerical_routes_preserved(monkeypatch):
    weights = make_weights(device="meta")
    monkeypatch.setenv("VLLM_SM70_DSV4_FP13_GEMV", "1")
    monkeypatch.setenv("VLLM_SM70_DSV4_FP16_GEMV", "1")
    assert (
        gemv.sm70_fused_fp16_aux_reason(weights, enabled=True)
        == "existing_fp13_route_preserved"
    )
    assert gemv.prepare_sm70_dsv4_fused_fp16_aux_weight(weights) is None
    monkeypatch.setenv("VLLM_SM70_DSV4_FP13_GEMV", "0")
    monkeypatch.setenv("VLLM_SM70_DSV4_FP16_GEMV", "0")
    envs.disable_envs_cache()
    assert (
        gemv.sm70_fused_fp16_aux_reason(weights, enabled=True)
        == "existing_exact_gemv_route_not_selected"
    )
    monkeypatch.setenv("VLLM_SM70_DSV4_FP16_GEMV", "1")
    envs.disable_envs_cache()
    assert (
        gemv.sm70_fused_fp16_aux_reason(weights, enabled=True)
        == "requires_cuda_weights"
    )
    assert (
        gemv.sm70_fused_fp16_aux_reason(weights, enabled=False)
        == "disabled_by_kernel_config"
    )


def test_policy_and_startup_reason():
    assert KernelConfig().fused_fp16_aux_gemv
    assert not KernelConfig(fused_fp16_aux_gemv=False).fused_fp16_aux_gemv
    layer = torch.nn.Module()
    layer._sm70_fused_fp16_aux_enabled = False
    layer._sm70_fused_fp16_aux_reason = "existing_fp13_route_preserved"
    report = loaded_sm70_preparations(layer)["variants"][""]
    assert report["flags"]["_sm70_fused_fp16_aux_enabled"] is False
    assert (
        report["reasons"]["_sm70_fused_fp16_aux_reason"]
        == "existing_fp13_route_preserved"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "k,rows",
    [
        (1024, (3, 5, 7)),
        (3072, (17, 31, 63)),
        (4096, (2048, 512, 64)),
        (5120, (256, 1024, 32)),
    ],
)
def test_graph_bitwise_existing_reduction(monkeypatch, k, rows):
    if not gemv.current_platform.is_device_capability((7, 0)):
        pytest.skip("requires SM70")
    monkeypatch.setenv("VLLM_SM70_DSV4_FP13_GEMV", "0")
    monkeypatch.setenv("VLLM_SM70_DSV4_FP16_GEMV", "1")
    envs.disable_envs_cache()
    torch.manual_seed(k)
    weights = tuple(
        torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.01 for n in rows
    )
    packed = torch.cat(weights)
    x = torch.randn((1, k), device="cuda", dtype=torch.float16)
    reference = tuple(
        torch.empty((1, n), device="cuda", dtype=dtype)
        for n, dtype in zip(rows, (torch.float32, torch.float32, torch.float16))
    )

    def run():
        for weight, out in zip(weights, reference):
            gemv._sm70_dsv4_fp16_gemv_kernel[(weight.shape[0],)](
                x, weight, out, K=k, BLOCK_K=gemv._BLOCK_K, num_warps=gemv._NUM_WARPS
            )
        return gemv.maybe_sm70_dsv4_fused_fp16_aux_gemv(x, packed, rows)

    actual = run()
    assert actual is not None
    for a, b in zip(actual, reference):
        assert torch.equal(a, b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(12):
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(actual, reference):
            assert torch.equal(a, b)


def test_loaded_preparation_clears_stale_weights_and_reports_reason(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.deepseek_v4 import attention

    cls = attention.DeepseekV4MultiHeadLatentAttentionWrapper
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    weights = tuple(w.zero_() for w in make_weights(rows=(256, 512, 64)))
    layer.compressor = SimpleNamespace(
        fused_wkv_wgate=SimpleNamespace(weight=weights[0])
    )
    layer.indexer = SimpleNamespace(
        compressor=SimpleNamespace(fused_wkv_wgate=SimpleNamespace(weight=weights[1])),
        weights_proj=SimpleNamespace(weight=weights[2]),
    )
    layer._fused_fp16_aux_policy = True
    layer.register_buffer("_sm70_fused_fp16_aux_packed_weight", None, persistent=False)
    monkeypatch.setattr(
        attention, "sm70_fused_fp16_aux_reason", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        attention,
        "prepare_sm70_dsv4_fused_fp16_aux_weight",
        lambda weights, **kwargs: torch.cat(weights),
    )
    layer.prepare_sm70_fused_fp16_aux_weight()
    assert layer._sm70_fused_fp16_aux_enabled
    assert layer._sm70_fused_fp16_aux_rows == (256, 512, 64)
    assert torch.equal(layer._sm70_fused_fp16_aux_packed_weight, torch.cat(weights))
    assert "_sm70_fused_fp16_aux_packed_weight" not in layer.state_dict()
    monkeypatch.setattr(
        attention,
        "sm70_fused_fp16_aux_reason",
        lambda *args, **kwargs: "existing_fp13_route_preserved",
    )
    layer.prepare_sm70_fused_fp16_aux_weight()
    assert not layer._sm70_fused_fp16_aux_enabled
    assert layer._sm70_fused_fp16_aux_packed_weight is None
    assert layer._fused_fp16_aux_policy
    assert (
        loaded_sm70_preparations(layer)["variants"][""]["reasons"][
            "_sm70_fused_fp16_aux_reason"
        ]
        == "existing_fp13_route_preserved"
    )


def test_fused_projection_output_order_and_stream_join(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.deepseek_v4 import attention

    cls = attention.DeepseekV4MultiHeadLatentAttentionWrapper
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.aux_stream_list = [object(), object(), object()]
    layer.ln_events = [object(), object(), object(), object()]
    layer.compressor = SimpleNamespace()
    layer.indexer = SimpleNamespace()
    layer._sm70_fused_fp16_aux_packed_weight = torch.empty(
        (3, 4096), dtype=torch.float16
    )
    layer._sm70_fused_fp16_aux_rows = (1, 1, 1)
    main = torch.tensor([0.0])
    outputs = (torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]))
    layer.fused_wqa_wkv = lambda x: (main, None)
    monkeypatch.setattr(
        attention, "can_use_sm70_dsv4_fused_fp16_aux_gemv", lambda *args: True
    )
    monkeypatch.setattr(
        attention, "maybe_sm70_dsv4_fused_fp16_aux_gemv", lambda *args: outputs
    )

    def execute(default_fn, aux_fns, start_event, done_events, aux_streams, enable):
        assert start_event is layer.ln_events[0]
        assert done_events == [layer.ln_events[1]]
        assert aux_streams == [layer.aux_stream_list[0]]
        assert enable
        return default_fn(), [fn() for fn in aux_fns]

    monkeypatch.setattr(attention, "execute_in_parallel", execute)
    result = layer.attn_gemm_parallel_execute(
        torch.empty((1, 4096), dtype=torch.float16)
    )
    assert all(actual is expected for actual, expected in zip(result, (main, *outputs)))
