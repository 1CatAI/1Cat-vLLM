# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""79t row-scale binding retains the frozen ordinary-attention error contract."""

import pytest
import torch

from tests.kernels.attention.test_sm70_splitd_late_scales import check_attention, inputs
from vllm.v1.attention.kv_codecs import INT8, INT8_PER_TOKEN_HEAD

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.fixture
def native():
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    load_fa2_library(torch.device("cuda"))
    ns = torch.ops._vllm_fa2_C
    assert ns.sm70_d256_gqa_architecture_late_scale_abi() == 1
    return ns


def execute(native, q, k, v, ks, vs, out):
    length = q.shape[1]
    name = (
        "sm70_d256_gqa_architecture" + ("_q8192" if length == 8192 else "") + "_kv_fwd"
    )
    k = k.view(1, -1, 1, 256)
    v = v.view_as(k)
    return getattr(native, name)(
        q, k, v, out, 0.0625, True, ks.view(1, -1, 1), vs.view(1, -1, 1)
    )


@pytest.mark.parametrize("codec", [INT8, INT8_PER_TOKEN_HEAD], ids=lambda c: c.name)
@pytest.mark.parametrize(
    "length,prefix", [(8000, 32), (8192, 0), (8192, 32), (8192, 8224)]
)
def test_79t_shared_algorithm_with_row_scales(native, codec, length, prefix):
    q, k, v, ks, vs = inputs(codec, 1, 1, length, length + prefix)
    out = torch.full_like(q, float("nan"))
    execute(native, q, k, v, ks, vs, out)
    check_attention(q, k, v, ks, vs, out)


def test_79t_all_query_rows_meet_error_contract(native):
    q, k, v, ks, vs = inputs(INT8, 1, 1, 8192, 8224)
    out = torch.empty_like(q)
    execute(native, q, k, v, ks, vs, out)
    keys = k.reshape(-1, 256).double()
    values = v.reshape(-1, 256).double()
    key_scales = ks.flatten().double()
    value_scales = vs.flatten().double()
    key_rows = torch.arange(keys.shape[0], device=q.device)
    # Bound the FP64 reference workspace while checking all rows and GQA heads.
    for start in range(0, q.shape[1], 128):
        rows = torch.arange(start, start + 128, device=q.device)
        queries = q[0, rows].permute(1, 0, 2).double()
        scores = (queries @ keys.T) * (key_scales * 0.0625)
        scores.masked_fill_(
            key_rows[None, None] > (32 + rows)[None, :, None], -torch.inf
        )
        expected = ((scores.softmax(-1) * value_scales) @ values).permute(1, 0, 2)
        torch.testing.assert_close(
            out[0, rows].double(), expected, atol=0.004, rtol=0.015
        )
    assert torch.isfinite(out).all()


def test_79t_row_scale_graph_mutates_values_and_resets_recovery(native):
    q, k, v, ks, vs = inputs(INT8, 1, 1, 8192, 8224)
    out = torch.empty_like(q)
    execute(native, q, k, v, ks, vs, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute(native, q, k, v, ks, vs, out)
    for factor in (0.5, 2.0, 1.0):
        ks.mul_(factor)
        vs.mul_(factor)
        graph.replay()
        check_attention(q, k, v, ks, vs, out)


@pytest.mark.parametrize("length", [8000, 8192])
def test_old_79t_entry_keeps_original_half_arithmetic(native, length):
    torch.manual_seed(949)
    q = torch.randn((1, length, 6, 256), device="cuda", dtype=torch.float16)
    k = torch.randn((1, length + 32, 1, 256), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    reference = torch.empty_like(q)
    name = "sm70_d256_gqa_architecture" + ("_q8192" if length == 8192 else "")
    getattr(native, name + "_fwd")(q, k, v, out, 0.0625, True)
    getattr(native, name + "_kv_fwd")(q, k, v, reference, 0.0625, True)
    assert torch.equal(out, reference)


@pytest.mark.parametrize("length,context", [(8001, 8032), (8096, 8128), (8191, 8224)])
def test_79t_common_padding_preserves_scales_and_causal_alignment(
    native, length, context
):
    from vllm.v1.attention.backends.flash_v100.dense_prefill import (
        _run_sm70_d256_gqa_79t_q8192_dispatch,
    )

    q, k, v, ks, vs = inputs(INT8, 2, 2, length, context)
    out = torch.full_like(q, float("nan"))
    k = k.reshape(2, context, 2, 256)
    v = v.reshape_as(k)
    ks = ks.reshape(2, context, 2)
    vs = vs.reshape_as(ks)
    result = _run_sm70_d256_gqa_79t_q8192_dispatch(
        q,
        k,
        v,
        out,
        softmax_scale=0.0625,
        architecture_q8192_op=native.sm70_d256_gqa_architecture_q8192_kv_fwd,
        key_scale=ks,
        value_scale=vs,
    )
    assert result is out
    check_attention(q, k, v, ks, vs, out)


def test_79t_executes_scaled_qk_and_tensor_core_pv(native, record_property):
    q, k, v, ks, vs = inputs(INT8, 1, 1, 8192, 8224)
    out = torch.empty_like(q)
    execute(native, q, k, v, ks, vs, out)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        execute(native, q, k, v, ks, vs, out)
        torch.accelerator.synchronize()
    names = [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    record_property("executed_cuda_kernels", names)
    assert any("batched_row_scale_qk_kernel" in name for name in names), names
    assert sum("batched_row_scale_pv_kernel" in name for name in names) == 8, names
    check_attention(q, k, v, ks, vs, out)


@pytest.mark.parametrize("peak_key", [1, 9505], ids=["prefix", "tail"])
def test_79t_large_finite_value_scale_does_not_overflow_partial(native, peak_key):
    q, k, v, ks, vs = inputs(INT8, 1, 1, 8192, 16416)
    # A peak absent from the stride-eight prefix scan can overflow a weighted
    # FP16 probability even when the score and normalized output are finite.
    q.zero_()
    q[..., 0] = 16
    k.zero_()
    k.reshape(-1, 256)[peak_key, 0] = 10
    ks.fill_(1)
    v.fill_(127)
    vs.fill_(500)
    out = torch.empty_like(q)
    execute(native, q, k, v, ks, vs, out)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out, torch.full_like(out, 127 * 500), atol=0.004, rtol=0.015
    )


def test_79t_engine_owner_profiles_scaled_workspace_before_capture(native):
    from tests.config.test_sm70_prefill_runtime import engine
    from vllm.config import set_current_vllm_config
    from vllm.runtime_resources import release_runtime_resources, runtime_resources_for
    from vllm.v1.attention.backends.flash_v100 import dense_prefill, ops, runtime

    config = engine(prefill_serial_tail=True)
    query = torch.zeros((8192, 6, 256), device="cuda", dtype=torch.float16)
    try:
        with set_current_vllm_config(config):
            runtime.prepare_prefill_runtime(native)
            dense_prefill.profile_sm70_prefill_workspace(query, 1, INT8)
            owner = runtime_resources_for(config)["sm70_prefill"]
            counts = owner.owner.observations()
            dense_prefill.profile_sm70_prefill_workspace(query, 1, INT8)
            assert owner.owner.observations() == counts
            assert counts[1] == 1
            operation = ops.get_sm70_d256_gqa_architecture_q8192_op()
            assert ops.prefill_row_scales_supported(operation)
            q, k, v, ks, vs = inputs(INT8, 1, 1, 8192, 8224)
            out = torch.empty_like(q)

            def run():
                operation(
                    q,
                    k.reshape(1, -1, 1, 256),
                    v.reshape(1, -1, 1, 256),
                    out,
                    0.0625,
                    True,
                    ks.reshape(1, -1, 1),
                    vs.reshape(1, -1, 1),
                )

            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for factor in (0.5, 2.0):
                ks.mul_(factor)
                vs.mul_(1 / factor)
                graph.replay()
                check_attention(q, k, v, ks, vs, out)
            del graph
            torch.accelerator.synchronize()
    finally:
        release_runtime_resources(config)
    with pytest.raises(RuntimeError, match="closed"):
        run()
