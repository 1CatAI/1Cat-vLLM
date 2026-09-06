# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from benchmarks.kernels.benchmark_sm70_flashinfer_gdn_conv import (
    capture,
    check_exclusive,
    load_weights,
)
from benchmarks.kernels.benchmark_sm70_flashinfer_qsa import make_case
from vllm import envs
from vllm.model_executor.layers import sm70_flashinfer_batch as fi
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first


@pytest.fixture(scope="module")
def cuda_bridge():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    check_exclusive()
    for key in ("VLLM_SM70_FLASHINFER_GDN_LIBRARY", "VLLM_SM70_FLASHINFER_QSA_LIBRARY"):
        torch.ops.load_library(os.environ[key])
    zero = torch.zeros(256, device="cuda", dtype=torch.float16)
    fi._QSA_ZERO[zero.device] = zero
    yield
    fi._QSA_ZERO.pop(zero.device)


@pytest.mark.parametrize("rows", [4, 8, 9, 15, 16])
@pytest.mark.parametrize("kv_heads,selected", [(1, 2051), (2, 2051), (1, 15), (1, 65)])
def test_qsa_bridge_graph_and_materialized_gate(
    cuda_bridge, monkeypatch, rows, kv_heads, selected
):
    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        _qsa_output_gate,
        qsa_sparse_paged_attention,
    )

    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SM70_FLASHINFER_BATCH", "0")
    torch.manual_seed(37)
    q, k, v, indices, table, requests = make_case(rows)
    if kv_heads != 1:
        q = q.repeat(1, kv_heads, 1)
        k = k.repeat(1, 1, kv_heads, 1)
        v = v.repeat(1, 1, kv_heads, 1)
    indices = indices[:, :selected]
    output = torch.empty_like(q)
    gate = torch.randn_like(q)

    def call():
        assert fi.try_qsa(q, k, v, indices, table, requests, output) is output
        _qsa_output_gate(output, gate)

    graph = capture(call)
    for cycle in range(4):
        q.normal_()
        gate.normal_()
        indices[:, :8] = cycle
        if cycle == 3:
            indices[-1].fill_(-1)
        reference = qsa_sparse_paged_attention(
            q, k, v, indices, table, requests, output_gate=gate
        )
        call()
        eager = output.clone()
        output.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(output, eager, atol=0, rtol=0)
        relative = (
            output.float() - reference.float()
        ).norm() / reference.float().norm()
        assert torch.isfinite(output).all() and relative < 5e-3


@pytest.mark.parametrize("empty_aot_placeholders", [False, True])
def test_gdn_bridge_independent_state_and_graph(cuda_bridge, empty_aot_placeholders):
    from flash_qla.ops.gated_delta_rule.chunk.sm70 import fused_fwd as qla
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    torch.manual_seed(17)
    model = os.environ.get("SM70_FLASHINFER_TEST_MODEL")
    if not model or not Path(model).is_dir():
        pytest.skip("Set SM70_FLASHINFER_TEST_MODEL to the checkpoint directory")
    h, hq, hv, wqkv, ba, cw, A, dt = load_weights(Path(model))
    rows, width, pool = 8, wqkv.shape[0], 11
    weight = torch.cat((wqkv, torch.randn(hv * 128, h, device="cuda").half() * 0.01))
    layer = NS(
        _sm70_fi_ready=True,
        num_v_heads=hv * 4,
        num_k_heads=hq * 4,
        tp_size=4,
        _sm70_fi_ba=ba.t().contiguous(),
        _sm70_fi_bias=ba.new_empty(0),
        _sm70_fi_op=torch.ops._C_flashinfer_gdn_sm70_h2560_q4_v12.run,
        conv1d=NS(weight=cw[:, None, :]),
        A_log=A,
        dt_bias=dt,
        in_proj_qkvz=lambda x: (torch.nn.functional.linear(x, weight), None),
    )
    hidden = torch.randn(rows, h, device="cuda", dtype=torch.float16)
    conv = torch.randn(pool, width, 3, device="cuda", dtype=torch.float16) * 0.1
    state = torch.randn(pool, hv, 128, 128, device="cuda") * 0.01
    raw_conv = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
    layer.kv_cache = (
        (raw_conv, state)
        if empty_aot_placeholders
        else (raw_conv.clone(), state.clone())
    )
    placeholder = hidden.new_empty(0)
    indices = torch.arange(rows, device="cuda", dtype=torch.int32)
    meta = NS(
        num_prefills=0,
        num_prefill_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_decodes=rows,
        num_decode_tokens=rows,
        non_spec_state_indices_tensor=indices,
    )
    z = hidden.new_empty(rows, hv, 128)
    output = torch.empty_like(z)
    ref_out = torch.empty_like(z)

    def call():
        assert fi.try_gdn(
            layer,
            hidden,
            z,
            output,
            placeholder if empty_aot_placeholders else raw_conv,
            placeholder if empty_aot_placeholders else state,
            meta,
        )

    graph = capture(call)
    ref_conv, ref_state = conv.clone(), state.clone()
    for cycle in range(8):
        hidden.normal_().mul_(0.5)
        indices.copy_(torch.randperm(pool, device="cuda")[:rows])
        if cycle == 7:
            indices[-1] = -1
        qkvz = torch.nn.functional.linear(hidden, weight)
        raw_qkv = qkvz[:, :width].clone()
        ba_ref = hidden @ ba.t()
        causal_conv1d_update(
            raw_qkv,
            ref_conv,
            cw,
            None,
            "silu",
            conv_state_indices=indices,
            validate_data=False,
        )
        qla.gdn_decode_mixed_qkv_global_state_sm70(
            raw_qkv,
            ba_ref[:, hv:].contiguous(),
            ba_ref[:, :hv].contiguous(),
            A,
            dt,
            ref_state,
            indices,
            ref_out,
        )
        old_c, old_s = conv.clone(), state.clone()
        call()
        expected = [t.clone() for t in (z, output, conv, state)]
        conv.copy_(old_c)
        state.copy_(old_s)
        output.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        for actual, eager in zip((z, output, conv, state), expected):
            torch.testing.assert_close(actual, eager, atol=0, rtol=0)
        torch.testing.assert_close(z, qkvz[:, width:].reshape_as(z), atol=0, rtol=0)
        torch.testing.assert_close(conv, ref_conv, atol=0, rtol=0)
        assert (state - ref_state).norm() / ref_state.norm() < 5e-3
        live = indices >= 0
        relative = (output[live].float() - ref_out[live].float()).norm()
        assert relative / ref_out[live].float().norm() < 5e-3
