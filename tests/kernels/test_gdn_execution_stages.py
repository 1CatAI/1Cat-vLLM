# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise old-operator/new-stage parity, including replay with changed indices."""

import pytest
import torch

from vllm.config.gdn_schedule import GdnScheduleConfig
from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update_mixed_qkv,
    fused_sigmoid_gating_delta_rule_update_mixed_qkv_out,
)
from vllm.model_executor.layers.fla.ops.gdn_chunk_kernels import create_chunk_kernels
from vllm.model_executor.layers.fla.ops.gdn_stages import (
    GdnHeadContract,
    convolve_decode,
    mixed_qkv_recurrence,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 operator and replay validation",
)


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize("row_stride", [2560, 4096])
def test_decode_stages_exact_and_replay(rows, use_out, row_stride):
    torch.manual_seed(89)
    dev, dtype = "cuda", torch.float16
    heads = GdnHeadContract(16, 48, 128, 128, 4)
    schedule = GdnScheduleConfig()
    schedule.resolve()
    x_seed = torch.randn(rows, row_stride, device=dev, dtype=dtype)
    a, b = (torch.randn(rows, 12, device=dev, dtype=dtype) for _ in range(2))
    A_log = torch.randn(12, device=dev)
    dt_bias = torch.randn(12, device=dev, dtype=dtype)
    weight = torch.randn(2560, 4, device=dev, dtype=dtype)
    indices = torch.tensor([0, 5, -1, 2], device=dev, dtype=torch.int32)[:rows]
    cu = torch.arange(rows + 1, device=dev, dtype=torch.int32)
    conv_seed = torch.randn(8, 2560, 3, device=dev, dtype=dtype)
    state_seed = torch.randn(8, 12, 128, 128, device=dev) * 0.01
    convs, states, outputs = [], [], []
    graphs, inputs = [], []
    for candidate in (False, True):
        storage = x_seed.clone()
        x = storage[:, :2560]
        inputs.append(storage)
        conv, state = conv_seed.clone(), state_seed.clone()
        output = torch.zeros(rows, 1, 12, 128, device=dev, dtype=dtype)

        def run(candidate=candidate, conv=conv, state=state, output=output, x=x):
            if candidate:
                mixed = convolve_decode(
                    x,
                    conv,
                    weight,
                    None,
                    "silu",
                    state_indices=indices,
                    validate_data=False,
                )
                result, _ = mixed_qkv_recurrence(
                    heads,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    a=a,
                    b=b,
                    mixed_qkv=mixed,
                    initial_state=state,
                    cu_seqlens=cu,
                    state_indices=indices,
                    out=output if use_out else None,
                    schedule=schedule,
                )
            else:
                mixed = causal_conv1d_update(
                    x,
                    conv,
                    weight,
                    None,
                    "silu",
                    conv_state_indices=indices,
                    validate_data=False,
                )
                kwargs = dict(
                    A_log=A_log,
                    dt_bias=dt_bias,
                    a=a,
                    b=b,
                    mixed_qkv=mixed,
                    initial_state=state,
                    cu_seqlens=cu,
                    ssm_state_indices=indices,
                    num_q_heads=4,
                    num_v_heads=12,
                    head_k_dim=128,
                    head_v_dim=128,
                    use_qk_l2norm_in_kernel=True,
                )
                if use_out:
                    fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
                        **kwargs, scale=128**-0.5, out=output
                    )
                    result = output
                else:
                    result, _ = fused_sigmoid_gating_delta_rule_update_mixed_qkv(
                        **kwargs, inplace_final_state=True
                    )
            if not use_out:
                output.copy_(result.reshape_as(output))

        # Warm both JIT routes before capture; reset state before each replay.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        convs.append(conv)
        states.append(state)
        outputs.append(output)
        graphs.append(graph)
    for replay in range(2):
        if replay:
            x_seed.add_(0.02)
            indices.copy_(
                torch.tensor([3, 1, -1, 6], device=dev, dtype=torch.int32)[:rows]
            )
        for storage, conv, state, output, graph in zip(
            inputs, convs, states, outputs, graphs
        ):
            storage.copy_(x_seed)
            conv.copy_(conv_seed)
            state.copy_(state_seed)
            output.zero_()
            graph.replay()
        torch.accelerator.synchronize()
        # Allocating recurrence leaves padding-row output unspecified; only
        # live rows are outputs, while every resident state slot is compared.
        live = indices >= 0
        assert torch.equal(outputs[0][live], outputs[1][live])
        assert torch.equal(convs[0], convs[1])
        assert torch.equal(states[0], states[1])


@pytest.mark.parametrize("tokens", [64, 300])
def test_prefill_engine_autotuners_preserve_output_and_state(tokens):
    torch.manual_seed(37)
    schedule = GdnScheduleConfig()
    schedule.resolve()
    kernels = create_chunk_kernels(schedule)
    assert kernels is not None
    q, k = (
        torch.randn(1, tokens, 4, 128, device="cuda", dtype=torch.float16)
        for _ in range(2)
    )
    v = torch.randn(1, tokens, 12, 128, device="cuda", dtype=torch.float16)
    g = -torch.rand(1, tokens, 12, device="cuda")
    beta = torch.rand_like(g)
    state = torch.randn(2, 12, 128, 128, device="cuda") * 0.01
    cu = torch.tensor([0, tokens // 2, tokens], device="cuda", dtype=torch.int32)
    outputs = []
    for bound in (None, kernels):
        outputs.append(
            chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=state.clone(),
                output_final_state=True,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
                kernels=bound,
            )
        )
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])


@pytest.mark.parametrize("direct_output", [False, True])
def test_native_provider_matches_direct_operator(direct_output):
    from vllm.model_executor.layers.fla.ops.sm70.gdn_verify import bind_native_verifier

    if not hasattr(torch.ops._C, "sm70_gdn_verify_out"):
        pytest.skip("native verifier unavailable")
    torch.manual_seed(53)
    tokens, h, hv, dim = 5, 4, 8, 128
    heads = GdnHeadContract(h, hv, dim, dim, 1)
    provider = bind_native_verifier(heads, enabled=True)
    # Allocate weights after binding, as during a reload/address replacement.
    A_log = torch.randn(hv, device="cuda") * 0.1
    bias = torch.randn(hv, device="cuda") * 0.1
    mixed = (
        torch.randn(tokens, (2 * h + hv) * dim, device="cuda", dtype=torch.float16)
        * 0.1
    )
    a, b = (
        torch.randn(tokens, hv, device="cuda", dtype=torch.float16) for _ in range(2)
    )
    seed = torch.randn(8, hv, dim, dim, device="cuda") * 0.01
    baseline_state, new_state = seed.clone(), seed.clone()
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    indices = torch.tensor([[0, 4, 2, 6, 1]], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([3], device="cuda", dtype=torch.int32)
    baseline = torch.zeros(tokens, hv, dim, device="cuda", dtype=torch.float16)
    target = torch.empty_like(baseline)
    torch.ops._C.sm70_gdn_verify_out(
        mixed,
        a,
        b,
        A_log,
        bias,
        baseline_state,
        baseline,
        cu,
        indices,
        accepted,
        h,
        hv,
        dim**-0.5,
        1,
        None,
        None,
        None,
        tokens,
    )
    output, state = provider(
        A_log,
        bias,
        1,
        mixed,
        a,
        b,
        new_state,
        target,
        tokens,
        direct_output,
        cu,
        indices,
        accepted,
    )
    assert state is new_state
    assert torch.equal(output.squeeze(0), baseline)
    assert torch.equal(new_state, baseline_state)
