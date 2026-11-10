# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Old operand/state protocol versus shared stages, with live graph inputs."""

import pytest
import torch

from vllm.config.gdn import GdnProfileConfig
from vllm.config.gdn_schedule import GdnScheduleConfig
from vllm.model_executor.layers.fla.ops import fused_post_conv_prep
from vllm.model_executor.layers.fla.ops.gdn_chunk_kernels import create_chunk_kernels
from vllm.model_executor.layers.fla.ops.gdn_prefill import GdnPrefill
from vllm.model_executor.layers.fla.ops.gdn_preparation import (
    GdnPreparation,
    fused_gdn_gating,
    unpack_mixed_qkv,
)
from vllm.model_executor.layers.fla.ops.gdn_profiling import GdnPrefillProfiler
from vllm.model_executor.layers.fla.ops.gdn_selector import (
    GDN_BACKEND_STAGES,
    GdnExecutionPlan,
)
from vllm.model_executor.layers.fla.ops.gdn_stages import GdnHeadContract
from vllm.model_executor.layers.fla.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 operand/state and replay validation",
)


@pytest.mark.parametrize("rows", [1, 8, 63, 64])
@pytest.mark.parametrize("strided", [False, True])
def test_packed_qkv_matches_original_split_cat(rows, strided):
    heads = GdnHeadContract(16, 48, 128, 128, 4)
    x = torch.randn(rows, 4096 if strided else 2560, device="cuda", dtype=torch.float16)
    x = x[:, :2560]
    expected = torch.cat([part.reshape(-1) for part in x.split([512, 512, 1536], -1)])
    for fused in (False, True):
        q, k, v = unpack_mixed_qkv(heads, x, fused_pack=fused)
        actual = torch.cat([part.flatten() for part in (q, k, v)])
        assert torch.equal(actual, expected)
        assert all(part.is_contiguous() for part in (q, k, v))


@pytest.mark.parametrize("backend", ["triton", "flashqla_vlk", "flashqla_indexed"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_prefill_preparation_state_commit_and_replay(backend, legacy, direct):
    torch.manual_seed(97)
    tokens, hq, hv, dim = 64, 4, 12, 128
    heads = GdnHeadContract(hq, hv, dim, dim, 1)
    selected = "triton" if backend == "triton" else "flashqla_sm70"
    indexed = backend == "flashqla_indexed"
    gate_exp = backend == "flashqla_vlk" and not legacy
    plan = GdnExecutionPlan(
        selected, GDN_BACKEND_STAGES[selected], indexed, indexed, direct, None
    )
    profile = GdnProfileConfig(enabled=False)
    profile.resolve()
    schedule = GdnScheduleConfig()
    schedule.resolve()
    native = None
    if backend == "flashqla_vlk":
        from flash_qla.ops.gated_delta_rule.chunk.sm70.fused_fwd import _load_ext

        native = _load_ext().GdnPolicy(-1)
    provider = GdnPrefill(
        plan, GdnPrefillProfiler(profile), create_chunk_kernels(schedule), native
    )
    storage = torch.randn(tokens, 4096, device="cuda", dtype=torch.float16) * 0.1
    mixed = storage[:, :2560]
    a, b = [
        torch.randn(tokens, hv, device="cuda", dtype=torch.float16) for _ in range(2)
    ]
    A_log = torch.randn(hv, device="cuda") * 0.1
    bias = torch.randn(hv, device="cuda", dtype=torch.float16) * 0.1
    seed = torch.randn(8, hv, dim, dim, device="cuda") * 0.01
    # Strided request views, with a row lacking cached state.
    slots = torch.tensor([4, 0, 1, 0], device="cuda", dtype=torch.int32)[::2]
    has_state = torch.tensor([True, False], device="cuda")
    cu = torch.tensor([0, 31, tokens], device="cuda", dtype=torch.int32)
    chunks, offsets = prepare_chunk_indices(cu, 64), prepare_chunk_offsets(cu, 64)
    preparation = GdnPreparation(heads, legacy=legacy, gate_is_exp=gate_exp)
    states = [seed.clone(), seed.clone()]
    outputs = [
        torch.empty(tokens, hv, dim, device="cuda", dtype=torch.float16)
        for _ in range(2)
    ]

    def run(candidate):
        state, output = states[candidate], outputs[candidate]
        if candidate:
            operands = preparation.prefill(mixed, a, b, A_log, bias)
            result, _ = provider.execute_prefill(
                *operands,
                ssm_state=state,
                state_indices=slots,
                has_initial_state=has_state,
                cu_seqlens=cu,
                chunk_indices=chunks,
                chunk_offsets=offsets,
                use_qk_l2norm_in_kernel=legacy,
                gate_is_exp=gate_exp,
                core_attn_out=output if direct else None,
                layer_name="layers.0",
                num_tokens=tokens,
            )
        else:
            # Original model-layer operand and state protocol, before extraction.
            if legacy:
                pieces = mixed.split([hq * dim, hq * dim, hv * dim], -1)
                flat = torch.cat([part.reshape(-1) for part in pieces])
                q, k, v = [
                    part.view(1, tokens, -1, dim)
                    for part in flat.split(
                        [tokens * hq * dim, tokens * hq * dim, tokens * hv * dim]
                    )
                ]
                g, beta = fused_gdn_gating(A_log, a, b, bias)
            else:
                q, k, v, g, beta = [
                    part.unsqueeze(0)
                    for part in fused_post_conv_prep(
                        conv_output=mixed,
                        a=a,
                        b=b,
                        A_log=A_log,
                        dt_bias=bias,
                        num_k_heads=hq,
                        head_k_dim=dim,
                        head_v_dim=dim,
                        apply_l2norm=True,
                        output_g_exp=gate_exp,
                    )
                ]
            initial = state if indexed else state[slots].contiguous()
            if not indexed:
                initial[~has_state] = 0
            extra = {}
            if selected == "flashqla_sm70":
                extra = dict(
                    state_indices=slots if indexed else None,
                    has_initial_state=has_state if indexed else None,
                    inplace_final_state=indexed,
                )
            result, final = provider._forward_method(
                q,
                k,
                v,
                g,
                beta,
                initial_state=initial,
                output_final_state=not indexed,
                cu_seqlens=cu,
                chunk_indices=chunks,
                chunk_offsets=offsets,
                use_qk_l2norm_in_kernel=legacy,
                gate_is_exp=gate_exp,
                core_attn_out=output if direct else None,
                **extra,
            )
            if not indexed:
                state[slots] = final.to(state.dtype)
        if not direct:
            output.copy_(result.squeeze(0))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for candidate in (0, 1):
            run(candidate)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(states[0], states[1])
    graphs = []
    for candidate in (0, 1):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(candidate)
        graphs.append(graph)
    for permutation, mask in (([3, 6], [False, True]), ([7, 0], [True, True])):
        mixed.add_(0.01)
        a.sub_(0.02)
        slots.copy_(torch.tensor(permutation, device="cuda", dtype=slots.dtype))
        has_state.copy_(torch.tensor(mask, device="cuda"))
        for state, graph in zip(states, graphs):
            state.copy_(seed)
            graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(states[0], states[1])
