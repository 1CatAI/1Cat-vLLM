# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-L2 real-weight GDN layer screen; unchanged TP collectives excluded.

Measure rollback through its following projections, which can evict dirty state
snapshots from L2. The isolated recurrent kernel can miss this deferred traffic.
This does not change model dispatch or claim a serving latency.
"""

import argparse
import json
import runpy
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update_mixed_qkv_out as reference,
)
from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn
from vllm.model_executor.layers.layernorm import (
    _sm70_dflash2_gemma_fused_add_rms_norm as gemma_norm,
)
from vllm.model_executor.layers.mamba.gdn.sm70_preprocess import conv_gate_zero


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accepted", type=int, default=3, choices=range(1, 9))
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--variant", type=int, default=3, choices=range(5))
    parser.add_argument(
        "--research-extension",
        type=Path,
        help="Optional isolated source extension for microbenchmarks only",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.research_extension:
        torch.ops.load_library(str(args.research_extension))
    if not hasattr(torch.ops._C, "gdn_wy_verify_sm70_out"):
        raise RuntimeError(
            "Build the WY operators from this source before benchmarking"
        )
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    assert torch.cuda.get_device_capability() == (7, 0)
    prefix = "model.language_model.layers.0."
    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as f:

        def read(name):
            return f.get_tensor(prefix + name)

        rows = list(range(512)) + list(range(2048, 2560)) + list(range(4096, 5632))
        codes = (
            torch.cat(
                (
                    read("linear_attn.in_proj_qkv.weight").view(torch.uint8)[rows],
                    read("linear_attn.in_proj_z.weight").view(torch.uint8)[:1536],
                )
            )
            .view(torch.float8_e4m3fn)
            .cuda()
        )
        scales = (
            torch.cat(
                (
                    read("linear_attn.in_proj_qkv.weight_scale")[rows],
                    read("linear_attn.in_proj_z.weight_scale")[:1536],
                )
            )
            .float()
            .cuda()
        )
        qcodes, qscales = ops.fp8_qpn8_prepare_sm70(codes, scales)
        outcodes, outscales = ops.fp8_qpn8_prepare_sm70(
            read("linear_attn.out_proj.weight")[:, :1536].contiguous().cuda(),
            read("linear_attn.out_proj.weight_scale").float().cuda(),
        )
        baweight = (
            torch.cat(
                (
                    read("linear_attn.in_proj_b.weight")[:12],
                    read("linear_attn.in_proj_a.weight")[:12],
                )
            )
            .half()
            .cuda()
        )
        conv = read("linear_attn.conv1d.weight").view(10240, 4)[rows].half().cuda()
        decay_log = read("linear_attn.A_log")[:12].float().cuda()
        bias = read("linear_attn.dt_bias")[:12].half().cuda()
        norm = read("linear_attn.norm.weight").half().cuda()
        inputnorm = read("input_layernorm.weight").half().cuda()
        postnorm = read("post_attention_layernorm.weight").half().cuda()
    helpers = runpy.run_path(
        str(Path(__file__).with_name("benchmark_sm70_nvfp4_qpn2.py"))
    )
    mlp = []
    for projection in helpers["_load_projection_shards"](args.model, 0, 0, 4):
        w, s = ops.nvfp4_qpn2_prepare_sm70(
            projection.packed.cuda(), projection.scales.cuda()
        )
        mlp.append((w, s, projection.inverse_global_scale))
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.125
    residual = torch.randn(8, 5120, device="cuda") * 0.125
    q, z = x.new_empty(8, 2560), x.new_empty(8, 1536)
    b, a = x.new_empty(8, 12), x.new_empty(8, 12)
    scratch_q, scratch_b = x.new_empty(8, 4096), x.new_empty(8, 24)
    core = x.new_empty(8, 1, 12, 128)
    projected, down = x.new_empty(8, 5120), x.new_empty(8, 5120)
    up = x.new_empty(8, 4352)
    history_storage_seed = (
        torch.randn(1, 10, 2560, device="cuda", dtype=x.dtype) * 0.125
    )
    history_storage = history_storage_seed.clone()
    history = history_storage.transpose(1, 2)
    candidate_history_seed = history_storage_seed.clone()
    candidate_history_seed[:, :3].copy_(
        history_storage_seed[:, args.accepted - 1 : args.accepted + 2]
    )
    initial = torch.randn(1, 12, 128, 128, device="cuda") * 0.01
    snapshots = initial.expand(8, -1, -1, -1).clone()
    compact = initial.clone()
    caches = [
        torch.empty(1, 8, 12, 128, device="cuda"),
        torch.empty(1, 8, 4, 128, device="cuda"),
        torch.empty(1, 8, 12, device="cuda"),
    ]
    pending = torch.full((1,), -1, device="cuda", dtype=torch.int32)
    no_previous = torch.zeros(1, device="cuda", dtype=torch.int32)
    batch_mapping = torch.zeros(1, device="cuda", dtype=torch.int32)
    descriptors = torch.tensor(
        [
            [
                compact.data_ptr(),
                compact.stride(0),
                caches[0].data_ptr(),
                caches[1].data_ptr(),
                caches[2].data_ptr(),
                pending.data_ptr(),
                0,
                1,
                1,
            ]
        ],
        device="cuda",
        dtype=torch.int64,
    )
    conv_descriptors = torch.tensor(
        [[history_storage.data_ptr(), history_storage.stride(0), 2560]],
        device="cuda",
        dtype=torch.int64,
    )
    table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    table_pointers = torch.tensor([table.data_ptr()], device="cuda", dtype=torch.int64)
    computed = torch.full((1,), 1024 + args.accepted, device="cuda", dtype=torch.int32)
    next_counts = torch.ones(1, device="cuda", dtype=torch.int32)
    indices = torch.arange(8, device="cuda", dtype=torch.int32)[None]
    compact_indices = torch.zeros_like(indices)
    accepted = torch.tensor([args.accepted], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    neutral = torch.ones_like(accepted)
    conv_indices = torch.zeros(1, device="cuda", dtype=torch.int32)

    def prepare(use_wy):
        normalized, res = gemma_norm(x, residual, inputnorm, 1e-6)
        ops.fp8_qpn8_dispatch_ba_split_sm70_out(
            q, z, b, a, scratch_q, scratch_b, 0, normalized, qcodes, qscales, baweight
        )
        transformed, g, beta = conv_gate_zero(
            q,
            history,
            conv,
            conv_indices,
            neutral if use_wy else accepted,
            cu,
            decay_log,
            a,
            b,
            bias,
            core,
        )
        return transformed, g.view(8, 12), beta.view(8, 12), res

    def run(use_wy, publish=True):
        mixed, g, beta, res = prepare(use_wy)
        if use_wy:
            qv, kv, vv = mixed.split([512, 512, 1536], dim=1)
            torch.ops._C.gdn_wy_verify_sm70_out(
                core.reshape(8, 1536),
                compact,
                *caches,
                qv,
                kv,
                vv,
                a,
                b,
                decay_log,
                bias,
                compact_indices,
                conv_indices,
                pending,
                cu,
                no_previous,
                *caches,
                128**-0.5,
                args.variant,
            )
        else:
            reference(
                A_log=decay_log,
                a=a,
                b=b,
                dt_bias=bias,
                mixed_qkv=mixed,
                num_q_heads=4,
                num_v_heads=12,
                head_k_dim=128,
                head_v_dim=128,
                scale=128**-0.5,
                initial_state=snapshots,
                out=core,
                cu_seqlens=cu,
                ssm_state_indices=indices,
                num_accepted_tokens=accepted,
                use_qk_l2norm_in_kernel=True,
                precomputed_g=g,
                precomputed_beta=beta,
                quantize_state_each_step=False,
                match_recurrent_schedule=True,
                match_recurrent_numerics=True,
            )
        normalized = rmsnorm_fn(
            core.view(96, 128),
            norm,
            None,
            z=z.reshape(96, 128),
            eps=1e-6,
            norm_before_gate=True,
        ).view(8, 1536)
        ops.fp8_qpn8_gemm_sm70_out(
            projected, normalized, outcodes, outscales, 12, 2, True, False
        )
        post, _ = gemma_norm(projected, res, postnorm, 1e-6)
        w, s, scale = mlp[0]
        ops.nvfp4_qpn2_gated_sm70_out(up, post, w, s, scale, 8, 1)
        w, s, scale = mlp[1]
        ops.nvfp4_qpn2_gemm_sm70_out(down, up, w, s, scale, 16, 2)
        if use_wy and publish:
            torch.ops._C.gdn_wy_commit_group_v2_sm70(
                [compact],
                [pending],
                descriptors,
                table_pointers,
                1,
                accepted,
                computed,
                batch_mapping,
                8192,
            )
            # No boundary is crossed in this microbenchmark. Publication still
            # normalizes convolution history and includes that launch cost.
            torch.ops._C.gdn_wy_finish_group_v2_sm70(
                [pending],
                [history_storage],
                descriptors,
                conv_descriptors,
                table_pointers,
                1,
                accepted,
                computed,
                batch_mapping,
                8192,
                next_counts,
            )

    history_storage.copy_(history_storage_seed)
    run(False)
    expected = down.clone()
    expected_state = snapshots[args.accepted - 1].clone()
    history_storage.copy_(candidate_history_seed)
    compact.copy_(initial)
    run(True)
    output_difference = (down - expected).abs().float()
    state_difference = (compact[0] - expected_state).abs()
    assert torch.isfinite(down).all() and torch.isfinite(compact).all()
    eviction = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    graphs, samples = [], [[], [], []]
    arms = [(False, False), (True, False), (True, True)]
    for use_wy, publish in arms:
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            snapshots.copy_(initial.expand_as(snapshots))
            compact.copy_(initial)
            history_storage.copy_(
                candidate_history_seed if use_wy else history_storage_seed
            )
            eviction.fill_(1)
            start.record()
            run(use_wy, publish)
            end.record()
        graphs.append((graph, start, end))
    # Actual publication uses one launch for all layers, not two launches per
    # layer. Keep its factor addresses distinct and measure it separately.
    pub_states_storage = initial.expand(48, 1, 12, 128, 128).clone()
    pub_states = list(pub_states_storage.unbind(0))
    pub_u = caches[0].expand(48, 1, 8, 12, 128).clone()
    pub_k = caches[1].expand(48, 1, 8, 4, 128).clone()
    pub_G = caches[2].expand(48, 1, 8, 12).clone()
    pub_pending_storage = torch.zeros(48, 1, device="cuda", dtype=torch.int32)
    pub_pending = list(pub_pending_storage.unbind(0))
    pub_history = history_storage_seed.expand(48, 1, 10, 2560).clone()
    pub_conv = list(pub_history.unbind(0))
    pub_desc = torch.tensor(
        [
            [
                pub_states[i].data_ptr(),
                pub_states[i].stride(0),
                pub_u[i].data_ptr(),
                pub_k[i].data_ptr(),
                pub_G[i].data_ptr(),
                pub_pending[i].data_ptr(),
                0,
                1,
                1,
            ]
            for i in range(48)
        ],
        device="cuda",
        dtype=torch.int64,
    )
    pub_conv_desc = torch.tensor(
        [[t.data_ptr(), t.stride(0), 2560] for t in pub_conv],
        device="cuda",
        dtype=torch.int64,
    )
    pub_graph = torch.cuda.CUDAGraph()
    pub_start = torch.cuda.Event(enable_timing=True, external=True)
    pub_end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(pub_graph):
        pub_states_storage.copy_(initial.expand_as(pub_states_storage))
        pub_history.copy_(history_storage_seed.expand_as(pub_history))
        pub_pending_storage.zero_()
        eviction.fill_(1)
        pub_start.record()
        torch.ops._C.gdn_wy_commit_group_v2_sm70(
            pub_states,
            pub_pending,
            pub_desc,
            table_pointers,
            1,
            accepted,
            computed,
            batch_mapping,
            8192,
        )
        torch.ops._C.gdn_wy_finish_group_v2_sm70(
            pub_pending,
            pub_conv,
            pub_desc,
            pub_conv_desc,
            table_pointers,
            1,
            accepted,
            computed,
            batch_mapping,
            8192,
            next_counts,
        )
        pub_end.record()
    graphs.append((pub_graph, pub_start, pub_end))
    samples.append([])
    for _ in range(10):
        for graph, _, end in graphs:
            graph.replay()
            end.synchronize()
    for _ in range(args.iters):
        for index, (graph, start, end) in enumerate(graphs):
            graph.replay()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000)
    baseline_us, wy_layer_us, wy_with_publication_us, publication_48_us = (
        statistics.fmean(s) for s in samples
    )
    result = dict(
        baseline_layer_us=baseline_us,
        wy_layer_us=wy_layer_us,
        wy_layer_with_single_layer_publication_us=wy_with_publication_us,
        estimated_48_layer_target_saving_ms=(baseline_us - wy_layer_us) * 48 / 1000,
        publication_48_layers_us=publication_48_us,
        estimated_48_layer_saving_ms=(
            (baseline_us - wy_layer_us) * 48 - publication_48_us
        )
        / 1000,
        publication_note=(
            "Target-layer saving minus measured 48-layer publication; "
            "TP collectives, head, and sampling remain excluded"
        ),
        accepted=args.accepted,
        max_layer_output_difference=output_difference.max().item(),
        mean_layer_output_difference=output_difference.mean().item(),
        max_state_difference=state_difference.max().item(),
        numerical_contract="Full-model teacher-forcing admission is separate",
        publication_and_conv_continuation_included=True,
        publication_batched_across_layers_in_production=True,
        variant=args.variant,
        layer=0,
        checkpoint=str(args.model),
        tp_shard=0,
        unchanged_tp_collectives_excluded=True,
        production_dispatch_changed=False,
    )
    (args.out / "layer-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
