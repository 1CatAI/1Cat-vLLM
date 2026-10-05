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
from benchmark_sm70_gdn_rollback_replay import candidate_module, launch
from safetensors import safe_open

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update_kernel as reference_kernel,
)
from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn
from vllm.model_executor.layers.layernorm import (
    _sm70_dflash2_gemma_fused_add_rms_norm as gemma_norm,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fused_gdn_gating
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accepted", type=int, default=3, choices=range(1, 9))
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
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
        alog = read("linear_attn.A_log")[:12].float().cuda()
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
    history_seed = torch.randn(1, 2560, 10, device="cuda", dtype=x.dtype) * 0.125
    history = history_seed.clone()
    initial = torch.randn(1, 12, 128, 128, device="cuda") * 0.01
    snapshots = initial.expand(8, -1, -1, -1).clone()
    compact = initial.clone()
    indices = torch.arange(8, device="cuda", dtype=torch.int32)[None]
    compact_indices = torch.zeros_like(indices)
    accepted = torch.tensor([args.accepted], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    neutral = torch.ones_like(accepted)
    conv_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    caches = [
        (
            initial.new_empty(8, 12, 128),
            initial.new_empty(8, 12, 128),
            initial.new_empty(8, 12),
        )
        for _ in range(2)
    ]
    candidate = candidate_module(args.out).rollback_replay_kernel

    def prepare():
        normalized, res = gemma_norm(x, residual, inputnorm, 1e-6)
        ops.fp8_qpn8_dispatch_ba_split_sm70_out(
            q, z, b, a, scratch_q, scratch_b, 0, normalized, qcodes, qscales, baweight
        )
        transformed = causal_conv1d_update(
            q,
            history,
            conv,
            None,
            "silu",
            conv_state_indices=conv_indices,
            num_accepted_tokens=accepted,
            query_start_loc=cu,
            max_query_len=8,
            validate_data=False,
        )
        g, beta = fused_gdn_gating(alog, a, b, bias, beta_dtype=torch.float32)
        return transformed, g.view(8, 12), beta.view(8, 12), res

    history.copy_(history_seed)
    mixed, g, beta, _ = prepare()
    launch(
        reference_kernel,
        mixed,
        g,
        beta,
        snapshots,
        indices,
        core,
        metadata=(cu, neutral),
    )
    launch(
        candidate,
        mixed,
        g,
        beta,
        compact,
        compact_indices,
        core,
        metadata=(cu, neutral),
        factors=(*caches[1], *caches[0]),
    )
    snapshot_seed = snapshots.clone()

    def run(use_replay):
        mixed, g, beta, res = prepare()
        if use_replay:
            launch(
                candidate,
                mixed,
                g,
                beta,
                compact,
                compact_indices,
                core,
                previous=(mixed, g, beta, accepted),
                metadata=(cu, accepted),
                factors=(*caches[0], *caches[1]),
            )
        else:
            launch(
                reference_kernel,
                mixed,
                g,
                beta,
                snapshots,
                indices,
                core,
                metadata=(cu, accepted),
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

    history.copy_(history_seed)
    run(False)
    expected = down.clone()
    history.copy_(history_seed)
    compact.copy_(initial)
    run(True)
    torch.testing.assert_close(down, expected, rtol=0, atol=0)
    eviction = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    graphs, samples = [], [[], []]
    for use_replay in (False, True):
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            snapshots.copy_(snapshot_seed)
            compact.copy_(initial)
            history.copy_(history_seed)
            eviction.fill_(1)
            start.record()
            run(use_replay)
            end.record()
        graphs.append((graph, start, end))
    for _ in range(10):
        for graph, _, end in graphs:
            graph.replay()
            end.synchronize()
    for _ in range(args.iters):
        for index, (graph, start, end) in enumerate(graphs):
            graph.replay()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000)
    baseline_us, candidate_us = (statistics.fmean(s) for s in samples)
    result = dict(
        baseline_layer_us=baseline_us,
        replay_layer_us=candidate_us,
        estimated_48_layer_saving_ms=(baseline_us - candidate_us) * 48 / 1000,
        accepted=args.accepted,
        final_layer_output_bitwise=True,
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
