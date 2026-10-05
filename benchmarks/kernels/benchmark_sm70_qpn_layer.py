# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight TP4 GDN layer graphs, including both collectives and norms.

Compare the installed native M8 route with an explicit K16 pack and with the
same packing absorbed into the preceding norm's output address. This is a
layer benchmark, not a model latency or quality admission.
"""

import argparse
import ctypes
import hashlib
import json
import os
import runpy
import statistics
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors import safe_open

from vllm import _custom_ops, _sm70_ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update_mixed_qkv_out as delta_update,
)
from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn
from vllm.model_executor.layers.layernorm import (
    _sm70_dflash2_gemma_fused_add_rms_norm as input_norm,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.mamba.gdn.sm70_preprocess import conv_gate_zero
from vllm.model_executor.layers.quantization.sm70_gdn_ba_verify import (
    apply_gdn_ba_verify,
)


class Projection(LinearBase):
    def __init__(self, weight, scales=None):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        if scales is not None:
            self.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
            self.sm70_fp8_qpn8 = True


def graph_counts(graph):
    runtime = ctypes.CDLL("libcudart.so.12")
    runtime.cudaGraphGetNodes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    runtime.cudaGraphNodeGetType.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    pointer = graph.raw_cuda_graph()
    count = ctypes.c_size_t()
    assert runtime.cudaGraphGetNodes(pointer, None, ctypes.byref(count)) == 0
    nodes = (ctypes.c_void_p * count.value)()
    assert runtime.cudaGraphGetNodes(pointer, nodes, ctypes.byref(count)) == 0
    types = []
    for node in nodes:
        kind = ctypes.c_int()
        assert runtime.cudaGraphNodeGetType(node, ctypes.byref(kind)) == 0
        types.append(kind.value)
    # No child graphs are emitted by this benchmark.
    assert 4 not in types
    return {
        "all_nodes": count.value,
        "kernel_nodes": types.count(0),
        "memcpy_nodes": types.count(1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--operand-library", type=Path, required=True)
    parser.add_argument("--norm-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ncu", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(20261005)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", rank), timeout=timedelta(seconds=120)
    )
    group = dist.new_group(backend="gloo")
    ca = CustomAllreduce(group, torch.device("cuda", rank), max_size=1024 * 1024)
    assert not ca.disabled and ca.fully_connected
    assert ca.sm70_tp4_push_buffer_ptrs is not None
    torch.ops.load_library(str(args.operand_library))
    torch.ops.load_library(str(args.norm_library))
    ext = torch.ops._qpn_operands700
    norm_ext = torch.ops._qpn_norm_layout700
    prefix = f"model.language_model.layers.{args.layer}."
    with safe_open(
        str(args.model / "model.safetensors"), framework="pt", device="cpu"
    ) as f:

        def read(name):
            return f.get_tensor(prefix + name)

        rows = (
            list(range(rank * 512, (rank + 1) * 512))
            + list(range(2048 + rank * 512, 2048 + (rank + 1) * 512))
            + list(range(4096 + rank * 1536, 4096 + (rank + 1) * 1536))
        )
        q = read("linear_attn.in_proj_qkv.weight").view(torch.uint8)[rows]
        z = read("linear_attn.in_proj_z.weight").view(torch.uint8)[
            rank * 1536 : (rank + 1) * 1536
        ]
        scales = (
            torch.cat(
                (
                    read("linear_attn.in_proj_qkv.weight_scale")[rows],
                    read("linear_attn.in_proj_z.weight_scale")[
                        rank * 1536 : (rank + 1) * 1536
                    ],
                )
            )
            .float()
            .cuda()
        )
        qpn_q, qpn_s = _sm70_ops.fp8_qpn8_prepare_sm70(
            torch.cat((q, z)).view(torch.float8_e4m3fn).cuda(), scales
        )
        outcode = (
            read("linear_attn.out_proj.weight")
            .view(torch.uint8)[:, rank * 1536 : (rank + 1) * 1536]
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        outq, outs = _sm70_ops.fp8_qpn8_prepare_sm70(
            outcode.cuda(), read("linear_attn.out_proj.weight_scale").float().cuda()
        )
        baweight = (
            torch.cat(
                (
                    read("linear_attn.in_proj_b.weight")[rank * 12 : (rank + 1) * 12],
                    read("linear_attn.in_proj_a.weight")[rank * 12 : (rank + 1) * 12],
                )
            )
            .half()
            .cuda()
        )
        conv = read("linear_attn.conv1d.weight").view(10240, 4)[rows].half().cuda()
        a_log = read("linear_attn.A_log")[rank * 12 : (rank + 1) * 12].float().cuda()
        bias = read("linear_attn.dt_bias")[rank * 12 : (rank + 1) * 12].half().cuda()
        norm = read("linear_attn.norm.weight").half().cuda()
        inputnorm = read("input_layernorm.weight").half().cuda()
        postnorm = read("post_attention_layernorm.weight").half().cuda()
    loader = runpy.run_path(
        str(args.source_root / "benchmarks/kernels/benchmark_sm70_nvfp4_qpn2.py")
    )["_load_projection_shards"]
    mlp = []
    for projection in loader(args.model, args.layer, rank, 4):
        w, s = _sm70_ops.nvfp4_qpn2_prepare_sm70(
            projection.packed.cuda(), projection.scales.cuda()
        )
        effective = (
            s.view(torch.float8_e4m3fn).float() * projection.inverse_global_scale
        ).half()
        assert torch.isfinite((effective.float() * 16384).half()).all().item()
        mlp.append((w, s, projection.inverse_global_scale))
    layer = SimpleNamespace(
        in_proj_qkvz=Projection(qpn_q, qpn_s),
        in_proj_ba=Projection(baweight),
        enable_sm70_dflash2_fused_gdn_verify=True,
        enable_sm70_gdn_ba_verify=True,
        gqa_interleaved_layout=False,
        disable_tp_for_ba_proj=False,
    )
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.125
    residual = torch.randn(8, 5120, device="cuda") * 0.125
    out = x.new_empty((8, 1536))
    projected, down, packed, post = [torch.empty_like(x) for _ in range(4)]
    next_residual = torch.empty_like(residual)
    up = x.new_empty((8, 4352))
    initial = torch.randn(11, 12, 128, 128, device="cuda") * 0.01
    history = (
        torch.randn(1, 10, 2560, device="cuda", dtype=torch.float16) * 0.125
    ).transpose(-1, -2)
    state = initial.clone()
    hist = history.clone(memory_format=torch.preserve_format)
    indices = torch.randperm(11, device="cuda").int()[:8]
    cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([4], device="cuda", dtype=torch.int32)
    conv_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    eviction = torch.ones(33554432, device="cuda", dtype=torch.int32)
    sink = torch.empty(256, device="cuda", dtype=torch.int64)
    buffer_bytes = _custom_ops.sm70_tp4_push_allreduce_buffer_size()

    def reset():
        state.copy_(initial)
        hist.copy_(history)

    def run(variant):
        normalized, _ = input_norm(x, residual, inputnorm, 1e-6)
        q, z, b, a = apply_gdn_ba_verify(layer, normalized)
        transformed, g, beta = conv_gate_zero(
            q,
            hist,
            conv,
            conv_indices,
            accepted,
            cu,
            a_log,
            a,
            b,
            bias,
            out.view(8, 12, 128),
        )
        delta_update(
            a_log,
            a,
            b,
            bias,
            transformed,
            4,
            12,
            128,
            128,
            out.view(8, 1, 12, 128),
            initial_state=state,
            cu_seqlens=cu,
            ssm_state_indices=indices[None],
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
            precomputed_g=g.view(8, 12),
            precomputed_beta=beta.view(8, 12),
            match_recurrent_schedule=True,
            match_recurrent_numerics=True,
        )
        core = rmsnorm_fn(
            out.view(96, 128),
            norm,
            None,
            z=z.reshape(96, 128),
            eps=1e-6,
            norm_before_gate=True,
        ).view(8, 1536)
        _sm70_ops.fp8_qpn8_gemm_sm70_out(
            projected, core, outq, outs, 12, 2, True, False
        )
        if variant in ["production", "explicit_pack"]:
            p, r = ca.sm70_tp4_all_reduce_gemma_rms_norm(
                projected, residual, postnorm, 1e-6
            )
            if variant == "explicit_pack":
                ext.pack(p, packed)
                p = packed
        else:
            norm_ext.run(
                post,
                next_residual,
                projected,
                residual,
                postnorm,
                ca.sm70_tp4_push_buffer_ptrs,
                rank,
                buffer_bytes,
                variant == "producer_pack",
            )
            p, r = post, next_residual
        if variant in ["production", "norm_control"]:
            _sm70_ops.nvfp4_qpn2_gated_sm70_out(up, p, *mlp[0], 8, 1)
            _sm70_ops.nvfp4_qpn2_gemm_sm70_out(down, up, *mlp[1], 16, 2)
        else:
            ext.run(up, p, *mlp[0], True, 7)
            ext.run(down, up, *mlp[1], False, 7)
        return ca.sm70_tp4_all_reduce_gemma_rms_norm(down, r, inputnorm, 1e-6)

    names = ["production", "norm_control", "explicit_pack", "producer_pack"]
    checks = []
    for amplitude in [0.0, 0.125, -0.125, 0.25]:
        x.copy_(torch.randn_like(x) * amplitude)
        reset()
        expected = [v.clone() for v in run("production")]
        for name in names[1:]:
            reset()
            actual = run(name)
            exact = all(
                torch.equal(a.view(torch.uint8), e.view(torch.uint8))
                for a, e in zip(actual, expected, strict=True)
            )
            checks.append(dict(variant=name, amplitude=amplitude, bitwise=exact))
            assert exact, (rank, checks[-1])
    graphs = []
    for name in names:
        for _ in range(5):
            reset()
            run(name)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        begin, end = [
            torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)
        ]
        with ca.capture(), torch.cuda.graph(graph):
            reset()
            ext.evict(eviction, sink)
            begin.record()
            run(name)
            end.record()
        graphs.append((graph, begin, end))
    counts = {
        name: graph_counts(g) for name, (g, _, _) in zip(names, graphs, strict=True)
    }
    if args.ncu:
        for graph, _, end in graphs:
            for _ in range(30):
                graph.replay()
            end.synchronize()
        dist.barrier(group=group)
        torch.cuda.cudart().cudaProfilerStart()
        for i in [0, 3]:
            graphs[i][0].replay()
            graphs[i][2].synchronize()
            dist.barrier(group=group)
        torch.cuda.cudart().cudaProfilerStop()
    if args.profile:
        for graph, _, end in graphs:
            for _ in range(30):
                graph.replay()
            end.synchronize()
        dist.barrier(group=group)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            for i in [0, 3]:
                with torch.profiler.record_function(names[i]):
                    for _ in range(15):
                        graphs[i][0].replay()
                    graphs[i][2].synchronize()
                dist.barrier(group=group)
        args.output.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.output / f"rank{rank}.trace.json"))
    samples = {name: [] for name in names}
    for repeat in range(45):
        for offset in range(len(names)):
            i = (repeat + offset) % len(names)
            graph, begin, end = graphs[i]
            dist.barrier(group=group)
            for _ in range(10):
                graph.replay()
            end.synchronize()
            if repeat >= 5:
                samples[names[i]].append(begin.elapsed_time(end) * 1000)
    result = dict(
        scope="Complete TP4 M8 GDN layer including both AR+norm boundaries",
        research_only=True,
        original_weights=True,
        layer=args.layer,
        rank=rank,
        cold_l2_bytes=eviction.numel() * eviction.element_size(),
        checks=checks,
        graph_nodes_including_reset_and_eviction=counts,
        timing={
            name: dict(mean_us=statistics.mean(s), samples_us=s)
            for name, s in samples.items()
        },
        library_sha256={
            str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [args.operand_library, args.norm_library]
        },
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                rank=rank,
                graph_counts=counts,
                mean_us={name: statistics.mean(s) for name, s in samples.items()},
            )
        ),
        flush=True,
    )
    for graph, _, _ in graphs:
        graph.reset()
    ca.close()
    dist.barrier(group=group)
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
