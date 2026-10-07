# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare exact paired gate/up warps in a complete real-weight TP4 GDN layer.

Control and candidate share bundled weights, down projection, state updates,
and norms/collectives. Extensions are research-only, not serving artifacts.
"""

import argparse
import ctypes
import importlib.util
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

from vllm import _sm70_ops as ops
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


def nodes(graph):
    driver = ctypes.CDLL("libcuda.so.1")
    handle = ctypes.c_void_p(graph.raw_cuda_graph())
    count = ctypes.c_size_t()
    assert driver.cuGraphGetNodes(handle, None, ctypes.byref(count)) == 0
    values = (ctypes.c_void_p * count.value)()
    assert driver.cuGraphGetNodes(handle, values, ctypes.byref(count)) == 0
    kinds = {}
    for value in values:
        kind = ctypes.c_int()
        assert (
            driver.cuGraphNodeGetType(ctypes.c_void_p(value), ctypes.byref(kind)) == 0
        )
        kinds[str(kind.value)] = kinds.get(str(kind.value), 0) + 1
    assert "4" not in kinds and "13" not in kinds
    return {"nodes": count.value, "types": kinds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--mode", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--prepared-qk", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=150)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", rank), timeout=timedelta(seconds=120)
    )
    ca = CustomAllreduce(
        dist.new_group(backend="gloo"), torch.device("cuda", rank), max_size=1024 * 1024
    )
    assert not ca.disabled and ca.fully_connected
    spec = importlib.util.spec_from_file_location(
        "qpn2_paired_gate_screen", args.extension
    )
    assert spec and spec.loader
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    table = torch.empty(0, dtype=torch.float16)
    with safe_open(
        str(args.model / "model.safetensors"), framework="pt", device="cpu"
    ) as f:

        def read(name):
            return f.get_tensor("model.language_model.layers.0." + name)

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
        qpn, qs = ops.fp8_qpn8_prepare_sm70(
            torch.cat((q, z)).view(torch.float8_e4m3fn).cuda(), scales
        )
        outcode = (
            read("linear_attn.out_proj.weight")
            .view(torch.uint8)[:, rank * 1536 : (rank + 1) * 1536]
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        outq, outs = ops.fp8_qpn8_prepare_sm70(
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
        str(Path(__file__).with_name("benchmark_sm70_nvfp4_qpn2.py"))
    )["_load_projection_shards"]
    mlp = []
    for projection in loader(args.model, 0, rank, 4):
        w, s = ops.nvfp4_qpn2_prepare_sm70(
            projection.packed.cuda(), projection.scales.cuda()
        )
        n, k = projection.packed.shape[0], projection.packed.shape[1] * 2
        bundle = torch.cat(
            (w.view(n // 32, k // 16, 256), s.view(n // 32, k // 16, 32)), dim=-1
        ).contiguous()
        mlp.append(
            (w, s, bundle, projection.inverse_global_scale, projection.gated_silu)
        )
    layer = SimpleNamespace(
        in_proj_qkvz=Projection(qpn, qs),
        in_proj_ba=Projection(baweight),
        enable_sm70_dflash2_fused_gdn_verify=True,
        enable_sm70_gdn_ba_verify=True,
        gqa_interleaved_layout=False,
        disable_tp_for_ba_proj=False,
    )
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.125
    residual = torch.randn(8, 5120, device="cuda") * 0.125
    core_out = x.new_empty(8, 1536)
    projected = torch.empty_like(x)
    down = torch.empty_like(x)
    up = x.new_empty(8, 4352)
    initial = torch.randn(11, 12, 128, 128, device="cuda") * 0.01
    history = (
        torch.randn(1, 10, 2560, device="cuda", dtype=torch.float16).transpose(-1, -2)
        * 0.125
    )
    state = initial.clone()
    hist = history.clone(memory_format=torch.preserve_format)
    indices = torch.randperm(11, device="cuda").int()[:8]
    cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([4], device="cuda", dtype=torch.int32)
    conv_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    def reset():
        state.copy_(initial)
        hist.copy_(history)

    def run(mode, prepare=False):
        normalized, _ = input_norm(x, residual, inputnorm, 1e-6)
        q, z, b, a = apply_gdn_ba_verify(layer, normalized)
        conv_fn, delta_fn = conv_gate_zero, delta_update
        if prepare:
            from sm70_gdn_prepared_qk_screen import conv_prepare, prepared_delta

            conv_fn, delta_fn = conv_prepare, prepared_delta
        transformed, g, beta = conv_fn(
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
            core_out.view(8, 12, 128),
        )
        delta_fn(
            a_log,
            a,
            b,
            bias,
            transformed,
            4,
            12,
            128,
            128,
            core_out.view(8, 1, 12, 128),
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
            core_out.view(96, 128),
            norm,
            None,
            z=z.reshape(96, 128),
            eps=1e-6,
            norm_before_gate=True,
        ).view(8, 1536)
        ops.fp8_qpn8_gemm_sm70_out(projected, core, outq, outs, 12, 2, True, False)
        p, r = ca.sm70_tp4_all_reduce_gemma_rms_norm(
            projected, residual, postnorm, 1e-6
        )
        for item, operand, result in ((mlp[0], p, up), (mlp[1], up, down)):
            w, s, bundle, scale, gated = item
            if gated and mode:
                extension.pair(result, operand, bundle, s, scale, mode)
            else:
                extension.launch(result, operand, bundle, s, table, scale, gated, 0)
        return ca.sm70_tp4_all_reduce_gemma_rms_norm(down, r, inputnorm, 1e-6)

    differences = []
    for amplitude in (0.01, 0.125, 1.0, 4.0):
        x.normal_().mul_(amplitude)
        reset()
        control_out = run(0)
        golden = [v.clone() for v in control_out] + [state.clone(), hist.clone()]
        reset()
        candidate_out = run(args.mode, args.prepared_qk)
        for result, reference in zip(list(candidate_out) + [state, hist], golden):
            bits = torch.int16 if result.element_size() == 2 else torch.int32
            exact = torch.equal(result.view(bits), reference.view(bits))
            differences.append(
                {
                    "amplitude": amplitude,
                    "bitwise": exact,
                    "max_abs": (result.float() - reference.float()).abs().max().item(),
                }
            )
            if not args.prepared_qk:
                assert exact
    graphs = []
    counts = []
    for arm in range(2):
        mode, prepare = (args.mode, args.prepared_qk) if arm else (0, False)
        reset()
        run(mode, prepare)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with ca.capture(), torch.cuda.graph(graph):
            reset()
            eviction.fill_(1)
            start.record()
            run(mode, prepare)
            end.record()
        counts.append(nodes(graph))
        graph.instantiate()
        graphs.append((graph, start, end))
    assert counts[0]["types"].get("0") == counts[1]["types"].get("0")
    samples = [[], []]
    for iteration in range(args.iters + 20):
        for index in (iteration % 2, 1 - iteration % 2):
            graph, start, end = graphs[index]
            graph.replay()
            end.synchronize()
            if iteration >= 20:
                samples[index].append(start.elapsed_time(end) * 1000)
    result = {
        "rank": rank,
        "mode": args.mode,
        "control_mean_us": statistics.mean(samples[0]),
        "candidate_mean_us": statistics.mean(samples[1]),
        "samples_us": samples,
        "graph_nodes": counts,
        "output_and_state_bitwise": all(d["bitwise"] for d in differences),
        "differences": differences,
        "prepared_qk": args.prepared_qk,
        "four_amplitudes": [0.01, 0.125, 1.0, 4.0],
        "scope": (
            "Complete real-weight TP4 layer0 GDN graph; unchanged norms/collectives "
            + (
                "and BV2 recurrence; precompute Q/K normalization in convolution."
                if args.prepared_qk
                else "and state update; shared activation registers in paired warps."
            )
        ),
    }
    (args.out / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k != "samples_us"}), flush=True
    )
    ca.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
