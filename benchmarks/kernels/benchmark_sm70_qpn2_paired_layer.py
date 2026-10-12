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
    parser.add_argument("--paired-control-extension", type=Path)
    parser.add_argument("--wheel-paired-control", action="store_true")
    parser.add_argument("--pipeline-extension", type=Path)
    parser.add_argument("--resident-mlp-extension", type=Path)
    parser.add_argument(
        "--pipeline-arm", choices=("both", "gate", "down"), default="both"
    )
    parser.add_argument("--prepared-qk", action="store_true")
    parser.add_argument("--projection-conv-extension", type=Path)
    parser.add_argument("--cooperative-core-extension", type=Path)
    parser.add_argument("--head-local-core-extension", type=Path)
    parser.add_argument("--gdn-out-stage-extension", type=Path)
    parser.add_argument("--head-local-convolution", action="store_true")
    parser.add_argument("--cooperative-no-qk-cache", action="store_true")
    parser.add_argument("--norm-partial-extension", type=Path)
    parser.add_argument("--norm-gate-extension", type=Path)
    parser.add_argument("--norm-weight-stage-extension", type=Path)
    parser.add_argument("--dump-weight-stage", action="store_true")
    parser.add_argument("--debug-weight-stage", action="store_true")
    parser.add_argument("--resident-down-norm-extension", type=Path)
    parser.add_argument("--resident-out-norm", action="store_true")
    parser.add_argument("--resident-out-only", action="store_true")
    parser.add_argument("--resident-weight-window", action="store_true")
    parser.add_argument("--resident-control-extension", type=Path)
    parser.add_argument("--resident-native-ca-buffers", action="store_true")
    parser.add_argument("--resident-fp16-norm-weights", action="store_true")
    parser.add_argument("--packed-input-extension", type=Path)
    parser.add_argument("--norm-packet-parts", type=int, choices=(5, 10, 20), default=5)
    parser.add_argument("--warp-publish", action="store_true")
    parser.add_argument("--dump-gdn-inputs", type=Path)
    parser.add_argument("--hybrid-snapshot", action="store_true")
    parser.add_argument("--hybrid-snapshots", type=int, choices=(1, 4), default=4)
    parser.add_argument("--inline-conv", action="store_true")
    parser.add_argument("--ba-order-extension", type=Path)
    parser.add_argument("--epilogue-scale-extension", type=Path)
    parser.add_argument("--ba-order", type=int, choices=(1, 2), default=1)
    parser.add_argument("--paired-accumulators", type=int, choices=(2, 4))
    parser.add_argument("--gate-group-layout", action="store_true")
    parser.add_argument("--accepted", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--ncu-direct", action="store_true")
    parser.add_argument("--ncu-graph", action="store_true")
    parser.add_argument("--ncu-candidate", action="store_true")
    parser.add_argument("--cuda-profiler-capture", action="store_true")
    parser.add_argument("--delta-bv", type=int, choices=(1, 2, 4))
    parser.add_argument("--gdn-weight-window", action="store_true")
    parser.add_argument("--gdn-prefetch-token", type=int, choices=range(8), default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["LOCAL_RANK"])
    rank_source = args.out / f"rank{rank}-source"
    rank_source.mkdir(parents=True, exist_ok=True)
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
        args.extension.name.split(".")[0], args.extension
    )
    assert spec and spec.loader
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    paired_control = None
    pipeline = None
    resident_mlp = None
    mlp_ready = None
    if args.resident_mlp_extension:
        assert args.wheel_paired_control
        spec = importlib.util.spec_from_file_location(
            args.resident_mlp_extension.name.split(".")[0], args.resident_mlp_extension
        )
        assert spec and spec.loader
        resident_mlp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resident_mlp)
        mlp_ready = torch.zeros(296, device="cuda", dtype=torch.int32)
    if args.pipeline_extension:
        assert args.wheel_paired_control
        spec = importlib.util.spec_from_file_location(
            args.pipeline_extension.name.split(".")[0], args.pipeline_extension
        )
        assert spec and spec.loader
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
    if args.paired_control_extension:
        spec = importlib.util.spec_from_file_location(
            args.paired_control_extension.name.split(".")[0],
            args.paired_control_extension,
        )
        assert spec and spec.loader
        paired_control = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(paired_control)
    if args.paired_accumulators is not None:
        assert args.extension.name.split(".")[0] in (
            "qpn2_paired_acc_screen",
            "qpn2_paired_static_acc_screen",
        )
        assert args.mode == {2: 1, 4: 2}[args.paired_accumulators]
    table = torch.empty(0, dtype=torch.float16)
    fused_projection = None
    if args.projection_conv_extension:
        assert not args.prepared_qk and args.mode == 0
        spec = importlib.util.spec_from_file_location(
            "gdn_projection_conv_screen", args.projection_conv_extension
        )
        assert spec and spec.loader
        fused_projection = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fused_projection)
    cooperative = None
    if args.cooperative_core_extension:
        assert not args.prepared_qk and not args.hybrid_snapshot
        assert fused_projection is None and args.mode == 0
        spec = importlib.util.spec_from_file_location(
            "gdn_cooperative_core_screen", args.cooperative_core_extension
        )
        assert spec and spec.loader
        cooperative = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cooperative)
    packet_norm = None
    if args.norm_partial_extension:
        assert not args.prepared_qk and not args.hybrid_snapshot
        assert fused_projection is None and cooperative is None
        name = args.norm_partial_extension.name.split(".")[0]
        spec = importlib.util.spec_from_file_location(name, args.norm_partial_extension)
        assert spec and spec.loader
        packet_norm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(packet_norm)
    norm_gate = None
    if args.norm_gate_extension:
        assert args.mode == 1 and packet_norm is None and cooperative is None
        assert fused_projection is None and not args.hybrid_snapshot
        name = args.norm_gate_extension.name.split(".")[0]
        spec = importlib.util.spec_from_file_location(name, args.norm_gate_extension)
        assert spec and spec.loader
        norm_gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(norm_gate)
    if args.warp_publish:
        assert packet_norm is not None and args.norm_packet_parts == 20
        assert args.mode == 0
    ba_order = None
    if args.ba_order_extension:
        assert args.mode == 0 and not args.inline_conv and not args.hybrid_snapshot
        assert fused_projection is None and cooperative is None and packet_norm is None
        spec = importlib.util.spec_from_file_location(
            args.ba_order_extension.name.split(".")[0], args.ba_order_extension
        )
        assert spec and spec.loader
        ba_order = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ba_order)
    epilogue_scale = None
    if args.epilogue_scale_extension:
        assert args.mode == 0 and not args.inline_conv and not args.hybrid_snapshot
        assert fused_projection is None and cooperative is None and packet_norm is None
        assert ba_order is None
        spec = importlib.util.spec_from_file_location(
            "qpn8_epilogue_scale_screen", args.epilogue_scale_extension
        )
        assert spec and spec.loader
        epilogue_scale = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(epilogue_scale)
    head_local = None
    head_extension = args.head_local_core_extension or args.gdn_out_stage_extension
    if head_extension:
        assert args.mode == 0 and not args.inline_conv and not args.hybrid_snapshot
        assert fused_projection is None and cooperative is None and packet_norm is None
        assert ba_order is None and epilogue_scale is None
        spec = importlib.util.spec_from_file_location(
            head_extension.name.split(".")[0],
            head_extension,
        )
        assert spec and spec.loader
        head_local = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(head_local)
        head_packets = torch.zeros(12, 8, 8, device=rank, dtype=torch.int64)
        head_generations = torch.zeros(12, device=rank, dtype=torch.int32)
        head_conv_ready = torch.zeros(4, 24, device=rank, dtype=torch.int32)
        if args.gdn_out_stage_extension:
            assert args.resident_out_norm and args.resident_native_ca_buffers
            head_out_ready = torch.zeros(256, device=rank, dtype=torch.int32)
    assert not args.head_local_convolution or head_local is not None
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
    norm_gate_buffers = None
    if norm_gate is not None:
        norm_gate_group = dist.new_group(backend="gloo")
        norm_gate_buffers = CustomAllreduce.create_shared_buffer(
            norm_gate.buffer_bytes(), norm_gate_group
        )
        norm_gate.initialize(norm_gate_buffers, rank)
        norm_gate_input = torch.empty(8, 5120, device="cuda", dtype=torch.float16)
        norm_gate_residual = torch.empty(8, 5120, device="cuda")
        norm_gate_weight = postnorm.float()
    candidate_gate_bundle = None
    if args.gate_group_layout:
        assert args.mode in (1, 2) and args.paired_accumulators is None
        assert args.extension.name.split(".")[0] == "qpn2_gate_group_layout_screen"
        candidate_gate_bundle = (
            mlp[0][2].view(2, 136, 320, 288).permute(1, 2, 0, 3).contiguous()
        )
    resident_down_norm = None
    resident_control = None
    if args.resident_control_extension:
        assert args.resident_out_norm and not args.resident_out_only
        spec = importlib.util.spec_from_file_location(
            args.resident_control_extension.name.split(".")[0],
            args.resident_control_extension,
        )
        assert spec and spec.loader
        resident_control = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resident_control)
    packed_input = None
    if args.packed_input_extension:
        assert args.resident_out_norm and not args.resident_out_only
        spec = importlib.util.spec_from_file_location(
            args.packed_input_extension.name.split(".")[0], args.packed_input_extension
        )
        assert spec and spec.loader
        packed_input = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(packed_input)
    if args.resident_down_norm_extension:
        assert (
            paired_control is not None or args.wheel_paired_control
        ) and args.mode == 0
        assert packet_norm is None and norm_gate is None
        spec = importlib.util.spec_from_file_location(
            args.resident_down_norm_extension.name.split(".")[0],
            args.resident_down_norm_extension,
        )
        assert spec and spec.loader
        resident_down_norm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resident_down_norm)
        if not hasattr(resident_down_norm, "fused"):
            resident_down_norm.fused = resident_down_norm.fused_packed
        if args.resident_native_ca_buffers:
            # Alternate ordinary and resident kernels in the actual
            # communicator's packet generations, with no private metadata.
            down_norm_buffers = ca.sm70_tp4_push_buffer_ptrs
        else:
            down_norm_group = dist.new_group(backend="gloo")
            down_norm_buffers = CustomAllreduce.create_shared_buffer(
                resident_down_norm.buffer_bytes(), down_norm_group
            )
            resident_down_norm.initialize(down_norm_buffers, rank)
        down_norm_output = torch.empty(8, 5120, device=rank, dtype=torch.float16)
        down_norm_residual = torch.empty(8, 5120, device=rank, dtype=torch.float32)
        down_norm_weight = (
            inputnorm if args.resident_fp16_norm_weights else inputnorm.float()
        ).contiguous()
        out_norm_weight = (
            postnorm if args.resident_fp16_norm_weights else postnorm.float()
        ).contiguous()
        out_norm_output = torch.empty_like(down_norm_output)
        out_norm_residual = torch.empty_like(down_norm_residual)
        if args.resident_out_norm:
            assert hasattr(resident_down_norm, "out")
    assert not args.resident_out_norm or resident_down_norm is not None
    assert not args.resident_out_only or args.resident_out_norm
    norm_weight_stage = None
    norm_stage_epochs = None
    stage_debug_prefix = None
    stage_debug_input = None
    if args.norm_weight_stage_extension:
        assert args.wheel_paired_control and args.mode == 0
        assert norm_gate is None and not args.resident_out_norm
        spec = importlib.util.spec_from_file_location(
            args.norm_weight_stage_extension.name.split(".")[0],
            args.norm_weight_stage_extension,
        )
        assert spec and spec.loader
        norm_weight_stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(norm_weight_stage)
        norm_gate_buffers = ca.sm70_tp4_push_buffer_ptrs
        norm_gate_input = torch.empty(8, 5120, device=rank, dtype=torch.float16)
        norm_gate_residual = torch.empty(8, 5120, device=rank, dtype=torch.float32)
        norm_gate_weight = postnorm.float()
        norm_stage_epochs = torch.zeros(176, device=rank, dtype=torch.int32)
        if args.debug_weight_stage:
            stage_debug_prefix = torch.empty(
                2 * 2 * 8 * 10 * 288 + 2 * 2 * 8 * 40 * 32 * 32,
                device=rank,
                dtype=torch.uint8,
            )
            stage_debug_input = torch.full(
                (2, 8, 40, 32, 16), float("nan"), device=rank, dtype=torch.float16
            )
            stage_debug_pure_prefix = torch.empty_like(stage_debug_prefix)
            stage_debug_pure_input = torch.empty_like(stage_debug_input)
            stage_debug_pure_gate = torch.empty(
                8, 4352, device=rank, dtype=torch.float16
            )
        if args.dump_weight_stage:
            codes = mlp[0][2]
            expected = (
                codes.view(2, 136, 8, 40, 288)[:, :, :, :10]
                .permute(1, 0, 2, 3, 4)
                .contiguous()
            )
            dumped = torch.empty_like(expected)
            diagnostics = []
            for workers in (128, 256):
                norm_weight_stage.dump_stage(dumped, codes, workers)
                error = dumped != expected
                diagnostics.append(
                    {
                        "workers": workers,
                        "errors": error.sum().item(),
                        "bitwise": torch.equal(dumped, expected),
                    }
                )
                if error.any():
                    torch.save(
                        {"expected": expected.cpu(), "actual": dumped.cpu()},
                        args.out / f"rank{rank}-stage-{workers}.pt",
                    )
            (args.out / f"rank{rank}-staging.json").write_text(
                json.dumps(diagnostics, indent=2)
            )
            assert all(d["bitwise"] for d in diagnostics), diagnostics
            golden = torch.empty(8, 4352, device=rank, dtype=torch.float16)
            candidate = torch.empty_like(golden)
            operand = torch.empty(8, 5120, device=rank, dtype=torch.float16)
            for amplitude in (0.01, 0.125, 1.0, 4.0):
                operand.normal_().mul_(amplitude)
                ops.nvfp4_qpn2_gated_sm70_out(
                    golden, operand, codes[..., :256], codes[..., 256:], mlp[0][3], 8, 1
                )
                norm_weight_stage.gate(candidate, operand, codes, mlp[0][1], mlp[0][3])
                errors = (
                    (golden.view(torch.int16) != candidate.view(torch.int16))
                    .sum()
                    .item()
                )
                diagnostics.append({"amplitude": amplitude, "gate_errors": errors})
            (args.out / f"rank{rank}-staging.json").write_text(
                json.dumps(diagnostics, indent=2)
            )
            assert not any(d.get("gate_errors", 0) for d in diagnostics), diagnostics
            return
    layer = SimpleNamespace(
        in_proj_qkvz=Projection(qpn, qs),
        in_proj_ba=Projection(baweight),
        enable_sm70_dflash2_fused_gdn_verify=True,
        enable_sm70_gdn_ba_verify=True,
        gqa_interleaved_layout=False,
        disable_tp_for_ba_proj=False,
    )
    gdn_window_hash = None
    gdn_window_kernel = None
    if args.gdn_weight_window:
        from sm70_gdn_weight_window_screen import candidate_module

        gdn_window_kernel, gdn_window_hash = candidate_module(rank_source)

    def delta_weight_window(*values, **kwargs):
        from sm70_gdn_weight_window_screen import launch

        return launch(
            gdn_window_kernel, outq, args.gdn_prefetch_token, *values, **kwargs
        )

    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.125
    residual = torch.randn(8, 5120, device="cuda") * 0.125
    core_out = x.new_empty(8, 1536)
    projected = torch.empty_like(x)
    down = torch.empty_like(x)
    if packet_norm is not None:
        packet_group = dist.new_group(backend="gloo")
        packet_buffers = CustomAllreduce.create_shared_buffer(
            packet_norm.buffer_bytes(), packet_group
        )
        packet_inputs = CustomAllreduce.create_shared_buffer(8 * 5120 * 2, packet_group)
        packet_norm.initialize(packet_buffers, rank)
        projected = packet_norm.alias(packet_inputs[rank], rank)
        down = projected
        packet_outputs = [
            (torch.empty_like(x), torch.empty_like(residual)) for _ in range(2)
        ]
        packet_weights = [postnorm.float(), inputnorm.float()]
        torch.cuda.synchronize()
        dist.barrier()
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
    accepted = torch.tensor([args.accepted], device="cuda", dtype=torch.int32)
    conv_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    hybrid_source_sha256 = None
    state_seeds = None
    prime_hybrid = False
    factor_caches = None
    if args.hybrid_snapshot:
        from benchmark_sm70_gdn_hybrid_snapshot import candidate_module, launch, recover

        assert not args.prepared_qk and fused_projection is None and args.mode == 0
        hybrid_kernel, hybrid_source_sha256 = candidate_module(
            rank_source, args.hybrid_snapshots
        )
        factor_caches = [
            [torch.empty(8, 12, 128, device=x.device) for _ in range(2)]
            + [torch.empty(8, 12, device=x.device)]
            for _ in range(2)
        ]

        def hybrid_delta(a_log, a, b, bias, qkv, h, hv, k, v, out, **kwargs):
            seen_routes.add("hybrid_snapshot")
            assert (h, hv, k, v) == (4, 12, 128, 128)
            launch(
                hybrid_kernel,
                qkv,
                kwargs["precomputed_g"],
                kwargs["precomputed_beta"],
                kwargs["initial_state"],
                kwargs["ssm_state_indices"],
                out,
                cu,
                accepted,
                factors=(*factor_caches[0], *factor_caches[1]),
                replay=not prime_hybrid,
            )

        recovered = torch.empty(8, 12, 128, 128, device=x.device)

    inline_kernel = None
    inline_source_sha256 = None
    if args.inline_conv:
        from sm70_gdn_inline_conv_screen import (
            candidate_module,
        )
        from sm70_gdn_inline_conv_screen import (
            launch as inline_launch,
        )

        assert args.mode == 0 and not args.hybrid_snapshot and not args.prepared_qk
        assert cooperative is None and packet_norm is None and fused_projection is None
        assert args.dump_gdn_inputs is None
        inline_kernel, inline_source_sha256 = candidate_module(rank_source)
        inline_ready = torch.zeros(4, 192, device=x.device, dtype=torch.uint32)

    def delta_bv(*values, **kwargs):
        from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating

        selector = fused_sigmoid_gating._select_fused_sigmoid_launch

        def select(*positional, **named):
            _, warps, stages = selector(*positional, **named)
            assert warps == 1
            return args.delta_bv, warps, stages

        fused_sigmoid_gating._select_fused_sigmoid_launch = select
        try:
            return delta_update(*values, **kwargs)
        finally:
            fused_sigmoid_gating._select_fused_sigmoid_launch = selector

    def reset(arm=0):
        state.copy_(initial if state_seeds is None else state_seeds[arm])
        hist.copy_(history)

    def boundary(value, residual_value, weight, site, prepare):
        if prepare and packet_norm is not None:
            output, residual_output = packet_outputs[site]
            launch_norm = (
                packet_norm.consume
                if args.warp_publish and site == 1
                else packet_norm.launch
            )
            launch_norm(
                output,
                residual_output,
                value,
                residual_value,
                packet_weights[site],
                packet_buffers,
                packet_inputs,
                rank,
                1,
            )
            return output, residual_output
        return ca.sm70_tp4_all_reduce_gemma_rms_norm(
            value, residual_value, weight, 1e-6
        )

    seen_routes = set()

    last_mlp_input = None

    def run(mode, prepare=False, capture_inputs=False, reference=False):
        nonlocal last_mlp_input
        normalized, _ = input_norm(x, residual, inputnorm, 1e-6)
        if prepare and fused_projection is not None:
            q, z = x.new_empty(8, 2560), x.new_empty(8, 1536)
            b, a = x.new_empty(8, 12), x.new_empty(8, 12)
            g = torch.empty((1, 8, 12), device=x.device, dtype=torch.float32)
            beta = torch.empty_like(g)
            fused_projection.launch(
                q,
                z,
                b,
                a,
                g,
                beta,
                core_out,
                normalized,
                qpn,
                qs,
                baweight,
                conv,
                hist,
                conv_indices,
                accepted,
                cu,
                a_log,
                bias,
            )
        elif prepare and epilogue_scale is not None:
            seen_routes.add("epilogue_scale")
            q, z = x.new_empty(8, 2560), x.new_empty(8, 1536)
            b, a = x.new_empty(8, 12), x.new_empty(8, 12)
            epilogue_scale.qkvz(q, z, b, a, normalized, qpn, qs, baweight)
        elif prepare and ba_order is not None:
            seen_routes.add("ba_order")
            q, z = x.new_empty(8, 2560), x.new_empty(8, 1536)
            b, a = x.new_empty(8, 12), x.new_empty(8, 12)
            ba_order.launch(q, z, b, a, normalized, qpn, qs, baweight, args.ba_order)
        else:
            q, z, b, a = apply_gdn_ba_verify(layer, normalized)
        conv_fn, delta_fn = conv_gate_zero, delta_update
        if prepare and args.gdn_weight_window:
            seen_routes.add("gdn_weight_window")
            delta_fn = delta_weight_window
        if prepare and args.delta_bv is not None:
            seen_routes.add(f"delta_bv{args.delta_bv}")
            delta_fn = delta_bv
        if prepare and args.prepared_qk:
            from sm70_gdn_prepared_qk_screen import conv_prepare, prepared_delta

            conv_fn, delta_fn = conv_prepare, prepared_delta
        if prepare and args.hybrid_snapshot:
            delta_fn = hybrid_delta
        if prepare and head_local is not None and args.head_local_convolution:
            transformed = q
            g = torch.empty((8, 12), device=x.device, dtype=torch.float32)
            beta = torch.empty_like(g)
        elif prepare and inline_kernel is not None:
            transformed, g, beta = q, None, None
        elif prepare and fused_projection is not None:
            transformed = q
        else:
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
        if capture_inputs:
            assert args.dump_gdn_inputs is not None
            args.dump_gdn_inputs.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "qkv": transformed.cpu(),
                    "g": g.view(8, 12).cpu(),
                    "beta": beta.view(8, 12).cpu(),
                    "state": state.cpu(),
                    "indices": indices[None].cpu(),
                    "scope": "FP16 GDN inputs derived from real layer0 TP4 weights",
                },
                args.dump_gdn_inputs / f"rank{rank}.pt",
            )
        if prepare and head_local is not None:
            seen_routes.add("head_local_core")
            core = torch.empty_like(core_out)
            extra = (
                (conv, hist, conv_indices, a, b, a_log, bias, head_conv_ready)
                if args.head_local_convolution
                else ()
            )
            if args.gdn_out_stage_extension:
                seen_routes.add("gdn_out_stage")
                head_local.launch(
                    out_norm_output,
                    out_norm_residual,
                    core,
                    residual,
                    out_norm_weight,
                    outq,
                    outs,
                    down_norm_buffers,
                    rank,
                    transformed,
                    z,
                    g.view(8, 12),
                    beta.view(8, 12),
                    norm,
                    state,
                    indices,
                    accepted,
                    cu,
                    core_out,
                    head_packets,
                    head_generations,
                    head_out_ready,
                )
            else:
                head_local.launch(
                    core,
                    core_out,
                    transformed,
                    z,
                    g.view(8, 12),
                    beta.view(8, 12),
                    norm,
                    state,
                    indices,
                    accepted,
                    cu,
                    head_packets,
                    head_generations,
                    *extra,
                )
        elif prepare and cooperative is not None:
            core = torch.empty_like(core_out)
            cooperative.launch(
                core,
                core_out,
                transformed,
                z,
                g.view(8, 12),
                beta.view(8, 12),
                norm,
                state,
                indices,
                accepted,
                cu,
                not args.cooperative_no_qk_cache,
            )
        else:
            if prepare and inline_kernel is not None:
                inline_launch(
                    inline_kernel,
                    q,
                    a,
                    b,
                    a_log,
                    bias,
                    conv,
                    hist,
                    conv_indices,
                    inline_ready,
                    state,
                    indices[None],
                    core_out.view(8, 1, 12, 128),
                    cu,
                    accepted,
                )
            else:
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
        resident_module = resident_control if reference else resident_down_norm
        resident_prepare = (reference and resident_control is not None) or prepare
        if prepare and args.gdn_out_stage_extension:
            seen_routes.add("resident_out_norm")
        elif resident_prepare and args.resident_out_norm:
            seen_routes.add("resident_out_norm")
            out_args = (
                out_norm_output,
                out_norm_residual,
                core,
                residual,
                out_norm_weight,
                outq,
                outs,
                down_norm_buffers,
                rank,
            )
            if args.resident_weight_window and not reference:
                out_args += (mlp[0][2],)
                seen_routes.add("resident_weight_window")
            resident_module.out(*out_args)
        elif prepare and epilogue_scale is not None:
            epilogue_scale.out(projected, core, outq, outs)
        else:
            ops.fp8_qpn8_gemm_sm70_out(projected, core, outq, outs, 12, 2, True, False)
        if prepare and norm_weight_stage is not None:
            seen_routes.add("norm_weight_stage")
            _, scales, bundle, scale, _ = mlp[0]
            stage_function = (
                norm_weight_stage.fused_debug
                if args.debug_weight_stage
                else norm_weight_stage.fused
            )
            stage_args = (
                up,
                norm_gate_input,
                norm_gate_residual,
                projected,
                residual,
                norm_gate_weight,
                bundle,
                scales,
                scale,
                norm_gate_buffers,
                rank,
                norm_stage_epochs,
            )
            if args.debug_weight_stage:
                stage_args += (stage_debug_prefix, stage_debug_input)
            stage_function(*stage_args)
            if args.debug_weight_stage:
                norm_weight_stage.gate_debug(
                    stage_debug_pure_gate,
                    norm_gate_input,
                    bundle,
                    scales,
                    scale,
                    stage_debug_pure_prefix,
                    stage_debug_pure_input,
                )
            p, r = norm_gate_input, norm_gate_residual
        elif prepare and norm_gate is not None:
            seen_routes.add("resident_norm_gate")
            _, scales, bundle, scale, _ = mlp[0]
            norm_gate.fused(
                up,
                norm_gate_input,
                norm_gate_residual,
                projected,
                residual,
                norm_gate_weight,
                bundle,
                scales,
                scale,
                norm_gate_buffers,
                rank,
            )
            p, r = norm_gate_input, norm_gate_residual
        elif resident_prepare and args.resident_out_norm:
            p, r = out_norm_output, out_norm_residual
        else:
            p, r = boundary(projected, residual, postnorm, 0, prepare)
        last_mlp_input = (p, r)
        if prepare and resident_mlp is not None:
            seen_routes.add("resident_mlp")
            resident_mlp.mlp(
                down, up, p, mlp[0][2], mlp[1][2], mlp_ready, mlp[0][3], mlp[1][3]
            )
            return boundary(down, r, inputnorm, 1, prepare)
        for item, operand, result in ((mlp[0], p, up), (mlp[1], up, down)):
            w, s, bundle, scale, gated = item
            if (
                gated
                and prepare
                and (norm_gate is not None or norm_weight_stage is not None)
            ):
                continue
            if gated and prepare and packed_input is not None:
                seen_routes.add("packed_gate_input")
                packed_input.warp_input(result, operand, bundle, s, scale, True, False)
                continue
            if (
                not gated
                and resident_prepare
                and resident_module is not None
                and not args.resident_out_only
            ):
                seen_routes.add("resident_down_norm")
                resident_module.fused(
                    down_norm_output,
                    down_norm_residual,
                    operand,
                    r,
                    down_norm_weight,
                    bundle,
                    s,
                    scale,
                    down_norm_buffers,
                    rank,
                )
                return down_norm_output, down_norm_residual
            if (
                prepare
                and pipeline is not None
                and (
                    args.pipeline_arm == "both"
                    or args.pipeline_arm == ("gate" if gated else "down")
                )
            ):
                seen_routes.add("pipeline_gate" if gated else "pipeline_down")
                if gated:
                    pipeline.pair(result, operand, bundle, s, scale, 1)
                else:
                    pipeline.down(result, operand, bundle, s, scale)
            elif args.wheel_paired_control:
                seen_routes.add("wheel_gate" if gated else "wheel_down")
                codes, scales = bundle[..., :256], bundle[..., 256:]
                if gated:
                    ops.nvfp4_qpn2_gated_sm70_out(
                        result, operand, codes, scales, scale, 8, 1
                    )
                else:
                    ops.nvfp4_qpn2_gemm_sm70_out(
                        result, operand, codes, scales, scale, 16, 2
                    )
            elif not gated and prepare and args.warp_publish:
                packet_norm.project(
                    result, operand, bundle, s, scale, packet_buffers, rank
                )
            elif gated and (mode or paired_control is not None):
                pair_bundle = (
                    candidate_gate_bundle
                    if candidate_gate_bundle is not None and not reference
                    else bundle
                )
                seen_routes.add("paired_gate")
                pair_extension = (
                    paired_control
                    if reference and paired_control is not None
                    else extension
                )
                pair_extension.pair(result, operand, pair_bundle, s, scale, mode or 1)
            else:
                extension.launch(result, operand, bundle, s, table, scale, gated, 0)
        return boundary(down, r, inputnorm, 1, prepare)

    if args.hybrid_snapshot:
        accepted.fill_(1)
        reset()
        run(0)
        full_seed = state.clone()
        reset()
        prime_hybrid = True
        run(0, True)
        prime_hybrid = False
        compact_seed = state.clone()
        for old, new in zip(*factor_caches):
            old.copy_(new)
        state_seeds = [full_seed, compact_seed]
        accepted.fill_(args.accepted)

    def candidate_state():
        if not args.hybrid_snapshot:
            return state
        recover[(64, 96)](
            state,
            indices,
            *factor_caches[1],
            recovered,
            state.stride(0),
            SNAPSHOTS=args.hybrid_snapshots,
            num_warps=1,
        )
        reconstructed = state.clone()
        reconstructed.index_copy_(0, indices.long(), recovered)
        return reconstructed

    if args.ncu_direct or args.ncu_graph:
        assert paired_control is not None or args.wheel_paired_control
        assert (
            not args.ncu_candidate
            or pipeline is not None
            or resident_mlp is not None
            or args.gdn_weight_window
            or resident_down_norm is not None
            or norm_weight_stage is not None
        )

        def profiled_run():
            return run(args.mode, args.ncu_candidate, reference=not args.ncu_candidate)

        reset()
        eviction.fill_(1)
        torch.cuda.synchronize()
        dist.barrier()
        profiled_run()
        torch.cuda.synchronize()
        dist.barrier()
        if args.ncu_graph:
            graph = torch.cuda.CUDAGraph()
            with ca.capture(), torch.cuda.graph(graph):
                reset()
                eviction.fill_(1)
                profiled_run()
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
        return

    differences = []
    for amplitude in (0.01, 0.125, 1.0, 4.0):
        x.normal_().mul_(amplitude)
        reset(0)
        control_out = run(
            1 if paired_control is not None else 0,
            capture_inputs=args.dump_gdn_inputs is not None and amplitude == 0.125,
            reference=True,
        )
        golden = [v.clone() for v in control_out] + [state.clone(), hist.clone()]
        if resident_mlp is not None or norm_weight_stage is not None:
            golden.extend((up.clone(), down.clone()))
        if norm_weight_stage is not None:
            golden.extend(v.clone() for v in last_mlp_input)
        reset(1)
        candidate_out = run(
            args.mode,
            args.prepared_qk
            or fused_projection is not None
            or args.hybrid_snapshot
            or cooperative is not None
            or packet_norm is not None
            or inline_kernel is not None
            or ba_order is not None
            or epilogue_scale is not None
            or head_local is not None
            or norm_gate is not None
            or args.delta_bv is not None
            or resident_down_norm is not None
            or pipeline is not None
            or resident_mlp is not None
            or args.gdn_weight_window
            or norm_weight_stage is not None,
        )
        candidate_values = list(candidate_out) + [candidate_state(), hist]
        if resident_mlp is not None or norm_weight_stage is not None:
            candidate_values.extend((up, down))
        if norm_weight_stage is not None:
            candidate_values.extend(last_mlp_input)
        for value_index, (result, reference) in enumerate(
            zip(candidate_values, golden)
        ):
            bits = torch.int16 if result.element_size() == 2 else torch.int32
            exact = torch.equal(result.view(bits), reference.view(bits))
            differences.append(
                {
                    "amplitude": amplitude,
                    "bitwise": exact,
                    "finite": bool(torch.isfinite(result).all()),
                    "max_abs": (result.float() - reference.float()).abs().max().item(),
                    "value_index": value_index,
                }
            )
            assert torch.isfinite(result).all(), "Nonfinite candidate output"
            if (
                not args.prepared_qk
                and fused_projection is None
                and cooperative is None
                and inline_kernel is None
                and head_local is None
                and norm_gate is None
                and epilogue_scale is None
                and args.paired_accumulators is None
                and not (packet_norm is not None and args.norm_packet_parts != 5)
            ):
                if not exact:
                    (args.out / f"rank{rank}-failure.json").write_text(
                        json.dumps(differences, indent=2)
                    )
                    failure = {
                        "reference": [v.cpu() for v in golden],
                        "candidate": [v.cpu() for v in candidate_values],
                    }
                    if args.debug_weight_stage:
                        failure["prefix"] = stage_debug_prefix.cpu()
                        failure["prefix_expected"] = (
                            mlp[0][2]
                            .view(2, 136, 8, 40, 288)[:, [0, 40], :, :10]
                            .permute(1, 0, 2, 3, 4)
                            .cpu()
                        )
                        failure["observed_input"] = stage_debug_input.cpu()
                        failure["pure_prefix"] = stage_debug_pure_prefix.cpu()
                        failure["pure_input"] = stage_debug_pure_input.cpu()
                        failure["pure_gate"] = stage_debug_pure_gate.cpu()
                        failure["stage_epochs"] = norm_stage_epochs.cpu()
                    torch.save(
                        failure,
                        args.out / f"rank{rank}-failure.pt",
                    )
                assert exact, f"value {value_index}: {differences[-1]}"
    if args.debug_weight_stage:
        (args.out / f"rank{rank}-debug-quality.json").write_text(
            json.dumps(differences, indent=2)
        )
        return
    graphs = []
    graph_outputs = []
    counts = []
    for arm in range(2):
        mode, prepare = (
            (
                args.mode,
                args.prepared_qk
                or fused_projection is not None
                or args.hybrid_snapshot
                or cooperative is not None
                or packet_norm is not None
                or inline_kernel is not None
                or ba_order is not None
                or epilogue_scale is not None
                or head_local is not None
                or norm_gate is not None
                or args.delta_bv is not None
                or resident_down_norm is not None
                or pipeline is not None
                or resident_mlp is not None
                or args.gdn_weight_window
                or norm_weight_stage is not None,
            )
            if arm
            else (
                1 if norm_gate is not None or paired_control is not None else 0,
                False,
            )
        )
        reset(arm)
        run(mode, prepare, reference=not arm)
        if arm and norm_gate is not None:
            assert "resident_norm_gate" in seen_routes
        if arm and head_local is not None:
            assert "head_local_core" in seen_routes
        if arm and epilogue_scale is not None:
            assert "epilogue_scale" in seen_routes
        if arm and ba_order is not None:
            assert "ba_order" in seen_routes, "CTA-order candidate was not selected"
        if arm and args.delta_bv is not None:
            assert f"delta_bv{args.delta_bv}" in seen_routes
        if arm and resident_down_norm is not None:
            assert (
                "resident_out_norm" if args.resident_out_only else "resident_down_norm"
            ) in seen_routes
        if arm and args.resident_out_norm:
            assert "resident_out_norm" in seen_routes
        if arm and pipeline is not None:
            for role in (
                ("gate", "down")
                if args.pipeline_arm == "both"
                else (args.pipeline_arm,)
            ):
                assert "pipeline_" + role in seen_routes
        if arm and resident_mlp is not None:
            assert "resident_mlp" in seen_routes
        if arm and args.gdn_weight_window:
            assert "gdn_weight_window" in seen_routes
        if arm and norm_weight_stage is not None:
            assert "norm_weight_stage" in seen_routes
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with ca.capture(), torch.cuda.graph(graph):
            reset(arm)
            eviction.fill_(1)
            start.record()
            graph_output = run(mode, prepare, reference=not arm)
            end.record()
        counts.append(nodes(graph))
        graph.instantiate()
        graphs.append((graph, start, end))
        graph_outputs.append(graph_output)
    difference = (
        int(
            fused_projection is not None
            or cooperative is not None
            or inline_kernel is not None
            or head_local is not None
            or norm_gate is not None
            or resident_down_norm is not None
        )
        + int(args.head_local_convolution)
        + int(args.resident_out_norm and not args.resident_out_only)
    )
    if resident_control is not None:
        difference -= 2
    if resident_mlp is not None:
        difference += 1
    if norm_weight_stage is not None:
        difference += 1
    if args.gdn_out_stage_extension:
        difference += 2
    assert counts[0]["types"].get("0") - difference == counts[1]["types"].get("0")
    replay_quality = []
    if norm_weight_stage is not None or args.gdn_out_stage_extension:
        # Cross-CTA publication must also survive changing inputs and graph
        # generations; eager checks alone can miss stale activation reads.
        for replay in range(20):
            x.normal_().mul_((0.01, 0.125, 1.0, 4.0)[replay % 4])
            graphs[0][0].replay()
            control_values = [v.clone() for v in (*graph_outputs[0], state, hist)]
            graphs[1][0].replay()
            for value_index, (candidate, reference) in enumerate(
                zip((*graph_outputs[1], state, hist), control_values)
            ):
                bits = torch.int16 if candidate.element_size() == 2 else torch.int32
                exact = torch.equal(candidate.view(bits), reference.view(bits))
                finite = bool(torch.isfinite(candidate).all())
                replay_quality.append(
                    {
                        "replay": replay,
                        "value_index": value_index,
                        "bitwise": exact,
                        "finite": finite,
                        "max_abs": (candidate.float() - reference.float())
                        .abs()
                        .max()
                        .item(),
                    }
                )
                assert finite, "Nonfinite graph replay"
                if norm_weight_stage is not None:
                    assert exact, f"Stale graph publication: {replay_quality[-1]}"
    samples = [[], []]
    for iteration in range(args.iters + 20):
        if args.cuda_profiler_capture and iteration == 20:
            torch.cuda.cudart().cudaProfilerStart()
        for index in (iteration % 2, 1 - iteration % 2):
            graph, start, end = graphs[index]
            if args.cuda_profiler_capture:
                torch.cuda.nvtx.range_push(
                    "layer/candidate" if index else "layer/control"
                )
            graph.replay()
            end.synchronize()
            if args.cuda_profiler_capture:
                torch.cuda.nvtx.range_pop()
            if iteration >= 20:
                samples[index].append(start.elapsed_time(end) * 1000)
    if args.cuda_profiler_capture:
        torch.cuda.cudart().cudaProfilerStop()
    result = {
        "rank": rank,
        "mode": args.mode,
        "paired_control": paired_control is not None,
        "wheel_paired_control": args.wheel_paired_control,
        "pipeline_arm": args.pipeline_arm if pipeline is not None else None,
        "resident_mlp": resident_mlp is not None,
        "norm_weight_stage": norm_weight_stage is not None,
        "control_mean_us": statistics.mean(samples[0]),
        "candidate_mean_us": statistics.mean(samples[1]),
        "samples_us": samples,
        "changed_input_graph_quality": replay_quality,
        "graph_nodes": counts,
        "output_and_state_bitwise": all(d["bitwise"] for d in differences),
        "differences": differences,
        "prepared_qk": args.prepared_qk,
        "projection_conv": fused_projection is not None,
        "hybrid_snapshot": args.hybrid_snapshot,
        "hybrid_snapshots": args.hybrid_snapshots,
        "cooperative_core": cooperative is not None,
        "cooperative_qk_cache": not args.cooperative_no_qk_cache,
        "norm_partial_packets": packet_norm is not None,
        "norm_packet_parts": args.norm_packet_parts,
        "warp_publish": args.warp_publish,
        "inline_conv": inline_kernel is not None,
        "ba_order": args.ba_order if ba_order is not None else None,
        "seen_routes": sorted(seen_routes),
        "paired_accumulators": args.paired_accumulators,
        "gate_group_layout": args.gate_group_layout,
        "epilogue_scale": epilogue_scale is not None,
        "head_local_core": head_local is not None,
        "gdn_out_stage": args.gdn_out_stage_extension is not None,
        "head_local_convolution": args.head_local_convolution,
        "resident_norm_gate": norm_gate is not None,
        "resident_down_norm": resident_down_norm is not None,
        "resident_out_norm": args.resident_out_norm,
        "resident_out_only": args.resident_out_only,
        "resident_weight_window": args.resident_weight_window,
        "resident_control": resident_control is not None,
        "resident_native_ca_buffers": args.resident_native_ca_buffers,
        "resident_fp16_norm_weights": args.resident_fp16_norm_weights,
        "packed_gate_input": packed_input is not None,
        "resident_norm_gate_single_ready": (
            bool(norm_gate.single_ready) if norm_gate is not None else None
        ),
        "inline_source_sha256": inline_source_sha256,
        "hybrid_source_sha256": hybrid_source_sha256,
        "accepted": args.accepted,
        "delta_bv": args.delta_bv,
        "gdn_weight_window": args.gdn_weight_window,
        "gdn_prefetch_token": args.gdn_prefetch_token,
        "gdn_window_source_sha256": gdn_window_hash,
        "four_amplitudes": [0.01, 0.125, 1.0, 4.0],
        "scope": (
            "Complete real-weight TP4 layer0 GDN graph; "
            + (
                "unchanged paired-gate operands/BV2 recurrence; resident norm prefix."
                if norm_gate is not None
                else "unchanged projections/BV2 recurrence; exact factor replay with "
                f"{args.hybrid_snapshots} state snapshots."
                if args.hybrid_snapshot
                else (
                    "unchanged projections/BV2 recurrence; "
                    "inline conv/gating/history/zero."
                )
                if inline_kernel is not None
                else "unchanged projections/core; per-part variance-generation packets."
                if packet_norm is not None
                else "and BV2 state tasks; cooperative delta/gated-norm phase boundary."
                if cooperative is not None
                else "and delta recurrence; projection/conv/gating/zero epilogue."
                if fused_projection is not None
                else "and BV2 recurrence; precompute Q/K normalization in convolution."
                if args.prepared_qk
                else "and BV2 warp tasks; head-local delta/gated-norm packets."
                if head_local is not None
                else "and FP8 weight precision; per-channel scaling after FP32 sum."
                if epilogue_scale is not None
                else "and BV2 recurrence; reorder already fused qkvz/b/a CTAs."
                if ba_order is not None
                else "and state update; shared activation registers in paired warps."
            )
        ),
    }
    (args.out / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k != "samples_us"}), flush=True
    )
    if packet_norm is not None:
        torch.cuda.synchronize()
        dist.barrier()
        CustomAllreduce.free_shared_buffer(packet_inputs, rank=rank)
        CustomAllreduce.free_shared_buffer(packet_buffers, rank=rank)
    if norm_gate_buffers is not None and norm_weight_stage is None:
        torch.cuda.synchronize()
        dist.barrier()
        CustomAllreduce.free_shared_buffer(norm_gate_buffers, rank=rank)
    ca.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
