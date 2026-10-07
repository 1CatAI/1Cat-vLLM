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
    parser.add_argument("--projection-conv-extension", type=Path)
    parser.add_argument("--cooperative-core-extension", type=Path)
    parser.add_argument("--head-local-core-extension", type=Path)
    parser.add_argument("--head-local-convolution", action="store_true")
    parser.add_argument("--cooperative-no-qk-cache", action="store_true")
    parser.add_argument("--norm-partial-extension", type=Path)
    parser.add_argument("--norm-packet-parts", type=int, choices=(5, 10, 20), default=5)
    parser.add_argument("--warp-publish", action="store_true")
    parser.add_argument("--dump-gdn-inputs", type=Path)
    parser.add_argument("--hybrid-snapshot", action="store_true")
    parser.add_argument("--inline-conv", action="store_true")
    parser.add_argument("--ba-order-extension", type=Path)
    parser.add_argument("--epilogue-scale-extension", type=Path)
    parser.add_argument("--ba-order", type=int, choices=(1, 2), default=1)
    parser.add_argument("--paired-accumulators", type=int, choices=(2, 4))
    parser.add_argument("--gate-group-layout", action="store_true")
    parser.add_argument("--accepted", type=int, choices=range(1, 9), default=4)
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
        args.extension.name.split(".")[0], args.extension
    )
    assert spec and spec.loader
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
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
    if args.warp_publish:
        assert packet_norm is not None and args.norm_packet_parts == 20
        assert args.mode == 0
    ba_order = None
    if args.ba_order_extension:
        assert args.mode == 0 and not args.inline_conv and not args.hybrid_snapshot
        assert fused_projection is None and cooperative is None and packet_norm is None
        spec = importlib.util.spec_from_file_location(
            "qpn8_ba_order_screen", args.ba_order_extension
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
    if args.head_local_core_extension:
        assert args.mode == 0 and not args.inline_conv and not args.hybrid_snapshot
        assert fused_projection is None and cooperative is None and packet_norm is None
        assert ba_order is None and epilogue_scale is None
        spec = importlib.util.spec_from_file_location(
            args.head_local_core_extension.name.split(".")[0],
            args.head_local_core_extension,
        )
        assert spec and spec.loader
        head_local = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(head_local)
        head_packets = torch.zeros(12, 8, 8, device=rank, dtype=torch.int64)
        head_generations = torch.zeros(12, device=rank, dtype=torch.int32)
        head_conv_ready = torch.zeros(4, 24, device=rank, dtype=torch.int32)
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
    candidate_gate_bundle = None
    if args.gate_group_layout:
        assert args.mode in (1, 2) and args.paired_accumulators is None
        assert args.extension.name.split(".")[0] == "qpn2_gate_group_layout_screen"
        candidate_gate_bundle = (
            mlp[0][2].view(2, 136, 320, 288).permute(1, 2, 0, 3).contiguous()
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

    state_seeds = None
    prime_hybrid = False
    factor_caches = None
    if args.hybrid_snapshot:
        from benchmark_sm70_gdn_hybrid_snapshot import candidate_module, launch, recover

        assert not args.prepared_qk and fused_projection is None and args.mode == 0
        hybrid_kernel, _ = candidate_module(args.out)
        factor_caches = [
            [torch.empty(8, 12, 128, device=x.device) for _ in range(2)]
            + [torch.empty(8, 12, device=x.device)]
            for _ in range(2)
        ]

        def hybrid_delta(a_log, a, b, bias, qkv, h, hv, k, v, out, **kwargs):
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
        inline_kernel, inline_source_sha256 = candidate_module(args.out)
        inline_ready = torch.zeros(4, 192, device=x.device, dtype=torch.uint32)

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

    def run(mode, prepare=False, capture_inputs=False):
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
        if prepare and epilogue_scale is not None:
            epilogue_scale.out(projected, core, outq, outs)
        else:
            ops.fp8_qpn8_gemm_sm70_out(projected, core, outq, outs, 12, 2, True, False)
        p, r = boundary(projected, residual, postnorm, 0, prepare)
        for item, operand, result in ((mlp[0], p, up), (mlp[1], up, down)):
            w, s, bundle, scale, gated = item
            if not gated and prepare and args.warp_publish:
                packet_norm.project(
                    result, operand, bundle, s, scale, packet_buffers, rank
                )
            elif gated and mode:
                pair_bundle = (
                    candidate_gate_bundle
                    if candidate_gate_bundle is not None
                    else bundle
                )
                seen_routes.add("paired_gate")
                extension.pair(result, operand, pair_bundle, s, scale, mode)
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
            num_warps=1,
        )
        reconstructed = state.clone()
        reconstructed.index_copy_(0, indices.long(), recovered)
        return reconstructed

    differences = []
    for amplitude in (0.01, 0.125, 1.0, 4.0):
        x.normal_().mul_(amplitude)
        reset(0)
        control_out = run(
            0, capture_inputs=args.dump_gdn_inputs is not None and amplitude == 0.125
        )
        golden = [v.clone() for v in control_out] + [state.clone(), hist.clone()]
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
            or head_local is not None,
        )
        for result, reference in zip(
            list(candidate_out) + [candidate_state(), hist], golden
        ):
            bits = torch.int16 if result.element_size() == 2 else torch.int32
            exact = torch.equal(result.view(bits), reference.view(bits))
            differences.append(
                {
                    "amplitude": amplitude,
                    "bitwise": exact,
                    "finite": bool(torch.isfinite(result).all()),
                    "max_abs": (result.float() - reference.float()).abs().max().item(),
                }
            )
            assert torch.isfinite(result).all(), "Nonfinite candidate output"
            if (
                not args.prepared_qk
                and fused_projection is None
                and cooperative is None
                and inline_kernel is None
                and head_local is None
                and epilogue_scale is None
                and args.paired_accumulators is None
                and not (packet_norm is not None and args.norm_packet_parts != 5)
            ):
                assert exact
    graphs = []
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
                or head_local is not None,
            )
            if arm
            else (0, False)
        )
        reset(arm)
        run(mode, prepare)
        if arm and head_local is not None:
            assert "head_local_core" in seen_routes
        if arm and epilogue_scale is not None:
            assert "epilogue_scale" in seen_routes
        if arm and ba_order is not None:
            assert "ba_order" in seen_routes, "CTA-order candidate was not selected"
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with ca.capture(), torch.cuda.graph(graph):
            reset(arm)
            eviction.fill_(1)
            start.record()
            run(mode, prepare)
            end.record()
        counts.append(nodes(graph))
        graph.instantiate()
        graphs.append((graph, start, end))
    difference = int(
        fused_projection is not None
        or cooperative is not None
        or inline_kernel is not None
        or head_local is not None
    ) + int(args.head_local_convolution)
    assert counts[0]["types"].get("0") - difference == counts[1]["types"].get("0")
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
        "projection_conv": fused_projection is not None,
        "hybrid_snapshot": args.hybrid_snapshot,
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
        "head_local_convolution": args.head_local_convolution,
        "inline_source_sha256": inline_source_sha256,
        "accepted": args.accepted,
        "four_amplitudes": [0.01, 0.125, 1.0, 4.0],
        "scope": (
            "Complete real-weight TP4 layer0 GDN graph; "
            + (
                "unchanged projections/BV2 recurrence; inline conv/gating/history/zero."
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
    ca.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
