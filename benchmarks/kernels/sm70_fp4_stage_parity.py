# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4 adapter A/B using exact admitted geometries and identical native bytes.

--root contains base/vllm/... from the reference commit. The candidate package
comes from PYTHONPATH. This is operator evidence, never model decode evidence.
"""

import argparse
import copy
import functools
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from vllm import _sm70_ops, envs
from vllm._sm70.policy import CONFIGURED_OPERATORS, NativeBindings
from vllm.config.sm70_moe import (
    MXFP4_ALIASES,
    NVFP4_ALIASES,
    Sm70MxFp4MoEConfig,
    Sm70NvFp4MoEConfig,
)
from vllm.forward_context import override_forward_context
from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.fused_moe.sm70.fp4_codec import Fp4MoECodec
from vllm.model_executor.layers.quantization import mxfp4_sm70_moe, nvfp4_sm70_moe
from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
    LayerWorkspaceView,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--family", choices=("nvfp4", "mxfp4"), required=True)
    parser.add_argument("--model", choices=("qwen38", "glm53"), default="qwen38")
    parser.add_argument("--grouped-only", action="store_true")
    args = parser.parse_args()
    root, family = args.root, args.family
    assert torch.cuda.get_device_capability() == (7, 0)
    spec = importlib.util.spec_from_file_location(
        "baseline_" + family,
        root
        / "base/vllm/model_executor/layers/quantization"
        / (family + "_sm70_moe.py"),
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    current = nvfp4_sm70_moe if family == "nvfp4" else mxfp4_sm70_moe
    clsname = (
        "ModelOptNvFp4SM70MoEMethod" if family == "nvfp4" else "Mxfp4SM70MoEMethod"
    )
    aliases = NVFP4_ALIASES if family == "nvfp4" else MXFP4_ALIASES
    config_cls = Sm70NvFp4MoEConfig if family == "nvfp4" else Sm70MxFp4MoEConfig
    native_calls = []
    reference_policy = ()
    for name in dir(_sm70_ops):
        native = getattr(_sm70_ops, name)
        if callable(native) and name.startswith(
            (
                family + "_moe_",
                family + "_qwen38_",
                family + "_glm53_",
                family + "_grouped_",
                "awq_moe_single_token_weighted_reduce",
                "sm70_glm53_",
            )
        ):

            def instrument(fn, name):
                @functools.wraps(fn)
                def call(*a, **kw):
                    if reference_policy and name in CONFIGURED_OPERATORS:
                        kw.setdefault("native_policy", reference_policy)
                    result = fn(*a, **kw)
                    native_calls.append(name)
                    return result

                return call

            setattr(_sm70_ops, name, instrument(native, name))

    def flags(values):
        for field, name in aliases.items():
            os.environ[name] = str(int(values.get(field, False)))
        envs.disable_envs_cache()
        policy = config_cls()
        policy.resolve()
        return policy

    # Preparation layouts are separate; strategy variants reuse the same banks.
    profiles = (
        ("plain", "interleaved", "raw", "raw_interleaved")
        if family == "nvfp4"
        else ("plain",)
    )
    if args.model == "glm53":
        assert family == "nvfp4"
        profiles = ("plain",)
    if args.grouped_only:
        assert family == "nvfp4" and args.model == "qwen38"
        profiles = ("plain", "interleaved")
    results = []
    for profile in profiles:
        interleaved = "interleaved" in profile
        raw = "raw" in profile
        preparation = dict(
            fused_swiglu_prefill=interleaved,
            raw_scale=raw,
            fast_prefill=interleaved,
            indexed_prefill=True,
            grouped_prefill=True,
            qpn_m1=True,
            w2_direct_reduce=True,
        )
        if family == "mxfp4":
            preparation = dict(qpn_m1=True, direct_top6=True, direct_order=True)
        if args.model == "glm53":
            preparation = dict(
                glm53_fused_permute=True, glm53_qpn_w13=True, grouped_expert_rows=True
            )
        if args.grouped_only:
            preparation.update(grouped_mtp5=True, qpn_mtp5=True, grouped_decode=True)
        preparation_policy = flags(preparation)
        prepared_binding = NativeBindings(preparation_policy.native.values)
        reference_policy = getattr(
            prepared_binding, "arguments", prepared_binding.values
        )
        h, i, e, k, group = (
            (2560, 160, 512, 10, 16) if family == "nvfp4" else (4096, 512, 256, 6, 32)
        )
        if args.model == "glm53":
            h, i, e, k, group = 4096, 256, 288, 8, 16
        cfg = NS(
            tp_size=8 if args.model == "glm53" else 4,
            hidden_dim=h,
            intermediate_size_per_partition=i,
            num_experts=e,
            experts_per_token=k,
            has_bias=False,
            moe_parallel_config=NS(use_all2all_kernels=False),
        )
        layer = torch.nn.Module()
        layer.moe_config = cfg
        layer.local_num_experts = layer.global_num_experts = e
        layer.top_k = k
        layer.activation = MoEActivation.SILU
        layer.apply_router_weight_on_input = False
        layer.expert_map = None
        layer.swiglu_limit = None if family == "nvfp4" else 7.0
        torch.manual_seed(54633)
        for stage, n, kd in (("w13", 2 * i, h), ("w2", h, i)):
            setattr(
                layer,
                stage + "_weight",
                torch.nn.Parameter(
                    torch.randint(
                        0, 256, (e, n, kd // 2), dtype=torch.uint8, device="cuda"
                    ),
                    requires_grad=False,
                ),
            )
            scale = torch.full(
                (e, n, kd // group),
                0.015625 if family == "nvfp4" else 120,
                dtype=torch.float16 if family == "nvfp4" else torch.uint8,
                device="cuda",
            )
            if family == "nvfp4":
                scale = scale.to(torch.float8_e4m3fn)
            setattr(
                layer,
                stage + "_weight_scale",
                torch.nn.Parameter(scale, requires_grad=False),
            )
            if family == "nvfp4":
                setattr(
                    layer,
                    stage + "_weight_scale_2",
                    torch.ones((e, 2) if stage == "w13" else (e,), device="cuda"),
                )
                setattr(layer, stage + "_input_scale", torch.ones(e, device="cuda"))
        old = object.__new__(getattr(baseline, clsname))
        old.moe = cfg
        candidate_layer = copy.deepcopy(layer)
        old.process_weights_after_loading(layer)
        new = object.__new__(getattr(current, clsname))
        new.moe = cfg
        new.sm70_moe_policy = config_cls()
        new.sm70_moe_policy.resolve()
        new.native_ops = NativeBindings(new.sm70_moe_policy.native.values)
        new.process_weights_after_loading(candidate_layer)
        expected = dict(layer.named_parameters())
        actual = dict(candidate_layer.named_parameters())
        assert expected.keys() == actual.keys()
        for name in expected:
            a, b = expected[name], actual[name]
            if "strided_ptrs" in name:
                # Separate loads have different addresses; compare offsets in
                # the owning bank, including split-W13 head/tail views.
                stage = name.split("_")[0]
                bank_name = stage + (
                    "_tm_weight" if name.endswith("_w") else "_tm_scales"
                )
                # Native StridedPtr is {uint64 address, int32 stride,
                # int32 unused padding}; padding bytes are not initialized.
                assert torch.equal(
                    a.view(torch.int32).view(-1, 4)[:, 2],
                    b.view(torch.int32).view(-1, 4)[:, 2],
                ), (family, profile, name, "stride")
                a = (
                    a.view(torch.int64).view(-1, 2)[:, 0]
                    - getattr(layer, bank_name).data_ptr()
                )
                b = (
                    b.view(torch.int64).view(-1, 2)[:, 0]
                    - getattr(candidate_layer, bank_name).data_ptr()
                )
            assert torch.equal(a, b), (family, profile, name)
        del expected, actual, layer
        layer = candidate_layer
        del candidate_layer
        layer.sm70_fp4_codec = Fp4MoECodec(
            family,
            LayerWorkspaceView(layer, ""),
            LayerWorkspaceView(layer, "sm70_" + family + "_"),
            raw,
            layer.swiglu_limit,
            bindings=getattr(new, "native_ops", None),
        )
        print(
            json.dumps(
                dict(event="prepared", family=family, profile=profile, banks_equal=True)
            ),
            flush=True,
        )
        if family == "nvfp4":
            routes = [
                ("dense", {}, [0, 1, 2, 5, 8, 17]),
                (
                    "qpn",
                    dict(qpn_m1=True, qpn_batch=True, qpn_dynamic=True, qpn_mtp5=True),
                    [1, 2, 3, 4, 5, 8, 16],
                ),
                ("mtp", dict(qpn_mtp5=True), [5]),
                (
                    "fused",
                    dict(
                        qpn_m1=True,
                        qpn_batch=True,
                        fused_batch_w13=True,
                        fused_batch_w2=True,
                        w2_direct_reduce=True,
                    ),
                    [1, 2, 4, 8, 16],
                ),
                (
                    "indexed",
                    dict(indexed_prefill=True, grouped_prefill=True),
                    [127, 128],
                ),
            ]
        else:
            routes = [
                ("dense", {}, [0, 1, 2, 5, 8, 9]),
                ("direct", dict(direct_top6=True), [1]),
                ("order", dict(direct_top6=True, direct_order=True), [1]),
                ("qpn", dict(direct_top6=True, direct_order=True, qpn_m1=True), [1]),
                (
                    "active",
                    dict(active_experts=True, active_expert_max_tokens=8),
                    [1, 2, 5, 8, 9],
                ),
                (
                    "grouped",
                    dict(
                        active_experts=True,
                        active_expert_max_tokens=8,
                        grouped_verifier=True,
                    ),
                    [2, 5, 8],
                ),
                (
                    "grouped_rows",
                    dict(
                        active_experts=True,
                        active_expert_max_tokens=8,
                        grouped_verifier=True,
                        grouped_expert_rows=True,
                    ),
                    [2, 5, 8],
                ),
            ]
        if args.model == "glm53":
            routes = [
                ("glm_dense", {}, [1, 8, 9]),
                (
                    "glm_permute",
                    dict(glm53_fused_permute=True, grouped_expert_rows=True),
                    [8],
                ),
                (
                    "glm_qpn",
                    dict(
                        glm53_fused_permute=True,
                        glm53_qpn_w13=True,
                        grouped_expert_rows=True,
                    ),
                    [8],
                ),
            ]
        if args.grouped_only:
            routes = [
                ("grouped_mtp", dict(grouped_mtp5=True, qpn_mtp5=True), [5]),
                ("grouped_decode", dict(grouped_decode=True), [8, 16]),
            ]
        for route, values, widths in routes:
            values = dict(
                values,
                fused_swiglu_prefill=interleaved,
                raw_scale=raw,
                fast_prefill=interleaved,
            )
            policy = flags(values)
            layer.sm70_moe_policy = new.sm70_moe_policy = policy
            # Each row emulates a distinct initialized engine. Old Python reads
            # these flags per call; the new native ABI freezes them per owner.
            # Bind both algorithms to this row's policy instead of comparing
            # an old process's first-use flags with a different prepared owner.
            new.native_ops = NativeBindings(policy.native.values)
            reference_policy = getattr(
                new.native_ops, "arguments", new.native_ops.values
            )
            layer.sm70_fp4_codec = replace(
                layer.sm70_fp4_codec, bindings=new.native_ops
            )
            # These are the post-load effective availability fields, unchanged
            # between A/B. Each combination is admitted by the real selector.
            if family == "nvfp4":
                layer.sm70_nvfp4_qwen38_indexed_prefill = values.get(
                    "indexed_prefill", False
                )
                layer.sm70_nvfp4_qwen38_fused_swiglu_decode = (
                    interleaved and values.get("qpn_m1", False)
                )
                layer.sm70_nvfp4_qwen38_w2_direct_reduce = values.get(
                    "w2_direct_reduce", False
                )
            if args.model == "glm53":
                layer.sm70_glm53_fused_permute_q8 = values.get(
                    "glm53_fused_permute", False
                )
                layer.sm70_glm53_qpn_w13_q8 = values.get("glm53_qpn_w13", False)
            if args.grouped_only:
                layer.sm70_nvfp4_grouped_mtp5 = values.get("grouped_mtp5", False)
                layer.sm70_nvfp4_grouped_decode = values.get("grouped_decode", False)
            for m in widths:
                x = torch.randn(m, h, device="cuda", dtype=torch.float16) * 0.0625
                ids = torch.randint(0, e, (m, k), device="cuda", dtype=torch.int32)
                weights = torch.softmax(torch.randn(m, k, device="cuda"), dim=-1)

                def call(method, layer=layer, x=x, weights=weights, ids=ids):
                    with override_forward_context(
                        NS(
                            attn_metadata={"attention": NS(max_query_len=1)},
                            additional_kwargs={},
                        )
                    ):
                        return method.apply(layer, x, weights, ids, None, None)

                for method in (old, new):
                    call(method)
                native_calls.clear()
                a = call(old).clone()
                old_calls = list(native_calls)
                native_calls.clear()
                b = call(new).clone()
                new_calls = list(native_calls)
                torch.accelerator.synchronize()
                assert torch.isfinite(a).all() and torch.isfinite(b).all(), (
                    family,
                    profile,
                    route,
                    m,
                    "nonfinite",
                )
                assert torch.equal(a, b), (
                    family,
                    profile,
                    route,
                    m,
                    (a - b).abs().max().item(),
                )
                assert old_calls == new_calls, (old_calls, new_calls)
                if args.grouped_only:
                    assert "nvfp4_grouped_w13_sm70_out" in new_calls
                if route == "glm_qpn":
                    assert "nvfp4_glm53_moe_q8_qpn_sm70_out" in new_calls
                if route in ("glm_permute", "glm_qpn"):
                    assert "sm70_glm53_moe_permute_q8_out" in new_calls
                replays = 0
                timings = []
                if 0 < m <= 8:
                    graphs = []
                    outs = []
                    for method in (old, new):
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            out = call(method)
                        graphs.append(g)
                        outs.append(out)
                    for rep in range(3):
                        x.normal_(std=0.0625)
                        ids.random_(e)
                        weights.copy_(torch.softmax(torch.randn_like(weights), dim=-1))
                        graphs[0].replay()
                        reference = outs[0].clone()
                        graphs[1].replay()
                        actual = outs[1].clone()
                        torch.accelerator.synchronize()
                        assert torch.equal(reference, actual), (
                            family,
                            profile,
                            route,
                            m,
                            "replay",
                            rep,
                        )
                        replays += 1
                    for g in graphs:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(40):
                            g.replay()
                        end.record()
                        end.synchronize()
                        timings.append(start.elapsed_time(end) * 1000 / 40)
                    del graphs, outs, g, out
                record = dict(
                    family=family,
                    profile=profile,
                    route=route,
                    m=m,
                    eager_equal=True,
                    replays=replays,
                    native_calls=old_calls,
                    graph_us=timings,
                )
                results.append(record)
                print(json.dumps(record), flush=True)
        del layer, old, new
        torch.accelerator.empty_cache()
    suffix = "grouped" if args.grouped_only else args.model
    (root / "artifacts" / f"{family}-{suffix}-ab.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
