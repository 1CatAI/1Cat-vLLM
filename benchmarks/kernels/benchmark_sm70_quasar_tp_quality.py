# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same real activation/weight inputs; isolate TP partition and kernel route.

No distributed model rollout here. Row-parallel reductions use FP32 summation
followed by FP16, with partial outputs retained for a real-collective check.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm import _sm70_ops
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    ref_nvfp4_quant_dequant,
)

_spec = importlib.util.spec_from_file_location(
    "quasar_tp_reference",
    Path(__file__).with_name("benchmark_sm70_quasar_nvfp4_oracle.py"),
)
assert _spec is not None and _spec.loader is not None
ref = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ref
_spec.loader.exec_module(ref)


def metrics(actual, expected):
    a, e = actual.float(), expected.float()
    d = a - e
    return {
        "max_abs": d.abs().max().item(),
        "relative_l2": (d.norm() / e.norm().clamp_min(1e-30)).item(),
        "max_row_relative_l2": (d.norm(dim=-1) / e.norm(dim=-1).clamp_min(1e-30))
        .max()
        .item(),
        "rmse": d.square().mean().sqrt().item(),
        "unequal": int((a != e).sum()),
        "elements": a.numel(),
        "finite": bool(torch.isfinite(a).all()),
        "reference_absmax": e.abs().max().item(),
    }


def projections(cp, layer):
    p = f"model.language_model.layers.{layer}"
    if layer % 4 != 3:
        yield (
            "gdn_qkvzba",
            "linear_attn.in_proj_qkvz",
            "column",
            tuple(
                p + ".linear_attn." + s
                for s in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
            ),
            [2048, 2048, 6144, 6144, 48, 48],
        )
        yield (
            "gdn_out",
            "linear_attn.out_proj",
            "row",
            (p + ".linear_attn.out_proj",),
            [5120],
        )
    else:
        yield (
            "attention_qkv",
            "self_attn.qkv_proj",
            "column",
            tuple(p + ".self_attn." + s for s in ["q_proj", "k_proj", "v_proj"]),
            [12288, 1024, 1024],
        )
        yield (
            "attention_out",
            "self_attn.o_proj",
            "row",
            (p + ".self_attn.o_proj",),
            [5120],
        )
    yield (
        "mlp_gate_up",
        "mlp.gate_up_proj",
        "column",
        (p + ".mlp.gate_proj", p + ".mlp.up_proj"),
        [17408, 17408],
    )
    yield "mlp_down", "mlp.down_proj", "row", (p + ".mlp.down_proj",), [5120]


def column_rows(widths, tp, rank):
    parts, offset = [], 0
    for width in widths:
        assert width % tp == 0
        parts.append(
            torch.arange(offset + rank * width // tp, offset + (rank + 1) * width // tp)
        )
        offset += width
    return torch.cat(parts)


def native(pr, x, route, tm_alignment=16):
    n, k = pr.packed.shape[0], pr.packed.shape[1] * 2
    if route == "tm":
        pn = (n + tm_alignment - 1) // tm_alignment * tm_alignment
        qw = ref._unpack_codes(pr.packed).cuda()
        scales = (
            (pr.scales.t().float() / pr.weight_global_divisor)
            .half()
            .contiguous()
            .cuda()
        )
        if pn != n:
            qw = F.pad(qw, (0, pn - n))
            scales = F.pad(scales, (0, pn - n))
        tw, ts, meta = _sm70_ops.nvfp4_sm70_prepare(qw, scales, 16, False)
        out = torch.empty((x.shape[0], pn), device=x.device, dtype=x.dtype)
        _sm70_ops.nvfp4_gemm_sm70_out(
            out, x, tw, ts, 16, int(meta[0]), int(meta[1]), False
        )
        y = out[:, :n].clone()
        gated = None
        if pr.name == "mlp_gate_up":
            gated = torch.empty((x.shape[0], n // 2), device=x.device, dtype=x.dtype)
            torch.ops._C.silu_and_mul(gated, y)
        return y, gated
    pn = (n + 31) // 32 * 32
    packed, scales = pr.packed.cuda(), pr.scales.cuda()
    if pn != n:
        p = torch.zeros((pn, packed.shape[1]), device=x.device, dtype=packed.dtype)
        s = torch.zeros((pn, scales.shape[1]), device=x.device, dtype=scales.dtype)
        p[:n].copy_(packed)
        s[:n].copy_(scales)
        packed, scales = p, s
    codes, qs = _sm70_ops.nvfp4_qpn2_prepare_sm70(packed, scales)
    split = 8 if pr.name in ["gdn_out", "attention_out", "mlp_gate_up"] else 16
    assert (k // 16) % split == 0
    out = torch.empty((x.shape[0], pn), device=x.device, dtype=x.dtype)
    _sm70_ops.nvfp4_qpn2_gemm_sm70_out(
        out, x, codes, qs, 1 / pr.weight_global_divisor, split, 2
    )
    y = out[:, :n].clone()
    gated = None
    if pr.name == "mlp_gate_up":
        assert pn == n
        gated = torch.empty((x.shape[0], n // 2), device=x.device, dtype=x.dtype)
        _sm70_ops.nvfp4_qpn2_gated_sm70_out(
            gated, x, codes, qs, 1 / pr.weight_global_divisor, split, 2
        )
    return y, gated


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--layers", nargs="+", type=int, default=list(range(64)))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tm-alignment", type=int, choices=[16, 32], default=32)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cp = ref.Checkpoint(args.model)
    tables = []
    positions = None
    input_ids = None
    for rank in range(4):
        d = torch.load(
            args.capture_dir / f"rank{rank}-step{args.step}.pt", weights_only=True
        )
        if positions is None:
            positions, input_ids = d["positions"], d["input_ids"]
        assert torch.equal(positions, d["positions"]) and torch.equal(
            input_ids, d["input_ids"]
        )
        tables.append(
            {(t["layer_idx"], t["label"]): t["tensor"] for t in d["tensors"].values()}
        )
    rows, partials = [], {}
    for layer in args.layers:
        for name, suffix, parallel, prefixes, widths in projections(cp, layer):
            tag = f"quant_input:language_model.model.layers.{layer}.{suffix}" + (
                ":gated" if name == "mlp_gate_up" else ":linear"
            )
            xs = [t[layer, tag] for t in tables]
            if parallel == "column":
                # Captured replicated tensors can already differ by rounding.
                # Fix rank-zero input for every shard in this operator oracle.
                fullx = xs[0].cuda()
                full = ref._load_column_parallel(cp, name, prefixes, 0, 1)
            else:
                fullx = torch.cat(xs, dim=-1).cuda()
                full = ref._load_row_parallel(cp, name, prefixes[0], 0, 1)
            weight = ref._dequantize_weight(full, torch.device("cuda"))
            assert tuple(weight.shape) == (sum(widths), fullx.shape[-1])
            oracle = fullx.float() @ weight.t()
            full_half = fullx.float() @ weight.half().float().t()
            qat_x = ref_nvfp4_quant_dequant(
                fullx,
                torch.tensor(
                    full.input_global_divisor, device=fullx.device, dtype=torch.float32
                ),
                16,
            )
            qat_out = qat_x.float() @ weight.t()
            assembled, gated_all = {}, {}
            part_sets = {}
            row = {
                "layer": layer,
                "operator": name,
                "parallel": parallel,
                "input_shape": list(fullx.shape),
                "weight_shape": list(weight.shape),
                "fp16_materialized_weight_vs_checkpoint": metrics(
                    weight.half(), weight
                ),
                "single_dense_fp16_weight_vs_checkpoint_output": metrics(
                    full_half, oracle
                ),
                "local_shapes": {},
            }
            if parallel == "column":
                row["captured_rank_inputs_vs_rank0"] = [metrics(x, xs[0]) for x in xs]
            row["qat_w4a4_activation_vs_w4a16"] = metrics(qat_x, fullx)
            row["qat_w4a4_output_vs_w4a16"] = metrics(qat_out, oracle)
            for tp in [2, 4]:
                products = {"tm": [], "qpn2": [], "fp32": [], "fp16": []}
                gates = {"tm": [], "qpn2": []}
                for rank in range(tp):
                    if parallel == "column":
                        pr = ref._load_column_parallel(cp, name, prefixes, rank, tp)
                        ids = column_rows(widths, tp, rank)
                        assert torch.equal(pr.packed, full.packed[ids])
                        assert torch.equal(
                            pr.scales.view(torch.uint8),
                            full.scales.view(torch.uint8)[ids],
                        )
                        x = fullx
                        local_ref = oracle[:, ids.cuda()]
                    else:
                        pr = ref._load_row_parallel(cp, name, prefixes[0], rank, tp)
                        sl = slice(
                            rank * fullx.shape[-1] // tp,
                            (rank + 1) * fullx.shape[-1] // tp,
                        )
                        assert torch.equal(
                            pr.packed,
                            full.packed[:, slice(sl.start // 2, sl.stop // 2)],
                        )
                        x = fullx[:, sl].contiguous()
                        local_ref = x.float() @ weight[:, sl].t()
                    row["local_shapes"][f"tp{tp}"] = {
                        "M": x.shape[0],
                        "N": pr.packed.shape[0],
                        "K": x.shape[-1],
                    }
                    products["fp32"].append(local_ref)
                    products["fp16"].append(local_ref.half())
                    for route in ["tm", "qpn2"]:
                        y, g = native(pr, x, route, tm_alignment=args.tm_alignment)
                        products[route].append(y)
                        if g is not None:
                            gates[route].append(g)
                for route, ys in products.items():
                    if parallel == "row":
                        total = torch.stack([y.float() for y in ys]).sum(0)
                        assembled[f"tp{tp}_{route}"] = (
                            total if route == "fp32" else total.half()
                        )
                        if route in ["tm", "qpn2"]:
                            part_sets[f"tp{tp}_{route}"] = torch.stack(ys).cpu()
                    else:
                        total = torch.empty_like(oracle)
                        for rank, y in enumerate(ys):
                            total[:, column_rows(widths, tp, rank).cuda()] = y.float()
                        assembled[f"tp{tp}_{route}"] = total
                for route, gs in gates.items():
                    if gs:
                        gated_all[f"tp{tp}_{route}"] = torch.cat(gs, dim=-1)
            row["vs_checkpoint_fp32"] = {
                k: metrics(v, oracle) for k, v in assembled.items()
            }
            row["tp2_vs_tp4"] = {
                route: metrics(assembled["tp4_" + route], assembled["tp2_" + route])
                for route in ["tm", "qpn2", "fp32", "fp16"]
            }
            row["production_tp4_qpn2_vs_tp2_tm"] = metrics(
                assembled["tp4_qpn2"], assembled["tp2_tm"]
            )
            row["tp4_qpn2_vs_tm"] = metrics(assembled["tp4_qpn2"], assembled["tp4_tm"])
            if gated_all:
                gate, up = oracle.chunk(2, -1)
                expected = F.silu(gate) * up
                row["gated_vs_fp32"] = {
                    k: metrics(v, expected) for k, v in gated_all.items()
                }
                row["gated_tp2_vs_tp4"] = {
                    route: metrics(gated_all["tp4_" + route], gated_all["tp2_" + route])
                    for route in ["tm", "qpn2"]
                }
                row["gated_production_tp4_qpn2_vs_tp2_tm"] = metrics(
                    gated_all["tp4_qpn2"], gated_all["tp2_tm"]
                )
            partials[f"{layer}:{name}"] = part_sets
            rows.append(row)
            print(
                layer,
                name,
                "TP delta",
                row["production_tp4_qpn2_vs_tp2_tm"]["relative_l2"],
                "same TM",
                row["tp2_vs_tp4"]["tm"]["relative_l2"],
                flush=True,
            )
            args.out.write_text(
                json.dumps(
                    {
                        "model": str(args.model),
                        "activation_capture": str(args.capture_dir),
                        "step": args.step,
                        "tm_alignment": args.tm_alignment,
                        "note": (
                            "TP2 QPN2 is a kernel counterfactual, not production "
                            "dispatch. FP32 summation emulates row collective; check "
                            "collectives separately. Reference uses checkpoint "
                            "dequantization, not the BF16 teacher."
                        ),
                        "rows": rows,
                    },
                    indent=2,
                )
            )
        torch.save(partials, args.out.with_suffix(".partials.pt"))


if __name__ == "__main__":
    main()
