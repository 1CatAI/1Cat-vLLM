# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router ablation with real routed/shared experts and production overlap."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sm70_router_benchmark_utils import pair, topk

from vllm.model_executor.layers.quantization.gguf_dense_hmma_formats import decode, pack
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.model_executor.layers.quantization.gguf_turbomind_moe import GGUFExpertBank
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import _pack_router_batch_weight
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def layer_data(tensors, layer):
    def tensor(name):
        return tensors[f"blk.{layer}.{name}.weight"]

    def value(name):
        t = tensor(name)
        return dequantize(t.data, int(t.tensor_type)).copy()

    def shared_segment(name, down=False):
        t = tensor(name)
        fmt, codes, scales, mins, group = decode(t.data, int(t.tensor_type))
        if down:
            codes, scales = codes[:, :160], scales[:, : 160 // group]
            if mins is not None:
                mins = mins[:, : 160 // group]
        else:
            codes, scales = codes[:160], scales[:160]
            if mins is not None:
                mins = mins[:160]
        return fmt, [
            torch.from_numpy(v).cuda() for v in pack(fmt, codes, scales, mins, group)
        ]

    sgfmt, sg = shared_segment("ffn_gate_shexp")
    sufmt, su = shared_segment("ffn_up_shexp")
    sdfmt, sd = shared_segment("ffn_down_shexp", True)
    wg = torch.from_numpy(value("ffn_gate_inp_shexp")).float().cuda().flatten()
    rw = torch.from_numpy(value("ffn_gate_inp")).half().cuda().reshape(512, 2560)
    packed = _pack_router_batch_weight(rw)
    sources = [tensor(f"ffn_{s}_exps") for s in ("gate", "up", "down")]
    typ, down_type = int(sources[0].tensor_type), int(sources[2].tensor_type)
    raw = [
        torch.from_numpy(
            np.stack(
                [
                    RawGGUFProjection.from_rows(v, typ).tp_slice(0, 4, axis=0).data
                    for v in t.data
                ]
            )
        ).cuda()
        for t in sources[:2]
    ]
    bank = GGUFExpertBank(down_type, 512, torch.device("cuda"), torch.float16)
    for e, row in enumerate(sources[2].data):
        bank.add(e, torch.from_numpy(row.copy()), 0, 4, axis=1)
    bank.finalize()
    return sgfmt, sg, sufmt, su, sdfmt, sd, wg, rw, packed, typ, down_type, raw, bank


def measure(tensors, layer, replays):
    sgfmt, sg, sufmt, su, sdfmt, sd, wg, rw, packed, typ, down_type, raw, bank = (
        layer_data(tensors, layer)
    )
    torch.manual_seed(layer + 320)
    x = torch.randn(5, 2560, device="cuda", dtype=torch.float16) * 0.25
    routing_weights = torch.empty(5, 10, device="cuda", dtype=torch.float32)
    ids = torch.empty(5, 10, device="cuda", dtype=torch.int32)
    source = torch.empty_like(ids)
    logits = torch.empty(5, 512, device="cuda", dtype=torch.float16)
    q8 = torch.empty(5, 80, 36, device="cuda", dtype=torch.uint8)
    hidden = torch.empty(5, 10, 5, 36, device="cuda", dtype=torch.uint8)
    routed, shared = torch.empty_like(x), torch.empty_like(x)
    sh = torch.empty(5, 160, device="cuda", dtype=torch.float16)
    gate = torch.empty(32, device="cuda", dtype=torch.float16)
    partial = torch.empty(1024 * 1024, device="cuda", dtype=torch.float32)
    counters = torch.zeros(1024, device="cuda", dtype=torch.int32)
    aux = torch.cuda.Stream()

    def shared_work():
        torch.ops._C.gguf_shared_gate_up_sm70_out(
            x, sg, su, [sgfmt, sufmt], wg, sh, gate, partial, counters, 5, 8
        )
        torch.ops._C.gguf_dense_segments_sm70_out(
            sh,
            [sd[0]],
            [sd[1]],
            [sd[2]],
            [shared],
            [sdfmt],
            [2560],
            160,
            1,
            8,
            partial,
            counters,
            gate,
        )

    def router_work(variant):
        torch.ops._C.qwen38_router_batch_sm70_out(logits, x, packed)
        if variant == "control":
            topk(logits, (routing_weights, ids, source))
        elif variant == "select_quantize":
            torch.ops.vllm_sm70_router.select_quantize(
                routing_weights, ids, source, q8, logits, x
            )
        else:
            torch.ops.vllm_sm70_router.top10(routing_weights, ids, source, logits)

    def chain(variant):
        for _ in range(8):
            aux.wait_stream(torch.cuda.current_stream())
            router_work(variant)
            if variant != "select_quantize":
                torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
            torch.ops._C.gguf_dp4a_gate_up_sm70_out(hidden, q8, ids, *raw, typ, True)
            torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
                routed,
                hidden,
                ids,
                routing_weights,
                bank.weight_ptrs,
                bank.stat_ptrs,
                down_type,
                512,
            )
            with torch.cuda.stream(aux):
                shared_work()
            torch.cuda.current_stream().wait_stream(aux)

    result = dict(
        layer=layer, quant=typ, scope="8 isolated complete MoE chains", variants={}
    )
    variants = ("top10", "select_quantize")
    for variant in variants:
        chain("control")
        torch.accelerator.synchronize()
        reference = [t.clone() for t in (routed, shared, ids, routing_weights)]
        chain(variant)
        torch.accelerator.synchronize()
        ref, shared_ref, ids_ref, weights_ref = reference
        check = dict(
            ids_exact=torch.equal(ids, ids_ref),
            shared_exact=torch.equal(shared, shared_ref),
            routed_relative_l2=(
                (routed.float() - ref.float()).norm() / ref.float().norm()
            ).item(),
            weights_max_abs=(routing_weights - weights_ref).abs().max().item(),
        )
        assert check["ids_exact"] and check["shared_exact"], check
        assert check["weights_max_abs"] < 3e-7, check
        assert check["routed_relative_l2"] < 1e-4, check
        timing = pair(
            lambda: chain("control"),
            lambda variant=variant: chain(variant),
            replays,
        )
        result["variants"][variant] = dict(check=check, timing=timing)
        print("MoE", layer, variant, check, timing, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    import vllm._sm70_router_C  # noqa: F401

    readers = [GGUFReader(f) for f in sorted(args.model_dir.glob("*IQ3_S*.gguf"))]
    tensors = {t.name: t for reader in readers for t in reader.tensors}
    result = []
    for layer in (17, 0, 1):
        result.append(measure(tensors, layer, args.replays))
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
