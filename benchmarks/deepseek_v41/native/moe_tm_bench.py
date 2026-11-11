# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill MoE on one TP rank of a V4.1 layer: current skinny QPN route vs the
TurboMind MXFP4 dense stage, and the cost of building the TurboMind stack.

moe_tm_bench.py [tokens ...]   (default 4096 8192)

384 experts, hidden 5120, 576 intermediate per rank, top-6, Zipf routing.
Times are CUDA-event medians (L2 flushed between runs). Also checks that the
two w13 outputs agree (same checkpoint codes and scales).
"""

import sys
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import (
    _prepare_mxfp4_sm70_experts,
)
from vllm.models.deepseek_v41 import native
from vllm.models.deepseek_v41.native import V41NativeMoEMethod, load_v41sk

DEV = torch.device("cuda")
E, H, I, TOPK = 384, 5120, 576, 6
FLUSH = torch.empty(64 << 20, dtype=torch.uint8, device=DEV)


def timed(fn, reps=5):
    times = []
    for _ in range(reps):
        FLUSH.sum()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.accelerator.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def checkpoint():
    g = torch.Generator(device=DEV).manual_seed(0)
    w13 = torch.randint(
        0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=DEV, generator=g
    )
    s13 = torch.randint(
        118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=DEV, generator=g
    )
    w2 = torch.randint(
        0, 256, (E, H, I // 2), dtype=torch.uint8, device=DEV, generator=g
    )
    s2 = torch.randint(
        118, 124, (E, H, I // 32), dtype=torch.uint8, device=DEV, generator=g
    )
    return w13, s13, w2, s2


def routing(tokens):
    weights = 1.0 / torch.arange(1, E + 1, device=DEV, dtype=torch.float32) ** 0.8
    weights = weights[torch.randperm(E, device=DEV)]
    ids = torch.multinomial(weights.expand(tokens, E), TOPK, replacement=False)
    probs = torch.rand(tokens, TOPK, device=DEV).softmax(-1)
    return ids.int(), probs


def load_tmg():
    import os

    import vllm._C  # noqa: F401  (the implementation lives there)
    from torch.utils.cpp_extension import load

    lib = os.path.dirname(vllm._C.__file__)
    return load(
        "tmg",
        [os.path.join(os.path.dirname(__file__), "tmg", "tmg.cpp")],
        extra_ldflags=[f"-L{lib}", "-l:_C.abi3.so", f"-Wl,-rpath,{lib}"],
        verbose=False,
    )


def main():
    load_v41sk()
    tmg = load_tmg()
    sizes = [int(v) for v in sys.argv[1:]] or [4096, 8192]
    w13, s13, w2, s2 = checkpoint()
    bytes_per_layer = sum(t.numel() for t in (w13, s13, w2, s2))
    print(f"expert bytes per layer per rank {bytes_per_layer / 2**30:.2f} GiB")

    # TurboMind stack from the checkpoint layout (what a transcode must build).
    torch.accelerator.synchronize()
    build = timed(
        lambda: (
            _prepare_mxfp4_sm70_experts(w13, s13),
            _prepare_mxfp4_sm70_experts(w2, s2),
        ),
        reps=1,
    )
    tm13w, tm13s, m13 = _prepare_mxfp4_sm70_experts(w13, s13)
    tm2w, tm2s, m2 = _prepare_mxfp4_sm70_experts(w2, s2)
    p13 = torch.ops._C.awq_moe_build_strided_ptrs(
        tm13w, tm13s, int(m13[0][0]), int(m13[0][1]), E
    )
    p2 = torch.ops._C.awq_moe_build_strided_ptrs(
        tm2w, tm2s, int(m2[0][0]), int(m2[0][1]), E
    )
    copy = timed(lambda: (tm13w.clone(), tm2w.clone()))
    print(
        f"TM stack build (per-expert prepare ops) {build:.1f} ms; "
        f"one D2D copy of the stack {copy:.1f} ms"
    )

    # Current route: the same checkpoint permuted into QPN fragment order.
    layer = SimpleNamespace(
        w13_weight=w13.clone(),
        w13_weight_scale=s13.clone(),
        w2_weight=w2.clone(),
        w2_weight_scale=s2.clone(),
    )
    V41NativeMoEMethod.process_weights_after_loading(None, layer)

    for tokens in sizes:
        x = (torch.randn(tokens, H, device=DEV) * 0.5).half()
        ids, probs = routing(tokens)
        skinny = timed(lambda: V41NativeMoEMethod.apply(None, layer, x, probs, ids))

        flat = ids.view(-1).long()
        order = flat.argsort(stable=True)
        counts = torch.bincount(flat, minlength=E)
        active = counts.nonzero().view(-1)
        offsets = torch.zeros(active.numel() + 1, dtype=torch.int32, device=DEV)
        offsets[1:] = counts[active].cumsum(0)
        experts = active.int()
        slots = flat.numel()
        xp = x[order // TOPK].contiguous()
        out13 = torch.empty(slots, 2 * I, dtype=torch.half, device=DEV)
        inter = torch.empty(slots, I, dtype=torch.half, device=DEV)
        out2 = torch.empty(slots, H, dtype=torch.half, device=DEV)

        def tm_w13():
            torch.ops._C.mxfp4_moe_dense_stage_sm70_out(
                out13,
                xp,
                offsets,
                experts,
                p13[0],
                p13[1],
                experts.numel(),
                H,
                2 * I,
                32,
            )

        def tm_w2():
            torch.ops._C.mxfp4_moe_dense_stage_sm70_out(
                out2, inter, offsets, experts, p2[0], p2[1], experts.numel(), I, H, 32
            )

        def tg_w13():
            tmg.grouped(
                out13, xp, offsets, p13[0], p13[1], experts.numel(), H, 2 * I, experts
            )

        def tg_w2():
            tmg.grouped(
                out2, inter, offsets, p2[0], p2[1], experts.numel(), I, H, experts
            )

        g13, g2 = timed(tg_w13), timed(tg_w2)
        tg_w13()
        torch.accelerator.synchronize()
        grouped13 = out13.float().clone()
        t13, t2 = timed(tm_w13), timed(tm_w2)
        flops = 2 * slots * (2 * I * H + I * H)
        # Same w13 product on the skinny route, for a numerical check.
        gate_up = torch.empty(slots, 2 * I, dtype=torch.half, device=DEV)
        perm, eids, offs = native._route_plan(ids)
        torch.ops.v41sk.skinny_moe_qpn_sm70(
            x,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.v41_g13,
            perm,
            eids,
            offs,
            TOPK,
            gate_up,
            False,
            tokens,
            layer.v41_split13,
            1,
            1,
        )
        tm_w13()
        torch.accelerator.synchronize()
        got = out13.float()
        rel = min(
            ((got - ref).norm() / ref.norm()).item()
            for ref in (gate_up.float(), gate_up[perm.long()].float())
        )
        print(
            f"tokens {tokens}: skinny route {skinny:.2f} ms | TM w13 {t13:.2f} + w2 {t2:.2f} "
            f"= {t13 + t2:.2f} ms ({flops / (t13 + t2) / 1e9:.1f} TFLOPS) | "
            f"active experts {experts.numel()} | w13 rel diff {rel:.2e}"
        )
        same = ((grouped13 - got).norm() / got.norm()).item()
        print(
            f"    TM grouped one launch: w13 {g13:.2f} + w2 {g2:.2f} = {g13 + g2:.2f} ms "
            f"({flops / (g13 + g2) / 1e9:.1f} TFLOPS), vs per-expert rel {same:.2e}"
        )


if __name__ == "__main__":
    main()
