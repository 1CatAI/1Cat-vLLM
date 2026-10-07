# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen four exact snapshots plus cached tail rank-one updates for GDN.

Inputs are captured from a real-weight TP4 layer, not invented kernel tensors.
Keep the first four snapshots; reconstruct longer accepted prefixes from exact
cached FP32 factors. No replays are needed for acceptance lengths one to four.
This is a research screen, not state-manager or serving admission.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as gdn
from vllm.model_executor.layers.fla.ops.op import exp
from vllm.triton_utils import tl, triton


def candidate_module(directory):
    source = Path(gdn.__file__).read_text()
    begin = source.index("@triton.heuristics(")
    end = source.index("\ndef fused_sigmoid_gating_delta_rule_update(", begin)
    kernel = (
        source[begin:end]
        .replace(
            "def fused_sigmoid_gating_delta_rule_update_kernel(",
            "def hybrid_snapshot_kernel(",
            1,
        )
        .replace(
            "    A_log,\n",
            "    old_k,\n    old_v,\n    old_g,\n"
            "    save_k,\n    save_v,\n    save_g,\n"
            "    REPLAY_PREVIOUS: tl.constexpr,\n    A_log,\n",
            1,
        )
    )
    initial = (
        "                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1"
    )
    assert kernel.count(initial) == 1
    kernel = kernel.replace(
        initial, initial + "\n                i_t = tl.minimum(i_t, 3)"
    )
    replay = """
    if REPLAY_PREVIOUS:
        count = tl.load(num_accepted_tokens + i_n)
        for step in range(4, count):
            rk = tl.load(old_k + (step * HV + i_hv) * K + o_k,
                         mask=mask_k, other=0)
            rv = tl.load(old_v + (step * HV + i_hv) * V + o_v,
                         mask=mask_v, other=0)
            rg = tl.load(old_g + step * HV + i_hv)
            decayed = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                args=[b_h, exp(rg)], dtype=tl.float32, is_pure=True, pack=1)
            b_h = tl.fma(rv[:, None], rk[None, :], decayed)

"""
    marker = "    for i_t in range(0, T):\n"
    assert kernel.count(marker) == 1
    kernel = kernel.replace(marker, replay + marker, 1)
    save = """
        if i_t >= 4:
            tl.store(save_v + (i_t * HV + i_hv) * V + o_v, b_v, mask=mask_v)
            if i_v == 0:
                tl.store(save_k + (i_t * HV + i_hv) * K + o_k, b_k, mask=mask_k)
                tl.store(save_g + i_t * HV + i_hv, b_g)
"""
    marker = "        b_v *= b_beta\n"
    assert kernel.count(marker) == 1
    kernel = kernel.replace(marker, marker + save, 1)
    begin = kernel.index("        # keep the states for multi-query tokens")
    end = kernel.index("        if MIXED_QKV:\n", begin)
    store = kernel[begin:end]
    kernel = (
        kernel[:begin]
        + "        if i_t < 4:\n"
        + "\n".join("    " + line if line else "" for line in store.splitlines())
        + "\n\n"
        + kernel[end:]
    )
    generated = directory / "hybrid_snapshot_generated.py"
    text = (
        "from vllm.triton_utils import triton, tl\n"
        "from vllm.model_executor.layers.fla.ops.op import exp\n\n" + kernel
    )
    generated.write_text(text)
    spec = importlib.util.spec_from_file_location(
        "hybrid_snapshot_generated", generated
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hybrid_snapshot_kernel, hashlib.sha256(text.encode()).hexdigest()


@triton.jit
def recover(State, Indices, KCache, VCache, GCache, Out, STRIDE: tl.constexpr):
    tile, sequence_head = tl.program_id(0), tl.program_id(1)
    snapshot, head = sequence_head // 12, sequence_head % 12
    k = tl.arange(0, 128)
    v = tile * 2 + tl.arange(0, 2)
    slot = tl.load(Indices + tl.minimum(snapshot, 3))
    value = tl.load(
        State + slot * STRIDE + head * 16384 + v[:, None] * 128 + k[None, :]
    )
    for step in range(4, snapshot + 1):
        rk = tl.load(KCache + (step * 12 + head) * 128 + k)
        rv = tl.load(VCache + (step * 12 + head) * 128 + v)
        rg = tl.load(GCache + step * 12 + head)
        decayed = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[value, exp(rg)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        value = tl.fma(rv[:, None], rk[None, :], decayed)
    tl.store(
        Out + snapshot * 196608 + head * 16384 + v[:, None] * 128 + k[None, :], value
    )


def launch(
    kernel, qkv, g, beta, state, indices, out, cu, accepted, factors=None, replay=True
):
    assert qkv.shape == (8, 2560) and qkv.dtype == torch.float16
    assert state.dtype == torch.float32 and indices.shape == (1, 8)
    kwargs = dict(
        A_log=g,
        a=g,
        b=beta,
        dt_bias=g,
        beta=1.0,
        threshold=20.0,
        q=qkv,
        k=qkv,
        v=qkv,
        mixed_qkv=qkv,
        o=out,
        h0=state,
        ht=state,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        ddtree_parent_ids=None,
        scale=128**-0.5,
        N=1,
        T=8,
        B=1,
        H=4,
        HV=12,
        K=128,
        V=128,
        BK=128,
        BV=2,
        Q_OFFSET=0,
        K_OFFSET=512,
        V_OFFSET=1024,
        QKV_STRIDE=2560,
        stride_init_state_token=state.stride(0),
        stride_final_state_token=state.stride(0),
        stride_indices_seq=indices.stride(0),
        stride_indices_tok=1,
        stride_parent_ids_seq=1,
        stride_parent_ids_tok=1,
        INPLACE_FINAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=True,
        IS_KDA=False,
        MIXED_QKV=True,
        PRECOMPUTED_GATING=True,
        QUANTIZE_BETA=False,
        QUANTIZE_STATE_EACH_STEP=False,
        MATCH_RECURRENT_NUMERICS=True,
        num_warps=1,
        num_stages=3,
    )
    if factors is not None:
        kwargs.update(
            old_k=factors[0],
            old_v=factors[1],
            old_g=factors[2],
            save_k=factors[3],
            save_v=factors[4],
            save_g=factors[5],
            REPLAY_PREVIOUS=replay,
        )
    kernel[(1, 64, 12)](**kwargs)


def capture(operation, reset, eviction):
    reset()
    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(graph):
        reset()
        eviction.fill_(1)
        begin.record()
        operation()
        end.record()
    return graph, begin, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    fixture = torch.load(args.fixture, weights_only=True)
    qkv, g, beta = (fixture[k].cuda() for k in ("qkv", "g", "beta"))
    indices = fixture["indices"].cuda()
    states = [fixture["state"].cuda() for _ in range(2)]
    outputs = [qkv.new_empty(8, 1, 12, 128) for _ in range(2)]
    caches = [
        [torch.empty(8, 12, 128, device="cuda") for _ in range(2)]
        + [torch.empty(8, 12, device="cuda")]
        for _ in range(2)
    ]
    cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([1], device="cuda", dtype=torch.int32)
    hybrid, generated_hash = candidate_module(args.out)
    reference = gdn.fused_sigmoid_gating_delta_rule_update_kernel
    launch(reference, qkv, g, beta, states[0], indices, outputs[0], cu, accepted)
    launch(
        hybrid,
        qkv,
        g,
        beta,
        states[1],
        indices,
        outputs[1],
        cu,
        accepted,
        factors=(*caches[0], *caches[1]),
        replay=False,
    )
    seeds = [s.clone() for s in states]
    for old, new in zip(caches[0], caches[1]):
        old.copy_(new)
    golden = torch.empty(8, 12, 128, 128, device="cuda")
    recover[(64, 96)](
        states[1], indices, *caches[0], golden, states[1].stride(0), num_warps=1
    )
    assert torch.equal(
        states[0][indices.flatten().long()].view(torch.int32), golden.view(torch.int32)
    )
    assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))

    def operation(arm):
        launch(
            reference if arm == 0 else hybrid,
            qkv,
            g,
            beta,
            states[arm],
            indices,
            outputs[arm],
            cu,
            accepted,
            factors=(*caches[0], *caches[1]) if arm else None,
        )

    def reset(arm):
        states[arm].copy_(seeds[arm])

    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    graphs = [
        capture(lambda arm=arm: operation(arm), lambda arm=arm: reset(arm), eviction)
        for arm in range(2)
    ]
    results = []
    for prefix in range(1, 9):
        accepted.fill_(prefix)
        for arm in range(2):
            reset(arm)
            operation(arm)
        recover[(64, 96)](
            states[1], indices, *caches[1], golden, states[1].stride(0), num_warps=1
        )
        assert torch.equal(
            states[0][indices.flatten().long()].view(torch.int32),
            golden.view(torch.int32),
        )
        assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
        samples = [[], []]
        for i in range(args.iters + 20):
            for arm in (i % 2, 1 - i % 2):
                graph, begin, end = graphs[arm]
                graph.replay()
                end.synchronize()
                if i >= 20:
                    samples[arm].append(begin.elapsed_time(end) * 1000)
        delta = [a - b for a, b in zip(*samples)]
        rng = random.Random(123)
        boot = sorted(
            statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(2000)
        )
        result = dict(
            accepted=prefix,
            reference_us=statistics.mean(samples[0]),
            hybrid_us=statistics.mean(samples[1]),
            saving_us=statistics.mean(delta),
            ci95_us=[boot[49], boot[1949]],
            samples_us=samples,
            output_and_recovered_state_bitwise=True,
        )
        results.append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "samples_us"}),
            flush=True,
        )
    (args.out / "result.json").write_text(
        json.dumps(
            dict(
                results=results,
                generated_sha256=generated_hash,
                fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                reference_source_sha256=hashlib.sha256(
                    Path(gdn.__file__).read_bytes()
                ).hexdigest(),
                research_only=True,
                state_manager_not_modified=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
