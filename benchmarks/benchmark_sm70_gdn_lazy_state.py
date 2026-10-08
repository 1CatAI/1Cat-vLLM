# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only FP32 recurrent update-log screen; no serving dispatch.

Reuse the installed recurrent verifier as control. Derive only persistence
and initial-state reconstruction from the owned source; no new q/k reduction
or quantization decoder is introduced. Two log banks prevent producer/read
races between V tiles, including CTAs that have not been scheduled yet.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
import uuid
from pathlib import Path

import torch

from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update_kernel as reference,
)

REPLAY = """
@triton.jit
def replay_history(state, decay, key, delta, prefix, HV: tl.constexpr,
                   K: tl.constexpr, V: tl.constexpr, head, ov, ok):
    for step in range(prefix):
        d = tl.load(decay + step * HV + head)
        k = tl.load(key + (step * HV + head) * K + ok)
        v = tl.load(delta + (step * HV + head) * V + ov)
        # Keep the materialized decay and fused rank-one update of the
        # original recurrence. The original matrix is also used by its dot
        # product between these operations, so decay rounds before the FMA.
        state = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=f,f,f", [state, d],
            dtype=tl.float32, is_pure=True, pack=1)
        state = tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
            [v[:, None], k[None, :], state],
            dtype=tl.float32, is_pure=True, pack=1)
    return state

@triton.jit
def materialize(base, decay, key, delta, out, prefix: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                BK: tl.constexpr, BV: tl.constexpr):
    head = tl.program_id(1)
    ov = tl.program_id(0) * BV + tl.arange(0, BV)
    ok = tl.arange(0, BK)
    mask = (ov[:, None] < V) & (ok[None, :] < K)
    off = head * V * K + ov[:, None] * K + ok[None, :]
    state = tl.load(base + off, mask=mask, other=0)
    state = replay_history(state, decay, key, delta, prefix, HV, K, V,
                           head, ov, ok)
    tl.store(out + off, state, mask=mask)
"""


SELECTED = """
@triton.jit
def materialize_selected_layers(base, decay, key, delta, state_ptrs, slots,
                                accepted, HV: tl.constexpr, K: tl.constexpr,
                                V: tl.constexpr, BK: tl.constexpr,
                                BV: tl.constexpr):
    layer = tl.program_id(2)
    head = tl.program_id(1)
    ov = tl.program_id(0) * BV + tl.arange(0, BV)
    ok = tl.arange(0, BK)
    prefix = tl.load(accepted)
    if prefix < 1 or prefix > 8:
        return
    slot = tl.load(slots + layer * 8 + prefix - 1)
    if slot < 0:
        return
    mask = (ov[:, None] < V) & (ok[None, :] < K)
    offset = head * V * K + ov[:, None] * K + ok[None, :]
    state = tl.load(base + layer * HV * V * K + offset,
                    mask=mask, other=0)
    state = replay_history(
        state, decay + layer * 8 * HV,
        key + layer * 8 * HV * K, delta + layer * 8 * HV * V,
        prefix, HV, K, V, head, ov, ok)
    destination = tl.load(state_ptrs + layer).to(tl.pointer_type(tl.float32))
    tl.store(destination + slot * HV * V * K + offset, state, mask=mask)
"""


def generate(root, output, selected=False):
    path = root / "vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py"
    original = path.read_text()
    start = original.index("@triton.heuristics(")
    end = original.index("\ndef fused_sigmoid_gating_delta_rule_update(", start)
    kernel = original[start:end].replace(
        "def fused_sigmoid_gating_delta_rule_update_kernel(",
        "def lazy_state_kernel(",
    )
    kernel = kernel.replace(
        "    MATCH_RECURRENT_NUMERICS: tl.constexpr,\n):",
        "    MATCH_RECURRENT_NUMERICS: tl.constexpr,\n"
        "    prev_decay, prev_key, prev_delta, next_decay, next_key, next_delta,\n"
        "    HAS_HISTORY: tl.constexpr,\n):",
    )
    if selected:
        kernel = kernel.replace(
            "    HAS_HISTORY: tl.constexpr,\n):",
            "    HAS_HISTORY: tl.constexpr,\n    base_target,\n):",
        )
    else:
        kernel = kernel.replace(
            "            p_h0 = h0 + state_idx * stride_init_state_token",
            "            p_h0 = h0 + i_n * stride_init_state_token",
        )
    anchor = "    for i_t in range(0, T):"
    replay = """    if HAS_HISTORY:
        prefix = tl.load(num_accepted_tokens + i_n)
        b_h = replay_history(b_h, prev_decay, prev_key, prev_delta, prefix,
                             HV, K, V, i_hv, o_v, o_k)
    # Persist the current round's base once, before any new update logs.
    tl.store(p_h0, b_h, mask=mask_h)

"""
    if selected:
        replay = """    p_base = base_target + i_n * HV * V * K
    p_base = p_base + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_base, b_h, mask=mask_h)

"""
    kernel = kernel.replace(anchor, replay + anchor, 1)
    store_start = kernel.index("        # keep the states for multi-query tokens")
    store_end = kernel.index("        if MIXED_QKV:", store_start)
    kernel = (
        kernel[:store_start]
        + """        tl.store(next_delta + (i_t * HV + i_hv) * V + o_v,
                 b_v, mask=mask_v)
        if i_v == 0:
            tl.store(next_key + (i_t * HV + i_hv) * K + o_k,
                     b_k, mask=mask_k)
            tl.store(next_decay + i_t * HV + i_hv, exp(b_g))

"""
        + kernel[store_end:]
    )
    prefix = original[: original.index("import os")]
    prefix += (
        "from vllm.triton_utils import triton, tl\n"
        "from vllm.model_executor.layers.fla.ops.op import exp\n"
    )
    generated = output / "gdn_lazy_state_generated.py"
    generated.write_text(
        prefix + REPLAY + (SELECTED if selected else "") + "\n" + kernel
    )
    return generated, hashlib.sha256(path.read_bytes()).hexdigest()


def logs(device):
    return (
        torch.empty(8, 12, device=device, dtype=torch.float32),
        torch.empty(8, 12, 128, device=device, dtype=torch.float32),
        torch.empty(8, 12, 128, device=device, dtype=torch.float32),
    )


def inputs(device, row_stride=2560):
    return {
        "mixed_qkv": (
            torch.randn(8, row_stride, device=device, dtype=torch.float16) * 0.1
        )[:, :2560],
        "a": torch.full((8, 12), -0.05, device=device, dtype=torch.float32),
        "b": torch.full((8, 12), 0.5, device=device, dtype=torch.float32),
        "A_log": torch.zeros(12, device=device),
        "dt_bias": torch.zeros(12, device=device),
        "cu_seqlens": torch.tensor([0, 8], device=device, dtype=torch.int32),
        "ssm_state_indices": torch.arange(8, device=device, dtype=torch.int32).view(
            1, 8
        ),
        "num_accepted_tokens": torch.tensor([3], device=device, dtype=torch.int32),
        "ddtree_parent_ids": None,
    }


def launch(
    module, tensors, state, out, banks=None, phase=0, history=True, base_target=None
):
    kwargs = dict(tensors)
    mixed = kwargs["mixed_qkv"]
    kwargs.update(
        q=mixed,
        k=mixed,
        v=mixed,
        h0=state,
        ht=state,
        o=out,
        beta=1.0,
        threshold=20.0,
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
        QKV_STRIDE=mixed.stride(0),
        stride_init_state_token=12 * 128 * 128,
        stride_final_state_token=12 * 128 * 128,
        stride_indices_seq=8,
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
    if banks is None:
        reference[(1, 64, 12)](**kwargs)
    else:
        prev, nxt = banks[1 - phase], banks[phase]
        kwargs.update(
            prev_decay=prev[0],
            prev_key=prev[1],
            prev_delta=prev[2],
            next_decay=nxt[0],
            next_key=nxt[1],
            next_delta=nxt[2],
            HAS_HISTORY=history,
        )
        if base_target is not None:
            kwargs["base_target"] = base_target
        module.lazy_state_kernel[(1, 64, 12)](**kwargs)


def benchmark(module, rank, tensors, states, base, banks):
    # Forty-eight independent layer states exceed L2 in both arms. Both
    # complete round-zero before round-one, matching recurrence reuse.
    chain = []
    for _ in range(48):
        chain.append(
            (
                {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in tensors.items()
                },
                states.clone(),
                base.clone(),
                [tuple(value.clone() for value in bank) for bank in banks],
                torch.empty(8, 12, 128, device=rank, dtype=torch.float16),
                torch.empty(8, 12, 128, device=rank, dtype=torch.float16),
            )
        )
    graphs = []
    for lazy in (False, True):

        def call(lazy=lazy):
            for phase in (0, 1):
                for x, ctrl, initial, logs_, ref_out, new_out in chain:
                    launch(
                        module,
                        x,
                        initial if lazy else ctrl,
                        new_out if lazy else ref_out,
                        logs_ if lazy else None,
                        phase,
                        True,
                    )

        call()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs.append(graph)
    samples = {False: [], True: []}
    timeline = []
    for arm in (False, True, True, False) * 5:
        torch.distributed.barrier()
        graph = graphs[int(arm)]
        for _ in range(5):
            graph.replay()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        wall_start = time.time()
        start.record()
        for _ in range(25):
            graph.replay()
        end.record()
        end.synchronize()
        wall_end = time.time()
        # Each graph contains two state-only verifier rounds (48 kernels each).
        us = start.elapsed_time(end) * 1000 / 50
        samples[arm].append(us)
        timeline.append(
            dict(lazy=arm, wall_start=wall_start, wall_end=wall_end, round_us=us)
        )
    # Both arms have advanced the same number of verifier rounds at this point.
    # Dynamic acceptance metadata and activation updates must remain safe in
    # the captured two-round chain, not only in the initial eager probes.
    for step in range(8):
        for x, *_ in chain:
            x["num_accepted_tokens"].fill_(step + 1)
            x["mixed_qkv"].normal_(std=0.1)
        graphs[0].replay()
        graphs[1].replay()
        torch.accelerator.synchronize()
        for *_, ref_out, new_out in chain:
            assert torch.equal(ref_out, new_out)
    for x, ctrl, initial, logs_, ref_out, new_out in chain:
        assert torch.equal(ref_out, new_out)
        recovered = torch.empty_like(initial)
        for prefix in range(1, 9):
            module.materialize[(64, 12)](
                initial,
                *logs_[1],
                recovered,
                prefix,
                HV=12,
                K=128,
                V=128,
                BK=128,
                BV=2,
                num_warps=1,
            )
            torch.accelerator.synchronize()
            assert torch.equal(recovered[0], ctrl[prefix - 1])
    clocks = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={rank}",
            "--query-gpu=clocks.sm,clocks.mem",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return dict(
        state_only=True,
        layer_count=48,
        rounds_per_graph=2,
        original_round_us=statistics.mean(samples[False]),
        lazy_round_us=statistics.mean(samples[True]),
        samples_us={str(key): value for key, value in samples.items()},
        timing_intervals=timeline,
        graph_after_replays_bitwise=True,
        changed_input_graph_replays=8,
        clocks_after_timing=clocks,
    )


def selected_benchmark(module, rank, tensors, states):
    layers = 48
    device = rank
    bases = torch.empty(layers, 12, 128, 128, device=device)
    log_arrays = (
        torch.empty(layers, 8, 12, device=device),
        torch.empty(layers, 8, 12, 128, device=device),
        torch.empty(layers, 8, 12, 128, device=device),
    )
    controls = [states.clone() for _ in range(layers)]
    cached = [states.clone() for _ in range(layers)]
    pointers = torch.tensor(
        [x.data_ptr() for x in cached], device=device, dtype=torch.uint64
    )
    slots = torch.arange(8, device=device, dtype=torch.int32).repeat(layers, 1)
    previous = [tensors["num_accepted_tokens"].clone() for _ in range(2)]
    chosen = torch.full((1,), 3, device=device, dtype=torch.int32)
    chain = []
    for layer in range(layers):
        x = dict(tensors)
        qkv = tensors["mixed_qkv"]
        # Preserve the packed projection's physical row stride in both arms.
        x["mixed_qkv"] = torch.empty_strided(
            qkv.shape, qkv.stride(), device=device, dtype=qkv.dtype
        )
        x["mixed_qkv"].copy_(qkv)
        x["a"] = tensors["a"].clone()
        x["b"] = tensors["b"].clone()
        outputs = [
            torch.empty(8, 12, 128, device=device, dtype=torch.float16)
            for _ in range(2)
        ]
        chain.append((x, outputs))

    def sweep():
        module.materialize_selected_layers[(8, 12, layers)](
            bases,
            *log_arrays,
            pointers,
            slots,
            chosen,
            HV=12,
            K=128,
            V=128,
            BK=128,
            BV=16,
            num_warps=4,
        )

    graphs = []
    for selected in (False, True):

        def call(selected=selected):
            arm = int(selected)
            for layer, (x, outputs) in enumerate(chain):
                args = dict(x, num_accepted_tokens=previous[arm])
                if selected:
                    bank = tuple(value[layer] for value in log_arrays)
                    launch(
                        module,
                        args,
                        cached[layer],
                        outputs[arm],
                        [bank, bank],
                        0,
                        False,
                        base_target=bases[layer : layer + 1],
                    )
                else:
                    launch(module, args, controls[layer], outputs[arm])
            if selected:
                sweep()
            # Mirrors the GPU metadata update needed before the next forward.
            # Both arms pay the same copy; reconstruction is included above.
            previous[arm].copy_(chosen)

        call()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs.append(graph)
    samples = {False: [], True: []}
    timeline = []
    for arm in (False, True, True, False) * 5:
        torch.distributed.barrier()
        graph = graphs[int(arm)]
        for _ in range(5):
            graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.time()
        start.record()
        for _ in range(50):
            graph.replay()
        end.record()
        end.synchronize()
        wall_end = time.time()
        us = start.elapsed_time(end) * 1000 / 50
        samples[arm].append(us)
        timeline.append(
            dict(selected=arm, wall_start=wall_start, wall_end=wall_end, round_us=us)
        )
    for step in range(8):
        chosen.fill_(step + 1)
        for x, _ in chain:
            x["mixed_qkv"].normal_(std=0.1)
            x["a"].uniform_(-10.0, -0.001)
            x["b"].uniform_(0.001, 0.999)
        graphs[0].replay()
        graphs[1].replay()
        torch.accelerator.synchronize()
        for layer, (_, outputs) in enumerate(chain):
            assert torch.equal(outputs[0], outputs[1]), (rank, step, layer)
            assert torch.equal(controls[layer][step], cached[layer][step])
    clocks = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={rank}",
            "--query-gpu=clocks.sm,clocks.mem",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return dict(
        state_only=True,
        layer_count=layers,
        rounds_per_graph=1,
        restoration_included=True,
        restoration_launches_per_round=1,
        selected_prefix_during_timing=3,
        original_round_us=statistics.mean(samples[False]),
        selected_round_us=statistics.mean(samples[True]),
        samples_us={str(key): value for key, value in samples.items()},
        timing_intervals=timeline,
        graph_after_replays_bitwise=True,
        changed_input_and_selected_prefix_graph_replays=8,
        packed_qkv_row_stride=tensors["mixed_qkv"].stride(0),
        clocks_after_timing=clocks,
    )


def selected_numeric(module, rank, tensors, states, base, banks):
    cached = states.clone()
    pointers = torch.tensor([cached.data_ptr()], device=rank, dtype=torch.uint64)
    ref_out = torch.empty(8, 12, 128, device=rank, dtype=torch.float16)
    new_out = torch.empty_like(ref_out)
    recovered = torch.empty_like(base)
    checks = []
    for step in range(20):
        tensors["mixed_qkv"].normal_(std=0.1)
        tensors["a"].uniform_(-10.0, -0.001)
        tensors["b"].uniform_(0.001, 0.999)
        launch(module, tensors, states, ref_out)
        launch(module, tensors, cached, new_out, banks, 0, False, base_target=base)
        torch.accelerator.synchronize()
        assert torch.equal(ref_out, new_out), (rank, step, "selected output")
        # All hypothetical prefixes are checked, but only the sampled prefix
        # is written into the real cache. Next forward reads that selected slot.
        for prefix in range(1, 9):
            module.materialize[(64, 12)](
                base,
                *banks[0],
                recovered,
                prefix,
                HV=12,
                K=128,
                V=128,
                BK=128,
                BV=2,
                num_warps=1,
            )
            torch.accelerator.synchronize()
            assert torch.equal(recovered[0], states[prefix - 1])
        chosen = step % 8 + 1
        tensors["num_accepted_tokens"].fill_(chosen)
        module.materialize_selected_layers[(8, 12, 1)](
            base,
            *banks[0],
            pointers,
            tensors["ssm_state_indices"],
            tensors["num_accepted_tokens"],
            HV=12,
            K=128,
            V=128,
            BK=128,
            BV=16,
            num_warps=4,
        )
        torch.accelerator.synchronize()
        assert torch.equal(cached[chosen - 1], states[chosen - 1])
        checks.append(
            dict(
                step=step,
                selected=chosen,
                output_bitwise=True,
                hypothetical_snapshots_bitwise=True,
                selected_cache_bitwise=True,
            )
        )
    return checks


def worker(rank, args, generated, source_hash):
    torch.accelerator.set_device_index(rank)
    spec = importlib.util.spec_from_file_location("gdn_lazy_state_generated", generated)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = inputs(rank, 4120 if args.selected_state else 2560)
    base = torch.randn(1, 12, 128, 128, device=rank) * 0.01
    states = base.repeat(8, 1, 1, 1)
    base = base.clone()
    banks = [logs(rank), logs(rank)]
    ref_out = torch.empty(8, 12, 128, device=rank, dtype=torch.float16)
    new_out = torch.empty_like(ref_out)
    materialized = torch.empty_like(base)
    if args.selected_state:
        if args.perf_only:
            prior = json.loads((args.numeric_proof / f"rank{rank}.json").read_text())
            assert prior["numerical_pass"]
            assert prior["original_source_sha256"] == source_hash
            assert (
                prior["generated_sha256"]
                == hashlib.sha256(generated.read_bytes()).hexdigest()
            )
            checks = prior["checks"]
        else:
            checks = selected_numeric(module, rank, x, states, base, banks)
        result = dict(
            research_only=True,
            serving_runtime=False,
            persistence="post_sampling_selected_state",
            rank=rank,
            numerical_pass=True,
            checks=checks,
            source_sha=args.source_sha,
            original_source_sha256=source_hash,
            generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
        )
        if args.perf_only:
            result["benchmark"] = selected_benchmark(module, rank, x, states)
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        torch.distributed.destroy_process_group()
        return
    numeric = []
    if args.perf_only:
        prior = json.loads((args.numeric_proof / f"rank{rank}.json").read_text())
        assert prior["numerical_pass"]
        assert prior["original_source_sha256"] == source_hash
        assert (
            prior["generated_sha256"]
            == hashlib.sha256(generated.read_bytes()).hexdigest()
        )
        x["num_accepted_tokens"].fill_(3)
        launch(module, x, states, ref_out)
        launch(module, x, base, new_out, banks, 1, False)
        torch.accelerator.synchronize()
        assert torch.equal(ref_out, new_out)
    for step in range(0 if args.perf_only else 20):
        accepted = step % 8 + 1
        x["num_accepted_tokens"].fill_(accepted)
        x["mixed_qkv"].normal_(std=0.1)
        launch(module, x, states, ref_out)
        launch(module, x, base, new_out, banks, step % 2, step > 0)
        torch.accelerator.synchronize()
        assert torch.equal(ref_out, new_out), (
            rank,
            step,
            "output",
            float((ref_out - new_out).abs().max()),
        )
        selected = banks[step % 2]
        for prefix in range(1, 9):
            module.materialize[(64, 12)](
                base,
                *selected,
                materialized,
                prefix,
                HV=12,
                K=128,
                V=128,
                BK=128,
                BV=2,
                num_warps=1,
            )
            torch.accelerator.synchronize()
            assert torch.equal(materialized[0], states[prefix - 1]), (
                rank,
                step,
                prefix,
                "state",
                float((materialized[0] - states[prefix - 1]).abs().max()),
            )
        numeric.append(dict(step=step, accepted=accepted, all_snapshots_equal=True))
    result = dict(
        research_only=True,
        serving_runtime=False,
        rank=rank,
        numerical_pass=True,
        numeric=numeric,
        source_sha=args.source_sha,
        original_source_sha256=source_hash,
        generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
        original_snapshot_bytes=8 * 12 * 128 * 128 * 4,
        lazy_base_write_and_log_bytes=12 * 128 * 128 * 4 + (12 + 12 * 128 * 2) * 8 * 4,
    )
    if args.perf_only:
        result["benchmark"] = benchmark(module, rank, x, states, base, banks)
        result["prior_numerical_proof"] = prior["numeric"]
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.distributed.destroy_process_group()


def distributed_worker(rank, args, generated, source_hash):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{args.rendezvous}",
        rank=rank,
        world_size=args.ranks,
    )
    worker(rank, args, generated, source_hash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-state", action="store_true")
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--numeric-proof", type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--ranks", type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.rendezvous = args.output / f"rendezvous-{uuid.uuid4().hex}"
    generated, source_hash = generate(args.root, args.output, args.selected_state)
    torch.multiprocessing.spawn(
        distributed_worker, args=(args, generated, source_hash), nprocs=args.ranks
    )
