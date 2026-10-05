# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen FP32 GDN rollback replay without changing production dispatch.

The candidate is generated from the exact installed recurrent kernel, retaining
its K reduction, V tile and arithmetic. It replaces token snapshots with one
round-start state and replays the accepted prefix before the next verification.
This is a kernel screen, not a model quality or end-to-end performance claim.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as gdn


def candidate_module(directory: Path):
    source = Path(gdn.__file__).read_text()
    begin = source.index("@triton.heuristics(")
    end = source.index("\ndef fused_sigmoid_gating_delta_rule_update(", begin)
    kernel = source[begin:end]
    kernel = kernel.replace(
        "def fused_sigmoid_gating_delta_rule_update_kernel(",
        "def rollback_replay_kernel(",
        1,
    ).replace(
        "    A_log,\n",
        "    replay_qkv,\n    replay_g,\n    replay_beta,\n    replay_count,\n"
        "    replay_k,\n    replay_v,\n    save_k,\n    save_v,\n    save_g,\n"
        "    CACHE_FACTORS: tl.constexpr,\n"
        "    REPLAY_PREFIX: tl.constexpr,\n    SAVE_HISTORY: tl.constexpr,\n"
        "    A_log,\n",
        1,
    )
    # Research route is deliberately restricted to precomputed scalar gating,
    # mixed QKV, FP32 state, linear chains and contiguous per-request prefixes.
    replay = """
    if not SAVE_HISTORY:
        if REPLAY_PREFIX:
            count = tl.load(replay_count + i_n)
            for step in range(count):
                if CACHE_FACTORS:
                    old_k = tl.load(replay_k + ((i_n * 8 + step) * HV + i_hv) * K
                        + o_k, mask=mask_k, other=0)
                    old_v = tl.load(replay_v + ((i_n * 8 + step) * HV + i_hv) * V
                        + o_v, mask=mask_v, other=0)
                else:
                    old_k = tl.load(replay_qkv + (i_n * 8 + step) * QKV_STRIDE
                        + K_OFFSET + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
                    old_v = tl.load(replay_qkv + (i_n * 8 + step) * QKV_STRIDE
                        + V_OFFSET + i_hv * V + o_v,
                        mask=mask_v, other=0).to(tl.float32)
                    old_k = old_k / tl.sqrt(tl.sum(old_k * old_k) + 1e-6)
                old_g = tl.load(replay_g + (i_n * 8 + step) * HV + i_hv)
                b_h *= exp(old_g)
                if not CACHE_FACTORS:
                    old_beta = tl.load(replay_beta + (i_n * 8 + step) * HV + i_hv)
                    old_v -= tl.sum(b_h * old_k[None, :], 1)
                    old_v *= old_beta
                b_h += old_v[:, None] * old_k[None, :]
        # Persist the start of this round, not its unaccepted final state.
        tl.store(p_h0, b_h, mask=mask_h)

"""
    marker = "    for i_t in range(0, T):\n"
    assert kernel.count(marker) == 1
    kernel = kernel.replace(marker, replay + marker, 1)
    save = """
        if CACHE_FACTORS:
            # Only persist the rank-1 update already computed by verification.
            # Separate previous/next buffers prevent inter-CTA overwrite races.
            tl.store(save_v + ((i_n * 8 + i_t) * HV + i_hv) * V + o_v,
                b_v, mask=mask_v)
            if i_v == 0:
                tl.store(save_k + ((i_n * 8 + i_t) * HV + i_hv) * K + o_k,
                    b_k, mask=mask_k)
                tl.store(save_g + (i_n * 8 + i_t) * HV + i_hv, b_g)
"""
    marker = "        b_v *= b_beta\n"
    assert kernel.count(marker) == 1
    kernel = kernel.replace(marker, marker + save, 1)
    begin_store = kernel.index("        # keep the states for multi-query tokens")
    end_store = kernel.index("        if MIXED_QKV:\n", begin_store)
    store = kernel[begin_store:end_store]
    kernel = (
        kernel[:begin_store]
        + "        if SAVE_HISTORY:\n"
        + "\n".join("    " + line if line else "" for line in store.splitlines())
        + "\n\n"
        + kernel[end_store:]
    )
    path = directory / "rollback_replay_generated.py"
    path.write_text(
        "from vllm.triton_utils import triton, tl\n"
        "from vllm.model_executor.layers.fla.ops.op import exp\n\n" + kernel
    )
    spec = importlib.util.spec_from_file_location("rollback_replay_generated", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def compile_only(kernel, directory, cache_factors):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    jit = kernel.fn
    signature = dict.fromkeys(("replay_qkv", "q", "k", "v", "mixed_qkv", "o"), "*fp16")
    signature.update(
        dict.fromkeys(
            (
                "replay_g",
                "replay_beta",
                "replay_k",
                "replay_v",
                "save_k",
                "save_v",
                "save_g",
                "A_log",
                "a",
                "b",
                "dt_bias",
                "h0",
                "ht",
            ),
            "*fp32",
        )
    )
    signature.update(
        dict.fromkeys(
            ("replay_count", "cu_seqlens", "ssm_state_indices", "num_accepted_tokens"),
            "*i32",
        )
    )
    signature.update(dict.fromkeys(("beta", "threshold", "scale"), "fp32"))
    signature.update(N="i64", T="i64")
    constants = dict(
        REPLAY_PREFIX=True,
        CACHE_FACTORS=cache_factors,
        SAVE_HISTORY=False,
        ddtree_parent_ids=None,
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
        stride_init_state_token=196608,
        stride_final_state_token=196608,
        stride_indices_seq=8,
        stride_indices_tok=1,
        stride_parent_ids_seq=1,
        stride_parent_ids_tok=1,
        USE_INITIAL_STATE=True,
        INPLACE_FINAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=True,
        IS_VARLEN=True,
        IS_CONTINUOUS_BATCHING=True,
        IS_SPEC_DECODING=True,
        IS_DDTREE=False,
        IS_KDA=False,
        MIXED_QKV=True,
        PRECOMPUTED_GATING=True,
        QUANTIZE_BETA=False,
        QUANTIZE_STATE_EACH_STEP=False,
        MATCH_RECURRENT_NUMERICS=True,
    )
    result = triton.compile(
        ASTSource(jit, signature=signature, constexprs=constants),
        target=GPUTarget("cuda", 70, 32),
        options={"num_warps": 1, "num_stages": 3},
    )
    (directory / "rollback-replay.cubin").write_bytes(result.asm["cubin"])
    (directory / "rollback-replay.ptx").write_text(result.asm["ptx"])
    record = dict(
        shared_bytes=result.metadata.shared,
        reference_source_sha256=hashlib.sha256(
            Path(gdn.__file__).read_bytes()
        ).hexdigest(),
        gpu_execution=False,
    )
    (directory / "compile.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def launch(
    kernel,
    qkv,
    g,
    beta,
    state,
    indices,
    out,
    previous=None,
    metadata=None,
    factors=None,
):
    rows = qkv.shape[0]
    requests = rows // 8
    hv, v, k = state.shape[-3:]
    h = (qkv.shape[1] - hv * v) // (2 * k)
    if metadata is None:
        cu = torch.arange(requests + 1, device="cuda", dtype=torch.int32) * 8
        accepted = torch.ones(requests, device="cuda", dtype=torch.int32)
    else:
        cu, accepted = metadata
    args = dict(
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
        scale=k**-0.5,
        N=requests,
        T=rows,
        B=1,
        H=h,
        HV=hv,
        K=k,
        V=v,
        BK=k,
        BV=2,
        Q_OFFSET=0,
        K_OFFSET=h * k,
        V_OFFSET=2 * h * k,
        QKV_STRIDE=qkv.stride(0),
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
    if kernel is not gdn.fused_sigmoid_gating_delta_rule_update_kernel:
        old_qkv, old_g, old_beta, count = previous or (qkv, g, beta, accepted)
        old_k, old_v, old_g, save_k, save_v, save_g = (
            factors if factors is not None else (g, g, old_g, g, g, g)
        )
        args.update(
            replay_qkv=old_qkv,
            replay_g=old_g,
            replay_beta=old_beta,
            replay_count=count,
            REPLAY_PREFIX=previous is not None,
            SAVE_HISTORY=False,
            CACHE_FACTORS=factors is not None,
            replay_k=old_k,
            replay_v=old_v,
            save_k=save_k,
            save_v=save_v,
            save_g=save_g,
        )
    kernel[(1, v // 2, requests * hv)](**args)


def graph_time(operation, reset, iters):
    reset()
    operation()
    torch.cuda.synchronize()
    eviction = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(graph):
        reset()
        eviction.fill_(1)
        start.record()
        operation()
        end.record()
    for _ in range(10):
        graph.replay()
    samples = []
    for _ in range(iters):
        graph.replay()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.fmean(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--raw-replay", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    candidate = candidate_module(args.out).rollback_replay_kernel
    if args.compile_only:
        compile_only(candidate, args.out, not args.raw_replay)
        return
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    assert torch.cuda.get_device_capability() == (7, 0)
    if args.snapshot:
        data = torch.load(args.snapshot, map_location="cuda", weights_only=True)
        qkv, g, beta, start_state = (
            data[k] for k in ("mixed_qkv", "g", "beta", "state")
        )
    else:
        qkv = torch.randn(8, 2560, device="cuda", dtype=torch.float16) * 0.125
        g = -torch.rand(8, 12, device="cuda") * 0.25
        beta = torch.rand(8, 12, device="cuda")
        start_state = torch.randn(1, 12, 128, 128, device="cuda") * 0.01
    assert qkv.shape == (8, 2560) and start_state.shape == (1, 12, 128, 128)
    assert g.dtype == beta.dtype == start_state.dtype == torch.float32
    reference = start_state.expand(8, -1, -1, -1).clone()
    compact = start_state.clone()
    indices = torch.arange(8, device="cuda", dtype=torch.int32)[None]
    compact_indices = torch.zeros_like(indices)
    out = torch.empty(8, 1, 12, 128, device="cuda", dtype=torch.float16)
    candidate_out = torch.empty_like(out)
    caches = [
        (g.new_empty(8, 12, 128), g.new_empty(8, 12, 128), g.new_empty(8, 12))
        for _ in range(2)
    ]
    initial_factors = None if args.raw_replay else (*caches[1], *caches[0])
    factors = None if args.raw_replay else (*caches[0], *caches[1])
    launch(
        gdn.fused_sigmoid_gating_delta_rule_update_kernel,
        qkv,
        g,
        beta,
        reference,
        indices,
        out,
    )
    launch(
        candidate,
        qkv,
        g,
        beta,
        compact,
        compact_indices,
        candidate_out,
        factors=initial_factors,
    )
    torch.testing.assert_close(candidate_out, out, rtol=0, atol=0)
    checks = []
    for count in range(1, 9):
        compact.copy_(start_state)
        previous = (
            qkv,
            g,
            beta,
            torch.tensor([count], device="cuda", dtype=torch.int32),
        )
        launch(
            candidate,
            qkv,
            g,
            beta,
            compact,
            compact_indices,
            candidate_out,
            previous=previous,
            factors=factors,
        )
        # Snapshot column count-1 contains count updates (including carry-in).
        torch.testing.assert_close(compact[0], reference[count - 1], rtol=0, atol=0)
        accepted_state = reference[count - 1 : count].expand(8, -1, -1, -1).clone()
        launch(
            gdn.fused_sigmoid_gating_delta_rule_update_kernel,
            qkv,
            g,
            beta,
            accepted_state,
            indices,
            out,
        )
        torch.testing.assert_close(candidate_out, out, rtol=0, atol=0)
        checks.append(count)
    timings = {}
    # Reset state outside the timed interval: screen arithmetic and traffic,
    # not a copy kernel. The benchmark operation only reads the start state.
    for count in range(0, 9):
        previous = (
            None
            if count == 0
            else (qkv, g, beta, torch.tensor([count], device="cuda", dtype=torch.int32))
        )
        metadata = (
            torch.tensor([0, 8], device="cuda", dtype=torch.int32),
            torch.tensor([max(count, 1)], device="cuda", dtype=torch.int32),
        )
        # The compact table aliases column zero, so the accepted-count tensor
        # remains compatible with the unmodified initial-state address logic.
        baseline_seed = start_state.expand(8, -1, -1, -1) if count == 0 else reference
        baseline_state = baseline_seed.clone()
        baseline_us = graph_time(
            lambda baseline_state=baseline_state, metadata=metadata: launch(
                gdn.fused_sigmoid_gating_delta_rule_update_kernel,
                qkv,
                g,
                beta,
                baseline_state,
                indices,
                out,
                metadata=metadata,
            ),
            lambda baseline_state=baseline_state, baseline_seed=baseline_seed: (
                baseline_state.copy_(baseline_seed)
            ),
            args.iters,
        )
        candidate_us = graph_time(
            lambda previous=previous, metadata=metadata: launch(
                candidate,
                qkv,
                g,
                beta,
                compact,
                compact_indices,
                candidate_out,
                previous=previous,
                metadata=metadata,
                factors=factors,
            ),
            lambda: compact.copy_(start_state),
            args.iters,
        )
        timings[str(count)] = dict(
            baseline_us=baseline_us,
            candidate_us=candidate_us,
            estimated_48_layer_saving_ms=(baseline_us - candidate_us) * 48 / 1000,
        )
    record = dict(
        bitwise_accepted_prefixes=checks,
        cold_l2_graph=timings,
        real_inputs=args.snapshot is not None,
        replay_format="raw_inputs" if args.raw_replay else "computed_rank1_factors",
        double_buffered_factor_bytes=2
        * sum(t.numel() * t.element_size() for t in caches[0]),
        saved_state_bytes_per_layer=7 * start_state.numel() * 4,
        production_dispatch_changed=False,
        note="Kernel screen only; layer and model validation pending.",
    )
    (args.out / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
