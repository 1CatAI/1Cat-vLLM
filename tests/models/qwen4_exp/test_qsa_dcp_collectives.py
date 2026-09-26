# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real TP4/DCP2 collectives and graph replay against independent attention."""

import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skip_global_cleanup


def _distributed_qsa_worker(rank: int, rendezvous: str) -> None:
    import faulthandler

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import get_dcp_group
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionImpl
    from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
    from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

    faulthandler.dump_traceback_later(90, exit=True)
    torch.accelerator.set_device_index(rank)
    set_custom_all_reduce(False)
    init_distributed_environment(
        world_size=4,
        rank=rank,
        local_rank=rank,
        distributed_init_method=rendezvous,
        backend="nccl",
    )
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=4,
            decode_context_parallel_size=2,
        )
    )
    with set_current_vllm_config(config):
        initialize_model_parallel(
            tensor_model_parallel_size=4,
            decode_context_model_parallel_size=2,
        )
    group = get_dcp_group()
    try:
        with torch.inference_mode():
            for kv_dtype in ("float16", "fp8_e4m3"):
                for rows in (1, 5, 33):
                    torch.manual_seed(98 + rank // 2)
                    heads, dim, page_size, pages, interleave = 6, 256, 16, 3, 4
                    length = 2 * page_size * pages
                    q_all = (
                        torch.randn(
                            rows, heads * 2, dim, device="cuda", dtype=torch.float16
                        )
                        * 0.3
                    )
                    q = q_all[
                        :,
                        group.rank_in_group * heads : (group.rank_in_group + 1) * heads,
                    ].contiguous()
                    torch.testing.assert_close(group.all_gather(q, dim=1), q_all)
                    probe = torch.full_like(q_all, float(rank), dtype=torch.float32)
                    torch.testing.assert_close(
                        group.reduce_scatter(probe, dim=1),
                        torch.full_like(
                            q, float(sum(group.ranks)), dtype=torch.float32
                        ),
                    )
                    k = torch.randn(length, 1, dim, device="cuda", dtype=torch.float16)
                    v = torch.randn_like(k)
                    gate = torch.randn_like(q)
                    k_scale, v_scale = 1.0, 1.0
                    if kv_dtype == "fp8_e4m3":
                        k_scale, v_scale = 0.01, 0.02
                        # Match the runtime's saturating KV write. Torch's
                        # raw FP8 cast emits NaN for out-of-range values
                        # (seed 99 includes K=4.8359375), unlike that writer.
                        k = (
                            (k.float() / k_scale)
                            .clamp(-448, 448)
                            .to(torch.float8_e4m3fn)
                            .view(torch.uint8)
                        )
                        v = (
                            (v.float() / v_scale)
                            .clamp(-448, 448)
                            .to(torch.float8_e4m3fn)
                            .view(torch.uint8)
                        )
                        dk = k.view(torch.float8_e4m3fn).float() * k_scale
                        dv = v.view(torch.float8_e4m3fn).float() * v_scale
                    else:
                        dk, dv = k.float(), v.float()
                    table = torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32)
                    owners = (torch.arange(length, device="cuda") // interleave) % 2
                    cache = torch.empty(
                        pages, 2, page_size, 1, dim, dtype=k.dtype, device="cuda"
                    )
                    cache[table[0].long(), 0] = k[owners == group.rank_in_group].view(
                        pages, page_size, 1, dim
                    )
                    cache[table[0].long(), 1] = v[owners == group.rank_in_group].view(
                        pages, page_size, 1, dim
                    )
                    selected = torch.arange(
                        67, device="cuda", dtype=torch.int32
                    ).repeat(rows, 1)
                    if rows > 1:
                        selected[-1].fill_(-1)
                    impl = object.__new__(Qwen4ExpQSAFlashAttentionImpl)
                    impl.dcp_world_size, impl.dcp_rank = 2, group.rank_in_group
                    impl.cp_kv_cache_interleave_size = interleave
                    impl.kv_cache_dtype = kv_dtype
                    impl.alibi_slopes, impl.sinks = None, None
                    impl.sliding_window = (-1, -1)
                    layer = SimpleNamespace(
                        topk_indices_buffer=selected,
                        _k_scale_float=k_scale,
                        _v_scale_float=v_scale,
                        qsa_dcp_sharded=True,
                        cp_kv_cache_interleave_size=interleave,
                        indexer=SimpleNamespace(token_topk=64, compress_ratio=4),
                        dcp_local_indices_buffer=torch.empty_like(selected),
                        dcp_partial_output_buffer=torch.empty_like(
                            q_all, dtype=torch.float32
                        ),
                        dcp_partial_lse_buffer=torch.empty(
                            q_all.shape[:2], device="cuda", dtype=torch.float32
                        ),
                    )
                    meta = SimpleNamespace(num_actual_tokens=rows, block_table=table)
                    req = torch.zeros(rows, dtype=torch.int32, device="cuda")
                    out = torch.empty_like(q)

                    run = partial(
                        impl.forward_qsa,
                        layer,
                        q,
                        q[:, :1],
                        q[:, :1],
                        cache,
                        meta,
                        out,
                        token_to_req=req,
                        output_gate=gate,
                    )
                    reference = partial(_gated_reference, q, dk, dv, selected, gate)

                    for combine in (cp_lse_ag_out_rs, dcp_a2a_lse_reduce):
                        impl.dcp_combine = combine
                        run()
                        torch.testing.assert_close(
                            out, reference(), atol=1e-3, rtol=5e-3
                        )
                        with (
                            graph_capture(q.device) as context,
                            group.graph_capture(context),
                        ):
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=context.stream):
                                run()
                        for token in (90, 2):
                            selected[0, 0] = token
                            graph.replay()
                            torch.testing.assert_close(
                                out, reference(), atol=1e-3, rtol=5e-3
                            )
                        # NCCL graph user objects must be released before the
                        # communicators are destroyed at worker shutdown.
                        graph.reset()
                        if rank in (0, 2):
                            print(
                                f"rank={rank} TP4/DCP2 {kv_dtype} rows={rows} "
                                f"{combine.__name__}: eager+graph OK",
                                flush=True,
                            )
    finally:
        torch.accelerator.synchronize()
        destroy_model_parallel()
        destroy_distributed_environment()
        faulthandler.cancel_dump_traceback_later()


def _gated_reference(q, k, v, selected, gate):
    keys = k[selected.clamp_min(0).long(), 0]
    values = v[selected.clamp_min(0).long(), 0]
    scores = torch.einsum("rhd,rkd->rhk", q.float(), keys) / math.sqrt(q.shape[-1])
    scores.masked_fill_(selected[:, None, :] < 0, -torch.inf)
    attn = torch.einsum("rhk,rkd->rhd", scores.softmax(-1).nan_to_num(), values)
    return (attn.half().float() * gate.float().sigmoid()).half()


@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="4 GPUs required")
def test_qsa_tp4_dcp2_collectives_and_graph(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_qsa_worker,
        args=(f"file://{tmp_path / 'dcp_init'}",),
        nprocs=4,
        join=True,
    )
