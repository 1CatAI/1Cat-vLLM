# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 required",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _gemma_rms_ref(
    reduced_f16: torch.Tensor,
    residual_f32: torch.Tensor,
    weight_f16: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    res_out = reduced_f16.float() + residual_f32
    var = res_out.pow(2).mean(dim=-1, keepdim=True)
    norm = (res_out * torch.rsqrt(var + eps) * (1.0 + weight_f16.float())).to(
        torch.float16
    )
    return norm, res_out


def _worker_rs_norm_ag(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    try:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        cpu_group = dist.new_group(backend="gloo")
        max_size = 80 * 1024 * 1024
        ca = CustomAllreduce(
            group=cpu_group,
            device=device,
            max_size=max_size,
            long_prefill_fusion_enabled=True,
        )
        assert not ca.disabled, f"CustomAllreduce disabled on TP={world_size}"
        # Verify meta scratch is bounded by dispatch_max_size (8 MiB), not 80 MiB.
        assert ca.buffer_ptrs[rank] is not None
        assert ca.dispatch_max_size == 8 * 1024 * 1024
        assert ca.meta_scratch_size == 8 * 1024 * 1024

        eps = 1e-6
        gen_shared = torch.Generator(device=device).manual_seed(12345)
        weight = (
            torch.randn(
                (5120,), generator=gen_shared, device=device, dtype=torch.float32
            )
            * 0.1
        ).to(torch.float16)

        for m in (128, 256, 512, 2048, 4096, 8192):
            gen_rank = torch.Generator(device=device).manual_seed(7000 + m * 10 + rank)
            local_in = (
                torch.randn(
                    (m, 5120), generator=gen_rank, device=device, dtype=torch.float32
                )
                * 0.25
            ).to(torch.float16)
            gen_res = torch.Generator(device=device).manual_seed(9000 + m)
            full_res = torch.randn(
                (m, 5120), generator=gen_res, device=device, dtype=torch.float32
            )

            # Reference: NCCL sum of float32 -> float16 AR semantics + GemmaRMSNorm
            ref_sum = local_in.float().clone()
            dist.all_reduce(ref_sum)
            ref_norm, ref_res = _gemma_rms_ref(
                ref_sum.to(torch.float16), full_res, weight, eps
            )

            assert ca.can_sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather(
                local_in, full_res, weight
            )
            out_norm, out_res = ca.sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather(
                local_in, full_res.clone(), weight, eps
            )
            torch.cuda.synchronize(device)

            shard = m // world_size
            lo, hi = rank * shard, (rank + 1) * shard
            torch.testing.assert_close(out_norm, ref_norm, atol=3e-3, rtol=3e-3)
            torch.testing.assert_close(
                out_res[lo:hi], ref_res[lo:hi], atol=3e-3, rtol=3e-3
            )

        # Multi-boundary closed residual chain (3 consecutive RS+Norm+AG boundaries):
        m_chain = 2048
        gen_init = torch.Generator(device=device).manual_seed(4242)
        res_ref = torch.randn(
            (m_chain, 5120), generator=gen_init, device=device, dtype=torch.float32
        )
        res_sharded = res_ref.clone()
        for step in range(3):
            gen_step = torch.Generator(device=device).manual_seed(
                8800 + step * 10 + rank
            )
            x_step = (
                torch.randn(
                    (m_chain, 5120),
                    generator=gen_step,
                    device=device,
                    dtype=torch.float32,
                )
                * 0.2
            ).to(torch.float16)
            ref_sum = x_step.float().clone()
            dist.all_reduce(ref_sum)
            norm_ref, res_ref = _gemma_rms_ref(
                ref_sum.to(torch.float16), res_ref, weight, eps
            )
            norm_act, res_sharded = (
                ca.sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather(
                    x_step, res_sharded, weight, eps
                )
            )
            torch.testing.assert_close(norm_act, norm_ref, atol=4e-3, rtol=4e-3)
        ca.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", (2, 4))
def test_sm70_rs_gemma_norm_ag_tp2_and_tp4(tp_size: int) -> None:
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Requires >= {tp_size} GPUs")
    port = _free_port()
    mp.spawn(_worker_rs_norm_ag, args=(tp_size, port), nprocs=tp_size, join=True)
