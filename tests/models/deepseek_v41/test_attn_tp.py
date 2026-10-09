# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-ATTN: tensor parallelism (2 ranks on the lane's two GPUs) and the profile/dummy-run path.

TP2: heads, o-groups and attn_sink are sharded, the indexer and compressor are replicated, wo_b partial sums are
all-reduced in FP32. Gates: both ranks return bitwise identical outputs and identical top-512 indices (the
replicated top-k invariant of PORT_DESIGN §4.1), and the outputs meet the same reference gates as TP1."""

from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = pytest.mark.sm70


def _tp_worker(rank: int, world: int, rendezvous: str, S: int) -> None:
    torch.cuda.set_device(rank)
    import torch.distributed as dist

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from .test_attn_harness import ref_config, synthetic_attn_weights, topology
    from .test_attn_layers import _assert_gates, run_and_compare

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method=rendezvous,
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(world, 1)
        cfg = ref_config()
        group = (2, 3, 20, 21, 24)
        dev = torch.device("cuda", rank)
        weights = {i: synthetic_attn_weights(cfg, topology(cfg, i), dev, seed=i) for i in group}
        captured: dict = {}
        metrics = run_and_compare(cfg, group, weights, S=S, seed=3, tp_rank=rank, tp_size=world,
                                  capture=captured)
        _assert_gates(metrics)
        # cross-rank: outputs and replicated top-k must be bitwise identical
        for key, t in sorted(captured.items()):
            t = t.contiguous()
            other = [torch.empty_like(t) for _ in range(world)]
            dist.all_gather(other, t)
            for r in range(1, world):
                assert torch.equal(other[0].view(torch.uint8), other[r].view(torch.uint8)), f"{key}: rank {r} differs"
        cleanup_dist_env_and_memory()


def test_tp2_matches_reference_and_ranks_agree(tmp_path) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("needs the lane's two GPUs (CUDA_VISIBLE_DEVICES=<GPU-d0eed91d>,<GPU-a629284e>)")
    rendezvous = (tmp_path / "tp2-rendezvous").as_uri()
    mp.spawn(_tp_worker, args=(2, rendezvous, 1200), nprocs=2, join=True)


def test_profile_run_without_metadata(dist_env) -> None:    # noqa: F811
    """Dummy/profile forwards carry no attention metadata: every layer must run its projections, reserve the
    attention workspace and return finite zeros-based outputs without touching any cache."""
    from vllm.forward_context import ForwardContext, override_forward_context

    from .test_attn_harness import ref_config, synthetic_attn_weights, topology
    from .test_attn_layers import DEV, Stage

    cfg = ref_config()
    group = (0, 2, 3, 20, 24)
    weights = {i: synthetic_attn_weights(cfg, topology(cfg, i), DEV, seed=i) for i in group}
    st = Stage(cfg, group, weights)
    x = torch.randn(64, 5120, device=DEV).half()
    pos = torch.arange(64, device=DEV)
    before = {n: c.tensor.clone() for n, c in st.sim.caches.items()}
    with override_forward_context(ForwardContext(no_compile_layers=st.ctx, attn_metadata=None, slot_mapping={},
                                                 is_dummy_run=True)):
        for i in group:
            y = st.attn[i](pos, x)
            assert y.shape == (64, 5120) and torch.isfinite(y).all()
    for n, c in st.sim.caches.items():
        assert torch.equal(torch.isnan(c.tensor), torch.isnan(before[n])), f"profile run wrote cache {n}"


from .test_attn_layers import dist_env  # noqa: E402,F401  (fixture)
