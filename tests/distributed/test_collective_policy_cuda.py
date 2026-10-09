# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise independent configured native owners without model weights."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _configured_owners(rank, rendezvous, reverse):
    from vllm.config import set_current_vllm_config
    from vllm.config.collective import CollectiveNativeConfig
    from vllm.config.execution_policy import CommunicationPolicy
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=120),
    )
    owners = {}
    try:
        for mode in ("invalid", "oneshot") if reverse else ("oneshot", "invalid"):
            policy = CommunicationPolicy(
                native=CollectiveNativeConfig(custom_allreduce_algo=mode)
            )
            policy.resolve()
            cfg = SimpleNamespace(parallel_config=SimpleNamespace(communication=policy))
            with set_current_vllm_config(cfg):
                owners[mode] = CustomAllreduce(dist.group.WORLD, rank, max_size=65536)
            assert not owners[mode].disabled
        x = torch.full((32,), rank + 1, device="cuda", dtype=torch.float16)
        # The parser error remains deferred until execution, and belongs only
        # to its native owner. Another engine and a later environment edit
        # cannot change this owner's selection.
        os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = "oneshot"
        with pytest.raises(RuntimeError, match="Invalid VLLM_CUSTOM_ALLREDUCE_ALGO"):
            owners["invalid"].custom_all_reduce(x)
        os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = "invalid"
        good = owners["oneshot"]
        assert torch.equal(good.custom_all_reduce(x), torch.full_like(x, 3))
        graph = torch.cuda.CUDAGraph()
        with good.capture(), torch.cuda.graph(graph):
            output = good.custom_all_reduce(x)
        for value in (2, 5, 1):
            x.fill_(value + rank)
            dist.barrier()
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(output, torch.full_like(x, 2 * value + 1))
    finally:
        for owner in owners.values():
            owner.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("reverse", [False, True])
def test_configured_native_owners_and_graph_replay(tmp_path, reverse):
    if torch.accelerator.device_count() < 2:
        pytest.skip("requires two peer-accessible CUDA GPUs")
    rendezvous = (tmp_path / "native-policy").as_uri()
    mp.spawn(_configured_owners, args=(rendezvous, reverse), nprocs=2, join=True)
