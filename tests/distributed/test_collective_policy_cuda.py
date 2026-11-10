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


def _sum2_generic_push_owners(rank, rendezvous):
    from vllm.config import set_current_vllm_config
    from vllm.config.collective import CollectiveNativeConfig
    from vllm.config.execution_policy import CommunicationPolicy
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=120),
    )
    owners = []
    try:
        # Disable model-specific selectors: the enabled owner must use the new
        # generic admission, while the others retain the registered fallback.
        for enabled, small in ((False, True), (True, True), (True, False)):
            policy = CommunicationPolicy(
                native=CollectiveNativeConfig(
                    sm70_tp4_sum2_push_any=enabled,
                    sm70_tp4_push_allreduce_small_messages=small,
                    sm70_tp4_push_allreduce_qwen38_batch=False,
                    sm70_tp4_push_allreduce_mtp5=False,
                    sm70_tp4_push_allreduce_sum2_m1=False,
                )
            )
            policy.resolve()
            cfg = SimpleNamespace(parallel_config=SimpleNamespace(communication=policy))
            with set_current_vllm_config(cfg):
                owner = CustomAllreduce(dist.group.WORLD, rank, max_size=262144)
            assert not owner.disabled and owner.fully_connected
            owners.append(owner)
        os.environ["VLLM_V41_SUM2_PUSH_ANY"] = "0"
        for dtype, sizes in (
            (torch.float16, (8, 40, 5 * 5120, 40960, 40968)),
            (torch.float32, (32,)),
        ):
            for size in sizes:
                a = torch.zeros(size, dtype=dtype, device="cuda")
                b = torch.zeros_like(a)
                graphs, outputs = [], []
                for owner in owners:
                    graph = torch.cuda.CUDAGraph()
                    with owner.capture(), torch.cuda.graph(graph):
                        output = owner.custom_all_reduce_sum2(a, b)
                    assert output is not None
                    graphs.append(graph)
                    outputs.append(output)
                for step in range(6):
                    # Different values per rank/replay expose stale epochs and
                    # aliased graph buffers. Eight FP16 summands may round in
                    # different orders; bounds allow a few FP16 ULPs near zero.
                    generator = torch.Generator(device="cuda").manual_seed(
                        1234 + rank * 19 + step
                    )
                    a.copy_(torch.rand(size, generator=generator, device="cuda") - 0.5)
                    b.copy_(torch.rand(size, generator=generator, device="cuda") - 0.5)
                    expected = a.float() + b.float()
                    gathered = [
                        torch.empty_like(expected, device="cpu") for _ in range(4)
                    ]
                    dist.all_gather(gathered, expected.cpu())
                    reference = torch.stack(gathered).sum(0).to(a.device)
                    for graph, output in zip(graphs, outputs):
                        dist.barrier()
                        graph.replay()
                        torch.accelerator.synchronize()
                        assert torch.isfinite(output).all()
                        torch.testing.assert_close(
                            output.float(), reference, rtol=0.003, atol=0.015625
                        )
                # The public sum2 API declines eager execution. Exercise the
                # caller's ordinary a+b all-reduce fallback after capture too.
                for owner in owners:
                    assert owner.custom_all_reduce_sum2(a, b) is None
                    torch.testing.assert_close(
                        owner.custom_all_reduce(a + b).float(),
                        reference,
                        rtol=0.003,
                        atol=0.015625,
                    )
    finally:
        for owner in owners:
            owner.close()
        dist.destroy_process_group()


def test_sum2_generic_push_boundaries_and_graph_replay(tmp_path):
    if torch.accelerator.device_count() < 4:
        pytest.skip("requires four fully connected SM70 CUDA GPUs")
    if any(torch.cuda.get_device_capability(i) != (7, 0) for i in range(4)):
        pytest.skip("requires SM70")
    mp.spawn(
        _sum2_generic_push_owners,
        args=((tmp_path / "sum2-policy").as_uri(),),
        nprocs=4,
        join=True,
    )
