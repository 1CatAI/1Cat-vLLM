# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real four-process loader, file reads, swaps and Tensor Core prefill."""

import datetime
import json
import multiprocessing as mp
import socket
import traceback
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
import torch.distributed as dist

from vllm.model_executor.layers.quantization.gguf_shared_runtime import SHARDS

pytestmark = pytest.mark.skipif(
    torch.accelerator.device_count() != 4, reason="Expose exactly four test GPUs"
)


class _Group:
    world_size = 4

    def __init__(self, rank):
        self.rank_in_group = rank
        self.cpu_group = dist.group.WORLD

    def broadcast_object(self, obj, src):
        values = [obj]
        dist.broadcast_object_list(values, src=src, group=self.cpu_group)
        return values[0]


def _run(rank, directory):
    from vllm import _custom_ops  # noqa: F401
    from vllm.config.kernel import Sm70GgufConfig
    from vllm.model_executor.layers.quantization.gguf_shared_storage import (
        SharedOriginalExpertCache,
    )

    config = Sm70GgufConfig(
        expert_cache_host_storage="shared_original",
        expert_cache_host_slack=0,
        expert_arena=True,
        expert_storage="original",
        expert_cache_prefill_tokens=256,
    )
    kinds = ((16, 16, 29), (17, 17, 16))
    h, intermediate, experts = 512, 768, 256
    method = SimpleNamespace(
        is_v41=True,
        hidden_size=h,
        intermediate_size=intermediate // 4,
        num_experts=experts,
    )
    layer = SimpleNamespace(tp_size=4, tp_rank=rank)
    layers = [
        (i, layer, method, dict(zip(SHARDS, formats)))
        for i, formats in enumerate(kinds)
    ]
    sources = {}
    for i, formats in enumerate(kinds):
        for shard, kind in zip(SHARDS, formats):
            block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(kind)]
            n, k = (h, intermediate) if shard == "w2" else (intermediate, h)
            source = np.memmap(
                f"{directory}/{i}-{shard}",
                mode="r",
                dtype=np.uint8,
                shape=(experts, n, k // block * size),
            )
            sources[i, shard] = SimpleNamespace(
                data=source, data_offset=0, tensor_type=kind
            )
    cache = SharedOriginalExpertCache(
        layers, config, torch.device("cuda"), _Group(rank)
    )
    try:
        # Projections and layers arrive in an interleaved loader order.
        for i, shard in (
            (1, "w2"),
            (0, "w3"),
            (1, "w1"),
            (0, "w1"),
            (1, "w3"),
            (0, "w2"),
        ):
            cache.store_original_bank(i, shard, sources[i, shard])
            geometry = cache.specs[i].layouts(rank)[SHARDS.index(shard)]
            expected = np.stack(
                [
                    geometry.pack(sources[i, shard].data[e])
                    for e in range(cache.specs[i].resident)
                ]
            )
            np.testing.assert_array_equal(
                cache.bank(i, shard)[: cache.specs[i].resident].cpu().numpy(), expected
            )
        assert cache.data is not None and cache.runtime is not None
        identity = vars(cache.data.identity)
        generator = torch.Generator(device=rank).manual_seed(41)
        x = torch.randn(263, h, device=rank, generator=generator).half() * 0.1
        # Include cold experts, duplicates, empty groups and both sub-batch tails.
        pattern = torch.tensor([0, 127, 128, 177, 180, 255], device=rank)
        ids = pattern[(torch.arange(263 * 6, device=rank).view(263, 6) * 7) % 6].int()
        probabilities = torch.rand(263, 6, device=rank, generator=generator)
        probabilities /= probabilities.sum(-1, keepdim=True)

        def reference(i):
            unique = ids.unique().cpu().tolist()
            hidden = torch.empty(
                263, 6, intermediate // 4, device=rank, dtype=torch.half
            )
            expected = torch.zeros_like(x, dtype=torch.float32)
            for expert in unique:
                values = []
                for shard, kind in zip(SHARDS, kinds[i]):
                    dense = (
                        torch.from_numpy(
                            gguf.quants.dequantize(
                                sources[i, shard].data[expert],
                                gguf.GGMLQuantizationType(kind),
                            )
                        )
                        .to(rank)
                        .half()
                    )
                    values.append(
                        dense[:, rank * 192 : (rank + 1) * 192]
                        if shard == "w2"
                        else dense[rank * 192 : (rank + 1) * 192]
                    )
                token, route = (ids == expert).nonzero(as_tuple=True)
                gate = (
                    (x[token].float() @ values[0].float().T)
                    .half()
                    .float()
                    .clamp(max=10)
                )
                up = (
                    (x[token].float() @ values[1].float().T)
                    .half()
                    .float()
                    .clamp(-10, 10)
                )
                hidden[token, route] = (
                    torch.nn.functional.silu(gate)
                    * up
                    * probabilities[token, route, None]
                ).half()
                down = (hidden[token, route].float() @ values[2].float().T).half()
                # index_add is only a component reference; production uses
                # the fixed top-k FP32 reduction kernel.
                expected.index_add_(0, token, down.float())
            return expected.half()

        errors = []
        expected_misses = [0, 0]
        for i in (1, 0):
            actual = cache.prefill_moe(i, x, ids, probabilities)
            assert cache.tm is not None
            from vllm.model_executor.layers.quantization.gguf_shared_prefill import (
                SharedOriginalPrefill,
            )

            full = SharedOriginalPrefill(cache, tile_experts=experts)
            expected_untiled = full.forward(i, x, ids, probabilities)
            torch.testing.assert_close(actual, expected_untiled, rtol=0, atol=0)
            assert cache.tm.tile_experts == 32
            assert cache.tm.storage_bytes() * 8 == full.storage_bytes()
            del full, expected_untiled
            expected = reference(i)
            relative = (actual.float() - expected.float()).norm(
                dim=1
            ) / expected.float().norm(dim=1).clamp_min(1e-6)
            assert relative.max().item() < 0.005, (rank, i, relative.max().item())
            errors.append(relative.max().item())
            # Prefill consumes the current cold table after real LRU swaps.
            for first in (160, 4, 220):
                requested = (
                    torch.arange(first, first + 36, device=rank)
                    .remainder(experts)
                    .int()
                )
                expected_misses[i] += int(
                    (cache.runtime.table[i, requested.unique().long()] < 0).sum()
                )
                resolved = cache.resolve(i, requested)
                torch.accelerator.synchronize(rank)
                for shard, geometry in zip(SHARDS, cache.specs[i].layouts(rank)):
                    expected_bytes = np.stack(
                        [
                            geometry.pack(sources[i, shard].data[e])
                            for e in requested.cpu().tolist()
                        ]
                    )
                    np.testing.assert_array_equal(
                        cache.bank(i, shard)[resolved.long()].cpu().numpy(),
                        expected_bytes,
                    )
            actual = cache.prefill_moe(i, x, ids, probabilities)
            torch.testing.assert_close(actual, expected, rtol=0.005, atol=2e-4)
        assert cache.runtime.statistics.cpu().tolist() == [
            [3, misses] for misses in expected_misses
        ]
        assert cache.tm is not None
        assert cache.tm.cold_transcodes.cpu().tolist() == [
            2 * (spec.experts - spec.resident) for spec in cache.specs
        ]
        return dict(
            rank=rank, passed=True, identity=identity, max_relative_errors=errors
        )
    finally:
        torch.accelerator.synchronize(rank)
        dist.barrier()
        cache.close()


def _worker(rank, connection, directory):
    try:
        torch.set_num_threads(1)
        torch.accelerator.set_device_index(rank)
        dist.init_process_group(
            "gloo",
            init_method=f"file://{directory}/rendezvous",
            rank=rank,
            world_size=4,
            timeout=datetime.timedelta(seconds=90),
        )
        connection.send(json.dumps(_run(rank, directory)).encode())
    except BaseException:
        connection.send(
            json.dumps(dict(rank=rank, error=traceback.format_exc())).encode()
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


def test_shared_original_file_loader_and_mixed_tensor_core_prefill(tmp_path):
    from test_v41_device_codecs import source

    for i, formats in enumerate(((16, 16, 29), (17, 17, 16))):
        for shard, kind in zip(SHARDS, formats):
            n, k = (512, 768) if shard == "w2" else (768, 512)
            values = np.stack(
                [source(kind, n, k, 0.00137, 41 + e + i * 256) for e in range(256)]
            )
            values.tofile(tmp_path / f"{i}-{shard}")
            del values
    context = mp.get_context("spawn")
    processes, connections = [], []
    try:
        for rank in range(4):
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            parent.settimeout(180)
            process = context.Process(target=_worker, args=(rank, child, str(tmp_path)))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        reports = [json.loads(c.recv(65536)) for c in connections]
        assert all(r.get("passed") for r in reports), reports
        assert (
            len({(r["identity"]["device"], r["identity"]["inode"]) for r in reports})
            == 1
        )
        for process in processes:
            process.join(15)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
        for connection in connections:
            connection.close()
