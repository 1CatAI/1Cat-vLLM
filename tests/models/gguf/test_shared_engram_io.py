# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four GPU consumers, one file reader, deduplication and graph replay."""

import datetime
import json
import multiprocessing as mp
import socket
import time
import traceback
from contextlib import suppress

import gguf
import numpy as np
import pytest
import torch
import torch.distributed as dist

from vllm.model_executor.layers.deepseek_v41_engram_io import SharedEngramIO

pytestmark = pytest.mark.skipif(
    torch.accelerator.device_count() != 4, reason="Expose four GPUs"
)


class _Group:
    world_size = 4

    def __init__(self, rank):
        self.rank_in_group, self.cpu_group = rank, dist.group.WORLD

    def broadcast_object(self, obj, src):
        values = [obj]
        dist.broadcast_object_list(values, src=src)
        return values[0]


def _run(rank, path):
    from vllm import _custom_ops  # noqa: F401

    io = SharedEngramIO(path, max_tokens=64, group=_Group(rank))
    runtime = io.runtime
    assert runtime is not None and runtime.control_host is not None
    control = runtime.control_host
    references = list(gguf.GGUFReader(path, mode="r").tensors)
    failed = False
    stage = "warmup"
    try:
        hashes = torch.zeros(6, 2, 24, dtype=torch.int64, device=rank)
        runtime.prefetch(hashes)
        runtime.lookup(0, 6)
        runtime.lookup(1, 6)
        expected_rows = [1, 1]
        torch.accelerator.synchronize(rank)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runtime.prefetch(hashes)
            outputs = [runtime.lookup(c, 6) for c in range(2)]
        for iteration, count in enumerate((1, 6, 64, 6, 1, 6, 64)):
            stage = f"iteration={iteration}, count={count}"
            current = (
                torch.arange(count * 48, device=rank).view(count, 2, 24) * 13
                + iteration * 37
            ).remainder(1024)
            current[:, :, 1] = current[:, :, 0]
            for channel in range(2):
                expected_rows[channel] += current[:, channel].unique().numel()
            if count == 6:
                hashes.copy_(current)
                graph.replay()
                actual = outputs
            else:
                runtime.prefetch(current)
                if rank == 3:
                    time.sleep(0.01)  # Hold one GPU's consumption fence.
                actual = [runtime.lookup(c, count) for c in range(2)]
            for channel in range(2):
                ids = current[:, channel].cpu().numpy()
                raw = references[channel].data[ids].reshape(-1, 144)
                expected = (
                    torch.from_numpy(
                        gguf.quants.dequantize(
                            raw, gguf.GGMLQuantizationType.IQ4_NL
                        ).copy()
                    )
                    .half()
                    .view(count, 6144)
                )
                torch.testing.assert_close(
                    actual[channel].cpu(), expected, rtol=0, atol=0
                )
        torch.accelerator.synchronize(rank)
        dist.barrier()
        unique = [int(control[c * runtime.channel_words + 2]) for c in range(2)]
        assert max(unique) < 64 * 24  # Duplicate rows caused only one file read.
        statistics = runtime.statistics.cpu().tolist()
        for channel, (requests, rows, byte_count, read_ns) in enumerate(statistics):
            assert requests == 8
            assert rows == expected_rows[channel]
            assert byte_count == rows * 144
            assert read_ns > 0
        return dict(
            rank=rank, passed=True, identity=vars(io.data.identity), unique=unique
        )
    except BaseException:
        failed = True
        records = []
        for c in range(2):
            base = c * runtime.channel_words
            records.append(
                dict(
                    header=control[base : base + 8].tolist(),
                    requests=[
                        control[
                            base + 8 + r * runtime.rank_words : base
                            + 11
                            + r * runtime.rank_words
                        ].tolist()
                        for r in range(4)
                    ],
                )
            )
        print(
            json.dumps(
                dict(
                    rank=rank,
                    stage=stage,
                    records=records,
                    reader_error=io.owner.reader.error()
                    if io.owner is not None
                    else None,
                )
            ),
            flush=True,
        )
        raise
    finally:
        if not failed:
            torch.accelerator.synchronize(rank)
            dist.barrier()
            io.close()
        else:
            # Preserve the first protocol/launch failure for diagnostics.
            with suppress(BaseException):
                io.close()


def _worker(rank, connection, path, rendezvous):
    try:
        torch.set_num_threads(1)
        torch.accelerator.set_device_index(rank)
        dist.init_process_group(
            "gloo",
            init_method="file://" + rendezvous,
            rank=rank,
            world_size=4,
            timeout=datetime.timedelta(seconds=90),
        )
        connection.send(json.dumps(_run(rank, path)).encode())
    except BaseException:
        connection.send(
            json.dumps(dict(rank=rank, error=traceback.format_exc())).encode()
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


def test_shared_engram_reader_and_iq4_gpu_dequantization(tmp_path):
    path = str(tmp_path / "engram.gguf")
    writer = gguf.GGUFWriter(path, "deepseek41-engram")
    for layer in (1, 14):
        blocks = np.random.default_rng(41 + layer).integers(
            0, 256, (1024, 8, 18), dtype=np.uint8
        )
        scale = np.full((1024, 8), 0.03125, dtype="<f2")
        blocks[..., :2] = scale.view(np.uint8).reshape(1024, 8, 2)
        writer.add_tensor(
            f"blk.{layer}.engram_embd.weight",
            blocks.reshape(1024, 144),
            raw_dtype=gguf.GGMLQuantizationType.IQ4_NL,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    context = mp.get_context("spawn")
    processes, connections = [], []
    try:
        for rank in range(4):
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            parent.settimeout(240)
            process = context.Process(
                target=_worker, args=(rank, child, path, str(tmp_path / "rendezvous"))
            )
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
            process.join(20)
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
