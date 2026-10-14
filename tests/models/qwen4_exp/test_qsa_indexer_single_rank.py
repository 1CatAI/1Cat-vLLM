# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated selection uses one reader, with live paged state on graph replay."""

import socket
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp


def _worker(rank, port, results):
    import traceback

    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import (
            get_tp_group,
            graph_capture,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.models.qwen4_exp.nvidia.indexer_qsa import QSAIndexer
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        torch.accelerator.set_device_index(rank)
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                4, rank, f"tcp://127.0.0.1:{port}", rank, "nccl"
            )
            initialize_model_parallel(4, 1)
            indexer = QSAIndexer.__new__(QSAIndexer)
            torch.nn.Module.__init__(indexer)
            indexer.token_topk, indexer.compress_ratio = 2048, 4
            indexer.shared_key_scoring = True
            indexer.shared_key_batch_scoring = True
            from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer_staging import (
                QSAIndexerStagingWorkspace,
            )

            indexer.indexer_staging_workspace = QSAIndexerStagingWorkspace()
            indexer.selection_tp_group = get_tp_group()
            report = []
            for rows in (1, 5, 10, 20, 30, 40, 64, 65):
                requests = (rows + 4) // 5
                torch.manual_seed(20261015 + rows)
                host = (
                    torch.randn(requests * 16, 64, 1, 128, dtype=torch.float16) * 0.125
                ).pin_memory()
                cache = get_accelerator_view_from_cpu_tensor(host)
                q = (
                    torch.randn(rows, 4, 128, device="cuda", dtype=torch.float16)
                    * 0.125
                )
                table = torch.arange(
                    requests * 16, device="cuda", dtype=torch.int32
                ).view(requests, 16)
                mapping = torch.arange(rows, device="cuda", dtype=torch.int32) // 5
                lengths = torch.full(
                    (requests,), 4096, device="cuda", dtype=torch.int32
                )
                positions = torch.full((rows,), 4095, device="cuda", dtype=torch.int64)
                starts = torch.tensor([*range(0, rows, 5), rows], dtype=torch.int32)
                metadata = SimpleNamespace(
                    block_table=table,
                    token_to_req=mapping,
                    logical_positions=positions,
                    seq_lens=lengths,
                    query_start_loc_cpu=starts,
                    max_seq_len=4096,
                )
                out = torch.empty(rows, 2051, dtype=torch.int32, device="cuda")
                expected = torch.empty_like(out)
                indexer._select_local_tokens(q, cache, metadata, expected)
                indexer._select_tokens(q, cache, metadata, out)
                calls: list[int] = []
                local_select = indexer._select_local_tokens

                def counted(*args, calls=calls, local_select=local_select):
                    calls.append(1)
                    return local_select(*args)

                indexer._select_local_tokens = counted
                graph = torch.cuda.CUDAGraph()
                with (
                    graph_capture(device=torch.device("cuda", rank)) as context,
                    torch.cuda.graph(graph, stream=context.stream),
                ):
                    result = indexer._select_tokens(q, cache, metadata, out)
                indexer._select_local_tokens = local_select
                assert len(calls) == int(rows > 64 or rank == 0), (rank, rows, calls)
                assert result.data_ptr() == out.data_ptr()
                for shift in (1, 2):
                    q.mul_(0.75)
                    table.copy_(table.roll(shift, 0))
                    lengths[0] = 0
                    positions.sub_(47)
                    mapping[-1] = -1
                    graph.replay()
                    indexer._select_local_tokens(q, cache, metadata, expected)
                    torch.accelerator.synchronize()
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                report.append(rows)
            results.put((rank, report))
    except BaseException:
        results.put((rank, {"error": traceback.format_exc()}))
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_single_rank_indexer_graph_replay_and_capacity_fallback():
    if torch.accelerator.device_count() != 4 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("Four visible SM70 GPUs required")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(target=_worker, args=(r, port, results)) for r in range(4)
    ]
    for worker in workers:
        worker.start()
    try:
        reports = [results.get(timeout=180) for _ in workers]
    finally:
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join()
    assert all(isinstance(report, list) for _, report in reports), reports
    assert all(worker.exitcode == 0 for worker in workers)
    assert sorted(rank for rank, _ in reports) == list(range(4))
