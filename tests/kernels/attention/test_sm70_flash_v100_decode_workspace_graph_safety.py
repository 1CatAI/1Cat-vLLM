# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph safety of the cached Flash-V100 decode workspace.

The decode workspace is allocated inside the capture region, so its pointer is
baked into the captured graph. Growing the same cache entry during a later
capture on the same stream (the MTP draft/verify pair is the realistic case)
would release the buffer that the earlier graph still writes into. Upstream
vLLM hit the mirror image of this in FlashMLA (#56902), where the stale buffer
was instead kept resident forever.
"""

import weakref

import pytest
import torch

flash_attn_interface = pytest.importorskip(
    "flash_attn_v100.flash_attn_interface",
    reason="flash_attn_v100 extension is required",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Flash-V100 decode workspaces only exist on SM70",
)


def _plan(*, partitions: int, partition_size: int = 256):
    return flash_attn_interface._DecodePlan(
        partition_size=partition_size,
        actual_num_partitions=partitions,
        launch_num_partitions=partitions,
        workspace_num_partitions=partitions,
    )


def _capture(fn, stream: torch.cuda.Stream) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    return graph


def test_second_capture_does_not_free_the_first_graphs_decode_workspace(
    monkeypatch,
):
    monkeypatch.setattr(flash_attn_interface, "_decode_workspace_cache", {})
    cache = flash_attn_interface._decode_workspace_cache

    q = torch.zeros((1, 6, 256), dtype=torch.float16, device="cuda")
    kwargs = dict(batch_capacity=1, num_heads=6, head_dim=256)

    stream = torch.cuda.Stream()
    # Warm up on the capture stream so the allocator does not synchronize
    # inside the capture region.
    with torch.cuda.stream(stream):
        flash_attn_interface._get_decode_workspace_for_plan(
            q, plan=_plan(partitions=4), **kwargs
        )
    cache.clear()

    first = _capture(
        lambda: flash_attn_interface._get_decode_workspace_for_plan(
            q, plan=_plan(partitions=4), **kwargs
        ),
        stream,
    )
    assert len(cache) == 1
    captured = next(iter(cache.values()))
    live = [weakref.ref(t) for t in (captured.tmp_out, captured.max_logits)]
    captured_ptr = captured.tmp_out.data_ptr()
    del captured

    # A second capture on the same stream hits the same cache key; only the
    # requested partition count is larger, which is what MTP draft/verify does.
    second = _capture(
        lambda: flash_attn_interface._get_decode_workspace_for_plan(
            q, plan=_plan(partitions=64), **kwargs
        ),
        stream,
    )

    assert all(ref() is not None for ref in live), (
        "the workspace captured into the first graph was released by the "
        "second capture; replaying the first graph now writes into freed "
        "memory"
    )
    grown = next(iter(cache.values()))
    assert grown.tmp_out.data_ptr() != captured_ptr

    # Replay must remain valid for both graphs.
    first.replay()
    second.replay()
    torch.accelerator.synchronize()


def test_freed_capture_workspace_is_handed_to_another_graph(monkeypatch):
    """The freed buffer is reused by the next capture in the shared pool.

    vLLM captures every decode shape into one shared graph memory pool, so a
    workspace released mid-capture does not merely dangle: the allocator hands
    the same block to the next graph, and the two graphs then write into the
    same memory.
    """
    monkeypatch.setattr(flash_attn_interface, "_decode_workspace_cache", {})
    cache = flash_attn_interface._decode_workspace_cache

    q = torch.zeros((1, 6, 256), dtype=torch.float16, device="cuda")
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream()

    def capture(
        batch_capacity: int, partitions: int, partition_size: int = 256
    ) -> torch.cuda.CUDAGraph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=pool):
            flash_attn_interface._get_decode_workspace_for_plan(
                q,
                batch_capacity=batch_capacity,
                num_heads=6,
                head_dim=256,
                plan=_plan(partitions=partitions, partition_size=partition_size),
            )
        return graph

    with torch.cuda.stream(stream):
        flash_attn_interface._get_decode_workspace_for_plan(
            q, batch_capacity=1, num_heads=6, head_dim=256, plan=_plan(partitions=4)
        )
    cache.clear()

    first = capture(batch_capacity=1, partitions=4)
    victim_ptr = next(iter(cache.values())).tmp_out.data_ptr()

    # Same key, more partitions: the small buffer the first graph still uses is
    # released back into the shared pool.
    second = capture(batch_capacity=1, partitions=64)

    # A different cache key that allocates the exact same shape now fits the
    # hole: partition_size only participates in the key, not in the size.
    third = capture(batch_capacity=1, partitions=4, partition_size=512)
    reused = [ws.tmp_out.data_ptr() for ws in cache.values() if ws.tmp_out.numel() > 0]

    assert victim_ptr not in reused, (
        "a later graph was given the exact buffer the first graph still "
        "writes into; the two captured graphs now alias the same workspace"
    )
    for graph in (first, second, third):
        graph.replay()
    torch.accelerator.synchronize()
