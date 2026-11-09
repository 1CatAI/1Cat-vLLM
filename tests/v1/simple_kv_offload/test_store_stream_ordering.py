# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stores read the live GPU KV cache; loads read host pinned memory.

Both facts constrain how the transfer may be issued. A store must be ordered
behind the compute stream and must keep STREAM source-access ordering, or the
driver is free to read blocks the compute stream has not finished writing --
which silently corrupts the offloaded copy while its block hash marks it valid.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload import cuda_mem_ops
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1
CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 3


def _caches() -> dict[str, torch.Tensor]:
    return {"layer.0": torch.zeros(4, 16, dtype=torch.int8)}


@pytest.mark.parametrize(
    "src_access_order_any,expected",
    [
        (False, CU_MEMCPY_SRC_ACCESS_ORDER_STREAM),
        (True, CU_MEMCPY_SRC_ACCESS_ORDER_ANY),
    ],
)
def test_build_params_source_access_order(
    monkeypatch: pytest.MonkeyPatch, src_access_order_any: bool, expected: int
) -> None:
    monkeypatch.setattr(cuda_mem_ops, "_batch_memcpy_fn", lambda *a, **kw: 0)
    stream = SimpleNamespace(cuda_stream=0)

    params = cuda_mem_ops.build_params(
        _caches(),
        _caches(),
        stream,  # type: ignore[arg-type]
        src_access_order_any=src_access_order_any,
    )

    assert params.attrs.srcAccessOrder == expected


def test_store_is_ordered_behind_the_compute_stream() -> None:
    """The transfer stream waits on compute before the copy is queued."""
    worker = SimpleCPUOffloadWorker(
        SimpleNamespace(),  # type: ignore[arg-type]
        kv_cache_config=None,
        cpu_capacity_bytes=0,
    )
    recorder = mock.Mock()
    worker.store_stream = recorder.store_stream
    worker._backend = recorder.backend
    worker.bind_connector_metadata(
        SimpleCPUOffloadMetadata(
            store_event=0, store_gpu_blocks=[7], store_cpu_blocks=[3]
        )
    )

    worker.get_finished(set())

    ordered = [call[0] for call in recorder.mock_calls]
    assert "store_stream.wait_stream" in ordered
    assert "backend.launch_copy" in ordered
    assert ordered.index("store_stream.wait_stream") < ordered.index(
        "backend.launch_copy"
    ), f"copy was queued before the barrier: {ordered}"


def test_load_is_not_gated_on_the_compute_stream() -> None:
    """Loads read host memory no GPU stream writes, so they need no barrier."""
    worker = SimpleCPUOffloadWorker(
        SimpleNamespace(),  # type: ignore[arg-type]
        kv_cache_config=None,
        cpu_capacity_bytes=0,
    )
    worker.store_stream = mock.Mock()
    worker._backend = mock.Mock()
    worker.bind_connector_metadata(
        SimpleCPUOffloadMetadata(load_event=0, load_gpu_blocks=[7], load_cpu_blocks=[3])
    )

    worker.get_finished(set())

    worker.store_stream.wait_stream.assert_not_called()
