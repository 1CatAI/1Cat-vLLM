# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native QSA over host history + hot cache equals device-history output."""

import pytest
import torch

pytest.importorskip("vllm._sm70_qsa_device_C")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
    pytest.skip("requires SM70", allow_module_level=True)

from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV  # noqa: E402
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor  # noqa: E402

WIDTH = 2051


def _workspace(rows, dev):
    splits = (WIDTH + 63) // 64
    return (
        torch.empty(rows * 6 * splits * 256, dtype=torch.float32, device=dev),
        torch.empty(rows * 6 * splits * 2, dtype=torch.float32, device=dev),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.uint8])
@pytest.mark.parametrize("rows", [1, 5])
def test_host_native_matches_device_history(dtype, rows):
    torch.manual_seed(rows)
    dev = torch.device("cuda")
    blocks, page, context = 160, 64, 9000
    state = HostQSAKV(
        blocks,
        page,
        256,
        dev,
        hot_tokens=1024,
        rows=8,
        width=WIDTH,
        dtype=dtype,
        direct_device=False,
    )
    # Write the whole context through the normal writer (fills host history).
    table = torch.randperm(blocks, device=dev, dtype=torch.int32)[None]
    slots = (
        table[0].long()[torch.arange(context, device=dev) // page] * page
        + torch.arange(context, device=dev) % page
    ).to(torch.int32)
    key = torch.randn(context, 1, 256, device=dev, dtype=torch.float16)
    value = torch.randn(context, 1, 256, device=dev, dtype=torch.float16)
    state.write(key, value, slots)
    torch.accelerator.synchronize()

    positions = torch.arange(context - rows, context, device=dev, dtype=torch.int64)
    requests = torch.zeros(rows, dtype=torch.int32, device=dev)
    lengths = torch.tensor([context], dtype=torch.int32, device=dev)
    indices = torch.stack(
        [torch.randperm(context, device=dev)[:WIDTH].sort().values for _ in range(rows)]
    ).to(torch.int32)
    indices[:, -7:] = -1  # padded selections
    query = torch.randn(rows, 6, 256, device=dev, dtype=torch.float16)
    gate = torch.randn(rows, 6, 256, device=dev, dtype=torch.float16)

    # Reference: identical history bytes resident on device.
    device_history = state.history.clone()
    device_scales = state.scales.clone()
    reference = torch.empty_like(query)
    torch.ops.vllm_sm70_qsa_device.run(
        query,
        device_history,
        device_scales,
        indices,
        table,
        requests,
        positions,
        lengths,
        reference,
        gate,
        *_workspace(rows, dev),
    )

    # The 1024-token hot cache holds only part of the 2051-wide selection:
    # warm it once, then resolve so both hot hits and host reads occur.
    state.resolve_hot(indices, table, requests, positions, lengths)
    resolved = state.resolve_hot(indices, table, requests, positions, lengths)
    hot_hits = int((resolved >= 0).sum())
    host_reads = int((resolved == -1).sum())
    out = torch.empty_like(query)
    torch.ops.vllm_sm70_qsa_device.run_hot(
        query,
        state.history,
        state.scales,
        indices,
        table,
        requests,
        positions,
        lengths,
        out,
        gate,
        *_workspace(rows, dev),
        state.hot_values,
        resolved,
    )
    torch.accelerator.synchronize()
    assert hot_hits > 0 and host_reads > 0  # both sources exercised
    assert torch.equal(out.view(torch.int16), reference.view(torch.int16))


def test_history_is_mapped_host_memory():
    dev = torch.device("cuda")
    state = HostQSAKV(
        4,
        64,
        256,
        dev,
        hot_tokens=64,
        rows=8,
        width=WIDTH,
        dtype=torch.uint8,
        direct_device=False,
    )
    assert state.host is not None and state.host.is_pinned()
    assert (
        state.history.data_ptr()
        == get_accelerator_view_from_cpu_tensor(state.host).data_ptr()
    )
