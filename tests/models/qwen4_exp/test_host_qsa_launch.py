# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import host_kv_attention as host


@pytest.mark.parametrize(
    ("rows", "heads", "dim", "sm70", "expected"),
    [
        (5, 6, 256, True, 2),
        (20, 6, 256, True, 2),
        (32, 6, 256, True, 2),
        (33, 6, 256, True, 4),
        (20, 5, 256, True, 4),
        (20, 6, 128, True, 4),
        (20, 6, 256, False, 4),
    ],
)
def test_host_partial_policy_preserves_unsupported_profiles(
    monkeypatch, rows, heads, dim, sm70, expected
):
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(kwargs)

            return launch

    monkeypatch.setattr(host.current_platform, "is_device_capability", lambda _: sm70)
    monkeypatch.setattr(host, "_qsa_sparse_paged_gqa_splitk_kernel", Kernel())
    monkeypatch.setattr(host, "_qsa_merge_splitk_kernel", Kernel())
    query = torch.empty(rows, heads, dim, dtype=torch.float16)
    indices = torch.zeros(rows, 2051, dtype=torch.int32)
    state = SimpleNamespace(
        dim=dim,
        page_size=816,
        blocks=2,
        hot_values=torch.empty(0),
        staging=torch.empty(0),
        lengths=torch.zeros(rows, dtype=torch.int32),
        resolve=lambda *args: indices,
    )
    host.host_qsa_attention(
        query,
        state,
        indices,
        torch.zeros(1, 2, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.arange(rows),
        torch.ones(1, dtype=torch.int32),
        torch.empty_like(query),
    )
    assert launches[0]["num_warps"] == expected
    assert launches[0]["BLOCK_N"] == 16
    assert launches[0]["NUM_SPLITS"] == (64 if rows <= 8 else 32 if rows < 32 else 8)
