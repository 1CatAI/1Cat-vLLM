# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The profile run tunes the chunked KDA prefill kernels at its own length."""

from types import SimpleNamespace

import torch

import vllm.models.glm5next.nvidia.kda as glm_kda
from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention


def _layer() -> Glm5NextLinearAttention:
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.local_num_heads = 4
    layer.head_dim = 16
    layer.kda_safe_gate = True
    layer.kda_lower_bound = -5.0
    layer.A_log = torch.nn.Parameter(torch.zeros(1, 1, 4, 1))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(64))
    layer.chunk_kernels = SimpleNamespace(name="engine kernels")
    layer._prefill_kernels_warmed_up = False
    layer.get_state_dtype = lambda: (torch.float16, torch.float32)  # type: ignore[method-assign]
    layer.get_state_shape = lambda: ((96, 3), (4, 16, 16))  # type: ignore[method-assign]
    return layer


def test_profile_run_tunes_the_prefill_kernels_once_at_its_length(monkeypatch):
    calls = []
    monkeypatch.setattr(
        glm_kda, "chunk_kda_with_fused_gate", lambda **kwargs: calls.append(kwargs)
    )
    layer = _layer()
    profile_tokens = torch.empty(512, 3 * 4 * 16, dtype=torch.float16)

    layer._warmup_prefill_kernels(profile_tokens)
    layer._warmup_prefill_kernels(profile_tokens)

    assert len(calls) == 1
    call = calls[0]
    for name in ("q", "k", "v", "raw_g"):
        assert call[name].shape == (1, 512, 4, 16)
        assert call[name].dtype == torch.float16
    # A real prefill passes the sigmoided beta in fp32, the recurrent state in
    # the cache dtype and int32 offsets; dtypes are part of the autotune keys.
    assert call["beta"].shape == (1, 512, 4)
    assert call["beta"].dtype == torch.float32
    assert call["initial_state"].shape == (1, 4, 16, 16)
    assert call["initial_state"].dtype == torch.float32
    assert call["cu_seqlens"].tolist() == [0, 512]
    assert call["cu_seqlens"].dtype == torch.int32
    assert call["kernels"] is layer.chunk_kernels
    assert call["use_qk_l2norm_in_kernel"] is True
