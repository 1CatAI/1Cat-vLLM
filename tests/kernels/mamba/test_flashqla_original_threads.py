# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired ordinary API validation of SM70 prefill launch tuning."""

import pytest
import torch

from vllm.platforms import current_platform

if not torch.cuda.is_available() or not current_platform.is_device_capability(70):
    pytest.skip("requires SM70", allow_module_level=True)

from flash_qla.ops.gated_delta_rule.chunk import (  # noqa: E402
    chunk_gated_delta_rule_fwd_sm70_tilelang,
)
from flash_qla.utils import prepare_chunk_indices, prepare_chunk_offsets  # noqa: E402


@pytest.mark.parametrize("rows", [4097, 6108])
@pytest.mark.parametrize("has_state", [False, True])
def test_indexed_state_continuation_and_untouched_slots(rows, has_state):
    torch.manual_seed(rows)
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(1, rows, 4, 128, device="cuda"), dim=-1
        ).half()
        for _ in range(2)
    ]
    v = torch.randn(1, rows, 12, 128, device="cuda", dtype=torch.float16) * 0.1
    g = -torch.rand(1, rows, 12, device="cuda") * 0.08
    beta = torch.rand_like(g)
    cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    chunks = prepare_chunk_indices(cu, 64)
    offsets, _ = prepare_chunk_offsets(cu, 64)
    seed = torch.randn(4, 12, 128, 128, device="cuda") * 0.01
    baseline, candidate = seed.clone(), seed.clone()
    for continuation in (False, True):
        outputs = []
        for state, threads in ((baseline, 128), (candidate, 256)):
            _, _, output, _, final = chunk_gated_delta_rule_fwd_sm70_tilelang(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                cu_seqlens=cu,
                chunk_indices=chunks,
                chunk_offsets=offsets,
                state_indices=torch.tensor([2], device="cuda", dtype=torch.int32),
                has_initial_state=torch.tensor(
                    [has_state or continuation], device="cuda"
                ),
                output_h=False,
                auto_cp=False,
                state_layout_vlk=True,
                inplace_final_state=True,
                sm70_threads=threads,
            )
            assert final is None
            outputs.append(output)
        torch.accelerator.synchronize()
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(baseline, candidate)
        assert torch.isfinite(outputs[1]).all() and torch.isfinite(candidate).all()
        assert torch.equal(candidate[:2], seed[:2])
        assert torch.equal(candidate[3:], seed[3:])


@pytest.mark.parametrize("rows", [4097, 6108])
@pytest.mark.parametrize("has_state", [False, True])
def test_direct_state_continuation_and_input_ownership(rows, has_state):
    torch.manual_seed(rows)
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(1, rows, 4, 128, device="cuda"), dim=-1
        ).half()
        for _ in range(2)
    ]
    v = torch.randn(1, rows, 12, 128, device="cuda", dtype=torch.half) * 0.1
    g = -torch.rand(1, rows, 12, device="cuda") * 0.08
    beta = torch.rand_like(g)
    cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    chunks = prepare_chunk_indices(cu, 64)
    offsets, _ = prepare_chunk_offsets(cu, 64)
    seed = torch.randn(1, 12, 128, 128, device="cuda") * 0.01
    if not has_state:
        seed.zero_()
    baseline, candidate = seed.clone(), seed.clone()
    for _ in range(2):
        outputs, finals = [], []
        for state, threads in ((baseline, 128), (candidate, 256)):
            saved = state.clone()
            _, _, output, _, final = chunk_gated_delta_rule_fwd_sm70_tilelang(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                cu_seqlens=cu,
                chunk_indices=chunks,
                chunk_offsets=offsets,
                output_final_state=True,
                output_h=False,
                auto_cp=False,
                state_layout_vlk=True,
                sm70_threads=threads,
            )
            assert torch.equal(state, saved)
            assert final is not None
            outputs.append(output)
            finals.append(final)
        torch.accelerator.synchronize()
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(finals[0], finals[1])
        assert torch.isfinite(outputs[1]).all() and torch.isfinite(finals[1]).all()
        baseline, candidate = finals
