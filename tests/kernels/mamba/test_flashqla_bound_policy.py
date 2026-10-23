# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native legacy/configured output/state parity with interleaved graph owners."""

import pytest
import torch

from flash_qla.ops.gated_delta_rule.chunk.sm70.fused_fwd import _load_ext

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA operator validation"
)


@pytest.mark.parametrize("groups", [-1, 1, 2, 4, 8])
@pytest.mark.parametrize("tokens", [0, 1, 33])
def test_bound_prefill_and_decode_match_legacy_after_environment_changes(
    monkeypatch, groups, tokens
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 validated contract")
    ext = _load_ext()
    owner = ext.GdnPolicy(groups)
    torch.manual_seed(91)
    alias = "FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK"
    monkeypatch.setenv(alias, "" if groups == -1 else str(groups))
    q, k = [
        torch.randn(1, tokens, 4, 128, device="cuda", dtype=torch.float16) * 0.01
        for _ in range(2)
    ]
    v = torch.randn(1, tokens, 8, 128, device="cuda", dtype=torch.float16) * 0.01
    g = torch.full((1, tokens, 8), -0.1, device="cuda")
    beta = torch.full((1, tokens, 8), 0.2, device="cuda")
    state = torch.randn(2, 8, 128, 128, device="cuda") * 0.01
    cu = torch.tensor([0, tokens // 2, tokens], device="cuda", dtype=torch.int32)
    args = (q, k, v, g, beta, state, cu, 128**-0.5, True, False, False, None)
    expected = ext.gdn_forward_vlk_varlen(*args)
    monkeypatch.setenv(alias, "poison-after-init")
    actual = owner.gdn_forward_vlk_varlen(*args)
    for left, right in zip(actual, expected):
        assert torch.equal(left, right)
    # Empty inputs retain the operator's existing supported semantics.
    if tokens == 0:
        return
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = owner.gdn_forward_vlk_varlen(*args)
    for delta in (0.01, -0.02):
        q.add_(delta)
        actual = owner.gdn_forward_vlk_varlen(*args)
        graph.replay()
        torch.accelerator.synchronize()
        for left, right in zip(captured, actual):
            assert torch.equal(left, right)


def test_two_decode_owners_preserve_reordered_state_and_capture(monkeypatch):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 validated contract")
    ext = _load_ext()
    torch.manual_seed(92)
    tokens, hq, hv, dim = 4, 4, 8, 128
    mixed = (
        torch.randn(tokens, (2 * hq + hv) * dim, device="cuda", dtype=torch.float16)
        * 0.01
    )
    a, b = [
        torch.randn(tokens, hv, device="cuda", dtype=torch.float16) for _ in range(2)
    ]
    A_log, bias = [torch.randn(hv, device="cuda") for _ in range(2)]
    seed = torch.randn(8, hv, dim, dim, device="cuda") * 0.01
    indices = torch.tensor([3, 0, -1, 6], device="cuda", dtype=torch.int32)
    records = []
    for groups in (1, 4):
        owner, state, expected_state = ext.GdnPolicy(groups), seed.clone(), seed.clone()
        output, expected = [
            torch.zeros(tokens, hv, dim, device="cuda", dtype=torch.float16)
            for _ in range(2)
        ]
        args = (mixed, a, b, A_log, bias, state, indices, output, dim**-0.5, True)
        monkeypatch.setenv("FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK", str(groups))
        ext.gdn_decode_mixed_qkv_global_state(
            mixed, a, b, A_log, bias, expected_state, indices, expected, dim**-0.5, True
        )
        monkeypatch.setenv("FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK", "poison")
        owner.gdn_decode_mixed_qkv_global_state(*args)
        assert torch.equal(state, expected_state)
        assert torch.equal(output, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            owner.gdn_decode_mixed_qkv_global_state(*args)
        records.append((owner, state, output, graph, args))
    for permutation in ([6, 2, -1, 0], [0, 4, -1, 3]):
        indices.copy_(torch.tensor(permutation, device="cuda", dtype=torch.int32))
        mixed.add_(0.01)
        for owner, state, output, graph, args in reversed(records):
            state.copy_(seed)
            owner.gdn_decode_mixed_qkv_global_state(*args)
            expected_state, expected = state.clone(), output.clone()
            state.copy_(seed)
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(state, expected_state)
            assert torch.equal(output, expected)


@pytest.mark.parametrize("raw", ["-1", "4294967295", "0", "bad"])
def test_native_binding_keeps_invalid_legacy_groups_invalid(monkeypatch, raw):
    from vllm.config.gdn import GdnConfig

    monkeypatch.setenv("FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK", raw)
    ext = _load_ext()
    policy = GdnConfig()
    policy.resolve()
    message = "must be one of 1, 2, 4, 8"
    with pytest.raises(RuntimeError, match=message):
        ext.resolve_column_groups_per_block(33, 4, 8)
    with pytest.raises(RuntimeError, match=message):
        ext.GdnPolicy(policy.flashqla_column_groups)
    # An explicit typed automatic choice overrides the invalid compatibility input.
    typed = GdnConfig(flashqla_column_groups=-1)
    typed.resolve()
    assert (
        ext.GdnPolicy(typed.flashqla_column_groups).resolve_column_groups_per_block(
            33, 4, 8
        )
        == 1
    )
