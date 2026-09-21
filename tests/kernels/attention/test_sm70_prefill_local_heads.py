# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent numerical checks for multi-head FP32-accumulating prefill."""

import pytest
import torch


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("batch", [1, 2])
def test_q8192_multihead_prefill_at_256k(heads, batch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm.v1.attention.backends.flash_attn_v100 import _run_sm70_gqa_groups
    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    torch.manual_seed(7541)
    q_len, length = 8192, 262144
    q = torch.randn(batch, q_len, heads * 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, length, heads, 256, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    # Normal chunked prefill runs outside capture in a CUDA-Graph-enabled
    # server. This check measures correctness only, not eager throughput.
    _run_sm70_gqa_groups(
        torch.ops._vllm_fa2_C.sm70_d256_gqa_architecture_q8192_fwd,
        q,
        k,
        v,
        out,
        0.0625,
        True,
    )
    assert torch.isfinite(out).all()
    rows = torch.tensor([0, 63, 4095, q_len - 1], device="cuda")
    positions = torch.arange(length, device="cuda")
    for item in range(batch):
        for head in range(heads):
            group_q = q[item : item + 1, :, head * 6 : (head + 1) * 6].contiguous()
            control = torch.empty_like(group_q)
            torch.ops._vllm_fa2_C.sm70_d256_gqa_architecture_q8192_fwd(
                group_q,
                k[item : item + 1, :, head : head + 1].contiguous(),
                v[item : item + 1, :, head : head + 1].contiguous(),
                control,
                0.0625,
                True,
            )
            assert torch.equal(out[item, :, head * 6 : (head + 1) * 6], control[0])
            query = q[item, rows, head * 6 : (head + 1) * 6].transpose(0, 1).double()
            scores = query @ k[item, :, head].double().T * 0.0625
            scores.masked_fill_(
                positions[None, None] > (length - q_len + rows)[None, :, None],
                -torch.inf,
            )
            ref = (scores.softmax(-1) @ v[item, :, head].double()).transpose(0, 1)
            actual = out[item, rows, head * 6 : (head + 1) * 6].double()
            # Preserve the accepted 75T reduction exactly. Its FP16 probability
            # tiles have a different error budget from the optional v37 core.
            assert float((actual - ref).norm() / ref.norm()) < 0.007


@pytest.mark.parametrize("q_len", [8000, 8192])
@pytest.mark.parametrize("heads,batch", [(2, 1), (4, 2)])
def test_multihead_prefill_graph_replay_after_other_capture(q_len, heads, batch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm.v1.attention.backends.flash_attn_v100 import _run_sm70_gqa_groups
    from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401

    op = (
        torch.ops._vllm_fa2_C.sm70_d256_gqa_architecture_fwd
        if q_len == 8000
        else torch.ops._vllm_fa2_C.sm70_d256_gqa_architecture_q8192_fwd
    )
    torch.manual_seed(7542)
    captures = []
    for length in (32768, 65536):
        q = torch.randn(
            batch, q_len, heads * 6, 256, device="cuda", dtype=torch.float16
        )
        k = torch.randn(batch, length, heads, 256, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        _run_sm70_gqa_groups(op, q, k, v, out, 0.0625, True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_sm70_gqa_groups(op, q, k, v, out, 0.0625, True)
        captures.append((graph, q, k, v, out))

    # The second capture must not overwrite metadata used by the first graph.
    # Changing all inputs also detects accidentally captured warmup outputs.
    for graph, q, k, v, out in captures:
        q.normal_()
        k.normal_()
        v.normal_()
        for _ in range(3):
            graph.replay()
        torch.accelerator.synchronize()
        assert torch.isfinite(out).all()
        reference = torch.empty_like(q)
        _run_sm70_gqa_groups(op, q, k, v, reference, 0.0625, True)
        torch.accelerator.synchronize()
        assert torch.equal(out, reference)
