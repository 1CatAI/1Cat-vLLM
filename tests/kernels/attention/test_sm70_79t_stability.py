# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-like scores must not overflow the Q8000/Q8192 FP32 MMA path."""

import pytest
import torch


def _capture_attention(op, q, k, v, output):
    # Initialize handles/workspaces outside capture, then validate only replay.
    op(q, k, v, output, 0.0625, True)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(q, k, v, output, 0.0625, True)
    graph.replay()
    return graph


@pytest.mark.parametrize(("query_len", "kv_len"), [(8001, 8032), (8160, 8160)])
@torch.inference_mode()
def test_short_first_chunk_graph_matches_unpadded_reference(query_len, kv_len):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _run_sm70_d256_gqa_79t_q8192_dispatch,
    )
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    load_fa2_library(torch.device("cuda"))

    native = getattr(
        torch.ops._vllm_fa2_C, "sm70_d256_gqa_architecture_q8192_fwd", None
    )
    if native is None:
        pytest.skip("SM70 architecture operator was not built")
    torch.manual_seed(8173)
    q = torch.randn(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k) + 1
    output = torch.empty_like(q)

    def dispatch(q, k, v, out, scale, causal):
        return _run_sm70_d256_gqa_79t_q8192_dispatch(
            q, k, v, out, softmax_scale=scale, architecture_q8192_op=native
        )

    graph = _capture_attention(dispatch, q, k, v, output)
    rows = torch.tensor([0, 1, 63, 255, 4095, query_len - 1], device="cuda")
    scores = torch.einsum("rhd,kd->hrk", q[0, rows].double(), k[0, :, 0].double()) / 16
    keys = torch.arange(kv_len, device="cuda")
    scores.masked_fill_(
        keys[None, None, :] > (kv_len - query_len + rows)[None, :, None], -torch.inf
    )
    reference = (scores.softmax(-1) @ v[0, :, 0].double()).permute(1, 0, 2)
    torch.testing.assert_close(
        output[0, rows].double(), reference, rtol=0.003, atol=0.003
    )
    assert torch.isfinite(output).all()
    del graph


@pytest.mark.parametrize("kv_len", [16000, 128000])
@pytest.mark.parametrize(
    ("query_len", "op_name"),
    [
        (8000, "sm70_d256_gqa_architecture_fwd"),
        (8192, "sm70_d256_gqa_architecture_q8192_fwd"),
    ],
)
@torch.inference_mode()
def test_large_scores_and_biased_values(kv_len, query_len, op_name):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    # The FA2 library is loaded per device on first use, not at import.
    load_fa2_library(torch.device("cuda"))

    if not hasattr(torch.ops._vllm_fa2_C, op_name):
        pytest.skip("SM70 architecture operator was not built")
    torch.manual_seed(173)
    q = torch.randn(1, query_len, 6, 256, device="cuda", dtype=torch.float16) * 4
    k = torch.randn(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k) + 8
    output = torch.empty_like(q)
    graph = _capture_attention(getattr(torch.ops._vllm_fa2_C, op_name), q, k, v, output)
    assert torch.isfinite(output).all()
    rows = torch.tensor([0, 63, 64, query_len // 2 - 1, query_len - 1], device="cuda")
    scores = torch.einsum("rhd,kd->hrk", q[0, rows].float(), k[0, :, 0].float()) / 16
    keys = torch.arange(kv_len, device="cuda")
    scores.masked_fill_(
        keys[None, None, :] > (kv_len - query_len + rows)[None, :, None], -torch.inf
    )
    reference = (scores.softmax(-1) @ v[0, :, 0].float()).permute(1, 0, 2)
    torch.testing.assert_close(output[0, rows].float(), reference, rtol=0.01, atol=0.03)
    del graph


@pytest.mark.parametrize(
    ("query_len", "op_name"),
    [
        (8000, "sm70_d256_gqa_architecture_fwd"),
        (8192, "sm70_d256_gqa_architecture_q8192_fwd"),
    ],
)
@torch.inference_mode()
def test_periodic_score_spikes_do_not_overflow(query_len, op_name):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    # The FA2 library is loaded per device on first use, not at import.
    load_fa2_library(torch.device("cuda"))

    if not hasattr(torch.ops._vllm_fa2_C, op_name):
        pytest.skip("SM70 architecture operator was not built")
    kv_len = 16000
    q = torch.zeros(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.zeros(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.zeros_like(k)
    q[..., 0] = 16
    # Put every large score at the same nonzero residue. This reproduces the
    # sparse, correlated numerator growth that overflowed long model requests.
    k[:, 3::128, :, 0] = 16
    v[:, 3::128, :, 0] = 1
    output = torch.empty_like(q)
    graph = _capture_attention(getattr(torch.ops._vllm_fa2_C, op_name), q, k, v, output)
    assert torch.isfinite(output).all()
    rows = torch.tensor([0, 63, 64, query_len // 2 - 1, query_len - 1], device="cuda")
    scores = torch.einsum("rhd,kd->hrk", q[0, rows].float(), k[0, :, 0].float()) / 16
    keys = torch.arange(kv_len, device="cuda")
    scores.masked_fill_(
        keys[None, None, :] > (kv_len - query_len + rows)[None, :, None], -torch.inf
    )
    reference = (scores.softmax(-1) @ v[0, :, 0].float()).permute(1, 0, 2)
    torch.testing.assert_close(output[0, rows].float(), reference, rtol=0.01, atol=0.01)
    del graph


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("location", ["tail"])
@pytest.mark.parametrize(
    ("query_len", "op_name"),
    [
        (8000, "sm70_d256_gqa_architecture_fwd"),
        (8192, "sm70_d256_gqa_architecture_q8192_fwd"),
    ],
)
@torch.inference_mode()
def test_unsampled_unequal_peaks_preserve_weights(query_len, op_name, location, sparse):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    load_fa2_library(torch.device("cuda"))

    if not hasattr(torch.ops._vllm_fa2_C, op_name):
        pytest.skip("SM70 architecture operator was not built")
    prefix = 8192
    kv_len = prefix + query_len
    q = torch.zeros(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.zeros(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.zeros_like(k)
    if sparse:
        q[:, 3653, 2, 0] = 16
    else:
        q[..., 0] = 16
    start = 0 if location == "prefix" else prefix
    # Both peaks evade the former stride-8 sample. Clipping their distinct
    # logits to the same value gives roughly zero instead of almost +/-1.
    v[:, start + 3, :, 0] = 1
    v[:, start + 5, :, 0] = -1
    output = torch.empty_like(q)
    k[:, start + 3, :, 0] = 16
    k[:, start + 5, :, 0] = 24
    graph = _capture_attention(getattr(torch.ops._vllm_fa2_C, op_name), q, k, v, output)
    rows = torch.tensor([255, 256, 3653, 4095, query_len - 1], device="cuda")
    keys = torch.arange(kv_len, device="cuda")
    for first, second in [(16, 24), (28, 20), (2, 3)]:
        # Replay must recompute maxima when tensor values change in place.
        k[:, start + 3, :, 0] = first
        k[:, start + 5, :, 0] = second
        graph.replay()
        assert torch.isfinite(output).all()
        scores = (
            torch.einsum("rhd,kd->hrk", q[0, rows].double(), k[0, :, 0].double()) / 16
        )
        scores.masked_fill_(
            keys[None, None, :] > (prefix + rows)[None, :, None], -torch.inf
        )
        reference = (scores.softmax(-1) @ v[0, :, 0].double()).permute(1, 0, 2)
        torch.testing.assert_close(
            output[0, rows].double(), reference, rtol=0.003, atol=0.001
        )


@pytest.mark.parametrize(
    ("query_len", "op_name"),
    [
        (8000, "sm70_d256_gqa_architecture_fwd"),
        (8192, "sm70_d256_gqa_architecture_q8192_fwd"),
    ],
)
@torch.inference_mode()
def test_rejects_partial_prefix_pv_tile(query_len, op_name):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    # The FA2 library is loaded per device on first use, not at import.
    load_fa2_library(torch.device("cuda"))

    if not hasattr(torch.ops._vllm_fa2_C, op_name):
        pytest.skip("SM70 architecture operator was not built")
    q = torch.empty(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.empty(1, query_len + 8, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.empty_like(k)
    output = torch.empty_like(q)
    with pytest.raises(RuntimeError, match="32-token alignment"):
        getattr(torch.ops._vllm_fa2_C, op_name)(q, k, v, output, 0.0625, True)


@torch.inference_mode()
def test_q8000_q8192_share_scores_and_preserve_graph_replay():
    if not torch.accelerator.is_available() or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("requires SM70")
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    # The FA2 library is loaded per device on first use, not at import.
    load_fa2_library(torch.device("cuda"))

    torch.manual_seed(732)
    cases = []
    for length, name in [
        (8192, "sm70_d256_gqa_architecture_q8192_fwd"),
        (8000, "sm70_d256_gqa_architecture_fwd"),
    ]:
        q = torch.randn(1, length, 6, 256, device="cuda", dtype=torch.float16)
        k = torch.randn(1, length + 8192, 1, 256, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        op = getattr(torch.ops._vllm_fa2_C, name)
        before = torch.accelerator.memory_allocated()
        op(q, k, v, out, 0.0625, True)
        if length == 8000:
            # The second family must not allocate another ~2.2 GiB scores buffer.
            assert torch.accelerator.memory_allocated() - before < 1024**3
        reference = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(q, k, v, out, 0.0625, True)
        cases.append((graph, out, reference, q, k, v, op))
    combined = torch.cuda.CUDAGraph()
    with torch.cuda.graph(combined):
        for _, out, _, q, k, v, op in cases:
            op(q, k, v, out, 0.0625, True)
    for _ in range(3):
        for graph, out, reference, *_ in cases:
            graph.replay()
            torch.testing.assert_close(out, reference, rtol=0, atol=0)
        combined.replay()
        for _, out, reference, *_ in cases:
            torch.testing.assert_close(out, reference, rtol=0, atol=0)


@pytest.mark.parametrize("query_len", [8000, 8192])
@pytest.mark.parametrize(
    ("location", "total_kv"),
    [
        ("prefix", 0),
        ("last_prefix_block", 0),
        ("tail", 0),
        ("last_prefix_block", 256000),
    ],
)
@torch.inference_mode()
def test_compact_score_range_recovery_with_changing_graph_inputs(
    query_len, location, total_kv
):
    """Finite inputs can overflow or lose score differences on an FP16 store."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    from vllm.vllm_flash_attn.flash_attn_interface import load_fa2_library

    load_fa2_library(torch.device("cuda"))

    name = (
        "sm70_d256_gqa_architecture_fwd"
        if query_len == 8000
        else "sm70_d256_gqa_architecture_q8192_fwd"
    )
    op = getattr(torch.ops._vllm_fa2_C, name, None)
    if op is None:
        pytest.skip("SM70 architecture operator was not built")
    kv_len = total_kv or 8224 + query_len
    prefix = kv_len - query_len
    q = torch.zeros(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.zeros(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.zeros_like(k)
    out = torch.empty_like(q)
    start = {
        "prefix": 0,
        "last_prefix_block": (prefix - 1) // 8192 * 8192,
        "tail": prefix,
    }[location]
    graph = _capture_attention(op, q, k, v, out)
    rows = torch.tensor(
        [0, 1, 63, 64, 255, 256, 403, 1023, 1024, 3653, query_len - 1],
        device="cuda",
    )
    keys = torch.arange(kv_len, device="cuda")
    for case in [
        "overflow",
        "rounded",
        "sample_gap",
        "negative",
        "large_constant_value",
        "ordinary",
    ]:
        q.zero_()
        k.zero_()
        v.zero_()
        v[:, start + 3, :, 0] = 1
        v[:, start + 5, :, 0] = -1
        # Only one head in scattered query rows is exceptional. Other heads
        # and rows still need their ordinary answer after selective recovery.
        q[:, rows, 2, 0] = 16
        if case == "overflow":
            q[:, rows, 2, 0] = 256
            k[:, start + 3, :, 0] = 4096
            k[:, start + 5, :, 0] = 4100
        elif case == "rounded":
            q[:, rows, 2, 1] = 16
            k[:, start + 3, :, 0] = 8192
            k[:, start + 5, :, 0] = 8192
            k[:, start + 3, :, 1] = 1
            k[:, start + 5, :, 1] = 2
        elif case == "sample_gap":
            # The missed peak is only nine above the sample: no clipping
            # repair, but the compact-score range guard must still cover it.
            k[:, start, :, 0] = 120
            k[:, start + 3, :, 0] = 129
            k[:, start + 5, :, 0] = 129.125
        elif case == "negative":
            q[:, rows, 2, 1] = 16
            k[..., 0] = -8192
            k[:, start + 3, :, 1] = 8
            k[:, start + 5, :, 1] = 9
        elif case == "large_constant_value":
            q[:, rows, 2, 1] = 16
            k[..., 0] = 8192
            k[..., 1] = -0.287353515625
            k[:, start, :, 1] = 0
            v.fill_(65504)
        else:
            # Clear previous flags as well as maxima and denominators.
            k[:, start + 3, :, 0] = 2
            k[:, start + 5, :, 0] = 3
        graph.replay()
        scores = (
            torch.einsum("rhd,kd->hrk", q[0, rows].double(), k[0, :, 0].double()) / 16
        )
        scores.masked_fill_(
            keys[None, None, :] > (prefix + rows)[None, :, None], -torch.inf
        )
        reference = (scores.softmax(-1) @ v[0, :, 0].double()).permute(1, 0, 2)
        assert torch.isfinite(out).all(), case
        torch.testing.assert_close(
            out[0, rows].double(), reference, rtol=0.003, atol=0.001, msg=case
        )
