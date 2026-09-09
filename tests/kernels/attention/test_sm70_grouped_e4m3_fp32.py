# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-quantized-KV FP64 oracle; these are not model-quality tests."""

from types import SimpleNamespace

import pytest
import torch


def _native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    module = pytest.importorskip("flash_attn_v100")
    if not module.flash_attn_grouped_e4m3_fp32_available():
        pytest.skip("rebuild native E4M3 FP32 entry")
    return module.flash_attn_grouped_e4m3_fp32_paged


@pytest.mark.parametrize("has_entry", [False, True])
@pytest.mark.parametrize("version", [None, 0, 1, 2, 3, 4, 5])
def test_precision_capability_rejects_stale_binary(monkeypatch, has_entry, version):
    interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    native = SimpleNamespace()
    if has_entry:
        native.grouped_e4m3_fp32_paged_fwd = object()
    if version is not None:
        native.grouped_e4m3_fp32_precision_version = lambda: version
    monkeypatch.setattr(interface, "flash_attn_v100_cuda", native)
    assert interface.flash_attn_grouped_e4m3_fp32_available() is (
        has_entry and version is not None and version >= 4
    )


@pytest.mark.parametrize(
    "rows,page,length",
    [
        *[(rows, 848, 8197) for rows in range(2, 9)],
        (5, 800, 8003),
        (8, 1616, 65536),
        (5, 1648, 131072),
        (5, 3296, 262144),
        # Small multi-tile q8 case for the online-softmax warp-state racecheck.
        (8, 3296, 512),
        # DFlash2 q8 uses the same repaired arithmetic at each context boundary.
        (8, 3296, 8192),
        (8, 3296, 65536),
        (8, 3296, 131072),
        (8, 3296, 262144),
        (8, 1728, 131072),
        (8, 3456, 262144),
    ],
)
def test_fp32_grouped_row_lengths_graph(rows, page, length):
    op = _native()
    torch.manual_seed(20260906)
    pages = (length + page - 1) // page
    capacity = pages * page
    q = torch.randn((rows, 6, 256), device="cuda", dtype=torch.float16)
    raw = torch.randn((2, capacity, 1, 256), device="cuda", dtype=torch.float16)
    encoded = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    backing = torch.empty((pages, 2, page, 1, 256), device="cuda", dtype=torch.uint8)
    k, v = backing.unbind(1)
    order = torch.randperm(pages, device="cuda")
    k[order] = encoded[0].reshape_as(k)
    v[order] = encoded[1].reshape_as(v)
    table = order.int()[None].contiguous()
    lengths = torch.arange(
        length - rows + 1, length + 1, device="cuda", dtype=torch.int32
    )
    initial = lengths.clone()
    out = torch.empty_like(q)
    ks, vs = 0.5, 1.25
    rk = encoded[0, :length, 0].view(torch.float8_e4m3fn).double() * ks
    rv = encoded[1, :length, 0].view(torch.float8_e4m3fn).double() * vs

    def call():
        return op(
            q,
            k,
            v,
            table,
            lengths,
            out=out,
            softmax_scale=0.0625,
            k_scale=ks,
            v_scale=vs,
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for state in ("live", "all_zero", "tail_zero", "first_zero", "restore"):
        lengths.copy_(initial)
        if state == "all_zero":
            lengths.zero_()
        elif state == "tail_zero":
            lengths[-1] = 0
        elif state == "first_zero":
            lengths[0] = 0
        graph.replay()
        score = q.transpose(0, 1).double() @ rk.T * 0.0625
        mask = torch.arange(length, device="cuda")[None] >= lengths[:, None]
        score.masked_fill_(mask[None], -torch.inf)
        expected = (score.softmax(-1).nan_to_num(0.0) @ rv).transpose(0, 1)
        assert bool(torch.isfinite(out).all())
        assert bool((out[lengths == 0] == 0).all())
        if state != "all_zero":
            relative_l2 = (out.double() - expected).norm() / expected.norm()
            assert float(relative_l2) < 0.001
            # A loose aggregate L2 gate admitted single-half probabilities
            # despite a repeatable model token flip. Bound the avoidable
            # arithmetic error relative to the unavoidable FP16 output floor.
            rounding_floor = (
                expected.half().double() - expected
            ).norm() / expected.norm()
            assert float(relative_l2) <= 1.02 * float(rounding_floor) + 2e-6
        if state == "live":
            original = out.clone()
        elif state == "restore":
            assert torch.equal(out, original)


def test_fp32_workspace_is_separate_from_legacy_half_workspace():
    _native()
    from flash_attn_v100.flash_attn_interface import _get_grouped_verify_workspace

    q = torch.zeros((5, 6, 256), dtype=torch.float16, device="cuda")
    half = _get_grouped_verify_workspace(q)
    full = _get_grouped_verify_workspace(q, partial_dtype=torch.float32)
    assert half.partial_out.dtype == torch.float16
    assert full.partial_out.dtype == torch.float32
    assert half.partial_lse.shape == (80, 8, 6)
    assert full.partial_lse.shape == (80, 8, 6, 2)
    assert half.partial_out.data_ptr() != full.partial_out.data_ptr()
    assert _get_grouped_verify_workspace(q) is half
    assert _get_grouped_verify_workspace(q, partial_dtype=torch.float32) is full


def test_precision_workspace_is_separate_from_request_major_batch():
    _native()
    from flash_attn_v100.flash_attn_interface import _get_grouped_verify_workspace

    q = torch.zeros((16, 6, 256), dtype=torch.float16, device="cuda")
    batched = _get_grouped_verify_workspace(q, 2)
    wide = _get_grouped_verify_workspace(q, 1)
    precise = _get_grouped_verify_workspace(q[:5], partial_dtype=torch.float32)
    assert batched.partial_out.shape == (2, 80, 8, 6, 256)
    assert wide.partial_out.shape == (40, 16, 6, 256)
    assert precise.partial_out.shape == (80, 8, 6, 256)
    assert precise.partial_lse.shape == (80, 8, 6, 2)
    assert len({item.partial_out.data_ptr() for item in (batched, wide, precise)}) == 3


@pytest.mark.parametrize("padding", ["token", "block", "both"])
def test_eight_byte_kv_strides_match_contiguous_graph(padding):
    """The explicit-row route must not inherit q8's 16-byte paired loader."""
    op = _native()
    torch.manual_seed(20260907)
    rows, pages, page = 5, 3, 848
    token_stride = 264 if padding in ("token", "both") else 256
    block_stride = page * token_stride + (8 if padding in ("block", "both") else 0)
    shape = (pages, page, 1, 256)
    strides = (block_stride, token_stride, 256, 1)
    caches = []
    for _ in range(2):
        encoded = torch.randn(shape, device="cuda").to(torch.float8_e4m3fn)
        padded = torch.empty_strided(shape, strides, device="cuda", dtype=torch.uint8)
        padded.copy_(encoded.view(torch.uint8))
        assert padded.data_ptr() % 16 == 0
        assert any(s % 16 == 8 for s in padded.stride()[:2])
        caches.append(padded)
    q = torch.randn((rows, 6, 256), device="cuda", dtype=torch.float16)
    table = torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.arange(
        pages * page - rows, pages * page, device="cuda", dtype=torch.int32
    )
    reference, actual = torch.empty_like(q), torch.empty_like(q)
    contiguous = [cache.contiguous() for cache in caches]

    def call(kv, output):
        op(q, *kv, table, lengths, out=output, softmax_scale=0.0625)

    call(contiguous, reference)
    call(caches, actual)
    assert torch.equal(actual, reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(caches, actual)
    for empty in (True, False):
        lengths[-1] = 0 if empty else pages * page - 1
        call(contiguous, reference)
        graph.replay()
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("offset", [1, 4, 7])
def test_contiguous_unaligned_q_matches_aligned_graph(offset):
    op = _native()
    torch.manual_seed(20260907)
    rows, page = 5, 848
    shape = (rows, 6, 256)
    backing = torch.full(
        (rows * 6 * 256 + 16,), -17.0, dtype=torch.float16, device="cuda"
    )
    q = backing[offset : offset + rows * 6 * 256].view(shape)
    assert q.is_contiguous() and q.data_ptr() % 16 != 0
    aligned = torch.randn(shape, dtype=torch.float16, device="cuda")
    q.copy_(aligned)
    kv = torch.randn((2, 3, page, 1, 256), device="cuda").to(torch.float8_e4m3fn)
    k, v = kv.view(torch.uint8).unbind(0)
    table = torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.arange(3 * page - rows, 3 * page, dtype=torch.int32, device="cuda")
    original_lengths = lengths.clone()
    expected, actual = torch.empty_like(aligned), torch.empty_like(aligned)

    def call(query, output):
        op(query, k, v, table, lengths, out=output, softmax_scale=0.0625)

    call(aligned, expected)
    call(q, actual)
    assert torch.equal(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(q, actual)
    for zero_state in ("none", "tail", "all", "restore"):
        aligned.normal_()
        q.copy_(aligned)
        lengths.copy_(original_lengths)
        if zero_state == "tail":
            lengths[-1] = 0
        elif zero_state == "all":
            lengths.zero_()
        call(aligned, expected)
        graph.replay()
        assert torch.equal(actual, expected)
        assert bool((backing[:offset] == -17).all())
        assert bool((backing[offset + rows * 6 * 256 :] == -17).all())


def test_scaled_residual_value_operand_is_exact_for_all_finite_e4m3():
    """The residual product's inverse scale must not re-quantize V."""
    values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).half()
    values = values[torch.isfinite(values)]
    assert torch.equal((values / 2048).double(), values.double() / 2048)


@pytest.mark.parametrize(
    "value",
    [2**-9, 1.0, 448.0, -(2**-9), -1.0, -448.0],
    ids=["min", "unit", "max", "negative_min", "negative_unit", "negative_max"],
)
def test_small_probability_residual_survives_fp16_storage(value):
    """Expose lost low-P residuals before final FP16 output rounding hides them."""
    op = _native()
    from flash_attn_v100.flash_attn_interface import _get_grouped_verify_workspace

    rows, page, length = 5, 800, 5120
    pages = (length + page - 1) // page
    capacity = pages * page
    q = torch.zeros((rows, 6, 256), device="cuda", dtype=torch.float16)
    q[..., 0] = 1
    keys = torch.zeros((capacity, 1, 256), device="cuda", dtype=torch.float16)
    keys[..., 0] = -8
    keys[::32, :, 0] = 0
    values = torch.full_like(keys, value)
    values[::32] = 0
    backing = torch.empty((pages, 2, page, 1, 256), device="cuda", dtype=torch.uint8)
    k, v = backing.unbind(1)
    k.copy_(keys.to(torch.float8_e4m3fn).view(torch.uint8).reshape_as(k))
    v.copy_(values.to(torch.float8_e4m3fn).view(torch.uint8).reshape_as(v))
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None]
    lengths = torch.full((rows,), length, device="cuda", dtype=torch.int32)
    op(q, k, v, table, lengths, out=torch.empty_like(q), softmax_scale=1.0)
    workspace = _get_grouped_verify_workspace(q, partial_dtype=torch.float32)
    # All 80 partitions contain two identical N32 tiles. QK is exact here:
    # one score is zero with V=0; the other 31 scores are -8 with V=value.
    probability = torch.tensor(-8.0, dtype=torch.float64).exp()
    expected = value * 31 * probability / (1 + 31 * probability)
    actual = workspace.partial_out[:, :rows].double()
    actual = actual / workspace.partial_lse[:, :rows, :, 1].double()[..., None]
    relative_error = (actual - expected).abs().max() / expected.abs()
    assert bool(torch.isfinite(actual).all())
    assert float(relative_error) < 3e-6


@pytest.mark.parametrize("value_bias", [0.0, 4.0])
def test_long_context_fp32_partials_before_output_rounding(value_bias):
    """Do not let the final FP16 rounding floor hide long PV accumulation loss."""
    op = _native()
    from flash_attn_v100.flash_attn_interface import _get_grouped_verify_workspace

    torch.manual_seed(20260908)
    rows, length, page = 5, 262144, 3296
    pages = (length + page - 1) // page
    capacity = pages * page
    q = torch.randn((rows, 6, 256), device="cuda", dtype=torch.float16) * 0.5
    raw = torch.randn((2, capacity, 1, 256), device="cuda", dtype=torch.float16)
    # Real V features can have a nonzero mean. Zero-mean random V alone
    # hides the biased error from repeatedly feeding a large C back to MMA.
    raw[1].add_(torch.linspace(-value_bias, value_bias, 256, device="cuda"))
    encoded = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    backing = torch.empty((pages, 2, page, 1, 256), device="cuda", dtype=torch.uint8)
    k, v = backing.unbind(1)
    order = torch.randperm(pages, device="cuda")
    k[order] = encoded[0].reshape_as(k)
    v[order] = encoded[1].reshape_as(v)
    table = order.int()[None].contiguous()
    lengths = torch.arange(
        length - rows + 1, length + 1, device="cuda", dtype=torch.int32
    )
    op(q, k, v, table, lengths, out=torch.empty_like(q), softmax_scale=0.0625)
    workspace = _get_grouped_verify_workspace(q, partial_dtype=torch.float32)
    actual = workspace.partial_out[:, :rows].double()
    actual = actual / workspace.partial_lse[:, :rows, :, 1].double()[..., None]
    keys = encoded[0, :length, 0].view(torch.float8_e4m3fn).double()
    values = encoded[1, :length, 0].view(torch.float8_e4m3fn).double()
    scores = q.transpose(0, 1).double() @ keys.T * 0.0625
    mask = torch.arange(length, device="cuda")[None] >= lengths[:, None]
    scores.masked_fill_(mask[None], -torch.inf)
    tiles = (length + 31) // 32
    base, extra = divmod(tiles, 80)
    expected = []
    for split in range(80):
        start = (split * base + min(split, extra)) * 32
        end = min(length, start + (base + (split < extra)) * 32)
        expected.append(
            (scores[..., start:end].softmax(-1) @ values[start:end]).transpose(0, 1)
        )
    reference = torch.stack(expected)
    relative_l2 = (actual - reference).norm() / reference.norm()
    assert torch.isfinite(actual).all()
    assert float(relative_l2) < 3e-6


@pytest.mark.parametrize("length", [131072, 262144])
def test_uniform_attention_midpoint_has_one_final_normalization(length):
    """Avoid partition-normalization drift at an exactly representable midpoint."""
    op = _native()
    rows, page = 5, 848
    pages = (length + page - 1) // page
    q = torch.zeros((rows, 6, 256), device="cuda", dtype=torch.float16)
    cache = torch.zeros((pages, 2, page, 1, 256), device="cuda", dtype=torch.uint8)
    k, v = cache.unbind(1)
    values = torch.ones((pages * page, 1, 256), device="cuda", dtype=torch.float16)
    values[::256] = 1.125
    v.copy_(values.to(torch.float8_e4m3fn).view(torch.uint8).reshape_as(v))
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None]
    lengths = torch.full((rows,), length, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    op(q, k, v, table, lengths, out=out, softmax_scale=0.0625)
    # Uniform scores, exactly 1/256 of values are 1.125: the mean is
    # 1 + 2**-11. Round-to-nearest-even selects FP16 1, not 1 + 2**-10.
    assert torch.equal(out, torch.ones_like(out))
