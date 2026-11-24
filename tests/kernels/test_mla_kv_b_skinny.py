# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny kv_b products of MLA decode (kernel_config.mla_kv_b_skinny)."""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.utils import requires_tensor_core_triton
from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.attention.mla_kv_b_skinny import (
    KV_B_SKINNY_MAX_ROWS,
    KV_B_SKINNY_MIN_ROWS,
    KV_B_SKINNY_TILES,
    kv_b_bmm_skinny,
    kv_b_skinny_tile,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import mma_probe

# GLM-5.3-Flash: 64 heads, nope 256, value 256, latent 512, rope 64 in q.
HEADS, NOPE, VALUE, LATENT, ROPE = 64, 256, 256, 512, 64
DEVICE_INDEX = 3

# --- host tests: configuration and dispatch -------------------------------


def _device(monkeypatch, capability: int) -> None:
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=None: DeviceCapability(capability // 10, capability % 10),
    )
    monkeypatch.setattr(mma_probe, "lowers_dot_to_mma", lambda capability: True)


def test_off_by_default():
    assert not KernelConfig().use_mla_kv_b_skinny(DEVICE_INDEX)


def test_needs_the_tensor_core_triton_setting():
    with pytest.raises(ValueError, match="triton_sm7x_tensor_cores"):
        KernelConfig(mla_kv_b_skinny=True).use_mla_kv_b_skinny(DEVICE_INDEX)


@pytest.mark.parametrize("capability, expected", [(70, True), (75, True), (80, False)])
def test_on_for_sm7x_cards(monkeypatch, capability, expected):
    _device(monkeypatch, capability)
    config = KernelConfig(mla_kv_b_skinny=True, triton_sm7x_tensor_cores=True)
    assert config.use_mla_kv_b_skinny(DEVICE_INDEX) is expected


def test_setting_keeps_the_compilation_hash():
    # The products run inside the opaque attention op: no graph depends on them.
    on = KernelConfig(mla_kv_b_skinny=True, triton_sm7x_tensor_cores=True)
    assert (
        on.compute_hash() == KernelConfig(triton_sm7x_tensor_cores=True).compute_hash()
    )


@pytest.mark.parametrize(
    "rows", [1, KV_B_SKINNY_MIN_ROWS, KV_B_SKINNY_MAX_ROWS, KV_B_SKINNY_MAX_ROWS + 1]
)
@pytest.mark.parametrize("skinny", [False, True])
def test_dispatch_takes_the_kernel_only_for_few_rows(monkeypatch, rows, skinny):
    calls: list[str] = []
    monkeypatch.setattr(
        mla_attention,
        "kv_b_bmm_skinny",
        lambda x, w, out, tile: calls.append("skinny"),
    )
    monkeypatch.setattr(
        torch, "bmm", lambda x, w, out: calls.append("cublas"), raising=True
    )
    tiles = KV_B_SKINNY_TILES[70] if skinny else None
    layer = SimpleNamespace(kv_b_skinny_tiles=tiles)
    x = torch.empty(HEADS, rows, 8)
    MLAAttention._kv_b_bmm(layer, x, x, x, 0)
    takes_kernel = skinny and KV_B_SKINNY_MIN_ROWS <= rows <= KV_B_SKINNY_MAX_ROWS
    assert calls == ["skinny" if takes_kernel else "cublas"]


# --- GPU tests --------------------------------------------------------------


def _tile(weight: torch.Tensor):
    """The tile the layer would pick for ``weight`` on this card."""
    capability = current_platform.get_device_capability(weight.device.index)
    return kv_b_skinny_tile(weight, capability.to_int())


def _kv_b_views(gen: torch.Generator):
    """W_UK_T and W_UV as MLAAttention.process_weights_after_loading builds them
    from the [heads * (nope + value), latent] weight of ColumnParallelLinear."""
    weight = (
        torch.randn(HEADS * (NOPE + VALUE), LATENT, generator=gen, device="cuda") * 0.05
    ).half()
    per_head = weight.T.view(LATENT, HEADS, NOPE + VALUE)
    w_uk, w_uv = per_head.split([NOPE, VALUE], dim=-1)
    return w_uk.permute(1, 2, 0), w_uv.transpose(0, 1)  # [H, 256, 512], [H, 512, 256]


def _cublas_reference_error(x, w, out_cublas, out_triton):
    """Error against an fp64 product, relative to sum |x||w| (rounding of the
    result to FP16 alone is up to 2^-11 of that; a drifting accumulation is not)."""
    xd, wd = x.double(), w.double()
    ref = torch.bmm(xd, wd)
    scale = torch.bmm(xd.abs(), wd.abs())
    return [((o.double() - ref).abs() / scale) for o in (out_cublas, out_triton)]


@requires_tensor_core_triton
@pytest.mark.parametrize("rows", list(range(1, KV_B_SKINNY_MAX_ROWS + 1)))
@pytest.mark.parametrize("layout", ["q_nope_to_latent", "latent_to_value"])
def test_matches_fp64_at_least_as_well_as_cublas(rows, layout):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    w_uk_t, w_uv = _kv_b_views(gen)
    if layout == "q_nope_to_latent":
        # (B, N, nope + rope) query, nope part transposed to (N, B, nope).
        q = torch.randn(rows, HEADS, NOPE + ROPE, generator=gen, device="cuda").half()
        x = q[..., :NOPE].transpose(0, 1)
        w = w_uk_t
        out_cublas = torch.empty(HEADS, rows, LATENT, dtype=torch.half, device="cuda")
        out_triton = torch.empty_like(out_cublas)
    else:
        # (B, N, latent) attention output; result written into (B, N * value)
        # through the transposed view of _v_up_proj.
        attn = torch.randn(rows, HEADS, LATENT, generator=gen, device="cuda").half()
        x = attn.transpose(0, 1)
        w = w_uv
        flat = [
            torch.full((rows, HEADS * VALUE), -7.0, dtype=torch.half, device="cuda")
            for _ in range(2)
        ]
        out_cublas = flat[0].view(-1, HEADS, VALUE).transpose(0, 1)
        out_triton = flat[1].view(-1, HEADS, VALUE).transpose(0, 1)

    torch.bmm(x, w, out=out_cublas)
    kv_b_bmm_skinny(x, w, out_triton, _tile(w))

    error_cublas, error_triton = _cublas_reference_error(x, w, out_cublas, out_triton)
    # Bound: FP16 rounding of the result plus a negligible FP32 accumulation term.
    assert error_triton.max() <= 2**-11 + 1e-6
    # And no worse than the kernel it replaces, in the mean and at the worst.
    assert error_triton.mean() <= error_cublas.mean() * 1.05
    assert error_triton.max() <= max(error_cublas.max() * 1.05, 2**-12)
    # Sanity of the oracle itself: the two products agree to FP16 rounding.
    torch.testing.assert_close(out_triton, out_cublas, rtol=2e-3, atol=2e-3)


@requires_tensor_core_triton
def test_writes_only_its_rows():
    gen = torch.Generator(device="cuda").manual_seed(0)
    w_uk_t, _ = _kv_b_views(gen)
    rows = 5
    x = torch.randn(HEADS, rows, NOPE, generator=gen, device="cuda").half()
    out = torch.full((HEADS, 16, LATENT), -7.0, dtype=torch.half, device="cuda")
    kv_b_bmm_skinny(x, w_uk_t, out[:, :rows], _tile(w_uk_t))
    assert (out[:, rows:] == -7.0).all()
    assert (out[:, :rows] != -7.0).any()


@requires_tensor_core_triton
def test_deterministic_between_calls():
    gen = torch.Generator(device="cuda").manual_seed(1)
    w_uk_t, _ = _kv_b_views(gen)
    x = torch.randn(HEADS, 7, NOPE, generator=gen, device="cuda").half()
    first = torch.empty(HEADS, 7, LATENT, dtype=torch.half, device="cuda")
    second = torch.empty_like(first)
    kv_b_bmm_skinny(x, w_uk_t, first, _tile(w_uk_t))
    kv_b_bmm_skinny(x, w_uk_t, second, _tile(w_uk_t))
    assert torch.equal(first, second)


@requires_tensor_core_triton
@pytest.mark.parametrize(
    "tile", sorted({t for pair in KV_B_SKINNY_TILES.values() for t in pair}), ids=str
)
def test_tiles_agree(tile):
    """Any tile gives the same answer up to FP32 summation order."""
    gen = torch.Generator(device="cuda").manual_seed(2)
    w_uk_t, _ = _kv_b_views(gen)
    x = torch.randn(HEADS, 6, NOPE, generator=gen, device="cuda").half()
    default = torch.empty(HEADS, 6, LATENT, dtype=torch.half, device="cuda")
    other = torch.empty_like(default)
    kv_b_bmm_skinny(x, w_uk_t, default, _tile(w_uk_t))
    kv_b_bmm_skinny(x, w_uk_t, other, tile)
    torch.testing.assert_close(other, default, rtol=2e-3, atol=2e-3)


def test_shape_checks_name_the_tile():
    weight = torch.empty(4, 100, 512, dtype=torch.half)
    with pytest.raises(ValueError, match="K=100"):
        kv_b_skinny_tile(weight, 70)
    with pytest.raises(ValueError, match="FP16"):
        kv_b_skinny_tile(weight.float(), 70)


def test_each_sm7x_card_has_its_tiles_and_others_none():
    weight = torch.empty(HEADS, NOPE, LATENT, dtype=torch.half)
    assert kv_b_skinny_tile(weight, 70) == KV_B_SKINNY_TILES[70][0]
    assert kv_b_skinny_tile(weight, 75) == KV_B_SKINNY_TILES[75][0]
    with pytest.raises(ValueError, match="capability 80"):
        kv_b_skinny_tile(weight, 80)
