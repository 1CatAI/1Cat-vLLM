# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped (is_bmm) block-FP8 layers on the QPN8 kernel, host-side logic.

DeepSeek-V4's wo_a on Turing used to be dequantized to FP16 at load (twice the
bytes, cuBLAS per call). The QPN8 kernel now keeps the FP8 pack, one pack per
group, and launches all groups at once for decode rows. The native operators
are replaced by recorders here; the GPU numerics live in
tests/kernels/quantization/test_sm70_qpn8_grouped_bmm.py.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.qpn import fp8_block
from vllm.model_executor.kernels.linear.qpn.fp8_block import (
    QPN8Fp8BlockScaledMMLinearKernel as QPN8,
)
from vllm.model_executor.layers.quantization import fp8

GROUPS = 4
K = 256
N_GROUP = 128


def _stacked_pack() -> tuple[torch.Tensor, torch.Tensor]:
    """Group g's codes are all g + 1; its scales are g + 1 in every block."""
    codes = torch.stack(
        [torch.full((K, N_GROUP), g + 1, dtype=torch.uint8) for g in range(GROUPS)]
    )
    scales = torch.stack(
        [
            torch.full((K // 128, N_GROUP // 32), g + 1, dtype=torch.float16)
            for g in range(GROUPS)
        ]
    )
    return codes, scales


def _record_native(monkeypatch) -> list[tuple[str, tuple[int, ...], int]]:
    """Fake native ops: out = row sum of x + the group marker of the codes."""
    calls: list[tuple[str, tuple[int, ...], int]] = []

    def fake(name):
        def op(out, x, codes, scales, *args):
            marker = int(codes.flatten()[0])
            calls.append((name, tuple(codes.shape), marker))
            if codes.dim() == 3:
                groups, k_dim, _ = codes.shape
                x_groups = x.view(x.shape[0], groups, k_dim).float().sum(-1)
                markers = codes[:, 0, 0].float()
                out.view(x.shape[0], groups, -1).copy_(
                    (x_groups + markers)[..., None].expand(
                        -1, -1, out.shape[1] // groups
                    )
                )
            else:
                out.copy_((x.float().sum(-1, keepdim=True) + marker).expand_as(out))

        return op

    monkeypatch.setattr(fp8_block.sm70_ops, "fp8_qpn8_gemm_sm70_out", fake("gemm"))
    monkeypatch.setattr(fp8_block.sm70_ops, "fp8_qpn8_mma_sm75_out", fake("mma"))
    return calls


@pytest.mark.parametrize("rows", [1, 5, 8])
def test_decode_rows_are_one_launch_over_all_groups(monkeypatch, rows) -> None:
    calls = _record_native(monkeypatch)
    codes, scales = _stacked_pack()
    x = torch.randn(rows, GROUPS * K).half()
    out = x.new_empty(rows, GROUPS * N_GROUP)

    fp8_block._qpn8_native_grouped_linear_write(out, x, codes, scales, 16, 2, False)

    assert calls == [("gemm", (GROUPS, K, N_GROUP), 1)]
    expected = x.view(rows, GROUPS, K).float().sum(-1) + torch.arange(1, GROUPS + 1)
    assert torch.equal(
        out.view(rows, GROUPS, N_GROUP)[..., 0].float(), expected.half().float()
    )


@pytest.mark.parametrize("rows", [9, 64, 512])
def test_prefill_rows_run_each_group_on_its_own_slices(monkeypatch, rows) -> None:
    calls = _record_native(monkeypatch)
    codes, scales = _stacked_pack()
    x = torch.randn(rows, GROUPS * K).half()
    out = x.new_empty(rows, GROUPS * N_GROUP)

    fp8_block._qpn8_native_grouped_linear_write(
        out, x, codes, scales, 16, 2, False, sm75_mma=True
    )

    assert calls == [("mma", (K, N_GROUP), g + 1) for g in range(GROUPS)]
    expected = x.view(rows, GROUPS, K).float().sum(-1) + torch.arange(1, GROUPS + 1)
    assert torch.equal(
        out.view(rows, GROUPS, N_GROUP)[..., 0].float(), expected.half().float()
    )
    # Every column of a group holds that group's value: nothing is left unwritten.
    assert torch.equal(
        out.view(rows, GROUPS, N_GROUP),
        out.view(rows, GROUPS, N_GROUP)[..., :1].expand(-1, -1, N_GROUP),
    )


def test_empty_batch_launches_nothing(monkeypatch) -> None:
    calls = _record_native(monkeypatch)
    codes, scales = _stacked_pack()
    x = torch.empty(0, GROUPS * K).half()
    out = x.new_empty(0, GROUPS * N_GROUP)

    fp8_block._qpn8_native_grouped_linear_write(out, x, codes, scales, 16, 2, False)

    assert calls == []


def test_split_groups_gives_each_group_its_own_bytes_and_scale_plane() -> None:
    groups, k_dim, n_dim = 4, 256, 512
    n_group = n_dim // groups
    # Whole-matrix pack: codes tile-major over 32 rows, scales [K / 128, N / 32].
    codes = (
        torch.arange(k_dim * n_dim, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(k_dim, n_dim)
    )
    scales = (
        torch.arange((k_dim // 128) * (n_dim // 32))
        .half()
        .view(k_dim // 128, n_dim // 32)
    )

    split_codes, split_scales = QPN8._split_groups(codes, scales, groups, k_dim)

    assert split_codes.shape == (groups, k_dim, n_group)
    flat = codes.flatten()
    for group in range(groups):
        run = flat[group * k_dim * n_group : (group + 1) * k_dim * n_group]
        assert torch.equal(split_codes[group].flatten(), run)
        tiles = slice(group * n_group // 32, (group + 1) * n_group // 32)
        assert torch.equal(split_scales[group], scales[:, tiles])
    assert split_scales.is_contiguous()


@pytest.mark.parametrize("groups,n_dim", [(0, 1024), (3, 1024), (8, 512)])
def test_group_count_must_split_rows_into_scale_blocks(groups, n_dim) -> None:
    layer = SimpleNamespace(is_bmm=True, bmm_batch_size=groups)
    with pytest.raises(ValueError, match="grouped BMM"):
        QPN8._bmm_groups(layer, n_dim)


def test_plain_layer_has_no_groups() -> None:
    assert QPN8._bmm_groups(SimpleNamespace(), 1024) == 0
    assert QPN8._bmm_groups(SimpleNamespace(is_bmm=False), 1024) == 0


def test_apply_grouped_calls_the_opaque_op_and_shapes_the_result(
    monkeypatch,
) -> None:
    codes, scales = _stacked_pack()
    layer = SimpleNamespace(
        _sm70_block_fp8_qpn8_packed_codes=codes,
        _sm70_block_fp8_qpn8_packed_scales=scales,
        _qpn8_cfg=(16, 2, False),
        _qpn8_sm75_mma=True,
        _qpn8_bmm_groups=GROUPS,
    )
    seen: dict[str, object] = {}

    def invoke(operation, *args):
        seen["operation"] = operation
        seen["x_shape"] = tuple(args[0].shape)
        seen["tail"] = args[3:]
        return torch.zeros(args[0].shape[0], GROUPS * N_GROUP, dtype=torch.float16)

    kernel = QPN8.__new__(QPN8)
    kernel.native_ops = SimpleNamespace(invoke=invoke, arguments=("token",))
    x = torch.zeros(3, GROUPS, K, dtype=torch.float16)

    out = kernel.apply_weights(layer, x)

    assert out.shape == (3, GROUPS, N_GROUP)
    assert seen["operation"] is torch.ops.sm70_fp8.qpn8_native_grouped_linear
    assert seen["x_shape"] == (3, GROUPS * K)
    assert seen["tail"] == (16, 2, False, ("token",), True)
    assert not kernel.supports_direct_output(layer)
    with pytest.raises(ValueError, match="must end in"):
        kernel.apply_weights(layer, torch.zeros(3, GROUPS + 1, K, dtype=torch.float16))


def test_qpn8_serves_the_grouped_layer_itself_instead_of_dequantizing() -> None:
    """The old load dequantized an is_bmm layer to FP16 and dropped the kernel."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros(GROUPS * 128, K, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.ones(GROUPS, K // 128, dtype=torch.float32), requires_grad=False
    )
    layer.orig_dtype = torch.float16
    layer.is_bmm = True
    layer.bmm_batch_size = GROUPS
    packed: list[torch.nn.Module] = []

    method = fp8.Fp8LinearMethod.__new__(fp8.Fp8LinearMethod)
    method.use_marlin = False
    method.use_sm70_fp8_turbomind = False
    method.use_sm70_dequant_fallback = False
    method.block_quant = True
    method.act_q_static = False
    method.weight_block_size = [128, 128]
    method.fp8_linear = QPN8.__new__(QPN8)
    method.fp8_linear.process_weights_after_loading = packed.append

    method.process_weights_after_loading(layer)

    assert packed == [layer]
    assert not getattr(layer, "dequantized_bmm", False)
    assert layer.weight.dtype == torch.float8_e4m3fn
