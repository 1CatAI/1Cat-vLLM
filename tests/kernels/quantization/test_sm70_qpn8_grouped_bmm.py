# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped block-FP8 BMM on the QPN8 kernel (DeepSeek-V4 wo_a on Turing).

The old Turing path dequantized the weight to FP16 at load and ran cuBLAS. The
grouped QPN8 route reads the FP8 pack directly: one launch over all groups for
decode rows, the per-group M ladder of the dense linears above that. These tests
need a Volta or Turing card and are the numerical gate; the dispatch logic that
needs no GPU is in tests/quantization/test_qpn8_grouped_bmm.py.

Oracle: the FP32 product of the activations with the exactly dequantized weight
(FP8 to FP16 is exact; the scales here are powers of two). The kernel
accumulates in FP32 and rounds once to FP16, so the error is the FP16 output
rounding (2**-11 relative), not an accumulation drift.
"""

import pytest
import torch

from vllm._sm70.policy import NativeBindings
from vllm.config.kernel import Sm70Fp8Config
from vllm.config.sm70_native import capture_linear_native_config
from vllm.model_executor.kernels.linear.qpn.fp8_block import (
    QPN8Fp8BlockScaledMMLinearKernel,
)
from vllm.utils.torch_utils import set_high_precision_cuda_matmul_defaults

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((7, 0), (7, 5)),
    reason="requires an SM70 or SM75 GPU",
)

BLOCK = 128
# DeepSeek-V4-Flash wo_a (one stage, no TP): 8 groups of 1024 rows over K = 4096.
PRODUCTION = (8, 1024, 4096)
SMALL = (2, 256, 512)


@pytest.fixture(autouse=True)
def worker_matmul_policy():
    """The numerical policy a worker sets before loading (gpu_worker.py): no
    reduced-precision FP16 reductions. Without it cuBLAS reduces split-K partial
    sums in FP16 and the large-M products drift from the FP32 oracle. The flags
    are restored afterwards; the split-K half is read-only and is written back
    together with its reduction flag as a tuple."""
    matmul = torch.backends.cuda.matmul
    saved: dict[str, object] = {}
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        split_k_name = f"{name}_split_k"
        saved[name] = (
            (getattr(matmul, name), getattr(matmul, split_k_name))
            if hasattr(matmul, split_k_name)
            else getattr(matmul, name)
        )
    if hasattr(matmul, "allow_fp16_accumulation"):
        saved["allow_fp16_accumulation"] = matmul.allow_fp16_accumulation
    set_high_precision_cuda_matmul_defaults()
    yield
    for name, value in saved.items():
        setattr(matmul, name, value)


def _weights(groups, n_group, k, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    weight = (
        torch.randn(groups * n_group, k, device="cuda", generator=generator) * 0.5
    ).to(torch.float8_e4m3fn)
    exponents = torch.randint(
        -9,
        -4,
        (groups * n_group // BLOCK, k // BLOCK),
        device="cuda",
        generator=generator,
    )
    scales = torch.pow(2.0, exponents.float())
    dense = weight.float() * scales.repeat_interleave(BLOCK, 0).repeat_interleave(
        BLOCK, 1
    )
    return weight, scales, dense.view(groups, n_group, k)


def _reference(x, dense):
    return torch.einsum("mgk,gnk->mgn", x.float(), dense)


def _assert_close(out, expected):
    torch.testing.assert_close(
        out.float(),
        expected,
        rtol=1e-3,
        atol=1e-3 * expected.abs().mean().item(),
    )


def _kernel():
    kernel = object.__new__(QPN8Fp8BlockScaledMMLinearKernel)
    kernel.policy = Sm70Fp8Config()
    kernel.policy.resolve()
    kernel.native_ops = NativeBindings(capture_linear_native_config("fp8").values)
    return kernel


def _layer(weight, scales, groups):
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scales, requires_grad=False)
    )
    layer.orig_dtype = torch.float16
    layer.is_bmm = True
    layer.bmm_batch_size = groups
    return layer


@pytest.mark.parametrize("shape", [SMALL, PRODUCTION])
@pytest.mark.parametrize("split_k", [8, 16])
@pytest.mark.parametrize("rows", range(1, 9))
@torch.no_grad()
def test_grouped_launch_matches_per_group_launches_bit_for_bit(shape, split_k, rows):
    """One launch over all groups does what G separate launches do, per element.

    Old code: the operator rejects a 3D code tensor ("expected 2D tensors").
    """
    import vllm._C  # noqa: F401

    groups, n_group, k = shape
    weight, scales, dense = _weights(groups, n_group, k)
    codes, group_scales = torch.ops._C.fp8_qpn8_prepare_sm70(weight, scales)
    stacked_codes, stacked_scales = QPN8Fp8BlockScaledMMLinearKernel._split_groups(
        codes, group_scales, groups, k
    )
    x = torch.randn(rows, groups * k, device="cuda", dtype=torch.float16)

    grouped = x.new_empty(rows, groups * n_group)
    torch.ops._C.fp8_qpn8_gemm_sm70_out(
        grouped, x, stacked_codes, stacked_scales, split_k, 2, True, False
    )

    separate = x.new_empty(rows, groups * n_group)
    for group in range(groups):
        per_group_codes, per_group_scales = torch.ops._C.fp8_qpn8_prepare_sm70(
            weight[group * n_group : (group + 1) * n_group].contiguous(),
            scales[
                group * n_group // BLOCK : (group + 1) * n_group // BLOCK
            ].contiguous(),
        )
        # The whole-matrix pack split per group is the per-group pack.
        assert torch.equal(per_group_codes, stacked_codes[group])
        assert torch.equal(per_group_scales, stacked_scales[group])
        part = x.new_empty(rows, n_group)
        torch.ops._C.fp8_qpn8_gemm_sm70_out(
            part,
            x[:, group * k : (group + 1) * k].contiguous(),
            per_group_codes,
            per_group_scales,
            split_k,
            2,
            True,
            False,
        )
        separate[:, group * n_group : (group + 1) * n_group] = part

    assert torch.equal(grouped.view(torch.int16), separate.view(torch.int16))
    _assert_close(
        grouped.view(rows, groups, n_group), _reference(x.view(rows, groups, k), dense)
    )


@pytest.mark.parametrize("shape", [SMALL, PRODUCTION])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 64, 512, 600, 1024])
@torch.no_grad()
def test_layer_matches_fp32_reference_on_every_m_branch(shape, rows):
    """Load, then apply: decode launch, per-group tensor-core GEMM, dense prefill.

    Old code: the layer was dequantized to FP16 (``dequantized_bmm``) and its
    weight was not a uint8 pack, so the first assertion fails.
    """
    import vllm._C  # noqa: F401

    groups, n_group, k = shape
    weight, scales, dense = _weights(groups, n_group, k)
    layer = _layer(weight, scales, groups)
    kernel = _kernel()
    kernel.process_weights_after_loading(layer)

    codes = layer._sm70_block_fp8_qpn8_packed_codes
    assert codes.dtype == torch.uint8 and tuple(codes.shape) == (groups, k, n_group)
    assert layer.weight.numel() == 0 and not hasattr(layer, "dequantized_bmm")

    x = torch.randn(rows, groups, k, device="cuda", dtype=torch.float16)
    out = kernel.apply_weights(layer, x)

    assert out.shape == (rows, groups, n_group) and out.dtype == torch.float16
    _assert_close(out, _reference(x, dense))


@pytest.mark.parametrize("rows", [129, 256, 600, 1024])
@torch.no_grad()
def test_dense_bmm_rows_equal_the_former_einsum_on_the_dequantized_weight(rows):
    """Above the grouped-launch range the result is the old route's result.

    The old route multiplied the FP16-dequantized weight with torch.einsum; the
    new one dequantizes into the shared workspace and runs one
    batched product. The decoded weights are identical, so with the worker's
    matmul policy (fixture) the outputs should agree bit for
    bit. If cuBLAS picks a different kernel for the transposed operand layout and
    this fails by a few ulp, compare against the FP32 oracle instead (the layer
    test above) and record the difference.

    Old code: the layer holds an FP16 weight and no packed codes.
    """
    import vllm._C  # noqa: F401

    groups, n_group, k = PRODUCTION
    weight, scales, dense = _weights(groups, n_group, k)
    layer = _layer(weight, scales, groups)
    kernel = _kernel()
    kernel.process_weights_after_loading(layer)
    x = torch.randn(rows, groups, k, device="cuda", dtype=torch.float16)

    out = kernel.apply_weights(layer, x)

    old_weight = (dense.half()).contiguous()  # exact: FP8 * power of two in FP16
    expected = torch.einsum("...gk,grk->...gr", x, old_weight)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@torch.no_grad()
def test_layer_resident_bytes_are_the_fp8_pack():
    """The VRAM claim: FP8 bytes plus scales, no FP16 copy of the weight."""
    import vllm._C  # noqa: F401

    groups, n_group, k = PRODUCTION
    weight, scales, _ = _weights(groups, n_group, k)
    layer = _layer(weight, scales, groups)
    del weight, scales
    _kernel().process_weights_after_loading(layer)

    resident = sum(
        buffer.numel() * buffer.element_size() for buffer in layer.buffers()
    ) + sum(p.numel() * p.element_size() for p in layer.parameters())
    fp8_bytes = groups * n_group * k
    assert resident < 1.01 * fp8_bytes  # scales are 1/16384 of the codes
    assert resident < 0.6 * (2 * fp8_bytes)  # the FP16 copy would be 2 * fp8_bytes


@torch.no_grad()
def test_decode_launch_replays_in_a_cuda_graph_on_changed_input():
    import vllm._C  # noqa: F401

    groups, n_group, k = PRODUCTION
    weight, scales, dense = _weights(groups, n_group, k)
    layer = _layer(weight, scales, groups)
    kernel = _kernel()
    kernel.process_weights_after_loading(layer)
    x = torch.randn(6, groups, k, device="cuda", dtype=torch.float16)
    kernel.apply_weights(layer, x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernel.apply_weights(layer, x)
    for scale in (0.25, -0.5):
        x.mul_(scale)
        graph.replay()
        _assert_close(captured, _reference(x, dense))
