# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device GGUF expert transcoding must equal the host codecs byte for byte."""

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization import gguf_device_transcode as dev
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_lut_transcode import (
    transcode_lut4,
)
from vllm.model_executor.layers.quantization.gguf_transcode import transcode_affine
from vllm.transformers_utils.gguf_tensor_reader import quant_size

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def packed_rows(weight_type, rows, k, seed):
    """Random blocks with finite, wide-ranging FP16 super-scales."""
    rng = np.random.default_rng(seed)
    block, size = quant_size(weight_type)
    data = rng.integers(0, 256, (rows, k // block, size), dtype=np.uint8)
    d = (
        rng.standard_normal((rows, k // block))
        * 2.0 ** rng.integers(-14, 4, (rows, k // block))
    ).astype(np.float16)
    data[..., :2] = d.view(np.uint8).reshape(rows, k // block, 2)
    return data.reshape(rows, -1)


@pytest.mark.parametrize("weight_type", sorted(dev.DEVICE_LATTICE_TYPES))
@pytest.mark.parametrize("rows,k", [(32, 256), (96, 2560)])
def test_lattice_storage_matches_host(weight_type, rows, k):
    data = packed_rows(weight_type, rows, k, weight_type * 1000 + k)
    codes, stats = transcode_lattice(data, weight_type).mma884_storage()
    stats = stats.view({4: np.int32, 8: np.int64}[stats.itemsize])
    actual_codes, actual_stats = dev.lattice_storage(
        torch.from_numpy(data).cuda(), weight_type
    )
    np.testing.assert_array_equal(actual_codes.cpu().numpy(), codes)
    assert (
        actual_stats.dtype
        == {np.int32: torch.int32, np.int64: torch.int64}[stats.dtype.type]
    )
    np.testing.assert_array_equal(actual_stats.cpu().numpy(), stats)


@pytest.mark.parametrize("weight_type", sorted(dev.DEVICE_LUT4_TYPES))
@pytest.mark.parametrize("rows,k", [(32, 256), (64, 640)])
def test_lut4_codes_match_host(weight_type, rows, k):
    if k % quant_size(weight_type)[0]:
        pytest.skip("shape does not hold whole blocks")
    data = packed_rows(weight_type, rows, k, weight_type * 1000 + k)
    expected = transcode_lut4(data, weight_type)
    codes, scales = dev.lut4_codes(torch.from_numpy(data).cuda(), weight_type)
    np.testing.assert_array_equal(codes.cpu().numpy(), expected.codes)
    np.testing.assert_array_equal(
        scales.cpu().numpy().view(np.uint16), expected.scales.view(np.uint16)
    )


@pytest.mark.parametrize("rows,k", [(32, 640), (64, 2560)])
def test_q2_0_codes_match_host(rows, k):
    data = packed_rows(42, rows, k, k)
    expected = transcode_affine(data, 42)
    codes, scales, mins = dev.affine_codes(torch.from_numpy(data).cuda(), 42)
    np.testing.assert_array_equal(codes.cpu().numpy(), expected.codes)
    for actual, ref in ((scales, expected.scales), (mins, expected.mins)):
        np.testing.assert_array_equal(
            actual.cpu().numpy().view(np.uint16), ref.view(np.uint16)
        )


def test_lattice_scale_overflow_is_rejected_like_host():
    data = packed_rows(21, 32, 256, 7)
    data.reshape(32, -1, 110)[0, 0, :2] = np.frombuffer(
        np.float16(60000).tobytes(), dtype=np.uint8
    )
    data.reshape(32, -1, 110)[0, 0, 106:] = 0xFF
    with pytest.raises(ValueError, match="overflows"):
        transcode_lattice(data, 21)
    with pytest.raises(ValueError, match="overflows"):
        dev.lattice_storage(torch.from_numpy(data).cuda(), 21)


native = pytest.mark.skipif(
    not all(
        hasattr(torch.ops._C, op)
        for op in (
            "gguf_lattice_sm70_prepare",
            "gguf_lut4_sm70_prepare",
            "gguf_affine_sm70_prepare",
        )
    ),
    reason="GGUF SM70 prepare operators required",
)


@native
@pytest.mark.parametrize(
    "weight_type,rows,k,axis",
    [
        (21, 128, 512, 0),
        (22, 128, 512, 0),
        (18, 128, 512, 0),
        (23, 128, 512, 0),
        (20, 256, 128, 1),
        (42, 256, 640, 1),  # TP4 K slice cuts Q2_0 blocks
    ],
)
@pytest.mark.parametrize("size", [1, 2, 4])
def test_expert_bank_device_matches_host(weight_type, rows, k, axis, size):
    from vllm.model_executor.layers.quantization.gguf_turbomind_moe import (
        GGUFExpertBank,
    )

    experts = 6
    data = np.stack(
        [
            packed_rows(weight_type, rows, k, 100 * e + weight_type)
            for e in range(experts)
        ]
    )
    retain_raw = axis == 0 and weight_type in (18, 21, 22)
    for rank in range(size):
        banks = []
        for device_transcode in (False, True):
            bank = GGUFExpertBank(
                weight_type,
                experts,
                torch.device("cuda"),
                torch.float16,
                retain_raw=retain_raw,
                device_transcode=device_transcode,
            )
            bank.DEVICE_CHUNK = 4  # exercise a partial last chunk
            for e in reversed(range(experts)):
                bank.add(e, torch.from_numpy(data[e]), rank, size, axis)
            bank.finalize()
            banks.append(bank)
        host, device = banks
        assert device.device_transcode
        for name in ("weights", "stats", "raw_weights"):
            a, b = getattr(host, name, None), getattr(device, name, None)
            assert (a is None) == (b is None), name
            if a is not None:
                assert a.dtype == b.dtype and a.shape == b.shape
                assert torch.equal(a, b), name
        for attr in ("n", "k", "group", "decoder"):
            assert getattr(host, attr, None) == getattr(device, attr, None), attr
        assert host.raw_capabilities == device.raw_capabilities


@native
def test_expert_bank_device_rejects_incomplete_bank():
    from vllm.model_executor.layers.quantization.gguf_turbomind_moe import (
        GGUFExpertBank,
    )

    bank = GGUFExpertBank(
        21, 4, torch.device("cuda"), torch.float16, device_transcode=True
    )
    data = packed_rows(21, 32, 256, 1)
    bank.add(0, torch.from_numpy(data), 0, 1, 0)
    with pytest.raises(ValueError, match="Incomplete"):
        bank.finalize()
