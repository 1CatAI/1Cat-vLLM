# SPDX-License-Identifier: Apache-2.0
"""SM70 software NVFP4 admission + dequant smoke tests.

These drive the shipped ModelOpt capability gate and the real
`dequantize_to_dtype` helper used by EmulationNvFp4LinearKernel on
Volta (no native FP4 tensor cores).
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)


def test_modelopt_nvfp4_min_capability_admits_sm70():
    """ModelOpt NVFP4 must load on V100/SM70 (software dequant path)."""
    assert ModelOptNvFp4Config.get_min_capability() <= 70


def test_dequantize_to_dtype_roundtrip_shape_and_nonzero():
    """Exercise the real dequant helper with packed E2M1-like bytes.

    Packs two nibble values per byte; with unit scales the dequant must
    produce a finite fp16 tensor of expanded K dimension.
    """
    out_f, in_f = 8, 32  # in_f multiple of group_size 16
    # packed weight: [out, in//2]
    packed = torch.zeros(out_f, in_f // 2, dtype=torch.uint8)
    # set first two nibbles to known E2M1 codes: 0x1 (0.5) and 0x2 (1.0) -> byte 0x21
    packed[:, 0] = 0x21
    weight_scale = torch.ones(out_f, in_f // 16, dtype=torch.float8_e4m3fn)
    # global scale as ModelOpt-style amax/(6*448) inverse-ish unit
    global_scale = torch.tensor(1.0, dtype=torch.float32)

    deq = dequantize_to_dtype(
        packed,
        weight_scale,
        global_scale,
        torch.float16,
        block_size=16,
        swizzle=False,
    )
    assert deq.shape == (out_f, in_f)
    assert deq.dtype == torch.float16
    assert torch.isfinite(deq).all()
    # first two columns should be non-zero for our packed nibbles
    assert deq[:, 0].abs().sum() > 0
    assert deq[:, 1].abs().sum() > 0


def test_compressed_tensors_w4a16_fp4_min_capability_sm70():
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4 import (
        CompressedTensorsW4A16Fp4,
    )

    assert CompressedTensorsW4A16Fp4.get_min_capability() <= 70
