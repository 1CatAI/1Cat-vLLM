# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
import torch


def _require_grouped_page4():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("grouped QSA page4 is SM70-only")
    interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    extension = interface.flash_attn_v100_cuda
    if not hasattr(extension, "grouped_sparse_page4_fwd"):
        pytest.skip("Flash-V100 extension lacks grouped sparse page4")
    return extension


def _grouped_page4_abi_version(extension) -> int:
    capability = getattr(extension, "grouped_sparse_page4_abi_version", None)
    if callable(capability):
        return int(capability())
    doc = getattr(extension.grouped_sparse_page4_fwd, "__doc__", "") or ""
    return 2 if "arg11:" in doc else 1 if "arg8:" in doc else 0


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@torch.inference_mode()
def test_sm70_qsa_grouped_page4_calibrated_kv(kv_cache_dtype: str) -> None:
    extension = _require_grouped_page4()
    abi_version = _grouped_page4_abi_version(extension)
    if kv_cache_dtype == "fp8_e4m3" and abi_version < 2:
        pytest.skip("installed grouped page4 ABI does not support quantized K/V")
    torch.manual_seed(7)
    query = torch.randn((8, 6, 256), dtype=torch.float16, device="cuda") * 0.2
    key = torch.randn((1, 4, 1, 256), dtype=torch.float16, device="cuda") * 0.35
    value = torch.randn_like(key) * 0.3
    k_scale = float(key.abs().max().item()) / 448.0
    v_scale = float(value.abs().max().item()) / 448.0
    if kv_cache_dtype == "fp8_e4m3":
        key_cache = (key / k_scale).to(torch.float8_e4m3fn).view(torch.uint8)
        value_cache = (value / v_scale).to(torch.float8_e4m3fn).view(torch.uint8)
        reference_key = key_cache.view(torch.float8_e4m3fn).float() * k_scale
        reference_value = value_cache.view(torch.float8_e4m3fn).float() * v_scale
    else:
        key_cache = key
        value_cache = value
        reference_key = key.float()
        reference_value = value.float()
        k_scale = v_scale = 1.0

    block_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    token_masks = torch.full((1, 1), 0xFFFFFFFF, dtype=torch.uint32, device="cuda")
    seq_lens = torch.tensor([4], dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    lse = torch.empty((8, 6), dtype=torch.float32, device="cuda")
    args = (
        query,
        key_cache,
        value_cache,
        output,
        block_table,
        token_masks,
        seq_lens,
        lse,
        256**-0.5,
    )
    if abi_version >= 2:
        extension.grouped_sparse_page4_fwd(
            *args,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    else:
        extension.grouped_sparse_page4_fwd(*args)

    reference_key = reference_key.view(4, 256)
    reference_value = reference_value.view(4, 256)
    scores = torch.matmul(query.float(), reference_key.transpose(0, 1)) / math.sqrt(256)
    probabilities = torch.softmax(scores, dim=-1)
    reference = torch.matmul(probabilities, reference_value)
    torch.testing.assert_close(output.float(), reference, atol=3e-2, rtol=3e-2)
