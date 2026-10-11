# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the standalone package's versioned binding boundary."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

source = (
    Path(__file__).resolve().parents[3]
    / "flash-attention-v100/flash_attn_v100/kv_binding.py"
)
spec = importlib.util.spec_from_file_location("_flash_v100_kv_binding_contract", source)
assert spec is not None and spec.loader is not None
binding_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = binding_module
spec.loader.exec_module(binding_module)

ALGORITHMS = (
    "decode_paged",
    "decode_paged_xqa",
    "grouped_fp32_paged",
    "paged_kv_to_fp16",
    "prefill_paged",
    "prefill_paged_splitkv",
    "prefill_paged_bfla",
    "decode_paged_wmma",
)
CODECS = ("auto", "fp8_e4m3", "fp8_e5m2", "int8", "int8_per_token_head")


def _native(algorithm, version=1):
    return SimpleNamespace(
        **{
            f"{algorithm}_kv_abi": version,
            f"{algorithm}_kv_codecs": CODECS,
            f"{algorithm}_kv_fwd": lambda *args: args,
            f"{algorithm}_fwd": lambda *args: args,
        }
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("codec", CODECS)
def test_every_format_uses_the_same_versioned_abi(algorithm, codec):
    native = _native(algorithm)
    scales = (
        (torch.ones(1), torch.ones(1)) if codec.startswith("int8") else (None, None)
    )
    kwargs = {"partial_dtype": torch.float32} if algorithm.startswith("grouped") else {}
    binding = binding_module.bind_kv(native, algorithm, codec, *scales, **kwargs)
    assert binding.operation is getattr(native, f"{algorithm}_kv_fwd")
    assert binding.scale_args[0] is scales[0]
    assert binding.scale_args[1] is scales[1]
    expected = (
        torch.float32
        if algorithm.startswith("grouped") or codec in ("fp8_e4m3", *CODECS[-2:])
        else torch.float16
    )
    assert binding.partial_dtype == expected


@pytest.mark.parametrize("version", [0, 2])
@pytest.mark.parametrize("algorithm", ALGORITHMS[:2])
def test_unknown_abi_can_only_use_qualified_legacy_formats(version, algorithm):
    native = _native(algorithm, version)
    binding = binding_module.bind_kv(native, algorithm, "auto", None, None)
    assert binding.operation is getattr(native, f"{algorithm}_fwd")
    assert binding.scale_args == ()
    for codec in ("int8", "int8_per_token_head"):
        with pytest.raises(RuntimeError, match="no explicit KV binding"):
            binding_module.bind_kv(
                native, algorithm, codec, torch.ones(1), torch.ones(1)
            )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_codec_scales_are_checked_before_an_operation_can_be_returned(algorithm):
    native = _native(algorithm)
    with pytest.raises(ValueError, match="both scale planes"):
        binding_module.bind_kv(native, algorithm, "int8", None, torch.ones(1))
    with pytest.raises(ValueError, match="tensor scales"):
        binding_module.bind_kv(native, algorithm, "auto", torch.ones(1), torch.ones(1))
    setattr(native, f"{algorithm}_kv_codecs", ("auto",))
    with pytest.raises(RuntimeError, match="no explicit KV binding"):
        binding_module.bind_kv(native, algorithm, "int8", torch.ones(1), torch.ones(1))
