# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import ModelRegistry
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, get_kv_quant_mode

# FP16, one KV head of size 64: 2 * 64 * 2 bytes of K and V per token.
ATTN_BYTES_PER_TOKEN = 256
KERNEL_BLOCK = 16


class _LooseBackend:
    """A backend that accepts smaller kernel blocks than the sharded layers."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(8)]


def _vllm_config(mamba_tokens: int, dcp: int):
    class _Model:
        @classmethod
        def get_mamba_specs_from_config(cls, vllm_config):
            elements = mamba_tokens * ATTN_BYTES_PER_TOKEN // 2
            return (
                MambaSpec(
                    shapes=((elements,),), dtypes=(torch.float16,), block_size=-1
                ),
            )

        @classmethod
        def get_kv_block_size_multiple(cls, vllm_config):
            return dcp

    model_config = SimpleNamespace(
        use_mla=False,
        dtype=torch.float16,
        architecture="FakeHybridForCausalLM",
        get_num_kv_heads=lambda parallel_config: 1,
        get_head_size=lambda: 64,
        hf_text_config=SimpleNamespace(),
    )
    cache_config = SimpleNamespace(
        cache_dtype="auto",
        block_size=KERNEL_BLOCK,
        mamba_cache_mode="align",
        mamba_block_size=None,
        user_specified_mamba_block_size=False,
        mamba_page_size_padded=None,
    )
    config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
    )
    return config, _Model


# The mamba page spans 1,610 attention tokens (MTP4-like, where a 16-token
# alignment yields 1,616) or 1,590 tokens (MTP3-like, 1,600 either way).
@pytest.mark.parametrize("mamba_tokens", [1610, 1590])
@pytest.mark.parametrize("dcp", [1, 2])
def test_dcp_share_keeps_the_block_alignment(monkeypatch, mamba_tokens, dcp) -> None:
    config, model_cls = _vllm_config(mamba_tokens, dcp)
    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda *args, **kwargs: (model_cls, "FakeHybridForCausalLM"),
    )

    Platform._align_hybrid_block_size(config, _LooseBackend)

    block_size = config.cache_config.block_size
    assert block_size % dcp == 0
    assert (block_size // dcp) % KERNEL_BLOCK == 0, block_size
    assert block_size * ATTN_BYTES_PER_TOKEN >= mamba_tokens * ATTN_BYTES_PER_TOKEN
    # No larger than the next aligned size: existing geometries do not move.
    assert block_size - KERNEL_BLOCK * dcp < mamba_tokens
    assert config.cache_config.mamba_block_size == block_size


@pytest.mark.parametrize("mamba_tokens", [20, 21, 41])
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8_e4m3", "int8"])
def test_hybrid_budget_uses_the_whole_physical_page(
    monkeypatch, mamba_tokens, cache_dtype
):
    config, model_cls = _vllm_config(mamba_tokens, 1)
    config.cache_config.cache_dtype = cache_dtype
    monkeypatch.setattr(
        ModelRegistry, "resolve_model_cls", lambda *args, **kwargs: (model_cls, "fake")
    )
    Platform._align_hybrid_block_size(config, _LooseBackend)
    cache = config.cache_config
    dtype = torch.float16 if cache_dtype == "auto" else torch.uint8

    def page(block):
        return FullAttentionSpec(
            block_size=block,
            num_kv_heads=1,
            head_size=64,
            dtype=dtype,
            kv_quant_mode=get_kv_quant_mode(cache_dtype),
        ).page_size_bytes

    state_bytes = mamba_tokens * ATTN_BYTES_PER_TOKEN
    assert page(cache.block_size) >= state_bytes
    assert page(cache.block_size - KERNEL_BLOCK) < state_bytes
    assert cache.mamba_page_size_padded in (None, page(cache.block_size))


class _GridBackend:
    """Resizes attention blocks per group later, like Flash-V100."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]

    @staticmethod
    def get_hybrid_block_multiple():
        return 64


# #1422: an FP8 target with an FP16 DFlash draft resolved 1648 tokens; the draft
# page cannot be halved to 824, so attention moved to 1664 and Mamba stayed at
# 1648 (171392-token prefixes). Such backends now start align mode on 64.
@pytest.mark.parametrize(
    "mamba_tokens,backend,mode,expected",
    [
        (1640, _GridBackend, "align", 1664),
        (1600, _GridBackend, "align", 1600),  # already a multiple of 64
        (1640, _LooseBackend, "align", 1648),  # backends without the hook
        (1640, _GridBackend, "none", 1648),  # no prefix grid without caching
    ],
)
def test_hybrid_block_multiple_keeps_one_prefix_grid(
    monkeypatch, mamba_tokens, backend, mode, expected
) -> None:
    config, model_cls = _vllm_config(mamba_tokens, 1)
    config.cache_config.mamba_cache_mode = mode
    monkeypatch.setattr(
        ModelRegistry, "resolve_model_cls", lambda *args, **kwargs: (model_cls, "fake")
    )

    Platform._align_hybrid_block_size(config, backend)

    assert config.cache_config.block_size == expected
    if mode == "align":
        assert config.cache_config.mamba_block_size == expected


def test_page_sharing_backend_requests_the_grid_multiple() -> None:
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert AttentionBackend.get_hybrid_block_multiple() is None
    backend = AttentionBackendEnum.FLASH_ATTN_V100.get_class()
    assert backend.get_hybrid_block_multiple() == 64
