# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.platforms as platforms
from vllm.config.attention import AttentionConfig
from vllm.config.vllm import apply_prefix_anchored_swa_constraints
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import PrefixAnchoredSWASpec


def _config(window: int = 128) -> SimpleNamespace:
    return SimpleNamespace(
        attention_config=AttentionConfig(
            prefix_anchored_decode_window=window,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            enable_prefix_caching=True,
            kv_offloading_size=None,
        ),
        model_config=SimpleNamespace(dtype=torch.float16, use_mla=False),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )


@pytest.fixture
def sm70_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            device_count=lambda: 1,
            is_device_capability=lambda capability, device_id=0: capability == (7, 0),
        ),
    )


def test_engine_contract_is_model_identity_independent(sm70_platform: None):
    cfg = _config()

    apply_prefix_anchored_swa_constraints(cfg)

    assert cfg.attention_config.backend == AttentionBackendEnum.FLASH_ATTN_V100
    assert not cfg.cache_config.enable_prefix_caching


@pytest.mark.parametrize("window", [0, -1])
def test_engine_option_rejects_nonpositive_window(window: int):
    with pytest.raises(ValueError):
        AttentionConfig(prefix_anchored_decode_window=window)


def test_engine_contract_rejects_non_fp16_kv(sm70_platform: None):
    cfg = _config()
    cfg.cache_config.cache_dtype = "fp8_e5m2"

    with pytest.raises(ValueError, match="requires an fp16 KV cache"):
        apply_prefix_anchored_swa_constraints(cfg)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("model_config.use_mla", True, "MLA attention"),
        ("attention_config.use_non_causal", True, "causal decoder"),
        ("parallel_config.decode_context_parallel_size", 2, "context parallelism"),
        ("cache_config.kv_offloading_size", 1.0, "KV-cache offloading"),
        (
            "kv_transfer_config",
            SimpleNamespace(kv_connector="ExampleConnector"),
            "KV connectors",
        ),
        ("speculative_config", object(), "speculative decoding"),
        (
            "attention_config.backend",
            AttentionBackendEnum.TRITON_ATTN,
            "requires FLASH_ATTN_V100",
        ),
    ],
)
def test_engine_contract_rejects_unsupported_runtime_combinations(
    sm70_platform: None,
    path: str,
    value: object,
    message: str,
):
    cfg = _config()
    target = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)

    with pytest.raises(ValueError, match=message):
        apply_prefix_anchored_swa_constraints(cfg)


def test_engine_contract_rejects_non_sm70(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            device_count=lambda: 1,
            is_device_capability=lambda capability, device_id=0: False,
        ),
    )

    with pytest.raises(ValueError, match="SM70 GPU"):
        apply_prefix_anchored_swa_constraints(_config())


def test_base_attention_emits_prefix_anchored_spec_without_model_subclass():
    layer = object.__new__(Attention)
    object.__setattr__(layer, "attn_type", AttentionType.DECODER)
    object.__setattr__(
        layer,
        "attn_backend",
        SimpleNamespace(get_name=lambda: "FLASH_ATTN_V100"),
    )
    object.__setattr__(layer, "kv_cache_dtype", "auto")
    object.__setattr__(layer, "kv_cache_torch_dtype", torch.float16)
    object.__setattr__(layer, "sliding_window", None)
    object.__setattr__(layer, "num_kv_heads", 4)
    object.__setattr__(layer, "head_size", 128)
    object.__setattr__(layer, "head_size_v", 128)
    object.__setattr__(layer, "has_sink", False)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        attention_config=SimpleNamespace(prefix_anchored_decode_window=256),
    )

    spec = Attention.get_kv_cache_spec(layer, vllm_config)

    assert isinstance(spec, PrefixAnchoredSWASpec)
    assert spec.decode_sliding_window == 256


def test_prefix_anchored_spec_rejects_nonpositive_window():
    with pytest.raises(ValueError, match="greater than zero"):
        PrefixAnchoredSWASpec(
            block_size=16,
            num_kv_heads=1,
            head_size=64,
            dtype=torch.float16,
            decode_sliding_window=0,
        )
