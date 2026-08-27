# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


class _QuantConfig:
    def __init__(self) -> None:
        self.ignore: list[str] = []
        self.config: dict[str, object] = {}
        self.quantized_layers: dict[str, dict[str, str]] = {}


@pytest.mark.parametrize(
    ("split_enabled", "norm_enabled"),
    [
        (False, False),
        (False, True),
    ],
)
def test_sm70_gdn_qpn8_ba_split_is_disabled_without_split_flag(
    monkeypatch,
    split_enabled: bool,
    norm_enabled: bool,
) -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    monkeypatch.setattr(gdn.envs, "VLLM_SM70_GDN_QPN8_BA_SPLIT", split_enabled)
    monkeypatch.setattr(gdn.envs, "VLLM_SM70_GDN_RMSNORM_ONEPASS", norm_enabled)

    assert not gdn._sm70_gdn_qpn8_ba_split_enabled()


def test_sm70_gdn_qpn8_ba_split_rejects_unpaired_route(monkeypatch) -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    monkeypatch.setattr(gdn.envs, "VLLM_SM70_GDN_QPN8_BA_SPLIT", True)
    monkeypatch.setattr(gdn.envs, "VLLM_SM70_GDN_RMSNORM_ONEPASS", False)

    with pytest.raises(RuntimeError, match="requires the accepted"):
        gdn._sm70_gdn_qpn8_ba_split_enabled()


def test_sm70_gdn_qpn8_ba_split_requires_source_built_ops(monkeypatch) -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    monkeypatch.setattr(gdn.envs, "VLLM_SM70_GDN_QPN8_BA_SPLIT", True)
    monkeypatch.setattr(gdn.envs, "VLLM_SM70_GDN_RMSNORM_ONEPASS", True)
    monkeypatch.setattr(
        gdn,
        "_missing_sm70_gdn_qpn8_ba_ops",
        lambda: ["fp8_qpn8_dispatch_ba_split_sm70_out"],
    )

    with pytest.raises(RuntimeError, match="requires the source-built"):
        gdn._sm70_gdn_qpn8_ba_split_enabled()


def test_sm70_gdn_qpn8_ba_split_accepts_complete_contract(monkeypatch) -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    monkeypatch.setattr(gdn.envs, "VLLM_SM70_GDN_QPN8_BA_SPLIT", True)
    monkeypatch.setattr(gdn.envs, "VLLM_SM70_GDN_RMSNORM_ONEPASS", True)
    monkeypatch.setattr(gdn, "_missing_sm70_gdn_qpn8_ba_ops", lambda: [])

    assert gdn._sm70_gdn_qpn8_ba_split_enabled()


@pytest.mark.parametrize("input_width", [2560, 5120])
def test_sm70_gdn_qpn8_ba_weight_contract_is_layout_based(input_width) -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    qkvz = SimpleNamespace(
        sm70_fp8_qpn8=True,
        sm70_fp8_prefill_exact_dense_workspace_ptr=42,
        weight=torch.empty((input_width, 4096), dtype=torch.uint8, device="meta"),
        weight_scale_inv=torch.empty((1, 4096), dtype=torch.float16, device="meta"),
        bias=None,
    )
    ba = SimpleNamespace(
        weight=torch.empty((24, input_width), dtype=torch.float16, device="meta"),
        bias=None,
    )
    layer = SimpleNamespace(
        enable_sm70_gdn_qpn8_ba_split=True,
        in_proj_qkvz=qkvz,
        in_proj_ba=ba,
    )

    assert gdn._sm70_gdn_qpn8_ba_weight_contract(layer)

    qkvz.bias = object()
    assert not gdn._sm70_gdn_qpn8_ba_weight_contract(layer)
    qkvz.bias = None
    qkvz.weight = torch.empty((4096, input_width), dtype=torch.uint8, device="meta")
    assert not gdn._sm70_gdn_qpn8_ba_weight_contract(layer)


def test_qwen3_5_split_gdn_detects_compressed_tensors_ignore():
    from vllm.model_executor.models.qwen3_5 import (
        _uses_split_gdn_input_projections,
    )

    quant_config = _QuantConfig()
    quant_config.ignore = [
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.in_proj_a",
    ]
    quant_config.config = {}

    assert _uses_split_gdn_input_projections(quant_config)


def test_qwen3_5_split_gdn_detects_compressed_tensors_config_ignore():
    from vllm.model_executor.models.qwen3_5 import (
        _uses_split_gdn_input_projections,
    )

    quant_config = _QuantConfig()
    quant_config.config = {
        "ignore": [
            "model.language_model.layers.0.linear_attn.in_proj_b",
            "model.language_model.layers.0.linear_attn.in_proj_a",
        ],
    }

    assert _uses_split_gdn_input_projections(quant_config)


def test_qwen3_5_split_gdn_detects_modelopt_mixed_unquantized_ba():
    from vllm.model_executor.models.qwen3_5 import (
        _uses_split_gdn_input_projections,
    )

    quant_config = _QuantConfig()
    prefix = "model.language_model.layers.0.linear_attn"
    quant_config.quantized_layers = {
        f"{prefix}.in_proj_qkv": {"quant_algo": "FP8"},
        f"{prefix}.in_proj_z": {"quant_algo": "FP8"},
    }

    assert _uses_split_gdn_input_projections(quant_config)


def test_qwen3_5_split_gdn_keeps_same_precision_projections_fused():
    from vllm.model_executor.models.qwen3_5 import (
        _uses_split_gdn_input_projections,
    )

    quant_config = _QuantConfig()
    prefix = "model.language_model.layers.0.linear_attn"
    quant_config.quantized_layers = {
        f"{prefix}.{name}": {"quant_algo": "FP8"}
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    }

    assert not _uses_split_gdn_input_projections(quant_config)


def test_qwen3_5_split_gdn_detects_different_modelopt_algorithms():
    from vllm.model_executor.models.qwen3_5 import (
        _uses_split_gdn_input_projections,
    )

    quant_config = _QuantConfig()
    prefix = "model.language_model.layers.0.linear_attn"
    quant_config.quantized_layers = {
        f"{prefix}.in_proj_qkv": {"quant_algo": "FP8"},
        f"{prefix}.in_proj_z": {"quant_algo": "FP8"},
        f"{prefix}.in_proj_b": {"quant_algo": "W4A16_NVFP4"},
        f"{prefix}.in_proj_a": {"quant_algo": "W4A16_NVFP4"},
    }

    assert _uses_split_gdn_input_projections(quant_config)


def test_qwen3_5_lm_head_receives_quant_config():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config
    mock_vllm_config.lora_config = None

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5Model") as MockModel,
        patch("vllm.model_executor.models.qwen3_5.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()

        Qwen3_5ForCausalLMBase(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_mtp_lm_head_receives_quant_config():
    from vllm.config import CompilationMode
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64
    mock_hf_config.quantization_config = None

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.model_config.hf_config = None
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.compilation_config.mode = CompilationMode.NONE
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor"),
        patch("vllm.model_executor.models.qwen3_5_mtp.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5_mtp.LogitsProcessor"),
        patch.dict("os.environ", {"VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS": "0"}),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3_5MTP(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config
