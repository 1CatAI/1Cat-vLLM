# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2 item 1: DeepseekV41Config flattening, config/model registration, quant-method claim order."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_get_config_parses_official_checkpoint(ds41_checkpoint_dir: Path) -> None:
    from vllm.transformers_utils.config import get_config
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    cfg = get_config(str(ds41_checkpoint_dir), trust_remote_code=False)
    assert isinstance(cfg, DeepseekV41Config)
    assert cfg.model_type == "deepseek_v41" and cfg.text_model_type == "deepseek_v41_text"
    assert cfg.get_text_config() is cfg and not hasattr(cfg, "text_config")
    expect = dict(hidden_size=5120, num_hidden_layers=40, num_attention_heads=64, head_dim=512, q_lora_rank=1280,
                  o_lora_rank=1024, o_groups=8, n_routed_experts=384, num_experts_per_tok=6,
                  moe_intermediate_size=2304, vocab_size=129280, rms_norm_eps=1e-20, hc_eps=1e-6, hc_mult=4,
                  hc_sinkhorn_iters=20, compress_rope_theta=160000, rope_theta=10000, index_topk=512,
                  index_n_heads=32, index_head_dim=128, sliding_window=128, expert_dtype="fp4",
                  tie_word_embeddings=False, max_position_embeddings=1048576,
                  kv_source_layer_ids=[2, 8, 14, 20], index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
                  engram_layer_ids=[1, 14], candidate_source_layer_id=20, dspark_target_layer_ids=[37, 38, 39])
    for key, value in expect.items():
        assert getattr(cfg, key) == value, key
    assert len(cfg.compress_ratios) == 43
    assert cfg.compress_ratios[:3] == [0, 0, 2] and cfg.compress_ratios[19:21] == [2, 1] and cfg.compress_ratios[40:] == [0, 0, 0]
    rope = cfg.rope_parameters
    assert rope["rope_type"] == "yarn" and rope["factor"] == 16 and rope["original_max_position_embeddings"] == 65536
    assert rope["beta_fast"] == 32 and rope["beta_slow"] == 1
    assert cfg.quantization_config["weight_block_size"] == [32, 32]
    assert cfg.vision_n_layers == 32


def test_flat_text_config(ds41_config_json: dict[str, Any]) -> None:
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    flat = dict(ds41_config_json["text_config"])
    flat.pop("model_type")
    flat["quantization_config"] = ds41_config_json["quantization_config"]
    cfg = DeepseekV41Config(**flat)
    assert cfg.hidden_size == 5120 and cfg.rope_parameters["factor"] == 16 and cfg.expert_dtype == "fp4"
    assert cfg.vision_n_layers == 0


def test_registrations() -> None:
    from vllm.model_executor.models.registry import _VLLM_MODELS
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    assert _VLLM_MODELS["DeepseekV41ForCausalLM"] == ("vllm.models.deepseek_v41", "DeepseekV41ForCausalLM")
    assert _VLLM_MODELS["DeepseekV4ForCausalLM"] == ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM")
    assert _CONFIG_REGISTRY["deepseek_v41"].__name__ == "DeepseekV41Config"
    assert _CONFIG_REGISTRY["deepseek_v41_text"].__name__ == "DeepseekV41Config"
    assert _CONFIG_REGISTRY["deepseek_v4"].__name__ == "DeepseekV4Config"


def _resolve_quant_method(quant_cfg: dict[str, Any], hf_config: Any, user_quant: str | None = None) -> str | None:
    """Mirror of ModelConfig._verify_quantization's override search (vllm/config/model.py)."""
    from vllm.model_executor.layers import quantization as me_quant

    overrides = ["auto_gptq", "gptq", "gptq_marlin", "awq_marlin", "inc", "moe_wna16", "modelopt", "modelopt_fp4",
                 "modelopt_mxfp8", "modelopt_mixed", "mxfp4", "gpt_oss_mxfp4", "deepseek_v4_fp8", "humming", "gguf"]
    names = [q for q in me_quant.QUANTIZATION_METHODS if q not in overrides] + overrides
    for name in names:
        method = me_quant.get_quantization_config(name)
        claimed = method.override_quantization_method(quant_cfg, user_quant, hf_config=hf_config)
        if claimed is not None:
            return claimed
    return None


def test_quant_method_claim(ds41_checkpoint_dir: Path) -> None:
    from types import SimpleNamespace

    from vllm.transformers_utils.config import get_config

    cfg = get_config(str(ds41_checkpoint_dir), trust_remote_code=False)
    assert _resolve_quant_method(dict(cfg.quantization_config), cfg) == "deepseek_v41_fp8"
    # V4 and a generic FP8 model are untouched
    v4 = SimpleNamespace(model_type="deepseek_v4")
    assert _resolve_quant_method({"quant_method": "fp8", "weight_block_size": [128, 128]}, v4) == "deepseek_v4_fp8"
    other = SimpleNamespace(model_type="llama")
    assert _resolve_quant_method({"quant_method": "fp8", "weight_block_size": [128, 128]}, other) is None


@pytest.mark.sm70
def test_vllm_model_config_parses_official_checkpoint(ds41_checkpoint_dir: Path) -> None:
    """ModelConfig resolves the architecture to the SM70 port (registry import needs a CUDA SM70 platform)."""
    import torch

    from vllm.config import ModelConfig
    from vllm.model_executor.models.registry import ModelRegistry

    mc = ModelConfig(model=str(ds41_checkpoint_dir), dtype="half", skip_tokenizer_init=True, max_model_len=32768)
    assert mc.architectures == ["DeepseekV41ForCausalLM"] and mc.quantization == "deepseek_v41_fp8"
    assert mc.dtype == torch.float16 and mc.is_moe and mc.hf_text_config is mc.hf_config
    assert mc.tokenizer_mode == "deepseek_v41"          # auto-selected for the architecture
    cls, arch = ModelRegistry.resolve_model_cls(mc.architectures, mc)
    assert cls.__module__ == "vllm.models.deepseek_v41.sm70.model" and cls.supports_pp


def test_breakable_cudagraph_architectures() -> None:
    """PORT_DESIGN §9 AM-8: V4.1 auto-enables the breakable CUDA graph like V4 (V4 entries unchanged)."""
    import inspect

    from vllm.config import vllm as vllm_config_module

    archs = vllm_config_module.BREAKABLE_CUDAGRAPH_ARCHITECTURES
    assert {"DeepseekV4ForCausalLM", "DeepSeekV4MTPModel", "DeepseekV41ForCausalLM"} <= set(archs)
    assert "a in BREAKABLE_CUDAGRAPH_ARCHITECTURES" in inspect.getsource(vllm_config_module.VllmConfig)


def test_cache_block_size_set_before_model_init() -> None:
    """L-INTEG: the V4.1 attention builds its cache specs at model construction, before the platform's
    update_block_size_for_backend; the model-config hook must set the V4.1 block size (default 16 is refused)."""
    from types import SimpleNamespace

    from vllm.config.cache import CacheConfig
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, DeepseekV41ForCausalLMConfig
    from vllm.models.deepseek_v41.sm70.sparse import PREFERRED_BLOCK_SIZE

    assert MODELS_CONFIG_MAP["DeepseekV41ForCausalLM"] is DeepseekV41ForCausalLMConfig
    assert DeepseekV41ForCausalLMConfig.PREFERRED_BLOCK_SIZE == PREFERRED_BLOCK_SIZE
    vc = SimpleNamespace(cache_config=CacheConfig())
    assert vc.cache_config.block_size == CacheConfig.DEFAULT_BLOCK_SIZE
    DeepseekV41ForCausalLMConfig.verify_and_update_config(vc)
    assert vc.cache_config.block_size == PREFERRED_BLOCK_SIZE
    user = SimpleNamespace(cache_config=CacheConfig(block_size=128))
    DeepseekV41ForCausalLMConfig.verify_and_update_config(user)
    assert user.cache_config.block_size == 128
