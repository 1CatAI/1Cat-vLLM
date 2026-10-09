# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model qualification and ordered runtime defaults (initialization only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig, VllmConfig

from typing import Any

import torch

from vllm.config.sm70_dflash2 import (
    SM70_DFLASH2_VERIFIER_DEFAULTS,
    SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_SM70_DFLASH2_VERIFIER_DEFAULTS = SM70_DFLASH2_VERIFIER_DEFAULTS


_SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS = SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS


def _is_sm70_dflash2_verifier_contract(
    model_config: Any,
    speculative_config: Any,
    parallel_config: Any,
) -> bool:
    from vllm.model_executor.models.config import sm70_dflash2_verifier_qualified

    return sm70_dflash2_verifier_qualified(
        model_config, speculative_config, parallel_config
    )


def _configure_sm70_dflash2_graph_cache(
    model_config: Any,
    speculative_config: Any,
    parallel_config: Any,
    cache_config: Any,
    *,
    defaults,
) -> bool:
    """Reuse compiled subgraphs without the unqualified AOT FX-graph reload."""
    if (
        defaults.value("VLLM_DISABLE_COMPILE_CACHE")
        or not _is_sm70_dflash2_verifier_contract(
            model_config, speculative_config, parallel_config
        )
        or model_config.quantization != "compressed-tensors"
        or not model_config.is_nvfp4_quantized()
        or parallel_config.tensor_parallel_size != 4
        or cache_config.cache_dtype != "fp8_e4m3"
    ):
        return False
    # The E4M3 release contract passes cold/warm quality with this cache path.
    # Explicit AOT selection remains an override; other model routes keep their
    # existing defaults until their own cache/quality qualification passes.
    defaults.setdefault("VLLM_USE_AOT_COMPILE", "0")
    return True


def _is_sm70_qwen38_decode_compile_contract(
    model_config: Any,
    speculative_config: Any,
    parallel_config: Any,
) -> bool:
    """Select the FP16 Qwen4Exp lane; each operator validates its own geometry.

    Quantization and KV precision do not describe the unquantized checkpoint
    projections. TP, model dimensions and speculative width are likewise not
    requirements of an M=1 GEMV. Do not gate all operators on one benchmark.
    """
    if model_config is None or parallel_config is None:
        return False
    architectures = set(getattr(model_config, "architectures", ()) or ())
    multimodal_config = getattr(model_config, "multimodal_config", None)
    supported_architecture = "Qwen4ExpForCausalLM" in architectures or (
        "Qwen4ExpForConditionalGeneration" in architectures
        and multimodal_config is not None
        and getattr(multimodal_config, "language_model_only", False)
    )
    return bool(
        supported_architecture and getattr(model_config, "dtype", None) == torch.float16
    )


def _apply_sm70_qwen38_decode_defaults(
    cfg: VllmConfig, *, is_sm70: bool, defaults
) -> tuple[str, ...]:
    """Enable shape-checked FP16 routes without coupling them to MoE/KV policy."""
    if not is_sm70 or not _is_sm70_qwen38_decode_compile_contract(
        cfg.model_config, cfg.speculative_config, cfg.parallel_config
    ):
        return ()
    parallel = cfg.parallel_config
    choices = {
        "VLLM_SM70_QWEN38_FP16_GEMV": "1",
        "VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16": "1",
        "VLLM_SM70_QWEN38_FUSED_HC_FP16": "1",
    }
    # These are collective/stream policies, not FP16 projection requirements.
    if not parallel.enable_expert_parallel and not parallel.enable_dbo:
        choices["VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP"] = "1"
        choices["VLLM_SM70_MOE_ADD_ALLREDUCE"] = "1"
    if cfg.speculative_config is not None and cfg.speculative_config.method == "mtp":
        # Keep M=1 draft graphs independently of the verifier query width.
        # Otherwise the prepared single-token operators never reach capture.
        choices["VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS"] = "1"
    applied = []
    for name, value in choices.items():
        if name not in defaults:
            defaults[name] = value
            applied.append(name)
    return tuple(applied)


def _configure_sm70_glm5_dflash_tp4_push_allreduce(
    model_config: ModelConfig | None,
    speculative_config: SpeculativeConfig | None,
    parallel_config: ParallelConfig,
    *,
    is_sm70: bool,
    defaults,
) -> None:
    """Keep the rejected GLM5 DFlash TP4 push collective diagnostic-only."""
    if (
        not is_sm70
        or model_config is None
        or getattr(model_config.hf_text_config, "model_type", None)
        not in ("glm5_next", "glm5_next_text")
        or speculative_config is None
        or speculative_config.method != "dflash"
        or parallel_config.tensor_parallel_size != 4
    ):
        return

    env_name = "VLLM_SM70_TP4_PUSH_ALLREDUCE"
    if env_name not in defaults:
        defaults[env_name] = "0"
        logger.info_once(
            "Auto-setting %s=0 for SM70 GLM5 DFlash2 TP4 because the push "
            "collective failed the retained output-quality audit. The ordinary "
            "custom all-reduce remains enabled; other model routes keep their "
            "existing default.",
            env_name,
        )
    elif defaults[env_name] != "0":
        logger.warning_once(
            "%s=%s explicitly enables a diagnostic-only GLM5 DFlash2 TP4 "
            "route that failed the retained output-quality audit. Remove the "
            "override or set it to 0 for the accepted production contract.",
            env_name,
            defaults[env_name],
        )


def _configure_sm70_glm5_dflash_tp8_pp1_verifier_path(
    model_config: ModelConfig | None,
    speculative_config: SpeculativeConfig | None,
    parallel_config: ParallelConfig,
    *,
    is_sm70: bool,
    defaults,
) -> bool:
    """Select the quality-qualified GLM-5.3 DFlash2 TP8 verifier path."""
    if (
        not is_sm70
        or model_config is None
        or getattr(model_config.hf_text_config, "model_type", None)
        not in ("glm5_next", "glm5_next_text")
        or getattr(model_config, "quantization", None) != "modelopt_fp4"
        or getattr(model_config, "dtype", None) != torch.float16
        or speculative_config is None
        or speculative_config.method != "dflash"
        or speculative_config.draft_sample_method != "probabilistic"
        or speculative_config.num_speculative_tokens != 7
        or parallel_config.tensor_parallel_size != 8
        or parallel_config.pipeline_parallel_size != 1
        or getattr(parallel_config, "enable_dbo", False)
        or int(getattr(parallel_config, "ubatch_size", 0) or 0) > 1
    ):
        return False

    configured = []
    overrides = []
    for name, value in _SM70_GLM5_DFLASH_TP8_PP1_DEFAULTS.items():
        if name not in defaults:
            defaults[name] = value
            configured.append(f"{name}={value}")
        elif defaults[name] != value:
            overrides.append(f"{name}={defaults[name]}")

    if configured:
        logger.info_once(
            "Auto-selecting the quality-qualified SM70 GLM-5.3 DFlash2 "
            "TP8/PP1 verifier path: %s.",
            ", ".join(configured),
        )
    if overrides:
        logger.warning_once(
            "Explicit SM70 GLM-5.3 DFlash2 overrides differ from the retained "
            "TP8/PP1 verifier contract: %s. Re-run the exactness, acceptance, "
            "output-quality, and verifier-latency gates before production use.",
            ", ".join(overrides),
        )
    return True


def _configure_sm70_glm5_dflash_tp4_pp2_acceptance_path(
    model_config: ModelConfig | None,
    speculative_config: SpeculativeConfig | None,
    parallel_config: ParallelConfig,
    *,
    is_sm70: bool,
    defaults,
) -> None:
    """Select the retained GLM-5.3 DFlash2 acceptance path on V100."""
    if (
        not is_sm70
        or model_config is None
        or getattr(model_config.hf_text_config, "model_type", None)
        not in ("glm5_next", "glm5_next_text")
        or getattr(model_config, "quantization", None) != "modelopt_fp4"
        or speculative_config is None
        or speculative_config.method != "dflash"
        or speculative_config.draft_sample_method != "probabilistic"
        or speculative_config.num_speculative_tokens != 7
        or parallel_config.tensor_parallel_size != 4
        or parallel_config.pipeline_parallel_size != 2
    ):
        return

    accepted_defaults = {
        "VLLM_SM70_DFLASH2_PROPOSAL_TEMPERATURE_SCALE": "0.8",
        "VLLM_SM70_DFLASH2_PROPOSAL_TOP_P": "0.95",
    }
    # A fixed partition must account for every layer. Other layer counts
    # retain the normal partitioner rather than inheriting a 45-layer split.
    if getattr(model_config.hf_text_config, "num_hidden_layers", None) == 45:
        accepted_defaults["VLLM_PP_LAYER_PARTITION"] = "24,21"
    configured = []
    overrides = []
    for name, value in accepted_defaults.items():
        if name not in defaults:
            defaults[name] = value
            configured.append(f"{name}={value}")
        elif defaults[name] != value:
            overrides.append(f"{name}={defaults[name]}")

    materialize = defaults.get("VLLM_GLM53_PP_MHC_MATERIALIZE")
    if materialize not in (None, "0"):
        overrides.append(f"VLLM_GLM53_PP_MHC_MATERIALIZE={materialize}")

    if configured:
        logger.info_once(
            "Auto-selecting the quality-qualified SM70 GLM-5.3 DFlash2 "
            "TP4/PP2 path: %s.",
            ", ".join(configured),
        )
    if overrides:
        logger.warning_once(
            "Explicit SM70 GLM-5.3 DFlash2 overrides differ from the retained "
            "TP4/PP2 acceptance contract: %s. Re-run the acceptance and "
            "output-quality gates before production use.",
            ", ".join(overrides),
        )


def apply_runtime_defaults(cfg, defaults, phase: str, *, is_sm70: bool):
    defaults.phase = f"model.{phase}"
    if phase == "collectives":
        args = (cfg.model_config, cfg.speculative_config, cfg.parallel_config)
        _configure_sm70_glm5_dflash_tp4_push_allreduce(
            *args, is_sm70=is_sm70, defaults=defaults
        )
        qualified = _configure_sm70_glm5_dflash_tp8_pp1_verifier_path(
            *args, is_sm70=is_sm70, defaults=defaults
        )
        _configure_sm70_glm5_dflash_tp4_pp2_acceptance_path(
            *args, is_sm70=is_sm70, defaults=defaults
        )
        return qualified
    if phase == "prefill":
        if (
            defaults.value("VLLM_FLASH_V100_BFLA_PREFILL")
            and cfg.model_config is not None
            and "VLLM_FLASH_V100_BFLA_KEEP_RATIO" not in defaults
        ):
            hf_text_config = cfg.model_config.hf_text_config
            num_attention_heads = getattr(hf_text_config, "num_attention_heads", None)
            num_key_value_heads = getattr(
                hf_text_config, "num_key_value_heads", num_attention_heads
            )
            head_dim = getattr(hf_text_config, "head_dim", None)
            if (
                num_attention_heads == 24
                and num_key_value_heads == 4
                and head_dim == 256
            ):
                defaults["VLLM_FLASH_V100_BFLA_KEEP_RATIO"] = "0.10"
                logger.info_once(
                    "Auto-setting VLLM_FLASH_V100_BFLA_KEEP_RATIO=0.10 "
                    "for the SM70 Flash-V100 BFLA Qwen3.5/3.6-27B "
                    "attention shape. Set it explicitly to override."
                )
        return None
    if phase == "projections":
        return _apply_sm70_qwen38_decode_defaults(
            cfg, is_sm70=is_sm70, defaults=defaults
        )
    if phase == "breakable":
        if (
            cfg.model_config is not None
            and "VLLM_USE_BREAKABLE_CUDAGRAPH" not in defaults
            and any(
                a in ("DeepseekV4ForCausalLM", "DeepSeekV4MTPModel")
                for a in cfg.model_config.architectures
            )
        ):
            defaults["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
            logger.info_once(
                "Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1 for DeepSeek V4. "
                "Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to opt out."
            )
        return None
    if phase == "graph_cache":
        return _configure_sm70_dflash2_graph_cache(
            cfg.model_config,
            cfg.speculative_config,
            cfg.parallel_config,
            cfg.cache_config,
            defaults=defaults,
        )
    raise ValueError(f"Unknown runtime defaults checkpoint: {phase}")


def ple_storage_rejection(cfg) -> str | None:
    """Model partition and checkpoint layout contract for resident PLE tiers."""
    text = cfg.model_config.hf_text_config
    layers = text.ple_layer_ids
    reason = None
    from vllm.models.qwen4_exp.common.ple import (
        check_ple_layers_on_first_pp_rank,
    )
    from vllm.models.qwen4_exp.nvidia.ple_layer import (
        _get_ple_embedding_quant_method,
    )

    try:
        check_ple_layers_on_first_pp_rank(
            text, cfg.parallel_config.pipeline_parallel_size
        )
    except (ValueError, RuntimeError) as exc:
        reason = str(exc)
    if reason is None:
        storage = str(getattr(text, "ple_embedding_dtype", "")).removeprefix("torch.")
        # The model's hf_to_vllm_mapper is applied to the quantization
        # config only when the model is built, so its layer metadata still
        # uses checkpoint names here. ple_layer_ids are 1-based: id L is
        # the PLE module of decoder layer L - 1.
        methods = [
            _get_ple_embedding_quant_method(
                cfg.quant_config,
                f"model.language_model.layers.{int(layer_id) - 1}"
                ".ple.ple_embedding.ngram_embedding",
                force_fp8_storage=storage == "float8_e4m3fn",
            )
            for layer_id in layers
        ]
        if any(method is None for method in methods):
            reason = (
                "checkpoint metadata does not provide raw E4M3 or packed GGUF "
                "PLE storage"
            )
    return reason


def sparse_execution_family(model_config):
    """Declare which existing sparse provider the model can consume."""
    text = getattr(model_config, "hf_text_config", None)
    if getattr(text, "indexer_n_heads", None) is not None:
        return "qsa"
    if getattr(text, "index_head_dim", None):
        return "indexer"
    return None


def uses_mtp_weight_policy(model, spec):
    """Target and independent draft configs share the same weight contract."""
    draft = getattr(spec, "draft_model_config", None)
    for candidate in (model, draft):
        if getattr(candidate, "architecture", None) in ("Qwen3_5MTP", "Qwen3_5MoeMTP"):
            return True
    text = getattr(model, "hf_text_config", None)
    return getattr(spec, "method", None) == "mtp" and getattr(
        text, "model_type", None
    ) in ("qwen3_5_text", "qwen3_5_moe_text")
