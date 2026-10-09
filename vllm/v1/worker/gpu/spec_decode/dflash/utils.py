# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.shared_weights import (
    DFLASH_DRAFT_WEIGHTS,
    share_embeddings,
    share_lm_head,
    validate_dflash_shared_weights,
)
from vllm.model_executor.models.shared_weights import (
    get_target_lm_head as get_target_lm_head,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _validate_dflash_shared_weights(
    dflash_model: nn.Module,
    shared_embed: bool,
    shared_lm_head: bool,
) -> None:
    validate_dflash_shared_weights(
        dflash_model,
        shared_embed,
        shared_lm_head,
        is_last_rank=get_pp_group().is_last_rank,
        log=logger,
    )


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    draft_cache_dtype = speculative_config.kv_cache_dtype
    if (
        draft_cache_dtype is None
        and str(vllm_config.cache_config.cache_dtype).startswith("fp8")
        and current_platform.is_cuda()
        and current_platform.is_device_capability(70)
    ):
        draft_cache_dtype = "auto"
        logger.info_once(
            "Using FP16 draft KV cache for SM70 DFlash while the target uses %s.",
            vllm_config.cache_config.cache_dtype,
        )
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        is_speculative_draft=True,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=draft_cache_dtype,
            )
            if draft_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    shared_embed = share_embeddings(
        dflash_model,
        target_language_model,
        DFLASH_DRAFT_WEIGHTS,
        log=logger,
    )
    shared_lm_head = share_lm_head(
        dflash_model,
        target_model,
        target_language_model,
        DFLASH_DRAFT_WEIGHTS,
        log=logger,
    )

    _validate_dflash_shared_weights(dflash_model, shared_embed, shared_lm_head)

    # Keep only the precision the loaded drafter consumes. This affects the
    # auxiliary snapshots, never the target model's hidden/residual tensors.
    # Other drafters retain their existing auxiliary-state contract.
    get_aux_dtype = getattr(dflash_model, "get_aux_hidden_state_dtype", None)
    aux_dtype = get_aux_dtype() if get_aux_dtype is not None else None
    if aux_dtype is not None:
        for module in target_model.modules():
            if isinstance(module, EagleModelMixin):
                module.aux_hidden_state_dtype = aux_dtype
        logger.info(
            "DFlash auxiliary snapshot compaction uses projection dtype %s.", aux_dtype
        )

    return dflash_model
