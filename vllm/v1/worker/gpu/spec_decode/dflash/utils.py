# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)


def _validate_dflash_shared_weights(
    dflash_model: nn.Module,
    shared_embed: bool,
    shared_lm_head: bool,
) -> None:
    if not get_pp_group().is_last_rank:
        return

    requires_shared_embed = not getattr(dflash_model, "has_own_embed_tokens", False)
    requires_shared_lm_head = not getattr(dflash_model, "has_own_lm_head", False)
    logger.info_once(
        "DFlash shared-weight contract on the final PP stage: "
        "embedding=%s lm_head=%s draft_has_own_embedding=%s "
        "draft_has_own_lm_head=%s",
        shared_embed,
        shared_lm_head,
        not requires_shared_embed,
        not requires_shared_lm_head,
    )
    if requires_shared_embed and not shared_embed:
        raise RuntimeError(
            "The DFlash checkpoint has no embedding, but the final pipeline "
            "stage could not share the target embedding. DFlash proposals "
            "would be invalid."
        )
    shared_embed_module = getattr(
        getattr(dflash_model, "model", None), "embed_tokens", None
    )
    if (
        requires_shared_embed
        and shared_embed
        and getattr(shared_embed_module, "_dflash_pp_replica_expected", False)
        and not getattr(shared_embed_module, "_dflash_pp_replica_loaded", False)
    ):
        raise RuntimeError(
            "The DFlash checkpoint has no embedding, but the shared target "
            "embedding replica on the final pipeline stage was not loaded."
        )
    if requires_shared_lm_head and not shared_lm_head:
        raise RuntimeError(
            "The DFlash checkpoint has no lm_head, but the final pipeline "
            "stage could not share the target lm_head. DFlash proposals "
            "would be invalid."
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
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
        target_inner, "embedding", None
    )
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    shared_embed = False
    if (
        target_embed is not None
        and not isinstance(target_embed, PPMissingLayer)
        and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        )
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed
        shared_embed = True

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    shared_lm_head = False
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head
        shared_lm_head = True

    _validate_dflash_shared_weights(dflash_model, shared_embed, shared_lm_head)

    return dflash_model
