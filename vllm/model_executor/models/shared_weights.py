# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned draft weight contracts and a single binding lifecycle."""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.models.utils import PPMissingLayer

logger = init_logger(__name__)


@dataclass(frozen=True)
class DraftWeightContract:
    legacy: bool = False
    embedding_on_pipeline_stage: bool = False
    share_mtp_auxiliary: bool = True


LEGACY_DRAFT_WEIGHTS = DraftWeightContract(legacy=True)
EAGLE_DRAFT_WEIGHTS = DraftWeightContract()
DFLASH_DRAFT_WEIGHTS = DraftWeightContract(
    embedding_on_pipeline_stage=True, share_mtp_auxiliary=False
)


def should_share(draft_model, flag, draft, target, *, legacy=False):
    """Keep the original CPU/eager-memory comparison qualification."""
    if not getattr(draft_model, flag, False) or (draft is None and not legacy):
        return True
    if target is None:
        return False
    if legacy:
        target_weight = getattr(target, "weight", None)
        draft_weight = getattr(draft, "weight", None)
        return (
            isinstance(target_weight, torch.Tensor)
            and isinstance(draft_weight, torch.Tensor)
            and torch.equal(target_weight.cpu(), draft_weight.cpu())
        )
    weight = draft.weight
    if (
        weight.is_cuda
        and torch.cuda.mem_get_info(weight.device)[0] < weight.numel() * 2
    ):
        return torch.equal(weight.cpu(), target.weight.cpu())
    return torch.equal(weight, target.weight)


def bind_shared(owner, name, value):
    """Release the draft copy before registering the borrowed target object."""
    if hasattr(owner, name):
        delattr(owner, name)
    setattr(owner, name, value)


def _legacy_decision_log(draft, name, share, log):
    flag = f"has_own_{name}"
    label = "embedding" if name == "embed_tokens" else name
    if not hasattr(draft, flag):
        log.info(
            "Detected MTP model. Sharing target model %s weights with the draft model.",
            label,
        )
    elif not getattr(draft, flag):
        log.info(
            "Detected EAGLE model without its own %s in the checkpoint. "
            "Sharing target model %s weights with the draft model.",
            name,
            label,
        )
    elif share:
        log.info(
            "Detected EAGLE model with %s identical to the target model. "
            "Sharing target model %s weights with the draft model.",
            name,
            label,
        )
    else:
        log.info(
            "Detected EAGLE model with distinct %s weights. "
            "Keeping separate %s weights from the target model.",
            name,
            label,
        )


def share_embeddings(draft, target_language_model, contract, *, pp_size=1, log=logger):
    if pp_size != 1 and not contract.embedding_on_pipeline_stage:
        if contract.legacy:
            log.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )
        return False
    if contract.legacy:
        target_inner = getattr(target_language_model, "model", None)
        if target_inner is None:
            raise AttributeError("Target model does not have 'model' attribute")
        if hasattr(target_inner, "embed_tokens"):
            target_embed = target_inner.embed_tokens
        elif hasattr(target_inner, "embedding"):
            target_embed = target_inner.embedding
        else:
            raise AttributeError(
                "Target model does not have 'embed_tokens' or 'embedding' attribute"
            )
    else:
        target_inner = (
            getattr(target_language_model, "model", target_language_model)
            if contract.embedding_on_pipeline_stage
            else target_language_model.model
        )
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        if target_embed is None or (
            contract.embedding_on_pipeline_stage
            and isinstance(target_embed, PPMissingLayer)
        ):
            return False
    draft_embed = (
        draft.model.embed_tokens
        if contract.legacy and getattr(draft, "has_own_embed_tokens", False)
        else getattr(draft.model, "embed_tokens", None)
    )
    share = should_share(
        draft, "has_own_embed_tokens", draft_embed, target_embed, legacy=contract.legacy
    )
    if contract.legacy:
        _legacy_decision_log(draft, "embed_tokens", share, log)
    if share:
        bind_shared(draft.model, "embed_tokens", target_embed)
    return share


def get_target_lm_head(target_model, target_language_model):
    return getattr(target_language_model, "lm_head", None) or getattr(
        target_model, "lm_head", None
    )


def share_lm_head(draft, target_model, target_language_model, contract, *, log=logger):
    target_head = (
        getattr(target_language_model, "lm_head", None)
        if contract.legacy
        else get_target_lm_head(target_model, target_language_model)
    )
    draft_head = (
        draft.lm_head
        if contract.legacy
        and getattr(draft, "has_own_lm_head", False)
        and hasattr(target_head, "weight")
        else getattr(draft, "lm_head", None)
    )
    share = should_share(
        draft, "has_own_lm_head", draft_head, target_head, legacy=contract.legacy
    )
    if contract.legacy:
        _legacy_decision_log(draft, "lm_head", share, log)
    shared = share and (
        hasattr(target_language_model, "lm_head")
        if contract.legacy
        else target_head is not None
    )
    if shared:
        bind_shared(draft, "lm_head", target_head)
        if contract.share_mtp_auxiliary:
            inner = getattr(draft, "model", None)
            layers = getattr(inner, "layers", None) if inner is not None else None
            if layers is not None:
                items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
                for layer in items:
                    head = getattr(layer, "shared_head", None)
                    if head is not None and hasattr(head, "head"):
                        bind_shared(head, "head", target_head)
                        if contract.legacy:
                            log.info(
                                "Shared target model lm_head with MTP shared_head.head."
                            )
    if contract.share_mtp_auxiliary:
        target_inner = target_language_model.model
        if hasattr(target_inner, "topk_indices_buffer"):
            bind_shared(
                draft.model, "topk_indices_buffer", target_inner.topk_indices_buffer
            )
            if contract.legacy:
                log.info(
                    "Detected MTP model with topk_indices_buffer. Sharing target model "
                    "topk_indices_buffer with the draft model."
                )
    return shared


def validate_dflash_shared_weights(
    dflash_model, shared_embed, shared_lm_head, *, is_last_rank, log=logger
):
    if not is_last_rank:
        return
    requires_shared_embed = not getattr(dflash_model, "has_own_embed_tokens", False)
    requires_shared_lm_head = not getattr(dflash_model, "has_own_lm_head", False)
    log.info_once(
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
