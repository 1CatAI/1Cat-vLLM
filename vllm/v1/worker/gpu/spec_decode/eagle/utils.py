# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.shared_weights import (
    EAGLE_DRAFT_WEIGHTS,
    share_embeddings,
    share_lm_head,
)
from vllm.model_executor.models.shared_weights import (
    get_target_lm_head as get_target_lm_head,
)


def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    with set_model_tag("eagle_head"):
        eagle_model = get_model(
            vllm_config=vllm_config,
            model_config=draft_model_config,
            load_config=speculative_config.draft_load_config,
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    share_embeddings(
        eagle_model,
        target_language_model,
        EAGLE_DRAFT_WEIGHTS,
        pp_size=get_pp_group().world_size,
    )
    share_lm_head(eagle_model, target_model, target_language_model, EAGLE_DRAFT_WEIGHTS)

    # Specialized draft packs must use the final shared checkpoint head and
    # be resident before KV allocation/graph warmup reduce startup headroom.
    prepare_head = getattr(eagle_model, "prepare_sm70_draft_head", None)
    if prepare_head is not None:
        prepare_head()

    return eagle_model
