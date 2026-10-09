# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 attention backend (SM70).

Modules, in dependency order:

- ``ops``: lazy loading of the native operators.
- ``routing``: route accounting, tracing, decode partition/XQA admission.
- ``debug``: debug and profiling switches.
- ``kv_layout``: paged/contiguous KV views and gathers.
- ``masks``: reference attention and block/tree masks.
- ``dense_prefill``: dense D256 prefill dispatch and workspaces.
- ``spec``: registered speculative metadata hooks and small-query preparation.
- ``metadata``: common metadata and its builder.
- ``impl``: the attention implementation.
- ``backend``: backend registration.

Modules reach each other through the module object (``_routing._record_route``)
so a rebound global or a monkeypatch has one owner.
"""

from typing import TYPE_CHECKING

from vllm.v1.attention.backends.flash_v100 import (  # noqa: F401
    backend,
    config,
    debug,
    debug_compare,
    decode,
    dense_prefill,
    impl,
    kv_layout,
    masks,
    metadata,
    ops,
    prefill,
    routing,
    state,
    verify,
    workspace,
)
from vllm.v1.attention.backends.flash_v100.compat import install_owner_aliases
from vllm.v1.attention.backends.flash_v100.plan import diagnostics
from vllm.v1.attention.backends.flash_v100.spec import attention as spec_attention
from vllm.v1.attention.backends.flash_v100.spec import (
    attention_policy as spec_attention_policy,
)
from vllm.v1.attention.backends.flash_v100.spec import (
    builder as spec_builder,
)
from vllm.v1.attention.backends.flash_v100.spec import contracts as spec_contracts
from vllm.v1.attention.backends.flash_v100.spec import diagnostics as spec_diagnostics
from vllm.v1.attention.backends.flash_v100.spec import (
    draft as spec_draft,
)
from vllm.v1.attention.backends.flash_v100.spec import policy as spec_policy
from vllm.v1.attention.backends.flash_v100.spec import prefill as spec_prefill
from vllm.v1.attention.backends.flash_v100.spec import (
    smallq_metadata,
)
from vllm.v1.attention.backends.flash_v100.spec import (
    tree as spec_tree,
)
from vllm.v1.attention.backends.flash_v100.spec import tree_masks as spec_tree_masks
from vllm.v1.attention.backends.flash_v100.spec import (
    verify_metadata as spec_verify_metadata,
)
from vllm.v1.attention.backends.flash_v100.spec.compatibility import PUBLIC_EXPORTS

if TYPE_CHECKING:
    from vllm.v1.attention.backends.flash_v100.backend import FlashAttnV100Backend
    from vllm.v1.attention.backends.flash_v100.dense_prefill import (
        clear_flash_attn_v100_workspaces,
        flash_v100_dense_prefill,
        flash_v100_dense_prefill_available,
        flash_v100_dense_prefill_lse,
        flash_v100_dense_prefill_lse_available,
    )
    from vllm.v1.attention.backends.flash_v100.impl import FlashAttnV100Impl
    from vllm.v1.attention.backends.flash_v100.metadata import (
        FlashAttnV100Metadata,
        FlashAttnV100MetadataBuilder,
    )
    from vllm.v1.attention.backends.flash_v100.ops import (
        flash_v100_turboquant_decode,
        flash_v100_turboquant_decode_available,
    )

# Modules searched by the flash_attn_v100 compatibility module.
SUBMODULES = (
    diagnostics,
    spec_prefill,
    config,
    ops,
    routing,
    debug,
    kv_layout,
    masks,
    dense_prefill,
    smallq_metadata,
    metadata,
    impl,
    backend,
    state,
    decode,
    prefill,
    verify,
    workspace,
    debug_compare,
    spec_builder,
    spec_draft,
    spec_tree,
    spec_tree_masks,
    spec_verify_metadata,
    spec_attention,
    spec_contracts,
    spec_policy,
    spec_attention_policy,
    spec_diagnostics,
)

dense_prefill.LEGACY_OBSERVATIONS = {
    name: (state, name)
    for name in (
        "_warned_prefill_dense_splitkv3_oom",
        "_warned_prefill_d256_gqa_architecture_oom",
        "_logged_prefill_fa2_d256",
        "_logged_prefill_dense_splitkv3",
        "_logged_prefill_d256_gqa_architecture",
    )
}
for _owner in SUBMODULES:
    install_owner_aliases(_owner)

# Renamed compatibility bindings resolve to their actual owner. Do not copy
# function values: old-name writes must also affect the public execution path.
COMPATIBILITY_ALIASES = {
    **{
        old: (module, new)
        for module in SUBMODULES
        for old, new in vars(module).get("LEGACY_ALIASES", {}).items()
    },
    **{name: (state, name) for name in state.LOG_KEYS},
    "_allocate_growing_workspace": (workspace, "allocate_growing_workspace"),
    "_VALID_DECODE_PARTITION_SIZES": (routing, "VALID_DECODE_PARTITION_SIZES"),
    **{
        name: (spec_policy, target)
        for name, target in spec_policy.COMPATIBILITY_ALIASES.items()
    },
    **{
        name: (spec_contracts, target)
        for name, target in spec_contracts.COMPATIBILITY_ALIASES.items()
    },
    **{
        name: (spec_tree_masks, target)
        for name, target in spec_tree_masks.COMPATIBILITY_ALIASES.items()
    },
}


def _compatibility_bindings(name: str):
    if name in COMPATIBILITY_ALIASES:
        return [COMPATIBILITY_ALIASES[name]]
    return [(module, name) for module in SUBMODULES if name in vars(module)]


__all__ = [
    *PUBLIC_EXPORTS,
    "FlashAttnV100Backend",
    "FlashAttnV100Impl",
    "FlashAttnV100Metadata",
    "FlashAttnV100MetadataBuilder",
    "clear_flash_attn_v100_workspaces",
    "flash_v100_dense_prefill",
    "flash_v100_dense_prefill_available",
    "flash_v100_dense_prefill_lse",
    "flash_v100_dense_prefill_lse_available",
    "flash_v100_turboquant_decode",
    "flash_v100_turboquant_decode_available",
]


# Resolve public re-exports through their owning modules. A copied binding would
# miss updates made through the legacy compatibility module (and monkeypatches).
def __getattr__(name: str):
    if name in __all__:
        for module in SUBMODULES:
            if name in vars(module):
                return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
