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
- ``smallq_metadata``: small-query (DFlash2/MTP) group metadata.
- ``metadata``: metadata and its builder.
- ``impl``: the attention implementation.
- ``backend``: backend registration.

Modules reach each other through the module object (``_routing._record_route``)
so a rebound global or a monkeypatch has one owner.
"""

from typing import TYPE_CHECKING

from vllm.v1.attention.backends.flash_v100 import (  # noqa: F401
    backend,
    debug,
    dense_prefill,
    impl,
    kv_layout,
    masks,
    metadata,
    ops,
    routing,
    smallq_metadata,
)

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
    from vllm.v1.attention.backends.flash_v100.smallq_metadata import (
        DFlash2SmallQGroupDescriptor,
        DFlash2SmallQPreparedMetadata,
        prepare_dflash2_smallq_group_metadata,
    )

# Modules searched by the flash_attn_v100 compatibility module.
SUBMODULES = (
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
)

__all__ = [
    "DFlash2SmallQGroupDescriptor",
    "DFlash2SmallQPreparedMetadata",
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
    "prepare_dflash2_smallq_group_metadata",
]


# Resolve public re-exports through their owning modules. A copied binding would
# miss updates made through the legacy compatibility module (and monkeypatches).
def __getattr__(name: str):
    if name in __all__:
        for module in SUBMODULES:
            if name in vars(module):
                return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
