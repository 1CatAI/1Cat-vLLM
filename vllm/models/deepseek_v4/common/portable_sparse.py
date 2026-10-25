# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability contract for the shared sparse-MLA and indexer provider.

GLM reuses this provider's FP16, NoPE and software-FP8 operations. Hardware
admission belongs beside those operations; callers need no model-name test.

These kernels use Triton FP16 arithmetic, software E4M3 and PyTorch GEMM;
they do not require a Volta binary or DeepGEMM's SM90 instructions. Keep
admission explicit: adding an architecture requires kernel parity evidence.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.platforms.interface import DeviceCapability

# SM72 admission predates the heterogeneous route. Preserve that contract;
# the new deployment/parity targets are SM70, SM86 and SM89.
_CAPABILITIES = frozenset({(7, 0), (7, 2), (8, 6), (8, 9)})


def supports_portable_sparse(capability: "DeviceCapability | None") -> bool:
    return (
        capability is not None
        and (
            capability.major,
            capability.minor,
        )
        in _CAPABILITIES
    )


def use_portable_sparse() -> bool:
    # Lazy import keeps platform discovery independent of model registration.
    # The platform resolves None to the current worker device, not ordinal 0.
    from vllm.platforms import current_platform

    return current_platform.is_cuda() and supports_portable_sparse(
        current_platform.get_device_capability()
    )


def sparse_fp16_gemm_max_tokens(
    capability: "DeviceCapability | None",
    num_heads: int,
    head_dim: int,
    index_width: int,
    dtype: torch.dtype,
) -> int:
    """Admit only measured small-batch tensor-core attention geometries."""
    if (
        capability is not None
        and (capability.major, capability.minor) == (7, 0)
        and (num_heads, head_dim) == (64, 512)
        # KPool reserves its incomplete tail before padding to 128 columns.
        and index_width in (2048, 2176)
        and dtype == torch.float16
    ):
        return 15
    return 0


def can_use_sparse_fp16_gemm_prefill(
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    max_decode_query_len: int,
    *,
    num_actual_tokens: int,
) -> bool:
    """Prove every query exceeds the decode bound without reading device data.

    For N queries, each other query is at most max_query_len, so every query
    is at least total_tokens - (N - 1) * max_query_len. Admit only when this
    lower bound exceeds the maximum decode length. This conservatively rejects
    some uneven prefill batches while excluding every mixed decode batch.
    """
    if (
        num_reqs <= 0
        or max_decode_query_len < 1
        or max_query_len <= max_decode_query_len
        or num_tokens != num_actual_tokens
        or not max_query_len <= num_actual_tokens <= num_reqs * max_query_len
    ):
        return False
    min_query_lower_bound = num_actual_tokens - (num_reqs - 1) * max_query_len
    return min_query_lower_bound > max_decode_query_len


def portable_sparse_mla_config_error(
    *,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> str | None:
    """The actual cache/query contract, independent of a model's name."""
    if dtype != torch.float16:
        return "portable sparse MLA requires FP16 queries"
    if kv_cache_dtype not in {None, "auto", "float16", "fp8", "fp8_e4m3"}:
        return "portable sparse MLA requires FP16 or packed E4M3 KV"
    if head_size != 512 or kv_lora_rank != 512 or qk_rope_head_dim != 0:
        return "portable sparse MLA requires NoPE MLA with kv_lora_rank=512"
    return None
