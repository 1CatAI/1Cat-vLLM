# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve graph capabilities once; dispatch retains live batch/context keys."""

from dataclasses import dataclass

from vllm.config.execution_policy import flash_v100_policy, graph_policy
from vllm.logger import init_logger
from vllm.model_executor.models.graph_contract import model_graph_contract
from vllm.platforms import current_platform

logger = init_logger(__name__)


def parse_context_buckets(raw, alias):
    if raw is None or raw == "" or raw == ():
        return ()
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return ()
        try:
            buckets = tuple(sorted({int(value.strip()) for value in raw.split(",")}))
        except ValueError as exc:
            raise ValueError(
                f"{alias} must be a comma-separated list of positive integers, "
                f"got {raw!r}"
            ) from exc
    else:
        buckets = tuple(sorted(set(raw)))
    if not buckets or buckets[0] <= 0:
        raise ValueError(f"{alias} must contain only positive integers, got {raw!r}")
    return buckets


@dataclass(frozen=True)
class GraphExecutionPlan:
    mtp_buckets: tuple[int, ...]
    mtp_explicit: bool
    compressed_buckets: tuple[int, ...]
    fp8_buckets: tuple[int, ...]
    batch_context_routing: bool
    wave_context_routing: bool
    wave_min_seq_len: int
    decode_only_capture: bool


def fp8_decode_shape(
    contract,
    *,
    sm70,
    enabled,
    ratios,
    cache_dtypes=("fp8_e5m2",),
    backends=("FLASH_ATTN_V100", "FLASHINFER_SM70"),
):
    return bool(
        not contract.speculative
        and sm70
        and enabled
        and contract.cache_dtype in cache_dtypes
        and contract.attention_backend in (None, *backends)
        and contract.gqa_shape(ratios)
    )


def resolve_graph_plan(cfg):
    policy = graph_policy(cfg)
    contract = model_graph_contract(cfg)
    cuda = current_platform.is_cuda()
    sm70 = cuda and current_platform.is_device_capability((7, 0))
    family70 = cuda and current_platform.is_device_capability_family(70)
    enabled = flash_v100_policy(cfg).enabled
    mtp = parse_context_buckets(
        policy.mtp_context_buckets, policy.aliases["mtp_context_buckets"]
    )
    compressed = (
        parse_context_buckets(
            policy.dsv4_context_buckets, policy.aliases["dsv4_context_buckets"]
        )
        if policy.dsv4_context_buckets is not None
        else contract.compressed_context_buckets()
        if family70
        else ()
    )
    if policy.fp8_context_buckets is not None:
        fp8 = parse_context_buckets(
            policy.fp8_context_buckets, policy.aliases["fp8_context_buckets"]
        )
    else:
        fp8 = (
            (8192,)
            if (
                fp8_decode_shape(contract, sm70=sm70, enabled=enabled, ratios=(6, 8))
                and isinstance(contract.max_model_len, int)
                and contract.max_model_len > 8192
            )
            else ()
        )
    if compressed and policy.dsv4_context_buckets is None:
        logger.info_once(
            "Auto-enabling SM70 compressed-index decode CUDA graph context buckets "
            "%s. Set %s explicitly to override or to an empty value to disable.",
            compressed,
            policy.aliases["dsv4_context_buckets"],
        )
    if fp8 and policy.fp8_context_buckets is None:
        logger.info_once(
            "Auto-enabling the SM70 E5M2 KV short-context decode CUDA graph "
            "at %d tokens. This keeps the short D256 GQA graph scalar-only while "
            "the unbounded graph retains the long-context XQA routes. Set %s "
            "explicitly to override or to an empty value to disable.",
            fp8[0],
            policy.aliases["fp8_context_buckets"],
        )
    e4m3 = contract.cache_dtype in ("fp8", "fp8_e4m3")
    batch = bool(
        policy.batch_context_routing
        and policy.decode_partition_size is None
        and (not e4m3 or policy.e4m3_batch_xqa)
        and fp8_decode_shape(
            contract,
            sm70=sm70,
            enabled=enabled,
            ratios=(6,),
            cache_dtypes=("fp8", "fp8_e4m3") if e4m3 else ("fp8_e5m2",),
            backends=("FLASH_ATTN_V100",)
            if e4m3
            else ("FLASH_ATTN_V100", "FLASHINFER_SM70"),
        )
    )
    return GraphExecutionPlan(
        mtp,
        policy.mtp_context_buckets is not None,
        compressed,
        fp8,
        batch,
        bool(
            batch and e4m3 and policy.e4m3_p64_p256_auto and policy.e4m3_wave_partitions
        ),
        max(
            1,
            int(
                policy.e4m3_p512_begin
                if policy.e4m3_p512_begin is not None
                else "49152"
            ),
        ),
        bool(policy.decode_only_capture),
    )
