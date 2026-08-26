# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental FlashInfer-shaped attention backend for SM70.

This backend intentionally has its own registry name so SM70 FlashInfer work
can be developed and measured without changing the accepted FLASH_ATTN_V100
baseline.  The initial implementation reuses the proven SM70 metadata and
execution path; kernels are replaced here only after their microbenchmark and
NCU promotion gates pass.
"""

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn_v100 import (
    FlashAttnV100Backend,
    FlashAttnV100Impl,
    FlashAttnV100MetadataBuilder,
    _is_cuda_graph_capturing,
    _record_route,
    _split_paged_kv_cache,
)
from vllm.v1.attention.backends.flashinfer_sm70_planner import (
    FlashInferSM70PlannerDecision,
    FlashInferSM70PrefillWorkDescription,
    SM70KernelResources,
    make_delegate_decision,
    plan_decode,
    plan_prefill_requests,
)

logger = init_logger(__name__)

_ACCEPTED_BM32_RESOURCES = SM70KernelResources(
    threads_per_cta=512,
    registers_per_thread=64,
    shared_bytes_per_cta=41_936,
)
_ACCEPTED_DECODE_RESOURCES = SM70KernelResources(
    threads_per_cta=256,
    registers_per_thread=186,
    shared_bytes_per_cta=41_220,
)
_PREFILL_CTA_TILE_Q = 32
_KV_TILE_SIZE = 128
_DECODE_PARTITION_SIZE = 256
_FIXED_PREFILL_DISPATCH = "flashinfer_sm70_fixed_entry"
_FIXED_PREFILL_KERNEL = (
    "flash_attn_prefill_paged_d256_bm32_allp_pair_scratch<true,true,true>"
)
_SPLITKV3_PREFILL_DISPATCH = "flashinfer_sm70_splitkv3_fast_visible"
_SPLITKV3_PREFILL_KERNEL = (
    "flash_attn_prefill_paged_d256_bm32_splitkv3_partial<false>+merge"
)
_PROMOTED_SPLITKV3_QUERY_LEN = 8_096
_PROMOTED_SPLITKV3_KV_LENS = (
    80_960,
    89_056,
    97_152,
    105_248,
    113_344,
    121_440,
)
_PROMOTED_SPLITKV3_NUM_HEADS = 6
_PROMOTED_SPLITKV3_NUM_KV_HEADS = 1


def _load_fixed_prefill_entry():
    try:
        from flash_attn_v100 import (
            flash_attn_prefill_paged_d256_bm32_allp_pair_scratch,
        )
    except (ImportError, OSError):
        return None
    return flash_attn_prefill_paged_d256_bm32_allp_pair_scratch


def _load_splitkv3_prefill_entry():
    try:
        from flash_attn_v100 import (
            flash_attn_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3,
        )
    except (ImportError, OSError):
        return None
    return flash_attn_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3


def _splitkv3_has_visible_anchor(query_len: int, kv_len: int) -> bool:
    if query_len <= 0 or kv_len < query_len:
        return False
    total_kv_tiles = (kv_len + _KV_TILE_SIZE - 1) // _KV_TILE_SIZE
    last_split_begin = (2 * total_kv_tiles // 3) * _KV_TILE_SIZE
    return total_kv_tiles >= 3 and last_split_begin <= kv_len - query_len


def _is_promoted_splitkv3_shape(
    *,
    query_len: int,
    kv_len: int,
    num_heads: int,
    num_kv_heads: int,
) -> bool:
    return (
        query_len == _PROMOTED_SPLITKV3_QUERY_LEN
        and kv_len in _PROMOTED_SPLITKV3_KV_LENS
        and num_heads == _PROMOTED_SPLITKV3_NUM_HEADS
        and num_kv_heads == _PROMOTED_SPLITKV3_NUM_KV_HEADS
        and _splitkv3_has_visible_anchor(query_len, kv_len)
    )


class FlashInferSM70MetadataBuilder(FlashAttnV100MetadataBuilder):
    """Graph-safe metadata bridge for the independent SM70 route."""

    @staticmethod
    def _cpu_ints(values) -> tuple[int, ...] | None:
        if values is None:
            return None
        host_values = values.tolist() if hasattr(values, "tolist") else values
        return tuple(int(value) for value in host_values)

    @classmethod
    def _prefill_request_lengths(
        cls,
        common_attn_metadata,
        attn_metadata,
    ) -> tuple[tuple[int, ...], tuple[int, ...], str]:
        num_reqs = int(common_attn_metadata.num_reqs)
        if num_reqs <= 0:
            raise ValueError("SM70 prefill planning requires at least one request")

        query_start_loc = cls._cpu_ints(
            getattr(common_attn_metadata, "query_start_loc_cpu", None)
        )
        query_is_exact = (
            query_start_loc is not None and len(query_start_loc) >= num_reqs + 1
        )
        if query_is_exact:
            assert query_start_loc is not None
            qo_lens = tuple(
                query_start_loc[index + 1] - query_start_loc[index]
                for index in range(num_reqs)
            )
        else:
            qo_lens = (int(common_attn_metadata.max_query_len),) * num_reqs

        exact_seq_lens = cls._cpu_ints(
            getattr(common_attn_metadata, "_seq_lens_cpu", None)
        )
        seq_is_exact = exact_seq_lens is not None and len(exact_seq_lens) >= num_reqs
        if seq_is_exact:
            assert exact_seq_lens is not None
            kv_lens = exact_seq_lens[:num_reqs]
        else:
            upper_seq_lens = cls._cpu_ints(
                getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
            )
            if upper_seq_lens is None:
                upper_seq_lens = cls._cpu_ints(
                    getattr(attn_metadata, "seq_lens_cpu", None)
                )
            if upper_seq_lens is None or len(upper_seq_lens) < num_reqs:
                kv_lens = (int(common_attn_metadata.max_seq_len),) * num_reqs
            else:
                kv_lens = upper_seq_lens[:num_reqs]

        planning_mode = "exact" if query_is_exact and seq_is_exact else "upper_bound"
        return qo_lens, kv_lens, planning_mode

    @staticmethod
    def _decode_seq_lens(common_attn_metadata, attn_metadata) -> list[int]:
        seq_lens = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        if seq_lens is None:
            seq_lens = getattr(
                common_attn_metadata,
                "seq_lens_cpu_upper_bound",
                None,
            )
        if seq_lens is None:
            seq_lens = getattr(attn_metadata, "seq_lens_cpu", None)
        if seq_lens is None:
            max_seq_len = int(common_attn_metadata.max_seq_len)
            return [max_seq_len] * max(1, int(common_attn_metadata.num_reqs))
        values = seq_lens.tolist() if hasattr(seq_lens, "tolist") else seq_lens
        return [int(value) for value in values]

    def _attach_planner_decision(
        self,
        attn_metadata,
        common_attn_metadata,
        *,
        stage: str,
    ):
        enable_cuda_graph = bool(
            getattr(attn_metadata, "flash_v100_cudagraph_capture", False)
        )
        prefill_work_description = None
        if int(common_attn_metadata.max_query_len) > 1:
            if self.num_heads_q % self.num_heads_kv != 0:
                raise ValueError("SM70 GQA planning requires divisible Q/KV heads")
            qo_lens, kv_lens, planning_mode = self._prefill_request_lengths(
                common_attn_metadata,
                attn_metadata,
            )
            prefill_work_description = plan_prefill_requests(
                qo_lens=qo_lens,
                kv_lens=kv_lens,
                num_qo_heads=self.num_heads_q,
                num_kv_heads=self.num_heads_kv,
                head_dim_vo=self.headdim,
                cta_tile_q=_PREFILL_CTA_TILE_Q,
                kv_tile_size=_KV_TILE_SIZE,
                resources=_ACCEPTED_BM32_RESOURCES,
                enable_cuda_graph=enable_cuda_graph,
                planning_mode=planning_mode,
            )
            plan = prefill_work_description.plan
            workload = "prefill"
        else:
            plan = plan_decode(
                seq_lens=self._decode_seq_lens(
                    common_attn_metadata,
                    attn_metadata,
                ),
                num_kv_heads=self.num_heads_kv,
                kv_tile_size=_KV_TILE_SIZE,
                resources=_ACCEPTED_DECODE_RESOURCES,
                enable_cuda_graph=enable_cuda_graph,
                fixed_split_size=_DECODE_PARTITION_SIZE,
            )
            workload = "decode"

        decision = make_delegate_decision(
            stage=stage,
            workload=workload,
            plan=plan,
        )
        attn_metadata.flashinfer_sm70_planner_decision = decision
        attn_metadata.flashinfer_sm70_prefill_work_description = (
            prefill_work_description
        )
        attn_metadata.flashinfer_sm70_route_proof = decision.route_proof
        attn_metadata.flashinfer_sm70_dispatch_observed = None
        attn_metadata.flashinfer_sm70_prefill_paged_routes_observed = ()
        attn_metadata.flashinfer_sm70_runtime_route_proof = None
        return attn_metadata

    def build_for_cudagraph_capture(self, common_attn_metadata):
        attn_metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        return self._attach_planner_decision(
            attn_metadata,
            common_attn_metadata,
            stage="capture",
        )

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build: bool = False,
        ddtree_parent_ids=None,
        ddtree_num_tree_tokens_cpu=None,
    ):
        attn_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
            ddtree_parent_ids,
            ddtree_num_tree_tokens_cpu,
        )
        return self._attach_planner_decision(
            attn_metadata,
            common_attn_metadata,
            stage="build",
        )

    def build_for_drafting(self, common_attn_metadata, draft_index: int):
        attn_metadata = super().build_for_drafting(
            common_attn_metadata,
            draft_index,
        )
        return self._attach_planner_decision(
            attn_metadata,
            common_attn_metadata,
            stage=f"draft:{draft_index}",
        )


class FlashInferSM70Impl(FlashAttnV100Impl):
    """SM70 implementation boundary for promoted FlashInfer-style kernels."""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "FLASHINFER_SM70 is in bootstrap mode: unpromoted shapes delegate "
            "to FLASH_ATTN_V100 and must not be reported as FlashInfer kernel "
            "speedups."
        )
        super().__init__(*args, **kwargs)
        self.flash_attn_prefill_paged_d256_bm32_allp_pair_scratch = (
            _load_fixed_prefill_entry()
        )
        self.flash_attn_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3 = (
            _load_splitkv3_prefill_entry()
        )
        self.last_route_proof: dict[str, object] | None = None
        self._flashinfer_sm70_active_metadata = None

    @staticmethod
    def _validate_route_proof(
        attn_metadata,
    ) -> FlashInferSM70PlannerDecision:
        decision = getattr(
            attn_metadata,
            "flashinfer_sm70_planner_decision",
            None,
        )
        if not isinstance(decision, FlashInferSM70PlannerDecision):
            raise RuntimeError(
                "FLASHINFER_SM70 metadata is missing its planner decision"
            )
        proof = getattr(attn_metadata, "flashinfer_sm70_route_proof", None)
        if proof != decision.route_proof:
            raise RuntimeError("FLASHINFER_SM70 route proof does not match its plan")
        if (
            decision.dispatch != "delegate"
            or decision.delegate_backend != "FLASH_ATTN_V100"
            or decision.delegate_dispatch != "flash_v100_runtime_dispatch"
            or decision.kernel_promoted
        ):
            raise RuntimeError("FLASHINFER_SM70 unpromoted route is not a delegate")
        return decision

    def _run_prefill_paged_call(self, *, route: str, **kwargs):
        attn_metadata = getattr(
            self,
            "_flashinfer_sm70_active_metadata",
            None,
        )
        if attn_metadata is not None:
            routes = getattr(
                attn_metadata,
                "flashinfer_sm70_prefill_paged_routes_observed",
                (),
            )
            attn_metadata.flashinfer_sm70_prefill_paged_routes_observed = (
                *routes,
                route,
            )
        return super()._run_prefill_paged_call(route=route, **kwargs)

    def _fixed_prefill_inputs(
        self,
        query,
        kv_cache,
        attn_metadata,
        output,
        output_scale,
        output_block_scale,
    ) -> (
        tuple[
            int,
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ):
        work = getattr(
            attn_metadata,
            "flashinfer_sm70_prefill_work_description",
            None,
        )
        if not isinstance(work, FlashInferSM70PrefillWorkDescription):
            return None
        if (
            work.enable_cuda_graph
            or bool(
                getattr(
                    attn_metadata,
                    "flash_v100_cudagraph_capture",
                    False,
                )
            )
            or not isinstance(query, torch.Tensor)
            or _is_cuda_graph_capturing(query)
            or not isinstance(output, torch.Tensor)
            or output_scale is not None
            or output_block_scale is not None
        ):
            return None
        if (
            not bool(getattr(attn_metadata, "causal", True))
            or getattr(self, "attn_type", None) != AttentionType.DECODER
            or bool(getattr(self, "use_triton_prefill", False))
            or not bool(getattr(self, "use_flash_v100_prefill_paged", False))
            or bool(getattr(self, "use_flash_v100_prefill_splitkv", False))
            or tuple(getattr(self, "sliding_window", (-1, -1)) or (-1, -1)) != (-1, -1)
            or bool(getattr(self, "use_flash_v100_prefill_bfla", False))
            or getattr(attn_metadata, "ddtree_parent_ids", None) is not None
            or getattr(attn_metadata, "ddtree_num_tree_tokens_cpu", None) is not None
            or getattr(self, "alibi_slopes", None) is not None
            or float(getattr(self, "logits_soft_cap", 0.0)) != 0.0
            or getattr(self, "sinks", None) is not None
        ):
            return None
        # This explicit backend intentionally prioritizes its validated fixed
        # paged entry over the inherited contiguous-dense prefill heuristic.
        # Therefore use_flash_v100_prefill_contig_dense is not an exclusion.
        if (
            not work.is_exact
            or work.split_kv
            or len(work.qo_lens) != 1
            or len(work.kv_lens) != 1
        ):
            return None

        query_len = work.qo_lens[0]
        kv_len = work.kv_lens[0]
        num_heads = int(getattr(self, "num_heads", 0))
        num_kv_heads = int(getattr(self, "num_kv_heads", 0))
        if (
            query_len < 32
            or kv_len <= query_len
            or num_heads <= 0
            or num_kv_heads <= 0
            or num_heads % num_kv_heads != 0
            or work.num_qo_heads != num_heads
            or work.num_kv_heads != num_kv_heads
            or work.packed_qo_lens[0] != query_len * (num_heads // num_kv_heads)
            or int(getattr(self, "head_size", 0)) != 256
            or int(getattr(attn_metadata, "num_actual_tokens", -1)) != query_len
            or getattr(self, "kv_cache_dtype", "auto") not in ("auto", "float16")
        ):
            return None
        if (
            query.ndim != 3
            or output.ndim != 3
            or query.shape[0] < query_len
            or output.shape[0] < query_len
            or query.shape[1:] != (num_heads, 256)
            or output.shape[1:] != (num_heads, 256)
            or query.dtype != torch.float16
            or output.dtype != torch.float16
        ):
            return None

        try:
            key_cache, value_cache = _split_paged_kv_cache(kv_cache)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        if (
            not isinstance(key_cache, torch.Tensor)
            or not isinstance(value_cache, torch.Tensor)
            or key_cache.ndim != 4
            or value_cache.shape != key_cache.shape
            or key_cache.shape[1:] != (784, num_kv_heads, 256)
            or key_cache.dtype != torch.float16
            or value_cache.dtype != torch.float16
            or key_cache.device != query.device
            or value_cache.device != query.device
        ):
            return None

        block_table = getattr(attn_metadata, "block_table", None)
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if (
            not isinstance(block_table, torch.Tensor)
            or not isinstance(seq_lens, torch.Tensor)
            or block_table.ndim != 2
            or block_table.shape[0] < 1
            or seq_lens.ndim != 1
            or seq_lens.shape[0] < 1
        ):
            return None
        return query_len, kv_len, key_cache, value_cache, block_table, seq_lens

    def _run_fixed_prefill(
        self,
        decision: FlashInferSM70PlannerDecision,
        query: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        fixed_inputs: tuple[
            int,
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        query_len, kv_len, key_cache, value_cache, block_table, seq_lens = fixed_inputs
        query_bhmd = query[:query_len].permute(1, 0, 2).unsqueeze(0).contiguous()
        splitkv3_shape = _is_promoted_splitkv3_shape(
            query_len=query_len,
            kv_len=kv_len,
            num_heads=int(self.num_heads),
            num_kv_heads=int(self.num_kv_heads),
        )
        if splitkv3_shape:
            splitkv3_entry = getattr(
                self,
                "flash_attn_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3",
                None,
            )
            if splitkv3_entry is None:
                raise RuntimeError(
                    "FLASHINFER_SM70 promoted split-KV prefill entry is unavailable"
                )
            output_bhmd, _ = splitkv3_entry(
                query_bhmd,
                key_cache,
                value_cache,
                block_table[:1],
                kv_len,
                softmax_scale=self.scale,
            )
            observed_dispatch = _SPLITKV3_PREFILL_DISPATCH
            executed_kernel = _SPLITKV3_PREFILL_KERNEL
        else:
            fixed_entry = getattr(
                self,
                "flash_attn_prefill_paged_d256_bm32_allp_pair_scratch",
                None,
            )
            if fixed_entry is None:
                raise RuntimeError(
                    "FLASHINFER_SM70 fixed D256 BM32 prefill entry is unavailable"
                )
            output_bhmd, _ = fixed_entry(
                query_bhmd,
                key_cache,
                value_cache,
                block_table[:1],
                seq_lens[:1],
                softmax_scale=self.scale,
            )
            observed_dispatch = _FIXED_PREFILL_DISPATCH
            executed_kernel = _FIXED_PREFILL_KERNEL
        output[:query_len].copy_(output_bhmd.squeeze(0).permute(1, 0, 2))
        _record_route(observed_dispatch)
        logger.info_once(
            "FLASHINFER_SM70 promoted BM32 prefill active "
            "(dispatch=%s, q=%d, kv=%d, heads_q=%d, heads_kv=%d, "
            "head_dim=256, page=784).",
            observed_dispatch,
            query_len,
            kv_len,
            self.num_heads,
            self.num_kv_heads,
        )

        runtime_proof = {
            "selected_backend": "FLASHINFER_SM70",
            "planner_stage": decision.stage,
            "workload": decision.workload,
            "planner_candidate": decision.planner_candidate,
            "planner_proof": decision.route_proof,
            "observed_dispatch": observed_dispatch,
            "executed_backend": "FLASHINFER_SM70",
            "executed_kernel": executed_kernel,
            "kernel_promoted": True,
            "query_len": query_len,
            "actual_n": kv_len,
            "splitkv3_fast_visible": splitkv3_shape,
        }
        attn_metadata.flashinfer_sm70_dispatch_observed = observed_dispatch
        attn_metadata.flashinfer_sm70_runtime_route_proof = runtime_proof
        self.last_route_proof = runtime_proof
        return output

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        if attn_metadata is None:
            self.last_route_proof = None
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        decision = self._validate_route_proof(attn_metadata)
        attn_metadata.flashinfer_sm70_dispatch_observed = None
        attn_metadata.flashinfer_sm70_prefill_paged_routes_observed = ()
        attn_metadata.flashinfer_sm70_runtime_route_proof = None
        self.last_route_proof = None
        fixed_inputs = self._fixed_prefill_inputs(
            query,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        if fixed_inputs is not None:
            return self._run_fixed_prefill(
                decision,
                query,
                attn_metadata,
                output,
                fixed_inputs,
            )

        self._flashinfer_sm70_active_metadata = attn_metadata
        try:
            result = super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        finally:
            self._flashinfer_sm70_active_metadata = None

        observed_dispatch = decision.delegate_dispatch
        attn_metadata.flashinfer_sm70_dispatch_observed = observed_dispatch
        runtime_proof = {
            **decision.route_proof,
            "observed_dispatch": observed_dispatch,
            "observed_prefill_paged_routes": list(
                attn_metadata.flashinfer_sm70_prefill_paged_routes_observed
            ),
        }
        attn_metadata.flashinfer_sm70_runtime_route_proof = runtime_proof
        self.last_route_proof = runtime_proof
        return result


class FlashInferSM70Backend(FlashAttnV100Backend):
    """Explicit-only FlashInfer-style backend for Volta GPUs."""

    @staticmethod
    def get_impl_cls():
        return FlashInferSM70Impl

    @staticmethod
    def get_builder_cls():
        return FlashInferSM70MetadataBuilder

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_SM70"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(7, 0)
