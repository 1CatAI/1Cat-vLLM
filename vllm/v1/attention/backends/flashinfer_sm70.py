# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental FlashInfer-shaped attention backend for SM70.

This backend intentionally has its own registry name so SM70 FlashInfer work
can be developed and measured without changing the accepted FLASH_ATTN_V100
baseline.  The initial implementation reuses the proven SM70 metadata and
execution path; kernels are replaced here only after their microbenchmark and
NCU promotion gates pass.
"""

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn_v100 import (
    FlashAttnV100Backend,
    FlashAttnV100Impl,
    FlashAttnV100MetadataBuilder,
)
from vllm.v1.attention.backends.flashinfer_sm70_planner import (
    FlashInferSM70PlannerDecision,
    SM70KernelResources,
    make_delegate_decision,
    plan_decode,
    plan_prefill,
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


class FlashInferSM70MetadataBuilder(FlashAttnV100MetadataBuilder):
    """Graph-safe metadata bridge for the independent SM70 route."""

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
        if int(common_attn_metadata.max_query_len) > 1:
            if self.num_heads_q % self.num_heads_kv != 0:
                raise ValueError("SM70 GQA planning requires divisible Q/KV heads")
            q_per_kv = self.num_heads_q // self.num_heads_kv
            plan = plan_prefill(
                packed_qo_len=int(common_attn_metadata.num_actual_tokens) * q_per_kv,
                kv_len=int(common_attn_metadata.max_seq_len),
                num_kv_heads=self.num_heads_kv,
                cta_tile_q=_PREFILL_CTA_TILE_Q,
                kv_tile_size=_KV_TILE_SIZE,
                resources=_ACCEPTED_BM32_RESOURCES,
                enable_cuda_graph=enable_cuda_graph,
            )
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
