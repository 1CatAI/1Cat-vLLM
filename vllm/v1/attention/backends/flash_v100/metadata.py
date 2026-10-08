# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 attention metadata, its builder and mixed decode-row plans."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import Any, cast

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100.spec.metadata_contracts import (
    INPUT_FIELDS,
    METADATA_FIELDS,
    STATE_FIELDS,
    MetadataInputs,
    MetadataOps,
    SpecMetadataPacket,
)
from vllm.v1.attention.backends.flash_v100.spec.metadata_state import SpecMetadataState
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import PrefixAnchoredSWASpec

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


class FlashAttnV100Metadata(TritonAttentionMetadata):
    """Common metadata and a single owned speculative packet."""

    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    causal: bool
    max_model_len: int
    flash_v100_cudagraph_capture: bool
    flash_v100_batch_context_routing: bool
    flash_v100_contig_dense_cache: dict[tuple[int, int, int, int, int], int]
    prefix_anchor_lens: torch.Tensor | None = None
    decode_sliding_window: int | None = None
    flash_v100_decode_max_seq_len_hint: int | None
    flash_v100_decode_workspace_seq_capacity_hint: int | None
    flash_v100_static_decode_seq_hint: int | None
    flash_v100_decode_active_num_partitions: torch.Tensor | None

    @property
    def spec_state(self) -> SpecMetadataPacket:
        attributes = vars(self)
        if "_spec_state" not in attributes:
            packet = SpecMetadataPacket()
            # Adoption keeps tensor identities and accepts existing legacy fields.
            for name in METADATA_FIELDS:
                if name in attributes:
                    setattr(packet, name, attributes.pop(name))
            attributes["_spec_state"] = packet
        return attributes["_spec_state"]

    def __getattr__(self, name: str) -> Any:
        if name in METADATA_FIELDS:
            return getattr(self.spec_state, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in METADATA_FIELDS:
            setattr(self.spec_state, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in METADATA_FIELDS:
            delattr(self.spec_state, name)
        else:
            super().__delattr__(name)

    def __copy__(self):
        result = object.__new__(type(self))
        vars(result).update(vars(self))
        if "_spec_state" in vars(self):
            vars(result)["_spec_state"] = copy(self.spec_state)
        return result


def _as_flash_v100_metadata(
    attn_metadata: TritonAttentionMetadata,
) -> FlashAttnV100Metadata:
    # The inherited builder creates this exact class. Adopt its object in place
    # so all existing tensor references and metadata identities remain valid.
    if type(attn_metadata) is TritonAttentionMetadata:
        attn_metadata.__class__ = FlashAttnV100Metadata
    if isinstance(attn_metadata, FlashAttnV100Metadata):
        _ = attn_metadata.spec_state
    return cast(FlashAttnV100Metadata, attn_metadata)


class FlashAttnV100MetadataBuilder(TritonAttentionMetadataBuilder):
    """Attach CPU metadata for the dense prefill path."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        cache_config = getattr(self.vllm_config, "cache_config", None)
        model_config = self.vllm_config.model_config
        hf_text_config = getattr(model_config, "hf_text_config", None)
        num_attention_heads = getattr(hf_text_config, "num_attention_heads", None)
        num_key_value_heads = getattr(hf_text_config, "num_key_value_heads", None)
        head_dim = getattr(hf_text_config, "head_dim", None)
        batch_context_shape_supported = (
            isinstance(num_attention_heads, int)
            and isinstance(num_key_value_heads, int)
            and num_key_value_heads > 0
            and num_attention_heads == 6 * num_key_value_heads
            and head_dim == 256
        )
        self._is_speculative_draft_model = (
            spec_config is not None
            and getattr(spec_config, "draft_model_config", None)
            is self.vllm_config.model_config
        )
        self._batch_context_routing_enabled = (
            _config.registered("VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING")
            and _config.registered("VLLM_FLASH_V100_DECODE_PARTITION_SIZE") is None
            and spec_config is None
            and _routing._batch_context_routing_cache_dtype_supported(
                getattr(cache_config, "cache_dtype", None)
            )
            and batch_context_shape_supported
        )
        self.initialize_spec_state(spec_config)
        # Prefix-anchored SWA: persistent per-request prompt-length buffer so
        # the device address stays stable across steps.
        kv_cache_spec = self.kv_cache_spec
        self.decode_sliding_window = (
            kv_cache_spec.decode_sliding_window
            if isinstance(kv_cache_spec, PrefixAnchoredSWASpec)
            else None
        )
        self.persistent_prefix_anchor_lens: torch.Tensor | None = None
        if self.decode_sliding_window is not None:
            self.persistent_prefix_anchor_lens = torch.empty(
                self.vllm_config.scheduler_config.max_num_seqs,
                dtype=torch.int32,
                device=self.device,
            )
        self._decode_active_num_partitions: torch.Tensor | None = None

    def initialize_spec_state(self, spec_config) -> None:
        self.spec_state = SpecMetadataState(
            MetadataInputs(
                id(self),
                self.vllm_config,
                self.device,
                getattr(self, "block_size", 0),
                self._is_speculative_draft_model,
            ),
            MetadataOps(
                base_build=super().build,
                attach_common=self._attach_common_flash_metadata,
                attach_prefix=self._attach_prefix_anchored_metadata,
                attach_shape_hints=self._attach_decode_shape_hints,
                update_active_partitions=self._update_decode_active_num_partitions,
            ),
            spec_config,
        )

    def __getattr__(self, name):
        # Legacy private callers continue to reach the single state owner.
        state = vars(self).get("spec_state")
        if state is not None:
            return getattr(state, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        state = vars(self).get("spec_state")
        if state is not None:
            if name in STATE_FIELDS:
                setattr(state, name, value)
                return
            if name in INPUT_FIELDS:
                state.inputs = replace(state.inputs, **{INPUT_FIELDS[name]: value})
        super().__setattr__(name, value)

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        **feature_inputs,
    ):
        return self.spec_state.build(
            common_prefix_len, common_attn_metadata, fast_build, **feature_inputs
        )

    def build_for_drafting(self, common_attn_metadata, draft_index):
        return self.spec_state.build_for_drafting(common_attn_metadata, draft_index)

    def _stabilize_draft_graph_metadata(self, attn_metadata, common_attn_metadata):
        self.spec_state._stabilize_draft_graph_metadata(
            attn_metadata, common_attn_metadata
        )

    def _attach_prefix_anchored_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        window = self.decode_sliding_window
        if window is None:
            return

        prefix_anchor_lens = common_attn_metadata.prefix_anchor_lens
        if prefix_anchor_lens is None:
            raise RuntimeError(
                "prefix-anchored SWA requires per-request prefix lengths"
            )
        assert self.persistent_prefix_anchor_lens is not None
        anchor_reqs = common_attn_metadata.num_reqs
        if prefix_anchor_lens.ndim != 1 or prefix_anchor_lens.numel() < anchor_reqs:
            raise RuntimeError(
                "prefix-anchored SWA prefix lengths must have shape [num_reqs]"
            )
        if anchor_reqs > self.persistent_prefix_anchor_lens.numel():
            raise RuntimeError(
                "prefix-anchored SWA request count exceeds metadata capacity"
            )

        prefix_anchor_lens = prefix_anchor_lens.to(
            device=self.device, dtype=torch.int32, non_blocking=True
        )
        persistent_anchor_lens = self.persistent_prefix_anchor_lens[:anchor_reqs]
        persistent_anchor_lens.copy_(prefix_anchor_lens[:anchor_reqs])
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.prefix_anchor_lens = persistent_anchor_lens
        flash_metadata.decode_sliding_window = window

    def _attach_common_flash_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        if seq_lens_cpu is None:
            # Async speculative decode keeps device seq_lens authoritative and
            # may deliberately omit the exact CPU shadow. Flash-V100 only uses
            # this CPU view for route/partition hints; the kernel still consumes
            # attn_metadata.seq_lens, so an upper bound is preferable to the
            # deprecated lazy seq_lens.to("cpu") sync.
            seq_lens_cpu = getattr(
                common_attn_metadata,
                "seq_lens_cpu_upper_bound",
                None,
            )
        flash_metadata.seq_lens_cpu = (
            seq_lens_cpu
            if seq_lens_cpu is not None
            else common_attn_metadata.seq_lens_cpu
        )
        flash_metadata.causal = common_attn_metadata.causal
        self.spec_state.attach_common(attn_metadata)
        flash_metadata.max_model_len = self.vllm_config.model_config.max_model_len
        flash_metadata.flash_v100_cudagraph_capture = False
        flash_metadata.flash_v100_batch_context_routing = (
            _routing._batch_context_routing_for_graph_variant(
                self._batch_context_routing_enabled,
                getattr(common_attn_metadata, "cudagraph_graph_variant", None),
            )
        )

    def _attach_decode_shape_hints(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
        *,
        static_decode: bool = False,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.flash_v100_decode_max_seq_len_hint = None
        flash_metadata.flash_v100_decode_workspace_seq_capacity_hint = None
        flash_metadata.flash_v100_static_decode_seq_hint = None

        max_query_len = int(getattr(common_attn_metadata, "max_query_len", 0) or 0)
        if max_query_len != 1:
            return

        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            max_seq_len_hint = int(seq_lens_cpu.max().item())
        else:
            max_seq_len_hint = int(getattr(common_attn_metadata, "max_seq_len", 0) or 0)
        if max_seq_len_hint <= 0:
            return

        flash_metadata.flash_v100_decode_max_seq_len_hint = max_seq_len_hint
        if not static_decode:
            return

        block_table = getattr(common_attn_metadata, "block_table_tensor", None)
        if block_table is None:
            block_table = getattr(attn_metadata, "block_table", None)
        raw_seq_capacity = (
            int(block_table.shape[1]) * int(self.block_size)
            if block_table is not None
            else max_seq_len_hint
        )
        static_seq_capacity = max(
            max_seq_len_hint,
            int(getattr(common_attn_metadata, "max_seq_len", 0) or 0),
        )
        workspace_seq_capacity = min(raw_seq_capacity, static_seq_capacity)
        if (
            raw_seq_capacity > max_seq_len_hint
            or workspace_seq_capacity > max_seq_len_hint
        ):
            flash_metadata.flash_v100_static_decode_seq_hint = workspace_seq_capacity
        flash_metadata.flash_v100_decode_workspace_seq_capacity_hint = (
            workspace_seq_capacity
        )

    def _ensure_decode_active_num_partitions(self) -> torch.Tensor:
        if self._decode_active_num_partitions is None:
            self._decode_active_num_partitions = torch.empty(
                (1,),
                dtype=torch.int32,
                device=self.device,
            )
        return self._decode_active_num_partitions

    def _update_decode_active_num_partitions(
        self,
        attn_metadata: TritonAttentionMetadata,
        *,
        stage: str,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.flash_v100_decode_active_num_partitions = None
        if not _routing._decode_dynamic_partitions_enabled():
            return

        max_seq_len_hint = getattr(
            attn_metadata,
            "flash_v100_decode_max_seq_len_hint",
            None,
        )
        if max_seq_len_hint is None:
            return

        if (
            getattr(
                attn_metadata,
                "flash_v100_decode_workspace_seq_capacity_hint",
                None,
            )
            is None
            and self._decode_active_num_partitions is None
        ):
            return

        partition_size = _routing._decode_partition_size_for_metadata(
            int(max_seq_len_hint)
        )
        active = max(1, (int(max_seq_len_hint) + partition_size - 1) // partition_size)
        active_num_partitions = self._ensure_decode_active_num_partitions()
        active_num_partitions.fill_(active)
        flash_metadata.flash_v100_decode_active_num_partitions = active_num_partitions
        _routing._trace_decode_active_metadata(
            stage=stage,
            max_seq_len_hint=int(max_seq_len_hint),
            workspace_seq_capacity_hint=getattr(
                attn_metadata,
                "flash_v100_decode_workspace_seq_capacity_hint",
                None,
            ),
            static_decode_seq_hint=getattr(
                attn_metadata,
                "flash_v100_static_decode_seq_hint",
                None,
            ),
            active=active,
            partition_size=partition_size,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        capture_seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        capture_seq_lens_cpu = (
            capture_seq_lens_cpu.clone()
            if capture_seq_lens_cpu is not None
            else common_attn_metadata.seq_lens.detach().cpu().clone()
        )
        attn_metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        self._attach_common_flash_metadata(attn_metadata, common_attn_metadata)
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        self._attach_prefix_anchored_metadata(attn_metadata, common_attn_metadata)
        flash_metadata.seq_lens_cpu = capture_seq_lens_cpu

        self.spec_state.prepare_capture(attn_metadata, common_attn_metadata)
        self.spec_state.debug_metadata(
            "capture",
            attn_metadata,
            common_attn_metadata,
        )
        self._attach_decode_shape_hints(
            attn_metadata,
            common_attn_metadata,
            static_decode=True,
        )
        flash_metadata.flash_v100_cudagraph_capture = True
        self._update_decode_active_num_partitions(attn_metadata, stage="capture")

        return attn_metadata
