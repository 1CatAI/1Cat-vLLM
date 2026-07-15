# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Set as AbstractSet
from dataclasses import replace
from itertools import product

from vllm import envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.lora.utils import get_captured_lora_counts

logger = init_logger(__name__)


def _get_sm70_mtp_context_buckets() -> tuple[int, ...]:
    """Parse optional one-request MTP attention graph context buckets."""
    raw = os.getenv("VLLM_SM70_MTP_CONTEXT_BUCKETS", "").strip()
    if not raw:
        return ()

    try:
        buckets = tuple(sorted({int(value.strip()) for value in raw.split(",")}))
    except ValueError as exc:
        raise ValueError(
            "VLLM_SM70_MTP_CONTEXT_BUCKETS must be a comma-separated list "
            f"of positive integers, got {raw!r}"
        ) from exc
    if not buckets or buckets[0] <= 0:
        raise ValueError(
            "VLLM_SM70_MTP_CONTEXT_BUCKETS must contain only positive integers, "
            f"got {raw!r}"
        )
    return buckets


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to dispatch keys for multiple set of
    cudagraphs.

    The dispatcher stores two sets of dispatch keys, one for PIECEWISE and one
    for FULL cudagraph runtime mode. The keys are initialized depending on
    attention support and what cudagraph mode is set in CompilationConfig. The
    keys stored in dispatcher are the only source of truth for valid
    cudagraphs that can be dispatched at runtime.

    At runtime, the dispatch method generates the runtime cudagraph mode (FULL,
    PIECEWISE, or NONE for no cudagraph) and the valid key (batch descriptor)
    based on the input key. After dispatching (communicated via forward
    context), the cudagraph wrappers will trust the dispatch key to either
    capture or replay (if the mode matches), or pass through to the underlying
    runnable without cudagraph (if the mode does not match or mode is NONE).
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.uniform_decode_query_len = (
            1
            if not self.vllm_config.speculative_config
            else 1 + self.vllm_config.speculative_config.num_speculative_tokens
        )
        self.sm70_mtp_context_buckets = _get_sm70_mtp_context_buckets()
        self._logged_sm70_mtp_context_bucket = False

        # Dict to store valid cudagraph dispatching keys.
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }

        from vllm.compilation.breakable_cudagraph import (
            is_breakable_cudagraph_enabled,
        )

        assert (
            not self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            or self.compilation_config.is_attention_compiled_piecewise()
            or is_breakable_cudagraph_enabled()
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.compilation_config.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )

        self.keys_initialized = False
        self.specialize_lora_count = (
            self.vllm_config.lora_config.specialize_active_lora
            if self.vllm_config.lora_config is not None
            else False
        )
        # Default cudagraph_mode to NONE until initialize_cudagraph_keys is called
        self.cudagraph_mode = CUDAGraphMode.NONE

    def _compute_bs_to_padded_graph_size(self) -> None:
        """Pre-compute the mapping from batch size to padded graph size."""
        max_size = self.compilation_config.max_cudagraph_capture_size
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        assert max_size is not None, (
            "Maximum cudagraph capture size must be set when cudagraphs are enabled."
        )
        assert capture_sizes is not None, (
            "Cudagraph capture sizes must be set when cudagraphs are enabled."
        )
        self._bs_to_padded_graph_size: list[int] = [0] * (max_size + 1)
        for end, start in zip(
            capture_sizes + [max_size + 1],
            [0] + capture_sizes,
        ):
            for bs in range(start, end):
                if bs == start:
                    self._bs_to_padded_graph_size[bs] = start
                else:
                    self._bs_to_padded_graph_size[bs] = end

        # Validate that compile_sizes won't be changed by padding.
        # Only validate when cudagraphs are actually being used.
        if (
            self.compilation_config.compile_sizes
            and self.cudagraph_mode != CUDAGraphMode.NONE
        ):
            for size in self.compilation_config.compile_sizes:
                size = int(size)
                if size <= max_size:
                    padded = self._bs_to_padded_graph_size[size]
                    if padded != size:
                        raise ValueError(
                            f"compile_sizes contains {size} which would be "
                            f"padded to {padded}. All compile_sizes must be "
                            "values that won't be changed by cudagraph padding. "
                            "Use values from cudagraph_capture_sizes."
                        )

    def _get_lora_cases(self) -> list[int]:
        """
        Returns list of has_lora values for CUDA graph capture.
        This is the single source of truth for LoRA capture cases.
        """
        lora_config = self.vllm_config.lora_config
        if lora_config is None:
            # No LoRA configured - single case with no LoRA
            return [0]

        # LoRA is enabled - capture graphs based on cudagraph_specialize_lora
        if self.compilation_config.cudagraph_specialize_lora:
            captured_counts = get_captured_lora_counts(
                lora_config.max_loras, self.specialize_lora_count
            )
            # Specialize: capture separate graphs for with and without LoRA
            return [0] + captured_counts
        else:
            # No specialization: only capture graphs with LoRA active
            return [lora_config.max_loras + 1]

    def _create_padded_batch_descriptor(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int = 0,
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
            assert num_tokens_padded % uniform_decode_query_len == 0, (
                f"num_tokens_padded={num_tokens_padded} must be divisible by "
                f"uniform_decode_query_len={uniform_decode_query_len} "
                f"for num_tokens={num_tokens}"
            )
        else:
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)

        return BatchDescriptor(
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )

    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    def _add_mtp_context_bucket_keys(self, batch_descriptor: BatchDescriptor) -> None:
        """Add short-context graphs only for the single-request MTP verifier."""
        if (
            not self.sm70_mtp_context_buckets
            or self.uniform_decode_query_len <= 1
            or not batch_descriptor.uniform
            or batch_descriptor.num_tokens != self.uniform_decode_query_len
        ):
            return

        for bucket in self.sm70_mtp_context_buckets:
            self.add_cudagraph_key(
                CUDAGraphMode.FULL,
                replace(batch_descriptor, attention_context_bucket=bucket),
            )

    def _get_mtp_context_bucket_descriptor(
        self,
        batch_descriptor: BatchDescriptor,
        attention_context_len: int | None,
    ) -> BatchDescriptor:
        """Return a bounded MTP graph key only when its capacity is sufficient."""
        if (
            attention_context_len is None
            or attention_context_len <= 0
            or not self.sm70_mtp_context_buckets
            or self.uniform_decode_query_len <= 1
            or not batch_descriptor.uniform
            or batch_descriptor.num_tokens != self.uniform_decode_query_len
        ):
            return batch_descriptor

        for bucket in self.sm70_mtp_context_buckets:
            if attention_context_len <= bucket:
                return replace(batch_descriptor, attention_context_bucket=bucket)
        return batch_descriptor

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
    ):
        # This should be called only after attention backend is initialized. So we can
        # get the correct cudagraph mode after backend support is resolved.
        self.cudagraph_mode = cudagraph_mode
        self.uniform_decode_query_len = uniform_decode_query_len

        # Early exit if cudagraphs are disabled
        if cudagraph_mode == CUDAGraphMode.NONE:
            self.keys_initialized = True
            return

        self._compute_bs_to_padded_graph_size()

        # Get LoRA cases to capture
        lora_cases = self._get_lora_cases()
        self.captured_lora_counts = [
            lora_count for lora_count in lora_cases if lora_count
        ]

        # Note: we create all valid keys for cudagraph here but do not
        # guarantee all keys would be used. For example, if we allow lazy
        # capturing in future PR, some keys may never be triggered.
        skip_sm70_mixed_capture = (
            envs.VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH
            and envs.VLLM_SM70_FLASH_V100_0DOT3_DECODE_ONLY_CAPTURE
            and cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
        )
        if skip_sm70_mixed_capture:
            logger.info_once(
                "Skipping mixed/piecewise CUDA graph capture for SM70 "
                "Flash-V100 0.0.3 decode-only baseline recovery; FULL decode "
                "graph capture remains enabled."
            )

        if (
            cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE
            and not skip_sm70_mixed_capture
        ):
            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when mixed mode is enabled."
            )
            for bs, num_active_loras in product(
                self.compilation_config.cudagraph_capture_sizes, lora_cases
            ):
                batch_desc = self._create_padded_batch_descriptor(
                    bs, False, num_active_loras > 0, num_active_loras
                )
                # Only relax for PIECEWISE mode. FULL mode needs exact num_reqs
                # because FA3's scheduler_metadata computation depends on it.
                if cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE:
                    batch_desc = replace(batch_desc, num_reqs=None, uniform=False)
                self.add_cudagraph_key(cudagraph_mode.mixed_mode(), batch_desc)

        # if decode cudagraph mode is FULL, and we don't already have mixed
        # mode full cudagraphs then add them here.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
        ):
            max_num_tokens = (
                uniform_decode_query_len
                * self.vllm_config.scheduler_config.max_num_seqs
            )
            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when full mode is enabled."
            )
            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens
                and x >= uniform_decode_query_len
                and (
                    self._bs_to_padded_graph_size[x]
                    % uniform_decode_query_len
                    == 0
                )
            ]
            for bs, num_active_loras in product(
                cudagraph_capture_sizes_for_decode, lora_cases
            ):
                batch_descriptor = self._create_padded_batch_descriptor(
                    bs, True, num_active_loras > 0, num_active_loras
                )
                self.add_cudagraph_key(CUDAGraphMode.FULL, batch_descriptor)
                self._add_mtp_context_bucket_keys(batch_descriptor)

        self.keys_initialized = True

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        num_active_loras: int = 0,
        attention_context_len: int | None = None,
        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """
        Given conditions(e.g.,batch descriptor and if using piecewise only),
        dispatch to a cudagraph runtime mode and the valid batch descriptor.
        A new batch descriptor is returned as we might dispatch a uniform batch
        to a graph that supports a more general batch (uniform to non-uniform).

        Args:
            num_tokens: Number of tokens in the batch.
            uniform_decode: Whether the batch is uniform decode (i.e. uniform and query
                length is uniform_decode_query_len).
            has_lora: Whether LoRA is active.
            num_active_loras: Number of distinct active LoRA adapters.
            attention_context_len: Maximum active KV length for optional
                attention-specialized MTP graph selection.
            valid_modes: Set of cudagraph modes that are allowed. None means
                all modes are allowed.
            invalid_modes: Set of cudagraph modes to exclude. Subtracted from
                valid_modes to compute allowed modes. (e.g., {FULL} for
                features like cascade attention not supported by full
                cudagraphs). None means no modes are excluded.
        """
        allowed_modes = valid_modes or CUDAGraphMode.valid_runtime_modes()

        if invalid_modes:
            allowed_modes -= invalid_modes

        assert len(allowed_modes) >= 1, (
            f"No allowed cudagraph modes: valid_modes={valid_modes}, "
            f"invalid_modes={invalid_modes}"
        )
        max_size = self.compilation_config.max_cudagraph_capture_size

        if (
            not self.keys_initialized
            or self.cudagraph_mode == CUDAGraphMode.NONE
            or max_size is None
            or num_tokens > max_size
            or allowed_modes <= {CUDAGraphMode.NONE}
        ):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        effective_num_active_loras = num_active_loras
        if has_lora and num_active_loras > 0:
            if self.specialize_lora_count:
                # Find the smallest captured `num_active_loras` that is >= the current
                # `num_active_loras`. This is because we only capture graphs for
                # a subset of possible `num_active_loras` values (powers of 2).
                import bisect

                idx = bisect.bisect_left(self.captured_lora_counts, num_active_loras)
                if idx < len(self.captured_lora_counts):
                    effective_num_active_loras = self.captured_lora_counts[idx]
            else:
                # When not specializing, graphs are captured only with max_loras + 1,
                # so we must use max_loras + 1 for dispatch to find a matching graph.
                assert self.vllm_config.lora_config is not None, (
                    "LoRA config must be set when has_lora is True."
                )
                effective_num_active_loras = self.vllm_config.lora_config.max_loras + 1

        normalized_uniform = uniform_decode and self.cudagraph_mode.separate_routine()
        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, normalized_uniform, has_lora, effective_num_active_loras
        )

        if CUDAGraphMode.FULL in allowed_modes:
            # check if key exists for full cudagraph
            batch_desc_to_check = self._get_mtp_context_bucket_descriptor(
                batch_desc, attention_context_len
            )
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.FULL]:
                if (
                    batch_desc_to_check.attention_context_bucket is not None
                    and not self._logged_sm70_mtp_context_bucket
                ):
                    logger.info(
                        "Using bounded SM70 MTP attention CUDA graph: "
                        "context=%d bucket=%d descriptor=%s",
                        attention_context_len,
                        batch_desc_to_check.attention_context_bucket,
                        batch_desc_to_check,
                    )
                    self._logged_sm70_mtp_context_bucket = True
                return CUDAGraphMode.FULL, batch_desc_to_check
            # A bounded graph might be absent because capture-size policy
            # excluded it. Preserve the existing full-context graph then.
            if batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_desc

        if CUDAGraphMode.PIECEWISE in allowed_modes:
            # also check if the relaxed key exists for more "general"
            # piecewise cudagraph
            batch_desc_to_check = replace(batch_desc, num_reqs=None, uniform=False)
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
                return CUDAGraphMode.PIECEWISE, batch_desc_to_check

        assert CUDAGraphMode.NONE in allowed_modes, (
            f"No matching cudagraph found and NONE is not in "
            f"allowed_modes={allowed_modes}"
        )
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

    def get_capture_descs(self) -> list[tuple[CUDAGraphMode, list[BatchDescriptor]]]:
        """
        Returns capture descriptors for cudagraph capturing.

        Returns:
            List of (runtime_mode, batch_descriptors) tuples, ordered PIECEWISE
            first then FULL. Batch descriptors are sorted largest-first for
            memory efficiency.
        """
        if not self.keys_initialized or self.cudagraph_mode == CUDAGraphMode.NONE:
            return []

        result = []
        # Return in order: PIECEWISE first, then FULL
        for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
            descs = list(self.cudagraph_keys[mode])
            if descs:
                # Capture the generic full-context graph before smaller
                # context-specialized graphs so its allocations seed the pool.
                descs.sort(
                    key=lambda d: (
                        d.num_tokens,
                        d.attention_context_bucket
                        if d.attention_context_bucket is not None
                        else float("inf"),
                        d.num_active_loras,
                    ),
                    reverse=True,
                )
                result.append((mode, descs))

        return result
