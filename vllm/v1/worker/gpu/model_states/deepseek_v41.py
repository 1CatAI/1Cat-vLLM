# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 model state for Model Runner V2.

V4.1 keeps private per-sequence state on its first pipeline stage (the Engram
hash history and the ratio-2 compressor states). The V1 runner commits that
state after every forward; this model state does the same through the MRV2
hooks, so the model code is unchanged:

* ``prepare_attn`` passes the per-request prefill flag that ``V41CacheBuilder``
  copies into its shared device flag, and marks capture and dummy batches as
  dummy (the builder then writes False), as the V1 runner does.
* ``add_request`` rewinds a same-session prefix hit (``model.restore_prefix``,
  which also resets the PP stage-1 commit protocol) and resets that protocol
  for a request that starts at position 0.
* ``postprocess_state`` commits the sampled counts
  (``model.commit_speculative_state``). On a first stage that is not the last
  this is the "sampled counts arrived" path of ``pp_commit.StageCommit``
  (``take_sampled``); on the last stage of PP2 the model ignores it; with PP1
  it is the commit itself, and a prefill chunk selects its full length on the
  device.
* The runner calls ``pp_before_forward`` / ``pp_after_forward`` on a first
  stage that is not the last (``model_runner.py``).

It also keeps the per-request inputs of V4.1 coupled drafts
(``VLLM_V41_COUPLED_DRAFTS``): the stream key on the device and a host copy of
the temperature, which the speculator reads without a sync.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank
from vllm.distributed.ec_transfer import has_ec_transfer
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.logger import init_logger
from vllm.sampling_params import SamplingType
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

V41_MODEL_TYPE = "deepseek_v41_text"


class _SkipCounter(Protocol):
    def step(
        self, skipped: bool, req_ids: list[str], num_scheduled: list[int]
    ) -> list[str]: ...


class _SkipHelpers(Protocol):
    """The deployed V1 helper contract (BASECHECK verifies its source)."""

    KNOB: str
    skip_step: Callable[[Iterable[int], Iterable[int], Iterable[int], int], bool]
    SkipCounter: Callable[[], _SkipCounter]

    def static_block_reasons(
        self,
        *,
        pooling: bool,
        spec_method: str | None,
        kv_transfer: bool,
        ec_transfer: bool,
    ) -> list[str]: ...

    def step_block_reasons(
        self,
        *,
        prompt_logprobs: bool,
        routed_experts: bool,
    ) -> list[str]: ...


@dataclass
class _PrefillSkip:
    """One owner for the runtime binding, block reasons and V1 accounting."""

    rows: int
    bind: Callable[[bool], None]
    helpers: _SkipHelpers
    pooling: bool
    spec_method: str | None
    report: bool
    counter: _SkipCounter
    logged: set[str] = field(default_factory=set)

    @cached_property
    def static_reasons(self) -> list[str]:
        # KV transfer initializes after load_model. Like V1, query it at the
        # first runtime plan, not while constructing the model state.
        return self.helpers.static_block_reasons(
            pooling=self.pooling,
            spec_method=self.spec_method,
            kv_transfer=has_kv_transfer_group(),
            ec_transfer=has_ec_transfer(),
        )

    def plan(
        self,
        batch: InputBatch,
        prompt: list[int],
        *,
        prompt_logprobs: bool,
        routed_experts: bool,
    ) -> bool:
        reasons = self.static_reasons or self.helpers.step_block_reasons(
            prompt_logprobs=prompt_logprobs, routed_experts=routed_experts
        )
        for reason in reasons:
            if reason not in self.logged:
                self.logged.add(reason)
                if self.report:
                    logger.info(
                        "%s: skip disabled (%s); logged once.",
                        self.helpers.KNOB,
                        reason,
                    )
        n = batch.num_reqs
        scheduled = batch.num_scheduled_tokens[:n].tolist()
        skip = not reasons and self.helpers.skip_step(
            batch.num_computed_tokens_np[:n].tolist(), scheduled, prompt, self.rows
        )
        if self.report:
            for line in self.counter.step(skip, batch.req_ids[:n], scheduled):
                logger.info("%s: %s", self.helpers.KNOB, line)
        return skip


def is_deepseek_v41(vllm_config: VllmConfig) -> bool:
    model_config = vllm_config.model_config
    text_config = getattr(model_config, "hf_text_config", None)
    return getattr(text_config, "model_type", None) == V41_MODEL_TYPE


def _language_model(model: nn.Module) -> nn.Module:
    get_language_model = getattr(model, "get_language_model", None)
    return get_language_model() if callable(get_language_model) else model


@dataclass
class V41AttnMetadata(ModelSpecificAttnMetadata):
    """The per-request prefill flag (CPU bool), read by V41CacheBuilder."""

    is_prefilling: torch.Tensor

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}


class DeepseekV41ModelState(DefaultModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        if self.rope_state is not None or self.supports_mm_inputs:
            raise ValueError(
                "DeepSeek-V4.1 under Model Runner V2 expects 1D positions and "
                "text-only inputs; got M-RoPE or multimodal inputs."
            )
        if vllm_config.kernel_config.tp_prefill_overlap_min_tokens > 0:
            raise ValueError(
                "DeepSeek-V4.1 under Model Runner V2 does not support "
                "tp_prefill_overlap_min_tokens > 0: its cache builder admits "
                "one request and one geometry per forward. Set it to 0."
            )
        language_model = _language_model(model)
        inner = getattr(language_model, "model", None)
        self._commit = getattr(language_model, "commit_speculative_state", None)
        self._restore_prefix = getattr(language_model, "restore_prefix", None)
        if self._commit is None or self._restore_prefix is None or inner is None:
            raise ValueError(
                "DeepSeek-V4.1 under Model Runner V2 needs the model's "
                "commit_speculative_state and restore_prefix hooks."
            )
        # None unless this is a first stage that is not the last.
        self._stage_commit = getattr(inner, "stage_commit", None)
        self._prefill_skip: _PrefillSkip | None = None
        self._prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.bool_)
        rows = int(getattr(inner, "_v41_skip_rows", 0))
        if rows > 0:
            bind = getattr(language_model, "prefill_skip_bind", None)
            if not callable(bind):
                raise ValueError(
                    "DeepSeek-V4.1 MRV2 prefill skip needs prefill_skip_bind."
                )
            # This dependency belongs to the deployed V1/model skip feature.
            # Import only when enabled; the original port predates that file.
            helpers = cast(
                _SkipHelpers, import_module("vllm.models.deepseek_v41.prefill_skip")
            )
            spec = vllm_config.speculative_config
            report = (
                get_pp_group().is_last_rank and get_tensor_model_parallel_rank() == 0
            )
            self._prefill_skip = _PrefillSkip(
                rows=rows,
                bind=bind,
                helpers=helpers,
                pooling=self.model_config.runner_type == "pooling",
                spec_method=spec.method if spec is not None else None,
                report=report,
                counter=helpers.SkipCounter(),
            )
            if getattr(inner, "_v41_skip_eligible", False):
                logger.info(
                    "%s=%d MRV2 skip active: per-forward prefill_skip_bind; "
                    "V1 static/step guards and chunk boundaries apply.",
                    helpers.KNOB,
                    rows,
                )

        # Coupled-draft inputs, indexed by request state index.
        self.keys_np = np.zeros(self.max_num_reqs, dtype=np.int64)
        self.keys_gpu = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self._keys_dirty = False
        # Host copy for the drafter's all-greedy check (the sampler, which
        # has its own, exists only on the last PP rank).
        self.temperature_np = np.zeros(self.max_num_reqs, dtype=np.float32)
        self._engine_seed = getattr(self.model_config, "seed", None) or 0
        logger.info(
            "DeepSeek-V4.1 Model Runner V2 state active: prefill flag through "
            "V4.1 attention metadata, sampled-count commit in postprocess_state, "
            "prefix restore on add_request (PP stage-1 commit protocol: %s).",
            "on" if self._stage_commit is not None else "not this stage",
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        params = new_req_data.sampling_params
        # Every stage sees new/resumed requests, unlike the last-stage sampler.
        # Replacing this slot also clears the prior request's logprob flag.
        self._prompt_logprobs[req_index] = (
            params is not None and params.prompt_logprobs is not None
        )
        if new_req_data.num_computed_tokens > 0:
            # A same-session prefix hit (or a resumed request): rewind the
            # private state to the cached position. prefill_token_ids holds
            # the prompt plus any outputs of a resumed request.
            assert new_req_data.prefill_token_ids is not None
            self._restore_prefix(
                new_req_data.num_computed_tokens, new_req_data.prefill_token_ids
            )
        elif self._stage_commit is not None:
            # A forward armed for the previous request is dropped: this one
            # starts at position 0, which resets the device protocol.
            self._stage_commit.reset()

        params = new_req_data.sampling_params
        if params is None:
            self.temperature_np[req_index] = 0.0
            return
        from vllm.v1.spec_decode import coupled_draw

        # The V1 runner's key (_fill_coupled_keys): the request seed when
        # given, else the engine seed and the request id.
        self.keys_np[req_index] = coupled_draw.request_key(
            params.seed, new_req_data.req_id, self._engine_seed
        )
        self._keys_dirty = True
        # V1 stores 0 for any greedy request (temperature below the epsilon).
        greedy = params.sampling_type == SamplingType.GREEDY
        self.temperature_np[req_index] = 0.0 if greedy else params.temperature

    @contextmanager
    def prefill_skip_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        cudagraph_mode: CUDAGraphMode,
        *,
        for_capture: bool = False,
    ) -> Iterator[None]:
        """UNBOUND -> bound for one runtime forward -> UNBOUND, even on error.

        Capture/profile and rows=0 never call the binding. FULL decode replays
        are bound False. A skip must execute eagerly: capturing the Python
        layer switch would bake an incomplete forward into future replays.
        """
        state = self._prefill_skip
        if state is None or input_batch.is_dummy_batch or for_capture:
            yield
            return
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            yield
            return
        indices = input_batch.idx_mapping_np[: input_batch.num_reqs]
        active_indices = list(req_states.req_id_to_index.values())
        skip = state.plan(
            input_batch,
            req_states.prompt_len.np[indices].tolist(),
            prompt_logprobs=bool(self._prompt_logprobs[active_indices].any()),
            routed_experts=self.model_config.enable_return_routed_experts,
        )
        if skip and cudagraph_mode != CUDAGraphMode.NONE:
            raise ValueError("DeepSeek-V4.1 MRV2 skip requires eager prefill.")
        try:
            state.bind(skip)
            yield
        finally:
            state.bind(False)

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        if self._keys_dirty:
            async_copy_to_gpu(self.keys_np, out=self.keys_gpu)
            self._keys_dirty = False

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        if isinstance(num_sampled, int):
            # A non-last stage with no sampled counts for this step (prefill
            # chunks or finishing rows). A prefill forward committed itself in
            # pp_after_forward; an armed decode forward stays armed and is
            # committed by pp_before_forward or dropped with its request.
            return
        if idx_mapping.shape[0] != 1:
            raise ValueError(
                "DeepSeek-V4.1 commits the private state of one request per "
                f"step; got {idx_mapping.shape[0]} rows (the model admits "
                "max_num_seqs=1)."
            )
        self._commit(num_sampled[:1])

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np[: input_batch.num_reqs]
        )
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            max_seq_len = input_batch.capture_context_len or self.max_model_len
        else:
            max_seq_len = int(seq_lens_cpu_upper_bound[:num_reqs].max().item())
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            model_specific_attn_metadata=V41AttnMetadata(is_prefilling),
            for_cudagraph_capture=for_capture,
            prefix_anchor_lens=input_batch.prefix_anchor_lens,
            is_dummy_batch=input_batch.is_dummy_batch or for_capture,
        )
