# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""DSpark speculator for Inkling 2 Mini (model runner V2, SM70).

Subclasses the fork's `DFlashSpeculator` (fill-in layout: the anchor is the
bonus token and every mask row predicts the token at its own position; the
non-causal draft block attends over the context K/V that the draft writes
from the target's features). DSpark adds a sequential recurrent Markov head
on top of the parallel block forward: the drafts are sampled left to right,
each step correcting the base logits with the previously sampled token.

Two ways through a step:

- the general one (`DFlashSpeculator.propose`): eager input preparation and
  context K/V, then one FULL CUDA graph (block forward + Markov head of all
  seven steps). Used after prefill steps and whenever the fast one cannot.
- the fast one (`VLLM_INKLING2_SM70_DSPARK_FAST`, decode steps): ONE graph
  per (batch size, draft width) holding everything: input preparation
  (`prepare.py`), context K/V, the block forward and exactly `width` Markov
  steps; at width 0 a small graph with the preparation and the context K/V
  only. The host stages three small tensors and replays; on the model runner
  V2 the eager launches of the general path are what bound a width-0 step.

With `VLLM_INKLING2_SM70_DSPARK_GREEDY_TARGET` the target's tokens of an
all-greedy batch come from a shard-local argmax (no vocabulary all-gather,
no dense sampler passes): the model's `get_top_tokens` if it has one, else
`ops/dspark_markov.py::shard_argmax` on the shared LM head.
"""

import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    _is_context_only_prefill,
)
from vllm.v1.worker.gpu.spec_decode.dspark.prepare import (
    MAX_GROUPS,
    prepare_dspark_inputs,
)
from vllm.v1.worker.gpu.spec_decode.dspark.widths import DSparkWidths

logger = init_logger(__name__)

# 0: the general path only. 1: decode steps replay ONE graph (preparation,
# context K/V, block forward, `width` Markov steps). 2: that graph only at
# width 0; drafting steps keep the eager preparation and context K/V and
# replay a graph of the block forward and exactly `width` Markov steps.
# Default: 1 when the workers wait for their next step in the C spin loop
# (`VLLM_USE_SPINLOOP_EXT=1`), else 0. With the Python wait loop a worker
# that finishes its step early delays its own output thread, and the
# pipeline stalls more often the less the worker does per step (measured:
# work/spd-draft/NOTES.md, ab_finU4 / ab_finU5).
_SPIN_EXT = (os.environ.get("VLLM_USE_SPINLOOP_EXT") or "0") not in ("0", "")
FAST_MODE = int(
    os.environ.get("VLLM_INKLING2_SM70_DSPARK_FAST", "1" if _SPIN_EXT else "0") or 0
)
FAST_PROPOSE = FAST_MODE != 0
GREEDY_TARGET = os.environ.get("VLLM_INKLING2_SM70_DSPARK_GREEDY_TARGET", "1") != "0"
# Sampling batches (temperature > 0, nothing that needs the whole vocabulary
# at once): shard-local rejection sampling instead of gathered vocabularies.
LOCAL_VERIFY = os.environ.get("VLLM_INKLING2_SM70_DSPARK_LOCAL_VERIFY", "1") != "0"
# Debug: after every proposal compare the draft tokens of all TP ranks (syncs).
CHECK_RANKS = os.environ.get("VLLM_INKLING2_SM70_DSPARK_CHECK", "0") == "1"
# Benchmark aid: a file with e.g. "fast=0 greedy=1 local=0" switches the three
# host-side paths at run time (read when it changes; change it only while no
# request is running, so that all TP ranks switch at the same step).
FLAGS_FILE = os.environ.get("VLLM_INKLING2_SM70_DSPARK_FLAGS_FILE")
# Prompt-lookup drafts (`dspark/lookup.py`): a request whose last tokens repeat an earlier span of at least
# LOOKUP_NUSE tokens drafts that span's continuation instead of the drafter's sample (exact: the verifier
# decides). Fast path only. Default off. LOOKUP_SEARCH: how far back the search looks (tokens).
LOOKUP = os.environ.get("VLLM_INKLING2_SM70_DSPARK_LOOKUP", "0") != "0"
LOOKUP_NUSE = int(os.environ.get("VLLM_INKLING2_SM70_DSPARK_LOOKUP_NUSE", "8"))
LOOKUP_NMIN = int(os.environ.get("VLLM_INKLING2_SM70_DSPARK_LOOKUP_NMIN", "4"))
LOOKUP_NMAX = int(os.environ.get("VLLM_INKLING2_SM70_DSPARK_LOOKUP_NMAX", "12"))
LOOKUP_LOG = os.environ.get("VLLM_INKLING2_SM70_DSPARK_LOOKUP_LOG", "0") == "1"
LOOKUP_SEARCH = int(os.environ.get("VLLM_INKLING2_SM70_DSPARK_LOOKUP_SEARCH", "65536"))


class _Live:
    fast = FAST_MODE
    greedy = GREEDY_TARGET
    local = LOCAL_VERIFY
    lookup = LOOKUP
    nuse = LOOKUP_NUSE
    mtime = 0

    @classmethod
    def poll(cls) -> None:
        try:
            mtime = os.stat(FLAGS_FILE).st_mtime_ns
            if mtime == cls.mtime:
                return
            cls.mtime = mtime
            with open(FLAGS_FILE) as f:
                for item in f.read().split():
                    key, _, value = item.partition("=")
                    if key == "fast":
                        cls.fast = int(value)
                    elif key in ("greedy", "local", "lookup"):
                        setattr(cls, key, value != "0")
                    elif key == "nuse":
                        cls.nuse = int(value)
                    elif key == "gil":
                        sys.setswitchinterval(float(value) / 1000.0)
        except (OSError, ValueError):
            pass


class Inkling2DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "Inkling2 DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        from vllm.models.inkling2.sm70.dspark import FUSED_MARKOV

        config = self.draft_model_config.hf_config
        assert not self.sample_from_anchor
        num_features = len(config.target_layer_ids)
        # Raw target features (each draft layer mixes them itself).
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            num_features * config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self.f_rnn = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device
        )
        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )
        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._widths = DSparkWidths(self.num_speculative_steps)
        self._next_width = self.num_speculative_steps
        self._skip_draft = False
        self._prof = os.environ.get("VLLM_INKLING2_SM70_DSPARK_PROFILE", "0")
        self._prof_stats: dict[str, list[float]] = {
            "propose": [],
            "context": [],
            "interval": [],
        }
        self._prof_last_end = 0.0
        self._cur_query_start_loc: torch.Tensor | None = None
        self._cur_num_rejected: torch.Tensor | None = None
        self._cur_num_reqs = 0

        self._fused = FUSED_MARKOV
        # Greedy drafts (`draft_sample_method` other than probabilistic) read
        # a zero temperature whatever the request asks for.
        self._zero_temperature = torch.zeros_like(self.temperature)
        # This rank's temperature-applied draft logits (probabilistic drafts):
        # row = request * steps + step. Allocated once the shard is known.
        self._dense_local: torch.Tensor | None = None
        # Per state row: whether `draft_logits` holds the row's current
        # drafts too (only a verification on the dense sampler reads those).
        # Per row, not per proposal: a step may propose for part of the
        # running requests (a joining request's prefill on its own), and the
        # others then verify drafts of an earlier proposal.
        self._row_gathered = np.ones(self.max_num_reqs, dtype=np.bool_)
        self._local_logged = False

        # --- fast path state
        self._fast = FAST_PROPOSE
        self._fast_ready = False
        self._graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self._ctx_rows: dict[int, list[int]] = {}
        # Per-step values of fresh tensors, staged for the in-graph kernel:
        # [request state row, sampled, rejected] per batch row.
        self._stage = torch.zeros(
            self.max_num_reqs, 3, dtype=torch.int32, device=device
        )
        self._f_rnn_rows = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        # 0 while graphs are captured: the in-graph preparation is inert.
        self._live = torch.zeros(1, dtype=torch.int32, device=device)
        self._tables: tuple[torch.Tensor, ...] | None = None
        self._table_ptrs_host: list[int] = []
        self._staged_sampling: tuple[bytes, bytes, bytes] | None = None
        self._req_states = None
        self._cur_batch: InputBatch | None = None
        self._sampler = None
        self._greedy_logged = False
        self._lookup = None
        # Debug: one JSON line per step (batch, verification path, tokens).
        self._trace = os.environ.get("VLLM_INKLING2_SM70_DSPARK_TRACE_FILE")
        self._trace_verify = None
        self._trace_n = 0

    # ------------------------------------------------------- runner hooks
    def set_req_states(self, req_states) -> None:
        self._req_states = req_states

    # Drafted steps always pass through `try_sample_target`; the runner also
    # routes steps without drafts there for speculators that set this.
    samples_plain_steps = GREEDY_TARGET or LOCAL_VERIFY or bool(FLAGS_FILE)

    # ------------------------------------------------------------ loading
    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32 like the target's head, so rejection sampling compares them
        # at the same precision.
        return torch.float32, 0.0

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.model_loader import get_model
        from vllm.model_executor.models.utils import get_draft_quant_config
        from vllm.runtime_resources import share_runtime_resources

        speculative_config = self.speculative_config
        draft_vllm_config = replace(
            self.vllm_config,
            is_speculative_draft=True,
            parallel_config=speculative_config.draft_parallel_config,
        )
        share_runtime_resources(draft_vllm_config, self.vllm_config)
        # The BF16 drafter takes none of the target's quantization.
        draft_vllm_config.quant_config = get_draft_quant_config(self.vllm_config)
        with set_model_tag("dspark_head"):
            model = get_model(
                vllm_config=draft_vllm_config, model_config=self.draft_model_config
            )
        target_lm = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        model.attach_target(target_lm)
        return model

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if self._fused and self.draft_logits is not None:
            shard = self.model.model.markov_head.fold_weight.shape[0]
            self._dense_local = torch.zeros(
                self.max_num_reqs * self.num_speculative_steps,
                shard,
                dtype=torch.float32,
                device=self.device,
            )

    def set_attn(self, *args, **kwargs) -> None:
        super().set_attn(*args, **kwargs)
        self._context_slot_mappings.fill_(PAD_SLOT_ID)

    # ------------------------------------------------- general path pieces
    def _prepare_proposal_runtime(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        self._cur_query_start_loc = input_batch.query_start_loc
        self._cur_num_rejected = num_rejected
        self._cur_num_reqs = input_batch.num_reqs

    def _combine_aux_hidden_states(self, aux_hidden_states: list[torch.Tensor]):
        return torch.cat(aux_hidden_states, dim=-1)

    def _precompute_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        if self._prof == "1":
            torch.accelerator.synchronize()
            t0 = time.perf_counter()
            self._precompute_context_kv_impl(hidden_states, positions, slots)
            torch.accelerator.synchronize()
            self._prof_stats["context"].append((time.perf_counter() - t0) * 1e3)
            return
        self._precompute_context_kv_impl(hidden_states, positions, slots)

    def _precompute_context_kv_impl(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        n = self._cur_num_reqs
        if (
            n
            and self._cur_query_start_loc is not None
            and self._cur_num_rejected is not None
        ):
            # Features of each request's last committed token feed the
            # recurrent head (a context shared by all of its draft steps).
            ends = self._cur_query_start_loc[1 : n + 1] - self._cur_num_rejected[:n]
            rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
            self.f_rnn[:n] = self.model.compute_f_rnn(rows)
        super()._precompute_context_kv(hidden_states, positions, slots)

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            return self.model(
                self.input_buffers.input_ids[:num_tokens],
                self.input_buffers.positions[:num_tokens],
            )

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if self.draft_logits is None:
            return logits.argmax(dim=-1)
        # sample_pos is the predicted token's position P; draws are keyed by
        # the position before the sampled token, P - 1.
        x = gumbel_sample(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            output_processed_logits=self.draft_logits,
            output_processed_logits_col=self._step_cols[step : step + 1],
            use_fp64=self.use_fp64_gumbel,
        )
        return x

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        model = self.model
        hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = model.compute_base_logits(hidden).view(num_reqs, n_spec, -1)
        context = model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
        # Per-step rows must be contiguous: the Gumbel kernel reads its
        # per-token request index and position arrays with unit stride.
        idx_map = (
            self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec).t().contiguous()
        )
        sample_pos = (
            self.sample_pos[:num_sample].view(num_reqs, n_spec).t().contiguous()
        )
        state = torch.zeros(
            num_reqs, model.config.markov_rank, dtype=torch.float32, device=self.device
        )
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        for i in range(n_spec):
            state, bias = model.model.markov_head.step(state, prev, context[i])
            logits = model.markov_logits(base_logits[:, i], bias)
            prev = self._sample_logits(logits, idx_map[i], sample_pos[i], i)
            self.draft_tokens[:num_reqs, i] = prev

    def _sample_fused(self, num_reqs: int, hidden: torch.Tensor, steps: int) -> None:
        """Shard-local Markov head on the `(request, step)`-ordered rows of
        `hidden` `(num_reqs * steps, hidden)`: `draft_tokens[:, :steps]`."""
        probabilistic = self.draft_logits is not None
        self.model.markov_sample(
            hidden,
            self.f_rnn[:num_reqs],
            self.draft_tokens,
            self.input_buffers.input_ids,
            self.num_query_per_req,
            self.sample_idx_mapping,
            self.sample_pos,
            self.temperature if probabilistic else self._zero_temperature,
            self.seeds,
            self._dense_local if probabilistic else None,
            self.num_speculative_steps,
            self.use_fp64_gumbel,
        )

    def _gather_dense(self, num_reqs: int, steps: int) -> None:
        """The dense draft logits of `steps` steps for rejection sampling:
        all ranks' shard rows into `draft_logits[state row, step]`."""
        from vllm.models.inkling2.sm70.ops.dspark_markov import (
            dense_rows,
            dense_scatter,
        )

        assert self._dense_local is not None and self.draft_logits is not None
        rows = dense_rows(
            self.sample_idx_mapping, num_reqs, steps, self.num_speculative_steps
        )
        gathered = self.model.gather_dense_logits(self._dense_local[rows])
        dense_scatter(gathered, self.draft_logits, self.sample_idx_mapping, steps)

    def _gather_dense_for(self, input_batch: InputBatch) -> None:
        """Eager catch-up before a dense verification whose proposal did not
        gather: the drafted rows of this batch into `draft_logits`."""
        assert self._dense_local is not None and self.draft_logits is not None
        per_req = input_batch.num_draft_tokens_per_req
        if per_req is None:
            return
        n_spec = self.num_speculative_steps
        rows = [
            int(state) * n_spec + i
            for state, n in zip(input_batch.idx_mapping_np.tolist(), per_req.tolist())
            for i in range(n)
        ]
        if not rows:
            return
        index = torch.tensor(rows, dtype=torch.int64, device=self.device)
        gathered = self.model.gather_dense_logits(self._dense_local[index])
        vocab = self.draft_logits.shape[-1]
        self.draft_logits.view(-1, vocab)[index] = gathered[:, :vocab]
        self._row_gathered[input_batch.idx_mapping_np] = True

    def _samples(self, input_batch: InputBatch) -> bool:
        """Whether a request of the batch has a temperature > 0 (greedy
        requests never read draft logits)."""
        if self._sampler is None:
            return True
        temps = self._sampler.sampling_states.temperature.np[input_batch.idx_mapping_np]
        return bool(np.any(temps != 0.0))

    def _needs_dense(self, input_batch: InputBatch, sampling: bool) -> bool:
        """Whether the verification of this proposal will read the gathered
        `draft_logits`: a sampling request that cannot verify shard by shard."""
        return sampling and not (_Live.local and self._local_verify_ok(input_batch))

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        if self._skip_draft:
            return
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        if not self._fused:
            self._sample_sequential(num_reqs, head_hidden)
            return
        n_spec = self.num_speculative_steps
        hidden = head_hidden[self.sample_indices[: num_reqs * n_spec]]
        self._sample_fused(num_reqs, hidden, n_spec)
        if self.draft_logits is not None:
            self._gather_dense(num_reqs, n_spec)

    # ------------------------------------------------------------ fast path
    def _refresh_tables(self) -> bool:
        """(Re)fill the block-table pointer table of the draft KV groups.
        The graphs read it from memory, so a moved table needs no capture."""
        gids = list(self.draft_kv_cache_group_ids)
        tables = [self.block_tables.input_block_tables[g] for g in gids]
        ptrs = [t.data_ptr() for t in tables]
        if ptrs == self._table_ptrs_host and self._tables is not None:
            return True
        if any(p % 16 for p in ptrs) or any(t.dtype != torch.int32 for t in tables):
            return False
        new = (
            torch.tensor(ptrs, dtype=torch.uint64, device=self.device),
            torch.tensor(
                [t.stride(0) for t in tables], dtype=torch.int64, device=self.device
            ),
            torch.tensor(
                [self.block_tables.kernel_block_sizes[g] for g in gids],
                dtype=torch.int32,
                device=self.device,
            ),
            torch.tensor(gids, dtype=torch.int32, device=self.device),
        )
        if self._tables is None:
            self._tables = new
        else:
            for old, cur in zip(self._tables, new):
                old.copy_(cur)
        self._table_ptrs_host = ptrs
        return True

    def _prepare(self, num_reqs: int, ctx_rows: int) -> None:
        assert self._tables is not None and self._req_states is not None
        prepare_dspark_inputs(
            num_reqs,
            self.input_buffers,
            self._require_query_slot_mappings(),
            self.context_positions,
            self._context_slot_mappings,
            self.sample_indices,
            self.sample_pos,
            self.sample_idx_mapping,
            self._f_rnn_rows,
            self.target_input_buffers,
            self._stage,
            self._req_states.last_sampled_tokens,
            self._req_states.next_prefill_tokens,
            *self._tables,
            self._live,
            self.parallel_drafting_token_id,
            self.num_query_per_req,
            self.num_speculative_steps,
            self.max_num_reqs,
            self.max_model_len,
            ctx_rows,
        )

    def _ctx_forward(self, num_reqs: int, rows: int, with_f_rnn: bool) -> None:
        assert self._layer_group_idx is not None
        slots = [self._context_slot_mappings[g][:rows] for g in self._layer_group_idx]
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:rows], self.context_positions[:rows], slots
        )
        if with_f_rnn:
            self.f_rnn[:num_reqs] = self.model.compute_f_rnn(
                self.hidden_states[self._f_rnn_rows[:num_reqs]]
            )

    def _ctx_step(self, num_reqs: int, rows: int) -> None:
        self._prepare(num_reqs, rows)
        self._ctx_forward(num_reqs, rows, False)

    def _draft_step(self, num_reqs: int, steps: int, attn_state) -> None:
        nq = self.num_query_per_req
        self._prepare(num_reqs, num_reqs * nq)
        self._ctx_forward(num_reqs, num_reqs * nq, True)
        attn_metadata, slot_mappings = attn_state
        head_hidden = self._run_model(num_reqs * nq, attn_metadata, slot_mappings, None)
        if not self._fused:
            self._sample_sequential(num_reqs, head_hidden)
            return
        hidden = head_hidden.view(num_reqs, nq, -1)[:, 1 : 1 + steps]
        self._sample_fused(num_reqs, hidden.reshape(num_reqs * steps, -1), steps)

    def _block_step(self, num_reqs: int, steps: int, attn_state) -> None:
        """Block forward and `steps` Markov steps on inputs prepared eagerly
        (the general path's preparation, context K/V and f_rnn)."""
        nq = self.num_query_per_req
        attn_metadata, slot_mappings = attn_state
        head_hidden = self._run_model(num_reqs * nq, attn_metadata, slot_mappings, None)
        hidden = head_hidden.view(num_reqs, nq, -1)[:, 1 : 1 + steps]
        self._sample_fused(num_reqs, hidden.reshape(num_reqs * steps, -1), steps)

    def _capture_fast(self) -> None:
        from vllm.distributed.parallel_state import graph_capture
        from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
            _prepare_dflash_inputs_to_capture,
        )

        manager = self.query_cudagraph_manager
        if (
            manager is None
            or not manager.cudagraph_mode
            or not manager.graphs
            or self._req_states is None
            or self._layer_group_idx is None
            or not 1 <= len(self.draft_kv_cache_group_ids) <= MAX_GROUPS
            or self.block_tables.cp_size != 1
            or self.dp_size != 1
            or self.max_num_reqs * self.num_query_per_req > self.max_num_tokens
            or not self._refresh_tables()
        ):
            logger.info(
                "%s: fast proposal graphs not available.", self._speculator_name
            )
            return
        nq, n_spec = self.num_query_per_req, self.num_speculative_steps
        table = self._widths.table
        max_batch = max(
            (
                b
                for b in range(1, self.max_num_reqs + 1)
                if table[min(b, len(table)) - 1] > 0
            ),
            default=1,
        )
        all_widths = (
            os.environ.get("VLLM_INKLING2_SM70_DSPARK_GRAPH_WIDTHS", "all") == "all"
        )
        self._live.zero_()
        self._stage.zero_()
        context = manager.get_capture_context()

        def capture(key: tuple, fn) -> None:
            fn()  # warm-up: compiles the kernels of this shape
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, manager.pool, stream=context.stream):
                fn()
            self._graphs[key] = graph

        with graph_capture(device=self.device, context=context):
            for b in range(1, self.max_num_reqs + 1):
                rows = sorted({b, b * nq})
                self._ctx_rows[b] = rows
                for r in rows:
                    capture(("ctx", b, r), lambda b=b, r=r: self._ctx_step(b, r))
                if b > max_batch:
                    continue
                attn_state = _prepare_dflash_inputs_to_capture(
                    b,
                    b * nq,
                    self.input_buffers,
                    self.block_tables,
                    self._require_query_slot_mappings(),
                    self.attn_groups,
                    self.kv_cache_config,
                    self.max_model_len,
                    skip_attn=False,
                    causal=self._group_causal,
                )
                widths = (
                    range(1, n_spec + 1)
                    if all_widths
                    else sorted({table[min(b, len(table)) - 1], n_spec} - {0})
                )
                for s in widths if self._fused else (n_spec,):
                    capture(
                        ("draft", b, s),
                        lambda b=b, s=s, a=attn_state: self._draft_step(b, s, a),
                    )
                    if self._fused:
                        capture(
                            ("block", b, s),
                            lambda b=b, s=s, a=attn_state: self._block_step(b, s, a),
                        )
                    if self._fused and self.draft_logits is not None:
                        capture(
                            ("dense", b, s), lambda b=b, s=s: self._gather_dense(b, s)
                        )
        if LOOKUP or FLAGS_FILE:
            self._setup_lookup()
        self._live.fill_(1)
        self._staged_sampling = None
        self._fast_ready = True
        logger.info(
            "%s: %d fast proposal graphs captured (batch sizes up to %d draft).",
            self._speculator_name,
            len(self._graphs),
            max_batch,
        )

    def _setup_lookup(self) -> None:
        """Allocate the lookup buffers and compile its kernels (inert: all lengths are 0 now)."""
        from vllm.v1.worker.gpu.spec_decode.dspark.lookup import DSparkLookup

        self._lookup = DSparkLookup(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.device,
            LOOKUP_NMIN,
            LOOKUP_NMAX,
            LOOKUP_NUSE,
            LOOKUP_SEARCH,
        )
        self._lookup_apply(1, 1, self._dense_local is not None)
        self._lookup.hits.zero_()
        self._lookup.steps = 0
        logger.info(
            "%s: prompt-lookup drafts available (match >= %d tokens, search %d back).",
            self._speculator_name,
            LOOKUP_NUSE,
            LOOKUP_SEARCH,
        )

    def _lookup_apply(self, num_reqs: int, width: int, sampling: bool) -> None:
        lookup = self._lookup
        lookup.nuse = _Live.nuse
        dense = self._dense_local if sampling else None
        shard = 0 if self._dense_local is None else self._dense_local.shape[1]
        lookup.apply(
            self._req_states.all_token_ids.gpu,
            self._req_states.total_len.gpu,
            self._stage,
            self.draft_tokens,
            num_reqs,
            width,
            dense,
            get_tensor_model_parallel_rank() * shard,
        )
        if LOOKUP_LOG and lookup.steps % 200 == 0:
            logger.info(
                "DSpark lookup: %d of %d proposals used a prompt match (nuse %d).",
                int(lookup.hits.item()),
                lookup.steps,
                lookup.nuse,
            )

    def _stage_sampling(
        self, input_batch: InputBatch, temperature: torch.Tensor, seeds: torch.Tensor
    ) -> bool:
        """Copy the requests' temperature and seed to the graph's buffers
        when the batch or its parameters changed; whether a request samples
        (temperature > 0)."""
        idx = input_batch.idx_mapping_np
        if self._sampler is None:
            self.temperature.copy_(temperature)
            self.seeds.copy_(seeds)
            return True
        states = self._sampler.sampling_states
        temps = states.temperature.np[idx]
        key = (idx.tobytes(), temps.tobytes(), states.seeds.np[idx].tobytes())
        if key != self._staged_sampling:
            self.temperature.copy_(temperature)
            self.seeds.copy_(seeds)
            self._staged_sampling = key
        return bool(np.any(temps != 0.0))

    def _propose_fast(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor],
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
    ) -> torch.Tensor | None:
        """One staged replay for a decode step; None if this step needs the
        general path."""
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        width = self._next_width
        if width > 0:
            if _Live.fast != 1:
                return None
            graph = self._graphs.get(("draft", num_reqs, width))
            dense = self._graphs.get(("dense", num_reqs, width))
        else:
            graph = dense = None
            for rows in self._ctx_rows.get(num_reqs, ()):
                if num_tokens <= rows:
                    graph = self._graphs.get(("ctx", num_reqs, rows))
                    break
        if (
            graph is None
            or num_tokens > num_reqs * self.num_query_per_req
            or int(input_batch.num_scheduled_tokens.max()) > self.num_query_per_req
            or _is_context_only_prefill(input_batch)
            or input_batch.positions.data_ptr()
            != self.target_input_buffers.positions.data_ptr()
            or input_batch.query_start_loc.data_ptr()
            != self.target_input_buffers.query_start_loc.data_ptr()
            or not self._refresh_tables()
        ):
            return None
        torch.cat(
            [h[:num_tokens] for h in aux_hidden_states],
            dim=-1,
            out=self.hidden_states[:num_tokens],
        )
        if num_sampled.dtype != torch.int32 or num_rejected.dtype != torch.int32:
            num_sampled, num_rejected = num_sampled.int(), num_rejected.int()
        torch.stack(
            (input_batch.idx_mapping, num_sampled, num_rejected),
            dim=1,
            out=self._stage[:num_reqs],
        )
        sampling = self._stage_sampling(input_batch, temperature, seeds)
        # The previous eager iteration may have queued copies the graph
        # cannot see (as `CudaGraphManager.run_fullgraph`).
        get_offloader().sync_prev_onload()
        graph.replay()
        if width > 0 and self._lookup is not None and _Live.lookup:
            self._lookup_apply(
                num_reqs, width, sampling and self._dense_local is not None
            )
        if dense is not None:
            gathered = self._needs_dense(input_batch, sampling)
            self._row_gathered[input_batch.idx_mapping_np] = gathered
            if gathered:
                dense.replay()
        return self.draft_tokens[:num_reqs]

    # ------------------------------------------------------------- width
    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        manager = self.query_cudagraph_manager
        if manager is not None and manager.cudagraph_mode:
            # The draft step is captured for every batch size that drafts
            # (width > 0), whatever `cudagraph_capture_sizes` rounds to.
            table = self._widths.table
            max_batch = max(
                (
                    b
                    for b in range(1, self.max_num_reqs + 1)
                    if table[min(b, len(table)) - 1] > 0
                ),
                default=1,
            )
            for b in range(1, max_batch + 1):
                size = b * self.num_query_per_req
                if size not in manager._capture_sizes:
                    manager._capture_sizes.append(size)
            manager._capture_sizes.sort()
            manager._init_candidates()
        if manager is not None:
            run_fullgraph = manager.run_fullgraph

            def run_unless_skipped(desc):
                if self._skip_draft:
                    return
                batch = self._cur_batch
                block = (
                    self._graphs.get(("block", desc.num_reqs, self._next_width))
                    if _Live.fast == 2
                    and batch is not None
                    and batch.num_reqs == desc.num_reqs
                    else None
                )
                if block is None:
                    # Block forward, all seven Markov steps, dense gather.
                    run_fullgraph(desc)
                    return
                get_offloader().sync_prev_onload()
                block.replay()
                dense = self._graphs.get(("dense", desc.num_reqs, self._next_width))
                gathered = dense is None or self._needs_dense(
                    batch, self._samples(batch)
                )
                self._row_gathered[batch.idx_mapping_np] = gathered
                if dense is not None and gathered:
                    dense.replay()

            manager.run_fullgraph = run_unless_skipped  # type: ignore[method-assign]

    def capture(self) -> None:
        super().capture()
        if self._fast or FLAGS_FILE:
            self._capture_fast()

    # The runner passes the scheduler's `next_spec_steps` (the width it chose
    # for the next step, see `AsyncScheduler`) to speculators defining this.
    multi_step_decode = True
    # A width-0 step hands no drafts on: the runner need not copy them
    # (part of the fast path: `VLLM_INKLING2_SM70_DSPARK_FAST=0` keeps the copy).
    skips_unused_draft_copy = FAST_PROPOSE

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        active = kwargs.pop("active_spec_steps", None)
        if active is None:
            self._next_width = self._widths.width(input_batch.num_reqs)
        else:
            self._next_width = max(0, min(int(active), self.num_speculative_steps))
        special = kwargs.get("dummy_run", False) or kwargs.get("is_profile", False)
        # Batches too large to verify any draft only keep the draft KV current.
        self._skip_draft = self._next_width == 0 and not special
        t0 = 0.0
        if self._prof != "0":
            if self._prof == "1":
                torch.accelerator.synchronize()
            t0 = time.perf_counter()
        drafts = None
        if FLAGS_FILE:
            _Live.poll()
        if (
            self._fast_ready
            and _Live.fast
            and not special
            and not args
            and aux_hidden_states
            and last_sampled is self._req_states.last_sampled_tokens
            and next_prefill_tokens is self._req_states.next_prefill_tokens
        ):
            drafts = self._propose_fast(
                input_batch,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                temperature,
                seeds,
            )
        fast = drafts is not None
        if drafts is None:
            # The general path gathers the dense draft logits itself (unless
            # `run_unless_skipped` replays a width-specific graph).
            if not special:
                self._row_gathered[input_batch.idx_mapping_np] = True
            self._cur_batch = None if special else input_batch
            drafts = super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                *args,
                **kwargs,
            )
        if CHECK_RANKS and not special and self._next_width > 0:
            self._check_ranks(input_batch.num_reqs)
        if self._prof != "0":
            self._profile(input_batch, t0, fast)
        if self._trace and not special:
            self._trace_dump(
                input_batch, num_sampled, num_rejected, last_sampled, drafts, fast
            )
        return drafts

    def _trace_dump(
        self, input_batch, num_sampled, num_rejected, last_sampled, drafts, fast
    ) -> None:
        import json

        idx = input_batch.idx_mapping_np.tolist()
        n = len(idx)
        row = {
            "n": self._trace_n,
            "idx": idx,
            "verify": self._trace_verify,
            "ns": num_sampled[:n].tolist(),
            "nr": num_rejected[:n].tolist(),
            "last": last_sampled[input_batch.idx_mapping].reshape(n, -1).tolist(),
            "fast": fast,
            "w": self._next_width,
            "gathered": bool(self._row_gathered[input_batch.idx_mapping_np].all()),
            "drafts": None if drafts is None else drafts[:n].tolist(),
            "sched": input_batch.num_scheduled_tokens.tolist(),
        }
        self._trace_n += 1
        self._trace_verify = None
        with open(f"{self._trace}.{os.getpid()}", "a") as f:
            f.write(json.dumps(row) + "\n")

    def _check_ranks(self, num_reqs: int) -> None:
        """Debug: every TP rank must hold the same draft tokens."""
        mine = self.draft_tokens[:num_reqs, : self._next_width].contiguous()
        if get_tensor_model_parallel_world_size() == 1:
            return
        every = tensor_model_parallel_all_gather(mine.to(torch.float32), dim=0)
        every = every.view(-1, num_reqs, mine.shape[1])
        if not bool((every == every[:1]).all().item()):
            logger.error(
                "DSpark draft tokens differ between TP ranks: %s", every.long().tolist()
            )

    def _profile(self, input_batch: InputBatch, t0: float, fast: bool) -> None:
        """`VLLM_INKLING2_SM70_DSPARK_PROFILE`: 1 = GPU-synchronous times (slows
        the engine), 2 = host time of the call only (no sync)."""
        if self._prof == "1":
            torch.accelerator.synchronize()
        t1 = time.perf_counter()
        st = self._prof_stats
        st["propose"].append((t1 - t0) * 1e3)
        if self._prof_last_end:
            st["interval"].append((t0 - self._prof_last_end) * 1e3)
        self._prof_last_end = t1
        if os.environ.get("VLLM_INKLING2_SM70_DSPARK_PROFILE_STEPS", "0") == "1":
            logger.info(
                "DSpark step: reqs=%d tokens=%d propose=%.2f interval=%.2f width_next=%d skip=%s fast=%s",
                input_batch.num_reqs,
                input_batch.num_tokens,
                st["propose"][-1],
                st["interval"][-1] if st["interval"] else -1.0,
                self._next_width,
                self._skip_draft,
                fast,
            )
        if len(st["propose"]) % 50 == 0:
            logger.info(
                "DSpark profile mode %s (last 50, ms): propose=%.3f (context+f_rnn=%.3f) "
                "between proposes=%.3f fast=%s width=%d",
                self._prof,
                sum(st["propose"][-50:]) / 50,
                sum(st["context"][-50:]) / max(len(st["context"][-50:]), 1),
                sum(st["interval"][-50:]) / max(len(st["interval"][-50:]), 1),
                fast,
                self._next_width,
            )

    def next_num_draft_tokens(self) -> int:
        """Drafts handed to the scheduler (structured-output path only; the
        async scheduler picks the width itself)."""
        return self._next_width

    # ----------------------------------------------------- target sampling
    def _all_greedy(self, input_batch: InputBatch) -> bool:
        """Whether argmax equals the full sampler for this batch (the checks
        of `Sampler.can_use_sm70_greedy_token_fastpath`)."""
        sampler = self._sampler
        if sampler is None or sampler.compute_nans:
            return False
        if input_batch.idx_mapping_np.size == 0:
            return False
        idx = input_batch.idx_mapping_np
        states = sampler.sampling_states
        return bool(
            np.all(states.temperature.np[idx] == 0.0)
            and states.max_num_logprobs(idx) == NO_LOGPROBS
            and sampler.logprob_token_ids_state.max_num_token_ids(idx) == 0
            and not np.any(sampler.logit_bias_state.use_logit_bias[idx])
            and not np.any(sampler.penalties_state.use_penalty[idx])
            and not np.any(sampler.bad_words_state.num_bad_words.np[idx])
        )

    def _local_verify_ok(self, input_batch: InputBatch) -> bool:
        """Whether this batch can be rejection-sampled shard by shard:
        nothing that needs the whole vocabulary at once."""
        sampler = self._sampler
        idx = input_batch.idx_mapping_np
        if sampler is None or sampler.compute_nans or idx.size == 0:
            return False
        states = sampler.sampling_states
        return bool(
            np.all(states.top_k.np[idx] == states.vocab_size)
            and np.all(states.top_p.np[idx] == 1.0)
            and np.all(states.min_p.np[idx] == 0.0)
            and states.max_num_logprobs(idx) == NO_LOGPROBS
            and sampler.logprob_token_ids_state.max_num_token_ids(idx) == 0
            and not np.any(sampler.logit_bias_state.use_logit_bias[idx])
            and not np.any(sampler.penalties_state.use_penalty[idx])
            and not np.any(sampler.bad_words_state.num_bad_words.np[idx])
        )

    @staticmethod
    def _head(model: nn.Module):
        """(LM head shard, softcap, valid rows, first token id) of the
        target, or None if its logits are not softcap(hidden @ head)."""
        processor = getattr(model, "logits_processor", None)
        lm_head = getattr(model, "lm_head", None)
        if (
            processor is None
            or lm_head is None
            or processor.scale != 1.0
            or processor.logits_as_input
        ):
            return None
        shard = lm_head.weight.shape[0]
        id_base = get_tensor_model_parallel_rank() * shard
        return (
            lm_head.weight,
            processor.soft_cap,
            min(max(processor.org_vocab_size - id_base, 0), shard),
            id_base,
        )

    def _target_argmax(
        self, model: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        """Greedy target token of every row, without the dense logits."""
        get_top_tokens = getattr(model, "get_top_tokens", None)
        if get_top_tokens is not None:
            return get_top_tokens(hidden_states).view(-1)
        from vllm.models.inkling2.sm70.dspark import mm_fp32
        from vllm.models.inkling2.sm70.ops.dspark_markov import shard_argmax

        head = self._head(model)
        if head is None:
            return None
        weight, softcap, num_valid, id_base = head
        return shard_argmax(
            mm_fp32(hidden_states, weight),
            softcap,
            num_valid,
            id_base,
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
            tensor_model_parallel_all_reduce,
        )

    def _local_verify(
        self, model: nn.Module, hidden_states: torch.Tensor, input_batch: InputBatch
    ) -> SamplerOutput | None:
        """Rejection sampling (and plain sampling of steps without drafts)
        on the vocabulary shards: `ops/dspark_verify.py`."""
        from vllm.models.inkling2.sm70.dspark import mm_fp32
        from vllm.models.inkling2.sm70.ops.dspark_verify import shard_rejection_sample

        head = self._head(model)
        if head is None:
            return None
        weight, softcap, num_valid, id_base = head
        states = self._sampler.sampling_states
        sampled, num_sampled = shard_rejection_sample(
            mm_fp32(hidden_states, weight),
            self._dense_local,
            input_batch.input_ids,
            input_batch.logits_indices,
            input_batch.cu_num_logits,
            input_batch.positions[input_batch.logits_indices],
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            states.temperature.gpu,
            states.seeds.gpu,
            self.num_speculative_steps,
            softcap,
            num_valid,
            id_base,
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
            self.use_fp64_gumbel,
            tensor_model_parallel_all_reduce,
        )
        if not self._local_logged:
            self._local_logged = True
            logger.info("%s: shard-local rejection sampling.", self._speculator_name)
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
        )

    def try_sample_target(
        self,
        model: nn.Module,
        rejection_sampler,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output,
        *,
        allow_graph: bool = True,
    ) -> SamplerOutput | None:
        if rejection_sampler is not None and self._sampler is None:
            # Also gives the proposal path the requests' temperatures.
            self._sampler = rejection_sampler.sampler
        if FLAGS_FILE:
            _Live.poll()
        output = self._sample_target(
            model, rejection_sampler, hidden_states, input_batch, grammar_output
        )
        if self._trace:
            per = input_batch.num_draft_tokens_per_req
            self._trace_verify = [
                "dense" if output is None else "local",
                None if per is None else per.tolist(),
                bool(self._row_gathered[input_batch.idx_mapping_np].all()),
                input_batch.idx_mapping_np.tolist(),
                None if output is None else output.sampled_token_ids.tolist(),
            ]
        if (
            output is None
            and not self._row_gathered[input_batch.idx_mapping_np].all()
            and input_batch.num_draft_tokens > 0
            and self.draft_logits is not None
            and self._dense_local is not None
            and self._samples(input_batch)
        ):
            # The dense sampler is about to read draft logits that the
            # proposal did not gather (the batch changed since).
            self._gather_dense_for(input_batch)
        return output

    def _sample_target(
        self,
        model: nn.Module,
        rejection_sampler,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output,
    ) -> SamplerOutput | None:
        if (
            grammar_output is not None
            or rejection_sampler is None
            or rejection_sampler.synthetic_conditional_rates is not None
        ):
            return None
        if not (_Live.greedy and self._all_greedy(input_batch)):
            if (
                _Live.local
                and self._local_verify_ok(input_batch)
                # Drafted rows need this rank's dense draft logits.
                and (
                    input_batch.num_draft_tokens == 0
                    or (self._fused and self._dense_local is not None)
                )
            ):
                return self._local_verify(model, hidden_states, input_batch)
            return None
        target_ids = self._target_argmax(model, hidden_states)
        if target_ids is None:
            return None
        if not self._greedy_logged:
            self._greedy_logged = True
            logger.info(
                "%s: greedy target tokens from the shard-local argmax.",
                self._speculator_name,
            )
        if input_batch.num_draft_tokens > 0:
            from vllm.v1.worker.gpu.spec_decode.sm70_greedy_verify import greedy_verify

            sampled, num_sampled = greedy_verify(
                target_ids,
                input_batch.input_ids[input_batch.logits_indices],
                input_batch.cu_num_logits,
                rejection_sampler.num_speculative_steps,
            )
        else:
            sampled = target_ids.view(-1, 1)
            num_sampled = input_batch.seq_lens.new_ones(input_batch.num_reqs)
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
        )
