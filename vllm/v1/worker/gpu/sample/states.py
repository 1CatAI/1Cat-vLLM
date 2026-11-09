# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_runtime import bind_topk_topp_runtime
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p

NO_LOGPROBS = -1
_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max


class SamplingStates:
    def __init__(self, max_num_reqs: int, vocab_size: int):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size
        self._topk_runtime = bind_topk_topp_runtime()

        self.temperature = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.top_k = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.top_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.min_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.seeds = UvaBackedTensor(max_num_reqs, dtype=torch.int64)

        # Initialize top_k and top_p manually because 0 is an invalid value for them.
        self.top_k.np.fill(self.vocab_size)
        self.top_k.copy_to_uva()
        self.top_p.np.fill(1.0)
        self.top_p.copy_to_uva()
        # Temperature is the SAME kind of hazard: UvaBackedTensor zero-inits,
        # and 0.0 is not a valid temperature -- `logits / temperature` becomes
        # inf and the distribution collapses to a point mass on the argmax.
        # Only live request slots are ever written by add_request, but the
        # graph kernels index this table by request slot, so a stale/rounded
        # slot must not read 0. Default to the 1.0 "no scaling" value.
        self.temperature.np.fill(1.0)
        self.temperature.copy_to_uva()

        self.num_logprobs = np.empty(self.max_num_reqs, dtype=np.int32)
        # -1 means no logprobs are requested.
        self.num_logprobs.fill(NO_LOGPROBS)

        # [2026-11-08] Graph-stable mirrors of the sampling parameters.
        # `UvaBackedTensor.copy_to_uva()` returns a buffer from a round-robin
        # pool (buffer_utils._DEFAULT_MAX_CONCURRENCY=2), so `.gpu` is a
        # DIFFERENT address after every `apply_staged_writes()`. Any CUDA graph
        # captured against `self.<field>.gpu` keeps reading the buffer it saw
        # at capture time, i.e. stale parameters. A wrong temperature flattens
        # the distribution and decodes fall onto vocabulary-edge tokens:
        # outputs repeat one fragment forever at T>=0.5, while T==0 looks
        # healthy because greedy never reads the value. That asymmetry is the
        # fingerprint of this bug.
        # These mirrors have fixed addresses and are refreshed with the pool.
        device = self.temperature.gpu.device
        self.temperature_stable = torch.empty(
            max_num_reqs, dtype=torch.float32, device=device
        )
        self.top_p_stable = torch.empty(max_num_reqs, dtype=torch.float32, device=device)
        self.min_p_stable = torch.empty(max_num_reqs, dtype=torch.float32, device=device)
        self.top_k_stable = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        self.seeds_stable = torch.empty(max_num_reqs, dtype=torch.int64, device=device)
        self._refresh_stable()

    def _refresh_stable(self) -> None:
        """Copy the live UVA tensors into the graph-stable mirrors."""
        self.temperature_stable.copy_(self.temperature.gpu)
        self.top_p_stable.copy_(self.top_p.gpu)
        self.min_p_stable.copy_(self.min_p.gpu)
        self.top_k_stable.copy_(self.top_k.gpu)
        self.seeds_stable.copy_(self.seeds.gpu)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        self.temperature.np[req_idx] = sampling_params.temperature
        self.top_p.np[req_idx] = sampling_params.top_p
        top_k = sampling_params.top_k
        if top_k <= 0 or top_k > self.vocab_size:
            top_k = self.vocab_size
        self.top_k.np[req_idx] = top_k
        self.min_p.np[req_idx] = sampling_params.min_p

        seed = sampling_params.seed
        if seed is None:
            seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
        self.seeds.np[req_idx] = seed

        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = NO_LOGPROBS
        elif num_logprobs == -1:
            # The public full-vocabulary request is not our internal None marker.
            num_logprobs = self.vocab_size
        self.num_logprobs[req_idx] = num_logprobs

    def apply_staged_writes(self) -> None:
        self.temperature.copy_to_uva()
        self.top_p.copy_to_uva()
        self.top_k.copy_to_uva()
        self.min_p.copy_to_uva()
        self.seeds.copy_to_uva()
        # Keep the graph-stable mirrors in lockstep with the rotating pool.
        self._refresh_stable()

    # [2026-11-08] The skip-branches below are CPU control flow, therefore
    # TRACE CONSTANTS for torch.cuda.graph: whichever branch is taken while
    # capturing is the only one baked into the graph, forever. DFlash2's
    # sampler graph captures `_dense()` during warm-up, when every request
    # still holds the default params (temperature 1.0, top_p 1.0, min_p 0).
    # The capture therefore baked in "skip", and every later replay sampled
    # an unprocessed (untempered) distribution: T==0 looked healthy because
    # greedy never wants temperature, while T>=0.5 degenerated into repeating
    # one fragment. Force the kernels into the graph instead.
    capture_mode = False

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        temp_np = self.temperature.np[idx_mapping_np]
        if not self.capture_mode and np.all((temp_np == 0.0) | (temp_np == 1.0)):
            # No request requires temperature. Skip the kernel launch.
            return

        apply_temperature(logits, expanded_idx_mapping, self.temperature_stable)

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if not self.capture_mode and np.all(self.min_p.np[idx_mapping_np] == 0.0):
            # No request uses min_p. Skip the kernel launch.
            return
        apply_min_p(logits, expanded_idx_mapping, self.min_p_stable)

    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        do_top_k = np.any(self.top_k.np[idx_mapping_np] != self.vocab_size)
        do_top_p = np.any(self.top_p.np[idx_mapping_np] != 1.0)
        if self.capture_mode:
            do_top_k = do_top_p = True
        elif not (do_top_k or do_top_p):
            return logits

        top_k = self.top_k_stable[expanded_idx_mapping] if do_top_k else None
        top_p = self.top_p_stable[expanded_idx_mapping] if do_top_p else None
        return apply_top_k_top_p(logits, top_k, top_p, runtime=self._topk_runtime)

    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int:
        return int(np.max(self.num_logprobs[idx_mapping_np]))
