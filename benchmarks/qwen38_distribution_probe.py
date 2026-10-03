# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-vocabulary teacher-forcing probes; excluded from speed measurements."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def distribution_metrics(reference, candidate):
    a = torch.as_tensor(reference, dtype=torch.float64)
    b = torch.as_tensor(candidate, dtype=torch.float64)
    if a.ndim != 1 or a.shape != b.shape or a.numel() < 2:
        raise ValueError("Expected matching full-vocabulary logit rows")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite valid-vocabulary logits")
    la = a - torch.logsumexp(a, dim=0)
    lb = b - torch.logsumexp(b, dim=0)
    delta = b - a
    return {
        "kl": max(0.0, float((la.exp() * (la - lb)).sum())),
        "reverse_kl": max(0.0, float((lb.exp() * (lb - la)).sum())),
        "top1_agreement": int(a.argmax()) == int(b.argmax()),
        "reference_top1": int(a.argmax()),
        "candidate_top1": int(b.argmax()),
        "reference_margin": float(a.topk(2).values.diff().abs()[0]),
        "max_logit_error": float(delta.abs().max()),
        "mean_logit_offset": float(delta.mean()),
        "centered_max_logit_error": float((delta - delta.mean()).abs().max()),
        "vocabulary": a.numel(),
    }


class DistributionProbeWorkerExtension:
    """Observe the current MRv2 sampler without switching model runners.

    This diagnostic hook is installed through the supported worker-extension
    RPC. Requests ask for one logprob to materialize the complete vocabulary
    instead of the greedy TP-local top-1 path. Neither timing nor sampling
    acceptance is taken from this instrumented run.
    """

    def configure_distribution_probe(self, requests):
        runner = self.model_runner
        if not hasattr(runner, "req_states") or not hasattr(
            runner.sampler, "sampling_states"
        ):
            raise RuntimeError("This probe requires the current MRv2 runner")
        self._distribution_requests = requests
        if hasattr(self, "_distribution_original_sample"):
            return
        sampler = runner.sampler
        self._distribution_original_sample = sampler.sample

        def observed(logits, expanded, indices, positions, inputs, local_positions):
            if logits.shape[0] != len(indices):
                raise RuntimeError("No-MTP probe received expanded speculative logits")
            positions_cpu = positions.cpu().tolist()
            inputs_cpu = inputs.cpu().tolist()
            for row, index in enumerate(indices):
                request_id = runner.req_states.index_to_req_id[int(index)]
                spec = self._distribution_requests[request_id]
                prompt = spec["prompt_token_ids"]
                step = int(positions_cpu[row]) - (len(prompt) - 1)
                if step < 0:  # Incomplete chunked prefill; no token is committed.
                    continue
                continuation = spec["continuation"]
                if not 0 <= step < len(continuation):
                    raise RuntimeError("Probe position is outside frozen continuation")
                expected_input = prompt[-1] if step == 0 else continuation[step - 1]
                if int(inputs_cpu[row]) != expected_input:
                    raise RuntimeError(
                        "Teacher-forcing input token differs from reference"
                    )
                vocabulary = spec["vocabulary"]
                if logits.shape[1] < vocabulary:
                    raise RuntimeError("Logit row does not cover valid vocabulary")
                if spec["capture"] and self.rank == 0:
                    destination = Path(spec["output"])
                    destination.mkdir(parents=True, exist_ok=True)
                    np.save(
                        destination / f"{step:04d}.npy",
                        logits[row, :vocabulary].detach().cpu().numpy(),
                        allow_pickle=False,
                    )
                    (destination / f"{step:04d}.json").write_text(
                        json.dumps(
                            {
                                "prefix_sha256": hashlib.sha256(
                                    json.dumps(prompt + continuation[:step]).encode()
                                ).hexdigest(),
                                "step": step,
                                "vocabulary": vocabulary,
                                "forced_next_token": continuation[step],
                                "active_width": len(indices),
                            }
                        )
                        + "\n"
                    )
                logits[row].fill_(-float("inf"))
                logits[row, continuation[step]] = 0
            return self._distribution_original_sample(
                logits, expanded, indices, positions, inputs, local_positions
            )

        sampler.sample = observed
