# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit experimental installation of context work behind the target probe.

The unchanged CPU cutoff guard consumes a pinned copy while the original
context graph runs on its original stream. Importing this file changes no
dispatch. KV storage, proposal, sampling arithmetic and RNG are untouched.
"""

import functools

import torch


def install_context_probe_candidate(*, shadow: bool = False) -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.dflash2 import sparse_rejection
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    capture_original = DFlash2Speculator.capture
    prepare_original = DFlash2Speculator.prepare_target_context
    sample_original = GPUModelRunner.sample
    guard_original = sparse_rejection._compact_target_requires_reference
    active = None

    class PendingContext:
        def __init__(self, owner):
            self.owner = owner
            self.arguments = None
            self.probe = torch.empty((8, 21), dtype=torch.float32, pin_memory=True)
            self.ready = torch.cuda.Event()
            self.calls = 0

        def flush(self):
            if self.arguments is not None:
                arguments = self.arguments
                self.arguments = None
                prepare_original(self.owner, *arguments)
                if shadow:
                    hidden = self.owner.hidden_states[:8].clone()
                    kv = [t.clone() for t in self.owner._context_projected_kv]
                    prepare_original(self.owner, *arguments)
                    assert torch.equal(
                        hidden.view(torch.uint8),
                        self.owner.hidden_states[:8].view(torch.uint8),
                    )
                    for left, right in zip(kv, self.owner._context_projected_kv):
                        assert torch.equal(
                            left.view(torch.uint8), right.view(torch.uint8)
                        )

    @functools.wraps(capture_original)
    def capture(self):
        result = capture_original(self)
        if self._context_compute_graph is not None:
            self._context_probe_candidate = PendingContext(self)
        return result

    @functools.wraps(prepare_original)
    def prepare(self, batch, hidden, aux):
        context = getattr(self, "_context_probe_candidate", None)
        if context is not None:
            assert context.arguments is None, "unconsumed previous context"
        if (
            context is None
            or batch.num_reqs != 1
            or batch.num_tokens != 8
            or batch.num_draft_tokens != 7
            or batch.is_prefilling_np[0]
        ):
            return prepare_original(self, batch, hidden, aux)
        self._prepared_context_batch = None
        context.arguments = (batch, hidden, aux)

    @functools.wraps(guard_original)
    def guard(probe_logits, temperature, top_p):
        context = active
        if context is None or context.arguments is None:
            return guard_original(probe_logits, temperature, top_p)
        if (
            probe_logits.shape != (8, 21)
            or probe_logits.dtype != torch.float32
            or probe_logits.device != context.owner.device
            or not probe_logits.is_contiguous()
        ):
            context.flush()
            return guard_original(probe_logits, temperature, top_p)
        context.probe.copy_(probe_logits.detach(), non_blocking=True)
        context.ready.record(torch.cuda.current_stream(context.owner.device))
        context.flush()
        # Wait for the probe copy, not for the context graph queued after it.
        context.ready.synchronize()
        result = guard_original(context.probe, temperature, top_p)
        if shadow:
            original_probe = probe_logits.detach().cpu()
            assert torch.equal(
                context.probe.view(torch.uint8), original_probe.view(torch.uint8)
            )
            assert result == guard_original(original_probe, temperature, top_p)
        context.calls += 1
        if context.calls == 1 or (shadow and context.calls % 64 == 0):
            print(
                "CONTEXT_PROBE_ROUTE "
                f"rank={torch.distributed.get_rank()} calls={context.calls} "
                f"shadow={shadow} guard={guard_original.__module__}",
                flush=True,
            )
        return result

    @functools.wraps(sample_original)
    def sample(self, *args, **kwargs):
        nonlocal active
        context = getattr(self.speculator, "_context_probe_candidate", None)
        if context is None:
            return sample_original(self, *args, **kwargs)
        assert active is None, "nested sampling would reuse the pinned probe"
        active = context
        try:
            result = sample_original(self, *args, **kwargs)
            # Structured output, full-vocabulary fallback and teacher forcing
            # may skip the compact guard. Complete their deferred preparation
            # before the caller mutates state or submits proposal consumers.
            context.flush()
            return result
        except BaseException:
            context.arguments = None
            raise
        finally:
            active = None

    DFlash2Speculator.capture = capture
    DFlash2Speculator.prepare_target_context = prepare
    GPUModelRunner.sample = sample
    sparse_rejection._compact_target_requires_reference = guard
