# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic NVTX annotations installed after graph capture.

Use only for the Nsight composition pass. Ordinary speed admission uses the
normal worker, without these wrappers or synchronized step profiling.
"""

from functools import wraps

import torch

from vllm.v1.worker.gpu_worker import Worker


def annotate(owner, name, label):
    original = getattr(owner, name)

    @wraps(original)
    def observed(*args, **kwargs):
        text = label(*args, **kwargs) if callable(label) else label
        with torch.cuda.nvtx.range(text):
            return original(*args, **kwargs)

    setattr(owner, name, observed)


class TraceWorker(Worker):
    def profile(self, is_start=True, profile_prefix=None):
        if is_start:
            # The benchmark reaches this boundary with an idle request queue.
            # Release unused eager-prefill allocations before CUPTI allocates
            # its buffers; graph-owned tensors and pointers remain live.
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
        return super().profile(is_start, profile_prefix)

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        runner = self.model_runner
        prepared = {}
        prepare = runner.prepare_inputs

        @wraps(prepare)
        def prepare_inputs(*args, **kwargs):
            with torch.cuda.nvtx.range("flashnext_batch.prepare"):
                batch = prepare(*args, **kwargs)
            prepared.update(
                requests=batch.num_reqs,
                tokens=batch.num_tokens,
                drafts=batch.num_draft_tokens,
            )
            return batch

        runner.prepare_inputs = prepare_inputs
        annotate(runner, "sample", "flashnext_batch.target_sample")
        annotate(runner, "postprocess_sampled", "flashnext_batch.target_state")
        annotate(
            runner.cudagraph_manager,
            "run_fullgraph",
            lambda desc: (
                f"flashnext_batch.target/M{desc.num_tokens}/R{desc.num_reqs}"
                f"/actualR{prepared['requests']}/T{prepared['tokens']}"
                f"/D{prepared['drafts']}"
            ),
        )
        draft = runner.speculator
        annotate(
            draft,
            "propose",
            lambda batch, *args, **kwargs: f"flashnext_batch.draft/R{batch.num_reqs}",
        )
        for manager, name in (
            (draft.prefill_cudagraph_manager, "draft_first"),
            (draft.decode_cudagraph_manager, "draft_rest"),
            (getattr(draft, "multistep_cudagraph_manager", None), "draft_chain"),
        ):
            if manager is None:
                continue
            annotate(
                manager,
                "run_fullgraph",
                lambda desc, name=name: f"flashnext_batch.{name}/M{desc.num_tokens}",
            )
        return result
