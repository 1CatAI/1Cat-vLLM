# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in full-vocabulary teacher logits from the actual MTP target graph."""

from pathlib import Path

import torch

from vllm.sm70_graph_observer import GraphParityWorkerExtension


class GGUFTeacherWorkerExtension(GraphParityWorkerExtension):
    def start_teacher_capture(self, directory: str, key: str):
        if hasattr(self, "_teacher_original_execute"):
            raise RuntimeError("Teacher capture is already active")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self._teacher_count = 0
        runner = self.model_runner
        original = runner.execute_model
        self._teacher_original_execute = original

        def execute(*args, **kwargs):
            result = original(*args, **kwargs)
            state = runner.execute_model_state
            if state is None or self._teacher_count:
                return result
            batch = state.input_batch
            if batch.num_reqs != 1 or batch.num_draft_tokens != 4:
                return result
            if batch.num_tokens_after_padding != 5 or state.hidden_states is None:
                raise RuntimeError("Teacher capture requires the actual M5 target")
            with torch.inference_mode():
                index = batch.logits_indices[:1]
                hidden = state.hidden_states[index]
                # All TP workers participate in the ordinary full LM head.
                # This runs after replay and before sampling constraints, and
                # cannot affect the target graph's hidden states or logits.
                logits = runner.model.compute_logits(hidden)
                if logits is None or logits.shape != (1, runner.vocab_size):
                    raise RuntimeError("Teacher logits must cover the full vocabulary")
                self._teacher_count += 1
                if self.rank == 0:
                    torch.save(
                        dict(
                            logits=logits.detach().float().cpu(),
                            position=batch.positions[index].cpu(),
                            input_ids=batch.input_ids[index].cpu(),
                            request_ids=list(batch.req_ids),
                        ),
                        root / f"{key}.pt",
                    )
            return result

        runner.execute_model = execute
        return {"rank": self.rank, "active": True}

    def stop_teacher_capture(self):
        original = getattr(self, "_teacher_original_execute", None)
        if original is None:
            raise RuntimeError("Teacher capture is not active")
        self.model_runner.execute_model = original
        del self._teacher_original_execute
        return {"rank": self.rank, "captured": self._teacher_count}
