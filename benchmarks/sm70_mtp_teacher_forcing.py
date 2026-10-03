# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic-only MTP4 forcing; install after unprofiled measurements.

Target retains its captured M5 graph. Draft runs eagerly so each of its four
conditional distributions can be recorded before enforcing the frozen tape.
These requests cannot supply timing or natural acceptance evidence.
"""

import types
from pathlib import Path

import torch


def install(worker, token_ids, prompt_length, prompt_sha256, folder):
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    runner = worker.model_runner
    draft = runner.speculator
    if type(runner).__name__ != "GPUModelRunner" or draft.method != "mtp":
        raise ValueError("This observer requires the V2 MTP runner")
    if hasattr(runner, "_mtp15_forcing"):
        raise ValueError("Observer already installed")
    if len(token_ids) <= prompt_length + 8:
        raise ValueError("Need a continuation and final draft padding")
    tape = torch.tensor(token_ids, device=runner.device, dtype=torch.int64)
    state = {"target": [], "draft": [], "tape": tape}
    runner._mtp15_forcing = state
    rank = torch.distributed.get_rank()
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    prepare = runner.prepare_inputs
    sample = runner.sample
    propose = draft.propose
    run_model = draft.run_model
    sample_draft = draft._sample_draft

    def record(role, positions, logits, ids):
        if rank != 0:
            return
        pos = positions.detach().cpu().clone().long()
        selected = pos >= prompt_length - 1
        if not selected.any():
            return
        row = {
            "positions": pos[selected],
            "tokens": ids.detach().cpu().clone().long()[selected],
            "logits": logits.detach().cpu().clone()[selected],
        }
        state[role].append(row)
        return row

    def prepare_fixed(*args, **kwargs):
        batch = prepare(*args, **kwargs)
        if batch.num_reqs != 1:
            raise ValueError("Teacher forcing requires one request")
        positions = batch.positions[: batch.num_tokens].long()
        batch.input_ids[: batch.num_tokens].copy_(tape[positions])
        return batch

    def sample_fixed(hidden, batch, grammar):
        positions = batch.positions[batch.logits_indices].long()
        if positions.numel() == 0 or int(positions.max()) < prompt_length - 1:
            return sample(hidden, batch, grammar)
        if grammar is not None:
            raise ValueError("Grammar transforms are not teacher-forcing logits")
        logits = runner.model.compute_logits(hidden[batch.logits_indices])
        record("target", positions, logits, tape[positions])
        next_ids = tape[positions + 1].view(1, -1).to(torch.int32)
        count = torch.full(
            (1,), next_ids.shape[1], device=runner.device, dtype=torch.int32
        )
        return (
            SamplerOutput(next_ids, None, None, count),
            count,
            torch.zeros_like(count),
        )

    def propose_fixed(*args, **kwargs):
        # Force only the diagnostic proposer eager, without altering its model
        # or target captured kernels. This guarantees Python logits observation.
        kwargs["is_profile"] = True
        return propose(*args, **kwargs)

    def draft_forward(num_tokens, *args, **kwargs):
        positions = draft.input_buffers.positions[:num_tokens].long()
        # MTP input token is shifted one position relative to target hidden.
        draft.input_buffers.input_ids[:num_tokens].copy_(tape[positions + 1])
        return run_model(num_tokens, *args, **kwargs)

    def sample_draft_fixed(self, hidden, idx_mapping, positions, step, draft_logits):
        logits = self.model.compute_logits(hidden)
        # The draft consumes token at pos+1 and predicts token at pos+2.
        row = record("draft", positions + 1, logits, tape[positions.long() + 1])
        if self.use_local_argmax_reduction:
            top_tokens = self.model.get_top_tokens(hidden)
            if row is not None:
                selected = (positions + 1).detach().cpu() >= prompt_length - 1
                row["optimized_token_ids"] = top_tokens.detach().cpu().clone()[selected]
        return tape[positions.long() + 2]

    runner.prepare_inputs = prepare_fixed
    runner.sample = sample_fixed
    draft.propose = propose_fixed
    draft.run_model = draft_forward
    draft._sample_draft = types.MethodType(sample_draft_fixed, draft)
    state["restore"] = {
        "prepare": prepare,
        "sample": sample,
        "propose": propose,
        "run_model": run_model,
        "sample_draft": sample_draft,
    }
    state["folder"] = str(root)
    state["prompt_sha256"] = prompt_sha256
    return {"rank": rank, "installed": True, "diagnostic_only": True}


def flush(worker, *, discard=False):
    """Save a diagnostic tape, restoring the runner even if saving fails."""
    runner = worker.model_runner
    state = runner._mtp15_forcing
    rank = torch.distributed.get_rank()
    counts = {}
    try:
        for role in ("target", "draft"):
            rows = state[role]
            counts[role] = sum(row["positions"].numel() for row in rows)
            if rank == 0 and not discard:
                if not rows:
                    raise RuntimeError(f"No {role} logits captured")
                dump = {
                    "role": role,
                    "prompt_sha256": state["prompt_sha256"],
                    "position_ids": torch.cat([row["positions"] for row in rows]),
                    "token_ids": torch.cat([row["tokens"] for row in rows]),
                    "logits": torch.cat([row["logits"] for row in rows]),
                }
                if all("optimized_token_ids" in row for row in rows):
                    dump["optimized_token_ids"] = torch.cat(
                        [row["optimized_token_ids"] for row in rows]
                    )
                torch.save(dump, Path(state["folder"]) / f"{role}.pt")
    finally:
        restore = state["restore"]
        runner.prepare_inputs = restore["prepare"]
        runner.sample = restore["sample"]
        runner.speculator.propose = restore["propose"]
        runner.speculator.run_model = restore["run_model"]
        runner.speculator._sample_draft = restore["sample_draft"]
        del runner._mtp15_forcing
    return {"rank": rank, "counts": counts, "restored": True}
