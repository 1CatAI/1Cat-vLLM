# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.sm70_gguf_quality import GGUFTeacherWorkerExtension


@pytest.mark.parametrize("rank", [0, 1])
def test_teacher_reads_unmasked_full_logits_once_after_verifier(tmp_path, rank):
    calls = []
    hidden = torch.arange(20).reshape(5, 4).float()
    batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=0,
        num_tokens_after_padding=5,
        logits_indices=torch.arange(5),
        positions=torch.arange(100, 105),
        input_ids=torch.arange(30, 35),
        req_ids=["teacher"],
    )

    def head(x):
        calls.append(x.clone())
        return torch.arange(7).reshape(1, 7).float()

    runner = SimpleNamespace(
        execute_model=lambda: None,
        vocab_size=7,
        model=SimpleNamespace(compute_logits=head),
        execute_model_state=SimpleNamespace(input_batch=batch, hidden_states=hidden),
    )
    worker = GGUFTeacherWorkerExtension()
    worker.rank, worker.model_runner = rank, runner
    original = runner.execute_model
    worker.start_teacher_capture(str(tmp_path), "position")
    runner.execute_model()
    assert not calls  # Prefill is not evidence for the M5 verifier.
    batch.num_draft_tokens = 4
    runner.execute_model()
    runner.execute_model()
    assert len(calls) == 1  # All TP workers must participate in the LM head.
    torch.testing.assert_close(calls[0], hidden[:1])
    result = worker.stop_teacher_capture()
    assert result["captured"] == 1 and runner.execute_model is original
    path = tmp_path / "position.pt"
    assert path.exists() == (rank == 0)
    if rank == 0:
        captured = torch.load(path, weights_only=True)
        assert captured["position"].item() == 100
        assert captured["input_ids"].item() == 30
        assert captured["logits"].shape == (1, 7)
        assert captured["logits"].dtype == torch.float32


def test_teacher_rejects_padded_verifier_instead_of_claiming_m5(tmp_path):
    worker = GGUFTeacherWorkerExtension()
    worker.rank = 0
    worker.model_runner = SimpleNamespace(
        execute_model=lambda: None,
        execute_model_state=SimpleNamespace(
            input_batch=SimpleNamespace(
                num_reqs=1, num_draft_tokens=4, num_tokens_after_padding=8
            ),
            hidden_states=torch.zeros(8, 4),
        ),
    )
    worker.start_teacher_capture(str(tmp_path), "padded")
    with pytest.raises(RuntimeError, match="actual M5"):
        worker.model_runner.execute_model()
    assert worker.stop_teacher_capture()["captured"] == 0
