# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import ple_layer as ple


@pytest.mark.parametrize("cascade", [False, True])
def test_hybrid_decode_waits_only_for_cascade_rows(monkeypatch, cascade):
    """A hybrid graph has no remote request to satisfy a semaphore wait."""
    monkeypatch.setattr(ple, "is_offload_process", lambda: False)
    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen4_exp_compute_ple_ngram_ids",
        lambda _ids, _starts, _context, out, _name: out.zero_(),
    )
    waits = []
    remote = torch.full((5, 2), 7.0)

    def wait(_hidden, count):
        waits.append(count)
        return remote

    def lookup(_ids, remote_rows=None):
        if cascade:
            assert remote_rows is remote
        else:
            assert remote_rows is None
        return torch.ones(5, 2, 3)

    layer = SimpleNamespace(
        _is_cpu_offloaded=True,
        _cascade=cascade,
        ngram_heads=2,
        layer_name="test.ple",
        wait_offloaded_output=wait,
        ngram_embedding=lookup,
    )
    output = ple.Qwen4ExpNGramEmbedding.forward_impl(
        layer,
        torch.zeros(5, 6),
        torch.zeros(5, dtype=torch.long),
        torch.tensor([0, 5]),
        torch.zeros(1, 2, dtype=torch.long),
    )
    assert output.shape == (5, 6)
    assert waits == ([5] if cascade else [])
