# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for upstream PLE priority and the broader output-buffer fallback."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import ple_layer as ple


def make_layer(tokens):
    layer = object.__new__(ple.Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(layer)
    layer.ngram_size = 3
    layer.heads_per_ngram = 8
    layer.ngram_heads = 16
    layer.eos_token_id = 151645
    layer.positions_buffer = torch.arange(tokens)
    layer.padded_buffer = torch.empty((32, tokens), dtype=torch.long)
    layer.layer_multipliers = torch.tensor([3, 5, 7])
    layer.ngram_heads_vocab_sizes = torch.arange(101, 117)
    layer.ngram_heads_offsets = torch.arange(16) * 117
    return layer


@pytest.mark.parametrize("tokens", [2, 5, 32])
def test_upstream_ngram_precedes_local_fusion_and_preserves_output(monkeypatch, tokens):
    layer = make_layer(tokens)
    ids = torch.arange(tokens)
    starts = torch.tensor([0, tokens])
    context = torch.zeros((1, 2), dtype=torch.long)
    output = torch.empty((tokens, 16), dtype=torch.long)
    native_result = torch.full_like(output, 19)
    calls = []
    monkeypatch.setattr(ple, "is_offload_process", lambda: False)
    monkeypatch.setattr(ple._fuse47, "unit_enabled", lambda _: True)
    monkeypatch.setattr(ple._fuse47, "p1_supported", lambda *_: True)
    monkeypatch.setattr(ple.SM70_PLE_NGRAM.__class__, "reason", lambda *_: None)

    def native(*args):
        calls.append(args)
        return native_result

    def unexpected_local(*args):
        raise AssertionError("qualified upstream route must take priority")

    monkeypatch.setattr(ple, "sm70_ple_ngram_ids", native)
    monkeypatch.setattr(ple._fuse47, "ple_ngram_ids", unexpected_local)
    monkeypatch.setattr(
        ple,
        "get_forward_context",
        lambda: SimpleNamespace(
            no_compile_layers={"layer": SimpleNamespace(ple_embedding=layer)}
        ),
    )

    ple.qwen4_exp_compute_ple_ngram_ids(ids, starts, context, output, "layer")

    assert len(calls) == 1
    assert calls[0][0] is ids or torch.equal(calls[0][0], ids)
    assert calls[0][1] is starts and calls[0][2] is context
    torch.testing.assert_close(output, native_result, rtol=0, atol=0)


def test_unqualified_40_row_mtp_batch_keeps_local_output_buffer(monkeypatch):
    layer = make_layer(40)
    ids = torch.arange(40)
    starts = torch.arange(0, 41, 5)
    context = torch.zeros((8, 2), dtype=torch.long)
    output = torch.empty((40, 16), dtype=torch.long)
    calls = []
    monkeypatch.setattr(ple, "is_offload_process", lambda: False)
    monkeypatch.setattr(ple._fuse47, "unit_enabled", lambda _: True)
    monkeypatch.setattr(ple._fuse47, "p1_supported", lambda *_: True)
    monkeypatch.setattr(
        ple.SM70_PLE_NGRAM.__class__,
        "reason",
        lambda *_: "outside_small_batch_band",
    )

    def local(layer_arg, ids_arg, starts_arg, context_arg, out):
        calls.append((layer_arg, ids_arg, starts_arg, context_arg, out))
        out.fill_(23)
        return out

    def unexpected_native(*args):
        raise AssertionError("unqualified route must keep its fallback")

    monkeypatch.setattr(ple._fuse47, "ple_ngram_ids", local)
    monkeypatch.setattr(ple, "sm70_ple_ngram_ids", unexpected_native)

    result = layer.compute_ngram_ids(ids, starts, context, out=output)

    assert result is output and len(calls) == 1
    assert calls[0][0] is layer and calls[0][4] is output
    assert torch.equal(result, torch.full_like(output, 23))
