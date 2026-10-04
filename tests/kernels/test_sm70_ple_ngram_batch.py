# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.ple.ngram import SM70_PLE_NGRAM
from vllm.models.qwen4_exp.nvidia import ple_layer


def make_layer(m):
    layer = object.__new__(ple_layer.Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(layer)
    layer.ngram_size = 3
    layer.heads_per_ngram = 8
    layer.ngram_heads = 16
    layer.eos_token_id = 151645
    layer.positions_buffer = torch.arange(m, device="cuda")
    layer.padded_buffer = torch.empty((32, m), dtype=torch.long, device="cuda")
    layer.layer_multipliers = torch.tensor(
        [-7046029254386353131, -4658895280553007687, -7723592293110705685],
        device="cuda",
    )
    layer.ngram_heads_vocab_sizes = torch.arange(20000101, 20000117, device="cuda")
    layer.ngram_heads_offsets = torch.arange(16, device="cuda") * 20000117
    return layer


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "m,starts",
    [
        (2, [0, 2]),
        (5, [0, 5]),
        (10, [0, 5, 10]),
        (20, [0, 5, 10, 15, 20]),
        (32, [0, 0, 4, 4, 20, 28]),
        (5, [0, 0, 0]),
    ],
)
def test_exact_legacy_ids_and_graph(monkeypatch, dtype, m, starts):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    monkeypatch.setattr(ple_layer, "is_offload_process", lambda: False)
    layer = make_layer(m)
    ids = (torch.arange(m, device="cuda") * 7919 + 12).to(dtype)
    ids[1::3] = layer.eos_token_id
    q_starts = torch.tensor(starts, device="cuda", dtype=dtype)
    requests = len(starts) - 1
    context = torch.full((requests, 2), 17, device="cuda", dtype=dtype)
    context[::2, 1] = layer.eos_token_id

    def reference():
        with monkeypatch.context() as patch:
            patch.setattr(
                ple_layer,
                "SM70_PLE_NGRAM",
                SimpleNamespace(
                    reason=lambda *args: "force_legacy_reference",
                ),
            )
            return layer.compute_ngram_ids(ids, q_starts, context)

    expected = reference()
    actual = layer.compute_ngram_ids(ids, q_starts, context)
    assert torch.equal(actual, expected)
    for _ in range(3):
        layer.compute_ngram_ids(ids, q_starts, context)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = layer.compute_ngram_ids(ids, q_starts, context)
    ids.add_(3)
    context.add_(7)
    q_starts[1:-1].copy_(q_starts[1:-1] // 2)
    expected = reference()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replay, expected)


def test_cpu_inputs_report_hardware_fallback():
    data = torch.zeros(5, dtype=torch.long)
    assert SM70_PLE_NGRAM.reason(data, data, data, data, data, data) == "requires_sm70"
