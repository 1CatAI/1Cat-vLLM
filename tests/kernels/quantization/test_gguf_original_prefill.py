# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.gguf import original_moe_prefill_capability


@pytest.fixture
def operators(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C_gguf", SimpleNamespace(ggml_moe_mmq=object()))


@pytest.mark.parametrize(
    "kind,k,n,m",
    [(kind, 2560, 160, 64) for kind in (18, 21, 22, 23)]
    + [(20, 160, 2560, 640), (42, 192, 2560, 640)],
)
def test_prefill_sized_inputs_use_batched_mmq(operators, kind, k, n, m):
    """Every chunk from 64 tokens up (640 routed down rows), not only 512."""
    cap = original_moe_prefill_capability(kind, k, n, 512, torch.float16, is_sm70=True)
    assert cap.reason is None and cap.graph_safe
    for admitted in (m, m + 1, 304 * m // 64, 512 * m // 64, 8192 * m // 64):
        assert cap.supports_m(admitted)
    for decode in (1, 5, 8, 20, m - 1):
        assert not cap.supports_m(decode)


@pytest.mark.parametrize(
    "supported,reason", [(True, None), (False, "mmq_not_supported_for_source_type")]
)
def test_installed_mmq_support_decides_source_admission(operators, supported, reason):
    cap = original_moe_prefill_capability(
        20, 2560, 160, 512, torch.float16, is_sm70=True, mmq_supported=supported
    )
    assert cap.reason == reason


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"enabled": False}, "disabled_by_kernel_config"),
        ({"is_sm70": False}, "requires_sm70"),
        ({"dtype": torch.float32}, "requires_fp16_activations"),
        ({"experts": 256}, "original_prefill_shape_or_source_unmeasured"),
        ({"k": 320}, "original_prefill_shape_or_source_unmeasured"),
        ({"source_type": 20}, "original_prefill_shape_or_source_unmeasured"),
    ],
)
def test_rejected_prefill_capabilities_explain_fallback(operators, changes, reason):
    args = dict(
        source_type=21,
        k=2560,
        n=160,
        experts=512,
        dtype=torch.float16,
        is_sm70=True,
    )
    args.update(changes)
    assert original_moe_prefill_capability(**args).reason == reason


def test_missing_prefill_operator_retains_fallback(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C_gguf", SimpleNamespace())
    cap = original_moe_prefill_capability(
        21, 2560, 160, 512, torch.float16, is_sm70=True
    )
    assert cap.reason == "operator_missing:ggml_moe_mmq"


@pytest.mark.parametrize(
    "m,selected",
    [(20, "mmvq"), (63, "mmvq"), (64, "mmq"), (304, "mmq"), (511, "mmq"), (512, "mmq")],
)
def test_projection_dispatch_counts_routed_down_rows(monkeypatch, m, selected):
    from vllm.model_executor.layers.fused_moe import MoEActivation
    from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod

    calls = []

    def operation(name):
        def run(x, weight, ids, kind, n, top_k, tokens):
            calls.append((name, tokens, top_k))
            return x.new_ones((tokens * top_k, n))

        return run

    monkeypatch.setattr(
        torch.ops,
        "_C_gguf",
        SimpleNamespace(ggml_moe_mmq=operation("mmq"), ggml_moe_mmvq=operation("mmvq")),
    )
    method = object.__new__(GGUFNativeMoEMethod)
    method.dp4a_admitted = False
    method.hidden_size, method.intermediate_size = 8, 4
    method.input_padding = {}
    method.weight_types = {"w1": 21, "w3": 21, "w2": 20}
    method.prefill_capabilities = {
        shard: original_moe_prefill_capability(
            kind,
            160 if shard == "w2" else 2560,
            2560 if shard == "w2" else 160,
            512,
            torch.float16,
            is_sm70=True,
        )
        for shard, kind in method.weight_types.items()
    }
    method.projection_capabilities = {
        shard: replace(cap, operator="ggml_moe_mmvq", min_m=1, max_m=None)
        for shard, cap in method.prefill_capabilities.items()
    }
    layer = SimpleNamespace(
        apply_router_weight_on_input=False,
        activation=MoEActivation.SILU,
        expert_map=None,
        gguf_w1=None,
        gguf_w3=None,
        gguf_w2=None,
    )
    out = method.apply(
        layer,
        torch.ones((m, 8), dtype=torch.float16),
        torch.ones((m, 10)),
        torch.zeros((m, 10), dtype=torch.int32),
        None,
        None,
    )
    assert out.shape == (m, 8) and torch.isfinite(out).all()
    assert calls == [(selected, m, 10), (selected, m, 10), (selected, m * 10, 1)]
