# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_layer_dump_obeys_nonaliasing_schema():
    from vllm.model_executor.models import qwen3_next  # noqa: F401

    op = torch.ops.vllm.sm70_qwen_layer_dump.default
    x = torch.randn(8, 32, device="cuda", dtype=torch.float16)
    torch.library.opcheck(
        op,
        (x, "schema_test", 0, "linear_attention"),
        test_utils=("test_schema", "test_faketensor"),
    )
    out = op(x, "schema_test", 0, "linear_attention")
    assert torch.equal(out, x)
    out.zero_()
    assert torch.count_nonzero(x) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_layer_dump_preserves_compiled_residual():
    from vllm.model_executor.models import qwen3_next  # noqa: F401

    def forward(x, residual):
        saved = x.float() + residual
        value = torch.ops.vllm.sm70_qwen_layer_dump(
            x, "compiled_test", 0, "linear_attention"
        )
        updated = value.float() + residual
        normalized = updated * torch.rsqrt(updated.square().mean(-1, True) + 1e-6)
        return saved, updated + normalized

    x = torch.randn(8, 32, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x, dtype=torch.float32)
    expected = forward(x, residual)
    actual = torch.compile(forward, backend="aot_eager", fullgraph=True)(x, residual)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
