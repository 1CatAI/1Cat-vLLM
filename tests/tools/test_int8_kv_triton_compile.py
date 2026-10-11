# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile the writer for SM70 without requiring a GPU or CUDA driver."""

import pytest

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("rne", [False, True])
def test_token_head_writer_compiles_for_sm70(tmp_path, monkeypatch, rne):
    triton = pytest.importorskip("triton")
    pytest.importorskip("torch")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        _reshape_cache_per_token_head as writer,
    )

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    constants = dict(
        block_size=16,
        head_size=80,
        head_size_v=80,
        HEAD_SIZE_PADDED=128,
        QUANT_MAX=127.0,
        QUANT_MIN=-128.0,
        ROUND_INT8_EVEN=rne,
    )
    signature = {name: "i64" for name in writer.arg_names if name not in constants}
    signature.update(
        key_ptr="*fp16",
        value_ptr="*fp16",
        key_cache_ptr="*i8",
        value_cache_ptr="*i8",
        k_scale_cache_ptr="*fp32",
        v_scale_cache_ptr="*fp32",
        slot_mapping_ptr="*i64",
    )
    kernel = triton.compile(
        ASTSource(writer, signature=signature, constexprs=constants),
        target=GPUTarget("cuda", 70, 32),
        options={"num_warps": 4},
    )
    assert kernel.asm["cubin"]
    assert "sm_70" in kernel.asm["ptx"]
    if rne:
        assert "div.rn.f32" in kernel.asm["ptx"]
        assert "cvt.rni.f32.f32" in kernel.asm["ptx"]
