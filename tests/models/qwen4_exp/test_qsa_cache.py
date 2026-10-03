# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.models.qwen4_exp.common.qsa_cache import (
    QSACompressedKeyCache,
    QSAKeyStateCache,
)
from vllm.models.qwen4_exp.nvidia.qsa import (
    Qwen4ExpQSAAttention,
    Qwen4ExpQSAFlashAttentionBackend,
    Qwen4ExpQSAFlashAttentionImpl,
)
from vllm.v1.worker.utils import bind_kv_cache


def test_qsa_does_not_claim_batch_invariant_reductions() -> None:
    assert not Qwen4ExpQSAFlashAttentionBackend.supports_batch_invariance()


@pytest.mark.parametrize(
    ("prefix", "expected_main_size", "sharded"),
    [
        # A 1600-token block holds 800 slots per rank of a sharded target
        # layer and the whole span of the replicated draft.
        ("model.layers.3.self_attn", 800, True),
        ("mtp.layers.48.self_attn", 1600, False),
    ],
)
def test_qsa_real_cache_specs_keep_only_target_main_sharded(
    prefix: str, expected_main_size: int, sharded: bool
) -> None:
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=1600),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )
    main = _bare_qsa_attention(output_width=4)
    main.layer_name = prefix
    main.num_kv_heads = 1
    main.head_dim = 256
    main.kv_cache_torch_dtype = torch.uint8
    main.kv_cache_dtype = "fp8_e4m3"
    main_spec = main.get_kv_cache_spec(config)

    side = object.__new__(QSACompressedKeyCache)
    torch.nn.Module.__init__(side)
    side.prefix = f"{prefix}.indexer.compressed_key_cache"
    side.head_size = 128
    side.dtype = torch.float16
    side.compress_ratio = 16
    side_spec = side.get_kv_cache_spec(config)

    assert main_spec.block_size == expected_main_size
    assert main_spec.dcp_sharded is sharded
    assert side_spec.block_size == 1600
    assert not side_spec.dcp_sharded
    assert main_spec.global_block_size(2) == side_spec.global_block_size(2)


def _bare_qsa_attention(output_width: int) -> Qwen4ExpQSAAttention:
    attention = object.__new__(Qwen4ExpQSAAttention)
    torch.nn.Module.__init__(attention)
    attention.indexer = SimpleNamespace(output_width=output_width)
    return attention


def test_qsa_attention_reuses_shared_topk_indices_buffer() -> None:
    attention = _bare_qsa_attention(output_width=7)
    shared = torch.empty(16, 7, dtype=torch.int32)

    attention._set_topk_indices_buffer(
        max_tokens=16,
        topk_indices_buffer=shared,
    )

    assert attention.topk_indices_buffer is shared
    assert "topk_indices_buffer" not in attention._buffers


def test_qsa_attention_keeps_private_buffer_fallback() -> None:
    attention = _bare_qsa_attention(output_width=7)

    attention._set_topk_indices_buffer(
        max_tokens=16,
        topk_indices_buffer=None,
    )

    assert attention.topk_indices_buffer.shape == (16, 7)
    assert attention.topk_indices_buffer.dtype == torch.int32
    assert "topk_indices_buffer" in attention._buffers


def test_qsa_attention_rejects_invalid_shared_topk_indices_buffer() -> None:
    attention = _bare_qsa_attention(output_width=7)
    invalid = torch.empty(16, 6, dtype=torch.int32)

    with pytest.raises(ValueError, match="shared top-k buffer"):
        attention._set_topk_indices_buffer(
            max_tokens=16,
            topk_indices_buffer=invalid,
        )


def test_bind_qsa_key_cache_builds_key_and_mrope_views() -> None:
    prefix = "model.layers.0.self_attn.raw_key_cache"
    static_forward_context: dict[str, Any] = {}
    layer = QSAKeyStateCache(
        head_size=128,
        dtype=torch.float16,
        cache_rope_positions=True,
        prefix=prefix,
        cache_config=SimpleNamespace(block_size=16),
        compress_ratio=4,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context=static_forward_context
            )
        ),
    )
    cache = torch.empty(2, 8, 1, layer.head_size, dtype=torch.float16)
    runner_kv_caches: list[torch.Tensor] = []

    bind_kv_cache({prefix: cache}, static_forward_context, runner_kv_caches)

    assert layer.kv_cache is cache
    assert layer.key_cache.shape == (2, 8, 1, 128)
    assert layer.key_cache.untyped_storage().data_ptr() == cache.data_ptr()
    assert layer.rope_position_cache.shape == (2, 8, 1, 3)
    assert layer.rope_position_cache.dtype == torch.int64
    assert runner_kv_caches == [cache]


def test_qsa_forward_splits_local_flash_cache_layout(monkeypatch) -> None:
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    num_blocks, block_size, head_size = 2, 16, 8
    kv_cache = torch.arange(
        num_blocks * 2 * block_size * head_size,
        dtype=torch.float16,
    ).view(num_blocks, 2, block_size, 1, head_size)
    query = torch.zeros(1, 2, head_size, dtype=torch.float16)
    output = torch.empty_like(query)
    logical_indices = torch.zeros(1, 4, dtype=torch.int32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    token_to_req = torch.zeros(1, dtype=torch.int32)
    query_positions = torch.zeros(1, dtype=torch.int64)
    sequence_lengths = torch.ones(1, dtype=torch.int32)
    captured = {}

    def fake_sparse_attention(
        query_arg,
        key_cache_arg,
        value_cache_arg,
        logical_indices_arg,
        block_table_arg,
        token_to_req_arg,
        output_arg,
        *,
        query_positions,
        sequence_lengths,
        kv_cache_dtype,
        k_scale,
        v_scale,
    ):
        captured["key_cache"] = key_cache_arg
        captured["value_cache"] = value_cache_arg
        assert torch.equal(query_arg, query)
        assert torch.equal(logical_indices_arg, logical_indices)
        assert torch.equal(block_table_arg, block_table)
        assert torch.equal(token_to_req_arg, token_to_req)
        captured["query_positions"] = query_positions
        captured["sequence_lengths"] = sequence_lengths
        captured["kv_cache_dtype"] = kv_cache_dtype
        captured["k_scale"] = k_scale
        captured["v_scale"] = v_scale
        output_arg.fill_(1)
        return output_arg

    monkeypatch.setattr(qsa_ops, "qsa_sparse_paged_attention", fake_sparse_attention)
    impl = object.__new__(Qwen4ExpQSAFlashAttentionImpl)
    impl.head_size = head_size
    impl.alibi_slopes = None
    impl.sinks = None
    impl.sliding_window = (-1, -1)
    impl.kv_cache_dtype = "float16"

    result = impl.forward_qsa(
        SimpleNamespace(
            topk_indices_buffer=logical_indices,
            _k_scale_float=1.0,
            _v_scale_float=1.0,
        ),
        query,
        query[:, :1],
        query[:, :1],
        kv_cache,
        SimpleNamespace(num_actual_tokens=1, block_table=block_table),
        output,
        token_to_req,
        query_positions=query_positions,
        sequence_lengths=sequence_lengths,
    )

    expected_key, expected_value = kv_cache.unbind(1)
    assert torch.equal(captured["key_cache"], expected_key)
    assert torch.equal(captured["value_cache"], expected_value)
    assert torch.equal(captured["query_positions"], query_positions)
    assert torch.equal(captured["sequence_lengths"], sequence_lengths)
    assert captured["kv_cache_dtype"] == "float16"
    assert captured["k_scale"] == 1.0
    assert captured["v_scale"] == 1.0
    assert result is output
    assert torch.equal(output, torch.ones_like(output))


def test_qsa_e4m3_forward_cannot_bypass_scale_finalization() -> None:
    layer = SimpleNamespace(
        _qsa_kv_scales_finalized=False,
        layer_name="model.layers.3.self_attn.attn",
    )
    tensor = torch.empty(0)
    with pytest.raises(RuntimeError, match="were not finalized"):
        Qwen4ExpQSAAttention._run_qsa(
            layer,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
        )
