# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable sparse admission and small real-kernel parity tests.

Run this file separately with each target GPU as CUDA_VISIBLE_DEVICES=0.
It needs no model weights and allocates only small attention/indexer caches.
Adding a device to the admission set requires these tests on that device.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.sm70_sparse import sparse_policy
from vllm.models.deepseek_v4.common.portable_sparse import (
    portable_sparse_mla_config_error,
    supports_portable_sparse,
    use_portable_sparse,
)
from vllm.models.deepseek_v4.sm70.indexer import (
    sm70_indexer_decode_logits,
    sm70_indexer_prefill_logits,
)
from vllm.models.glm5next.nvidia import attention as attention_module
from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    _hadamard128_torch,
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
)
from vllm.models.glm5next.sm70 import sparse as sparse_module
from vllm.models.glm5next.sm70.sparse import (
    Glm5NextSM70SparseBackend,
    Glm5NextSM70SparseImpl,
)
from vllm.platforms import current_platform
from vllm.platforms.cuda import _get_backend_priorities
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import indexer as indexer_metadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.workspace import WorkspaceManager


@pytest.mark.parametrize(
    "cap,expected",
    [
        (None, False),
        ((7, 0), True),
        ((7, 2), True),  # Existing SM72 contract is preserved.
        ((7, 5), False),
        ((8, 0), False),
        ((8, 6), True),
        ((8, 7), False),
        ((8, 9), True),
        ((9, 0), False),
        ((10, 0), False),
    ],
)
def test_portable_sparse_admission(cap, expected):
    capability = DeviceCapability(*cap) if cap is not None else None
    assert supports_portable_sparse(capability) is expected
    if capability is not None:
        assert (
            Glm5NextSM70SparseBackend.supports_compute_capability(capability)
            is expected
        )
        assert (
            AttentionBackendEnum.GLM5_SM70_SPARSE
            in _get_backend_priorities(True, capability)
        ) is expected


def test_portable_sparse_uses_the_worker_device(monkeypatch):
    asked = []

    def capability(device_id=None):
        asked.append(device_id)
        return DeviceCapability(8, 6)

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "get_device_capability", capability)
    assert use_portable_sparse()
    # None means the platform resolves the current worker's device. Explicit
    # ordinal zero would silently use the wrong GPU in a heterogeneous process.
    assert asked == [None]
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    assert not use_portable_sparse()


@pytest.mark.parametrize("cap", [(7, 0), (8, 6), (8, 9)])
def test_portable_indexer_keeps_queries_fp16(monkeypatch, cap):
    """Run the real model routing without CUDA projections/compiled leaves."""
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: DeviceCapability(*cap)
    )
    monkeypatch.setattr(attention_module, "_fused_indexer_k_norm", lambda x, *args: x)
    monkeypatch.setattr(
        attention_module, "fwht128", lambda x: _hadamard128_torch(x.float()).half()
    )
    monkeypatch.setattr(
        attention_module, "_fused_indexer_weight_only_scale", lambda w, s: w * s
    )
    monkeypatch.setattr(
        attention_module,
        "_pad_indexer_heads",
        lambda x, p: torch.cat((x, x.new_zeros(x.shape[0], p, *x.shape[2:])), 1),
    )

    def forbidden_quantize(q):
        pytest.fail("portable queries entered the FP8/DeepGEMM route")

    monkeypatch.setattr(attention_module, "fwht128_quant_fp8", forbidden_quantize)
    hidden = torch.ones(3, 4, dtype=torch.float16)
    query = torch.ones(3, 16 * 128, dtype=torch.float16)
    captured = []

    def score(hidden, q, k, weights, **kwargs):
        captured.append(weights)
        return q

    indexer = SimpleNamespace(
        wq_b=lambda x: (query, None),
        wk_weights_proj=lambda x: (torch.ones(3, 128, dtype=torch.float16), None),
        _wp_fp32=torch.ones(4, 16),
        n_head=16,
        head_dim=128,
        rope_dim=0,
        quant_block_size=128,
        scale_fmt="ue8m0",
        softmax_scale=128**-0.5,
        k_norm=SimpleNamespace(weight=None, bias=None, eps=1e-6),
        index_kpool_compress_gate=torch.ones(128, 4, dtype=torch.float16),
        index_kpool_compress_ape=torch.zeros(4, 128),
        index_kpool=4,
        indexer_op=score,
    )
    result = attention_module.Indexer.forward(indexer, hidden, hidden, None, None)
    assert result.dtype == torch.float16 and result.shape == (3, 32, 128)
    assert not result[:, 16:].count_nonzero()
    torch.testing.assert_close(captured[0][:, :16], torch.full((3, 16), 128**-0.5))
    assert not captured[0][:, 16:].count_nonzero()


@pytest.mark.parametrize(
    "overrides,expected_error",
    [
        ({}, None),
        ({"kv_cache_dtype": "fp8_e4m3"}, None),
        ({"dtype": torch.bfloat16}, "FP16 queries"),
        ({"head_size": 576}, "NoPE MLA"),
        ({"kv_lora_rank": 256}, "NoPE MLA"),
        ({"qk_rope_head_dim": 64}, "NoPE MLA"),
        ({"kv_cache_dtype": "fp8_e5m2"}, "packed E4M3"),
        ({"kv_cache_dtype": "fp8_ds_mla"}, "packed E4M3"),
    ],
)
def test_portable_sparse_shape_dtype_contract(overrides, expected_error):
    kwargs = dict(
        head_size=512,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        kv_lora_rank=512,
        qk_rope_head_dim=0,
    )
    kwargs.update(overrides)
    error = portable_sparse_mla_config_error(**kwargs)
    if expected_error is None:
        assert error is None
    else:
        assert error is not None and expected_error in error


@pytest.mark.parametrize("cap", [(7, 0), (8, 6), (8, 9), (9, 0)])
@pytest.mark.parametrize("compress_ratio", [1, 4])
def test_indexer_metadata_skips_unusable_deepgemm(monkeypatch, cap, compress_ratio):
    """Installed DeepGEMM must not execute SM90 kernels on portable devices."""
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: DeviceCapability(*cap)
    )
    monkeypatch.setattr(indexer_metadata, "has_deep_gemm", lambda: True)
    called = []

    def schedule(seq_lens, block_size, num_sms):
        called.append((seq_lens.clone(), block_size, num_sms))
        return torch.ones(num_sms + 1, 2, dtype=torch.int32)

    monkeypatch.setattr(indexer_metadata, "get_paged_mqa_logits_metadata", schedule)
    monkeypatch.setattr(
        indexer_metadata,
        "get_compressed_slot_mapping",
        lambda *args, out: out[:2],
    )
    builder = indexer_metadata.DeepseekV32IndexerMetadataBuilder.__new__(
        indexer_metadata.DeepseekV32IndexerMetadataBuilder
    )
    builder.compress_ratio = compress_ratio
    builder.reorder_batch_threshold = 1
    builder.use_flattening = False
    builder.num_speculative_tokens = 0
    builder.num_sms = 2
    builder.kv_cache_spec = SimpleNamespace(block_size=64, storage_block_size=64)
    builder.decode_lens_buffer = torch.zeros(2, dtype=torch.int32)
    builder.per_req_decode_lens_buffer = torch.zeros(2, dtype=torch.int32)
    builder.expanded_seq_lens_buffer = torch.zeros(2, dtype=torch.int32)
    builder.compressed_slot_mapping_buffer = torch.zeros(2, dtype=torch.int64)
    builder.scheduler_metadata_buffer = torch.zeros(3, 2, dtype=torch.int32)
    query_starts = torch.tensor([0, 1, 2], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_starts,
        query_start_loc_cpu=query_starts.clone(),
        seq_lens=torch.tensor([32, 48], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=48,
        block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.tensor([31, 111]),
        causal=True,
    )
    metadata = builder.build(0, common)
    assert metadata.decode is not None
    assert bool(called) is (compress_ratio == 1 or cap == (9, 0))
    torch.testing.assert_close(
        metadata.decode.seq_lens.flatten(), common.seq_lens // compress_ratio
    )
    assert common.seq_lens.tolist() == [32, 48]


@pytest.fixture
def portable_device():
    if not current_platform.is_cuda() or not torch.accelerator.is_available():
        pytest.skip("requires an admitted CUDA device")
    if not use_portable_sparse():
        pytest.skip("requires SM70/72/86/89")
    with set_current_vllm_config(VllmConfig()):
        yield torch.device("cuda")


def _unpack_main_cache(cache):
    """Independent CPU decoder for block-major E4M3 + eight UE8M0 scales."""
    num_blocks, block_size, _ = cache.shape
    pages = cache.cpu().reshape(num_blocks, -1)
    values = pages[:, : block_size * 512].contiguous().view(torch.float8_e4m3fn)
    values = values.float().reshape(num_blocks, block_size, 8, 64)
    exponents = (
        pages[:, block_size * 512 :].float().reshape(num_blocks, block_size, 8, 1)
    )
    return (values * torch.exp2(exponents - 127)).reshape(-1, 512)


@pytest.mark.parametrize("num_tokens", [1, 4, 9])
@pytest.mark.parametrize("cache_dtype", ["float16", "fp8_e4m3"])
def test_portable_backend_forward_mqa(
    portable_device, monkeypatch, num_tokens, cache_dtype
):
    """Exercise the production B1/B<=8/online dispatch and page translation."""
    torch.manual_seed(20261022)
    device = portable_device
    block_size, num_heads, index_width = 64, 32, 128
    kv = torch.randn(256, 512, dtype=torch.float16, device=device) * 0.2
    q = torch.randn(num_tokens, num_heads, 512, dtype=torch.float16, device=device)
    block_table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device=device)
    req_ids = torch.arange(num_tokens, device=device, dtype=torch.int32) % 2
    selected = [0, 3, 5, 17, 63, 64, 66, 79, 91]
    indices = torch.full(
        (num_tokens, index_width), -1, dtype=torch.int32, device=device
    )
    indices[:, : len(selected)] = torch.tensor(selected, device=device)
    if num_tokens > 1:
        indices[-1].fill_(-1)  # Empty row must produce finite zeros.

    manager = WorkspaceManager(device)
    monkeypatch.setattr(sparse_module, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(
        sparse_module,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16)
        ),
    )
    impl = Glm5NextSM70SparseImpl(
        num_heads=num_heads,
        head_size=512,
        scale=512**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=cache_dtype,
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        indexer=SimpleNamespace(topk_indices_buffer=indices),
        qk_rope_head_dim=0,
        kv_lora_rank=512,
    )
    if cache_dtype == "float16":
        cache = kv.reshape(4, block_size, 512)
        reference_kv = kv.float().cpu()
    else:
        cache = torch.zeros(4, block_size, 520, dtype=torch.uint8, device=device)
        impl.do_kv_cache_update(
            kv,
            kv[:, :0],
            cache,
            torch.arange(256, device=device, dtype=torch.int64),
            cache_dtype,
            torch.ones((), device=device),
        )
        reference_kv = _unpack_main_cache(cache)
        # Check insertion against an independent CPU E4M3 conversion too.
        groups = kv.float().cpu().reshape(256, 8, 64)
        scales = torch.exp2(
            torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448))
        )
        quantized = (groups / scales[..., None]).to(torch.float8_e4m3fn).float()
        expected_kv = (quantized * scales[..., None]).reshape(256, 512)
        torch.testing.assert_close(reference_kv, expected_kv, rtol=0, atol=0)

    metadata = SimpleNamespace(
        req_id_per_token=req_ids,
        block_table=block_table,
        block_size=block_size,
        slot_mapping=torch.zeros(num_tokens, device=device, dtype=torch.int64),
    )
    actual, lse = impl.forward_mqa(q, cache, metadata, None)
    actual = actual.clone()
    repeated, _ = impl.forward_mqa(q, cache, metadata, None)
    assert lse is None
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)

    expected = torch.zeros_like(q, device="cpu")
    for row in range(num_tokens):
        logical = indices[row].cpu().long()
        logical = logical[logical >= 0]
        if not logical.numel():
            continue
        physical = block_table[row % 2].cpu()[logical // block_size].long()
        keys = reference_kv[physical * block_size + logical % block_size]
        probs = torch.softmax(q[row].float().cpu() @ keys.T * 512**-0.5, -1)
        expected[row] = (probs @ keys).half()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=2e-3)


def _indexer_keys(device):
    generator = torch.Generator().manual_seed(1234)
    palette = torch.tensor([0, 0x30, 0x38, 0x3C, 0x40, 0xB0, 0xB8, 0xC0])
    picks = torch.randint(8, (6, 64, 128), generator=generator)
    bits = palette[picks].to(torch.uint8)
    scales = torch.rand(6, 64, generator=generator) + 0.5
    values = bits.view(torch.float8_e4m3fn).float() * scales[..., None]
    packed = torch.cat(
        (bits.reshape(6, -1), scales.contiguous().view(torch.uint8)), dim=1
    ).reshape(6, 64, 132)
    return bits.to(device), scales.to(device), values, packed.to(device)


def _indexer_reference(q, weights, k):
    per_head = torch.einsum("mhd,nd->mhn", q.float().cpu(), k.float().cpu())
    return torch.einsum("mhn,mh->mn", per_head.relu(), weights.float().cpu())


@pytest.mark.parametrize("use_cublas", [False, True])
def test_portable_indexer_prefill(portable_device, monkeypatch, use_cublas):
    torch.manual_seed(10)
    monkeypatch.setattr(sparse_policy(), "indexer_prefill_cublas", use_cublas)
    bits, scales, values, _ = _indexer_keys(portable_device)
    q = torch.randn(5, 32, 128, dtype=torch.float16, device=portable_device) * 0.1
    weights = torch.randn(5, 32, device=portable_device)
    actual = sm70_indexer_prefill_logits(
        q, bits.reshape(-1, 128).view(torch.float8_e4m3fn), scales.flatten(), weights
    )
    expected = _indexer_reference(q, weights, values.reshape(-1, 128))
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("next_n", [1, 4])
def test_portable_indexer_decode(portable_device, next_n):
    torch.manual_seed(20)
    _, _, values, cache = _indexer_keys(portable_device)
    table = torch.tensor([[5, 1], [4, 2]], dtype=torch.int32, device=portable_device)
    lengths = torch.tensor([[67], [97]], device=portable_device, dtype=torch.int32)
    lengths = lengths + torch.arange(next_n, device=portable_device)
    q = (
        torch.randn(2 * next_n, 32, 128, device=portable_device, dtype=torch.float16)
        * 0.1
    )
    weights = torch.randn(2 * next_n, 32, device=portable_device)
    actual = sm70_indexer_decode_logits(
        q, cache, weights, lengths, table, 100, indexer_decode_cublas=True
    )
    for row, length in enumerate(lengths.cpu().flatten().tolist()):
        ids = torch.arange(length)
        physical = table[row // next_n].cpu().long()[ids // 64]
        keys = values[physical, ids % 64]
        expected = _indexer_reference(q[row : row + 1], weights[row : row + 1], keys)
        torch.testing.assert_close(
            actual[row, :length].cpu(), expected[0], rtol=1e-2, atol=1e-2
        )
        # The scorer leaves columns beyond the live sequence uninitialized;
        # top-k consumers mask them with these same per-row sequence lengths.


@pytest.mark.parametrize("pool_size", [4, 16])
def test_portable_kpool_prefill_decode(portable_device, pool_size):
    """Both writers must match an independent FP32 pooling/E4M3 reference."""
    torch.manual_seed(30)
    device = portable_device
    keys = torch.randn(2, pool_size, 128, device=device, dtype=torch.float16)
    scores = torch.randn_like(keys)
    ape = torch.randn(pool_size, 128, device=device)
    slots = torch.tensor([65, 3], device=device, dtype=torch.int64)
    prefill = torch.zeros(2, 64, 132, device=device, dtype=torch.uint8)
    decode = torch.zeros_like(prefill)
    tail = torch.zeros(3, 2, pool_size, 128, device=device, dtype=torch.float16)
    kpool_compress_and_write_cache(prefill, keys, scores, ape, slots, pool_size)
    positions = torch.arange(pool_size, dtype=torch.int32, device=device).expand(2, -1)
    tail_slots = positions + torch.tensor([[1], [2]], device=device) * pool_size
    writes = torch.full((2, pool_size), -1, device=device, dtype=torch.int32)
    writes[:, -1] = slots.int()
    kpool_decode_update_and_maybe_write_cache_batched(
        decode, tail, tail_slots, keys, scores, ape, writes, positions, pool_size
    )
    torch.testing.assert_close(decode, prefill, rtol=0, atol=0)

    pooled = (
        torch.softmax(scores.float().cpu() + ape.cpu(), dim=1) * keys.float().cpu()
    ).sum(1)
    # Match the specified FP16 activation boundary on each side of rotation.
    rotated = _hadamard128_torch(pooled.half().float()).half().float()
    scale = torch.exp2(
        torch.ceil(torch.log2(rotated.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    quantized = (rotated / scale[:, None]).to(torch.float8_e4m3fn).float()
    reference = quantized * scale[:, None]
    pages = decode.cpu().reshape(2, -1)
    vals = (
        pages[:, : 64 * 128]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(128, 128)
    )
    scales = pages[:, 64 * 128 :].contiguous().view(torch.float32).reshape(128)
    actual = vals[slots.cpu()] * scales[slots.cpu(), None]
    # A tiny FP32 reduction difference may change a value at an FP8 midpoint.
    torch.testing.assert_close(actual, reference, rtol=0.125, atol=2e-3)
    torch.testing.assert_close(tail[1:, 0], keys, rtol=0, atol=0)
    torch.testing.assert_close(tail[1:, 1], scores, rtol=0, atol=0)
