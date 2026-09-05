# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.models.deepseek_v4.sm70.sparse_kernels import (
    sm70_sparse_attention_gathered,
)
from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    _hadamard128_torch,
    fwht128,
)
from vllm.models.glm5next.sm70 import sparse as sparse_module
from vllm.models.glm5next.sm70.fp8_kv import (
    sm70_glm5_fp8_kv_insert,
    sm70_glm5_sparse_attention_paged_fp8,
    sm70_glm5_sparse_attention_paged_fp8_batched_gemm,
    sm70_glm5_sparse_attention_paged_fp8_gemm,
)
from vllm.models.glm5next.sm70.sparse import Glm5NextSM70SparseBackend
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def test_glm53_model_arch_uses_mla_cache_width():
    config = Glm5NextConfig(
        architectures=["Glm5NextForConditionalGeneration"],
        text_config={
            "head_dim": 0,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 0,
        },
    )
    converted = ModelArchConfigConvertorBase(config, config.text_config).convert()

    assert converted.is_deepseek_mla
    assert converted.head_size == 512


def test_glm53_sm70_sparse_backend_contract():
    invalid = Glm5NextSM70SparseBackend.validate_configuration(
        head_size=512,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=1152,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(major=7, minor=0),
        attn_type="decoder",
    )
    assert invalid == []


def test_glm53_sm70_sparse_backend_uses_structural_gates(monkeypatch):
    text_config = SimpleNamespace(
        model_type="generic_sparse_mla",
        qk_rope_head_dim=0,
        kv_lora_rank=512,
    )
    monkeypatch.setattr(
        sparse_module,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=text_config)
        ),
    )

    assert (
        Glm5NextSM70SparseBackend.supports_combination(
            head_size=512,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=1152,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            device_capability=DeviceCapability(major=7, minor=0),
        )
        is None
    )

    text_config.qk_rope_head_dim = 64
    assert "requires NoPE" in (
        Glm5NextSM70SparseBackend.supports_combination(
            head_size=512,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=1152,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            device_capability=DeviceCapability(major=7, minor=0),
        )
        or ""
    )


def test_glm53_sm70_fp8_kv_layout_contract():
    invalid = Glm5NextSM70SparseBackend.validate_configuration(
        head_size=512,
        dtype=torch.float16,
        kv_cache_dtype="fp8_e4m3",
        block_size=1152,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(major=7, minor=0),
        attn_type="decoder",
    )
    assert invalid == []
    assert Glm5NextSM70SparseBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        cache_dtype_str="fp8_e4m3",
    ) == (3, 64, 520)

    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_e4m3",
        model_version="glm5_next",
    )
    assert spec.real_page_size_bytes == 64 * 520

    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_cache_dtype = "fp8_e4m3"
    layer.head_size = 512
    layer.attn_backend = Glm5NextSM70SparseBackend
    emitted = layer.get_kv_cache_spec(
        SimpleNamespace(
            cache_config=SimpleNamespace(block_size=64, cache_dtype="fp8_e4m3"),
            model_config=SimpleNamespace(dtype=torch.float16),
        )
    )
    assert isinstance(emitted, MLAAttentionSpec)
    assert emitted.model_version == "glm5_next"
    assert emitted.real_page_size_bytes == 64 * 520


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability((7, 0)),
    reason="requires an exact SM70 CUDA device",
)
def test_glm53_sm70_sparse_attention_matches_reference():
    torch.manual_seed(7)
    device = torch.device("cuda")
    q = torch.randn(2, 16, 512, dtype=torch.float16, device=device) * 0.1
    kv = torch.randn(48, 512, dtype=torch.float16, device=device) * 0.1
    indices = torch.tensor(
        [
            [0, 3, 5, 9, 11, 17, 23, 31] + [-1] * 8,
            [2, 4, 8, 12, 18, 24, 32, 40, 45] + [-1] * 7,
        ],
        dtype=torch.int32,
        device=device,
    )
    lengths = torch.tensor([8, 9], dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    repeat = torch.empty_like(q)
    scale = 512**-0.5

    sm70_sparse_attention_gathered(q, kv, indices, lengths, scale, None, out)
    sm70_sparse_attention_gathered(q, kv, indices, lengths, scale, None, repeat)
    torch.accelerator.synchronize()

    refs = []
    for row, length in enumerate(lengths.cpu().tolist()):
        selected = kv[indices[row, :length].long()].float()
        probs = torch.softmax(q[row].float() @ selected.T * scale, dim=-1)
        refs.append(probs @ selected)
    ref = torch.stack(refs).half()

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-3)
    assert torch.equal(out, repeat)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability((7, 0)),
    reason="requires an exact SM70 CUDA device",
)
def test_glm53_sm70_fp8_kv_sparse_attention_matches_reference():
    torch.manual_seed(20260827)
    device = torch.device("cuda")
    num_kv = 48
    block_size = 64
    kv = torch.randn(num_kv, 512, dtype=torch.float16, device=device) * 0.1
    cache = torch.zeros(2, block_size, 520, dtype=torch.uint8, device=device)
    slots = torch.arange(num_kv, dtype=torch.int64, device=device)
    sm70_glm5_fp8_kv_insert(kv, cache, slots)

    q = torch.randn(2, 16, 512, dtype=torch.float16, device=device) * 0.1
    indices = torch.tensor(
        [
            [0, 3, 5, 9, 11, 17, 23, 31] + [-1] * 8,
            [2, 4, 8, 12, 18, 24, 32, 40, 45] + [-1] * 7,
        ],
        dtype=torch.int32,
        device=device,
    )
    lengths = torch.tensor([8, 9], dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    repeat = torch.empty_like(q)
    scale = 512**-0.5
    sm70_glm5_sparse_attention_paged_fp8(q, cache, indices, lengths, scale, out)
    sm70_glm5_sparse_attention_paged_fp8(q, cache, indices, lengths, scale, repeat)
    torch.accelerator.synchronize()

    refs = []
    for row, length in enumerate(lengths.cpu().tolist()):
        selected = kv[indices[row, :length].long()].float()
        probs = torch.softmax(q[row].float() @ selected.T * scale, dim=-1)
        refs.append(probs @ selected)
    ref = torch.stack(refs).half()

    torch.testing.assert_close(out, ref, rtol=3e-2, atol=4e-3)
    assert torch.equal(out, repeat)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability((7, 0)),
    reason="requires an exact SM70 CUDA device",
)
def test_glm53_sm70_fp8_kv_gemm_decode_matches_reference():
    torch.manual_seed(20260828)
    device = torch.device("cuda")
    num_kv = 48
    index_width = 64
    kv = torch.randn(num_kv, 512, dtype=torch.float16, device=device) * 0.1
    cache = torch.zeros(1, 64, 520, dtype=torch.uint8, device=device)
    sm70_glm5_fp8_kv_insert(
        kv,
        cache,
        torch.arange(num_kv, dtype=torch.int64, device=device),
    )
    q = torch.randn(1, 16, 512, dtype=torch.float16, device=device) * 0.1
    selected_ids = [0, 3, 5, 9, 11, 17, 23, 31, 40, 45]
    indices = torch.tensor(
        [selected_ids + [-1] * (index_width - len(selected_ids))],
        dtype=torch.int32,
        device=device,
    )
    lengths = torch.tensor([len(selected_ids)], dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    repeat = torch.empty_like(q)
    gathered_kv = torch.empty(index_width, 512, dtype=torch.float16, device=device)
    scores = torch.empty(16, index_width, dtype=torch.float16, device=device)
    probs = torch.empty_like(scores)
    scale = 512**-0.5

    sm70_glm5_sparse_attention_paged_fp8_gemm(
        q,
        cache,
        indices,
        lengths,
        scale,
        out,
        gathered_kv,
        scores,
        probs,
    )
    sm70_glm5_sparse_attention_paged_fp8_gemm(
        q,
        cache,
        indices,
        lengths,
        scale,
        repeat,
        gathered_kv,
        scores,
        probs,
    )
    torch.accelerator.synchronize()

    selected = kv[indices[0, : len(selected_ids)].long()].float()
    ref_probs = torch.softmax(q[0].float() @ selected.T * scale, dim=-1)
    ref = (ref_probs @ selected).half().unsqueeze(0)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=4e-3)
    assert torch.equal(out, repeat)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability((7, 0)),
    reason="requires an exact SM70 CUDA device",
)
def test_glm53_sm70_fp8_kv_batched_gemm_matches_b1_decode():
    torch.manual_seed(20260901)
    device = torch.device("cuda")
    num_tokens = 8
    num_heads = 16
    num_kv = 256
    index_width = 2048
    kv = torch.randn(num_kv, 512, dtype=torch.float16, device=device) * 0.1
    cache = torch.zeros(4, 64, 520, dtype=torch.uint8, device=device)
    sm70_glm5_fp8_kv_insert(
        kv,
        cache,
        torch.arange(num_kv, dtype=torch.int64, device=device),
    )
    q = (
        torch.randn(
            num_tokens,
            num_heads,
            512,
            dtype=torch.float16,
            device=device,
        )
        * 0.1
    )
    indices = torch.full(
        (num_tokens, index_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    lengths = torch.arange(131, 139, dtype=torch.int32, device=device)
    for row, length in enumerate(lengths.cpu().tolist()):
        indices[row, :length] = torch.randperm(num_kv, device=device)[:length]

    actual = torch.empty_like(q)
    expected = torch.empty_like(q)
    gathered = torch.empty(
        num_tokens,
        index_width,
        512,
        dtype=torch.float16,
        device=device,
    )
    scores = torch.empty(
        num_tokens,
        num_heads,
        index_width,
        dtype=torch.float16,
        device=device,
    )
    probs = torch.empty_like(scores)
    b1_gathered = torch.empty(index_width, 512, dtype=torch.float16, device=device)
    b1_scores = torch.empty(
        num_heads,
        index_width,
        dtype=torch.float16,
        device=device,
    )
    b1_probs = torch.empty_like(b1_scores)
    scale = 512**-0.5

    sm70_glm5_sparse_attention_paged_fp8_batched_gemm(
        q,
        cache,
        indices,
        lengths,
        scale,
        actual,
        gathered,
        scores,
        probs,
    )
    for row in range(num_tokens):
        sm70_glm5_sparse_attention_paged_fp8_gemm(
            q[row : row + 1],
            cache,
            indices[row : row + 1],
            lengths[row : row + 1],
            scale,
            expected[row : row + 1],
            b1_gathered,
            b1_scores,
            b1_probs,
        )
    torch.accelerator.synchronize()

    assert torch.equal(actual, expected)
    eager = actual.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sm70_glm5_sparse_attention_paged_fp8_batched_gemm(
            q,
            cache,
            indices,
            lengths,
            scale,
            actual,
            gathered,
            scores,
            probs,
        )
    graph.replay()
    torch.accelerator.synchronize()

    assert torch.equal(actual, eager)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability((7, 0)),
    reason="requires an exact SM70 CUDA device",
)
def test_glm53_sm70_fp16_fwht_matches_reference():
    torch.manual_seed(11)
    q = torch.randn(37, 128, dtype=torch.float16, device="cuda") * 0.1

    actual = fwht128(q)
    repeat = fwht128(q)
    reference = _hadamard128_torch(q.float()).half()
    torch.accelerator.synchronize()

    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=2e-3)
    assert torch.equal(actual, repeat)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability((7, 0)),
    reason="requires an exact SM70 CUDA device",
)
def test_glm53_sm70_fused_q_kv_rmsnorm_matches_reference():
    torch.manual_seed(13)
    q = torch.randn(7, 1536, dtype=torch.float16, device="cuda")
    kv = torch.randn(7, 512, dtype=torch.float16, device="cuda")
    q_weight = torch.randn(1536, dtype=torch.float16, device="cuda")
    kv_weight = torch.randn(512, dtype=torch.float16, device="cuda")
    eps = 1e-6

    q_actual, kv_actual = fused_q_kv_rmsnorm(q, kv, q_weight, kv_weight, eps)
    q_ref = q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + eps)
    kv_ref = kv.float() * torch.rsqrt(kv.float().square().mean(-1, keepdim=True) + eps)
    q_ref = (q_ref * q_weight.float()).half()
    kv_ref = (kv_ref * kv_weight.float()).half()

    torch.testing.assert_close(q_actual, q_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(kv_actual, kv_ref, rtol=2e-3, atol=2e-3)
