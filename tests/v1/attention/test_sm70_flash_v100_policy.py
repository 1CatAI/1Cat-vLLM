# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for SM70 FlashAttention-V100 routing policy."""

import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VLLM_C_EXTENSIONS = ("vllm._C", "vllm._C_stable_libtorch")


@pytest.fixture
def local_flash_v100_model(tmp_path: Path) -> Callable[[], str]:
    def make_model() -> str:
        model_dir = tmp_path / "flash-v100-test-model"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "config.json").write_text(
            """
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 1024,
  "intermediate_size": 4096,
  "num_hidden_layers": 1,
  "num_attention_heads": 4,
  "num_key_value_heads": 1,
  "head_dim": 256,
  "vocab_size": 32000,
  "max_position_embeddings": 2048,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "rope_theta": 10000.0
}
""",
            encoding="utf-8",
        )
        return str(model_dir)

    return make_model


def _load_selector():
    for module_name in VLLM_C_EXTENSIONS:
        sys.modules.setdefault(module_name, types.ModuleType(module_name))

    import vllm.envs as envs
    from vllm.platforms import cuda as cuda_platform
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    envs.disable_envs_cache()
    cuda_platform._get_backend_priorities.cache_clear()
    return (
        cuda_platform._get_backend_priorities,
        DeviceCapability,
        AttentionBackendEnum,
    )


@pytest.fixture(autouse=True)
def clear_backend_priority_cache():
    _get_backend_priorities, _, _ = _load_selector()
    _get_backend_priorities.cache_clear()
    yield
    _get_backend_priorities, _, _ = _load_selector()
    _get_backend_priorities.cache_clear()


def test_sm70_flash_v100_priority_default_on(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FLASH_ATTN_V100", raising=False)
    _get_backend_priorities, DeviceCapability, AttentionBackendEnum = _load_selector()

    backends = _get_backend_priorities(
        use_mla=False,
        device_capability=DeviceCapability(major=7, minor=0),
    )

    assert backends[:2] == [
        AttentionBackendEnum.FLASH_ATTN_V100,
        AttentionBackendEnum.TRITON_ATTN,
    ]
    assert AttentionBackendEnum.FLASH_ATTN not in backends


def test_split_paged_kv_cache_prefers_standard_axis_for_two_blocks():
    from vllm.v1.attention.backends.flash_attn_v100 import _split_paged_kv_cache

    kv_cache = torch.empty((2, 2, 4, 1, 8), dtype=torch.float16)
    kv_cache[:, 0].fill_(3)
    kv_cache[:, 1].fill_(7)

    key_cache, value_cache = _split_paged_kv_cache(kv_cache)

    assert torch.equal(key_cache, kv_cache[:, 0])
    assert torch.equal(value_cache, kv_cache[:, 1])


def test_sm70_mla_prefill_rejects_unsupported_configs(monkeypatch):
    from vllm.v1.attention.backends.mla.prefill import flash_attn as mod

    monkeypatch.setattr(mod, "_is_sm70_flash_v100_platform", lambda: True)
    monkeypatch.setattr(mod, "_flash_v100_dense_prefill_lse_usable", lambda: True)
    # The generic CUDA predicate can report a stub as available. SM70 selection
    # must remain independent of it.
    monkeypatch.setattr(mod, "is_flash_attn_varlen_func_available", lambda: True)
    capability = SimpleNamespace(major=7, minor=0)

    def reasons(dtype=torch.float16, qk_head_dim=256, v_head_dim=256):
        config = SimpleNamespace(
            dtype=dtype,
            is_r1_compatible=False,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
        )
        return mod.FlashAttnPrefillBackend.validate_configuration(capability, config)

    assert reasons() == []
    assert any(
        "uniform" in reason for reason in reasons(qk_head_dim=192, v_head_dim=128)
    )
    assert any(
        "no compiled kernel" in reason
        for reason in reasons(qk_head_dim=80, v_head_dim=80)
    )
    assert any("requires float16" in reason for reason in reasons(dtype=torch.bfloat16))


def test_sm70_mla_selector_extracts_uniform_head_dims():
    from vllm.v1.attention.backends.mla.prefill.selector import _mla_head_dims

    hf_text_config = SimpleNamespace(
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_text_config)
    )

    assert _mla_head_dims(vllm_config) == (256, 256)


def test_sm70_mla_prefill_accepts_equal_independent_causal_metadata(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as flash_mod
    from vllm.v1.attention.backends.mla.prefill import flash_attn as mod

    monkeypatch.setattr(mod, "_is_sm70_flash_v100_platform", lambda: True)
    monkeypatch.setattr(mod, "_flash_v100_dense_prefill_lse_usable", lambda: True)

    calls = []

    def dense_lse(query, key, value, output, softmax_lse, *args, **kwargs):
        calls.append((args, kwargs))
        output.copy_(value.expand_as(output))
        softmax_lse.zero_()
        return output, softmax_lse

    monkeypatch.setattr(flash_mod, "flash_v100_dense_prefill_lse", dense_lse)
    impl = object.__new__(mod.FlashAttnPrefillBackend)
    impl.flash_attn_varlen_func = lambda *args, **kwargs: pytest.fail(
        "SM70 MLA prefill fell through to the unsupported FA2 stub"
    )

    query = torch.zeros((3, 1, 256), dtype=torch.float16)
    key = torch.zeros_like(query)
    value = torch.ones_like(query)
    cu_q = torch.tensor([0, 3], dtype=torch.int32)
    cu_k = cu_q.clone()

    output, lse = impl._flash_attn_varlen_diff_headdims(
        query,
        key,
        value,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        causal=True,
        return_softmax_lse=True,
    )

    assert len(calls) == 1
    assert torch.equal(output, value)
    assert torch.count_nonzero(lse) == 0


def test_sm70_flash_v100_priority_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLASH_ATTN_V100", "0")
    _get_backend_priorities, DeviceCapability, AttentionBackendEnum = _load_selector()

    backends = _get_backend_priorities(
        use_mla=False,
        device_capability=DeviceCapability(major=7, minor=0),
    )

    assert backends[:3] == [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.FLASHINFER,
        AttentionBackendEnum.TRITON_ATTN,
    ]
    assert AttentionBackendEnum.FLASH_ATTN_V100 not in backends


def test_flash_v100_priority_is_sm70_only(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLASH_ATTN_V100", "1")
    _get_backend_priorities, DeviceCapability, AttentionBackendEnum = _load_selector()

    backends = _get_backend_priorities(
        use_mla=False,
        device_capability=DeviceCapability(major=7, minor=5),
    )

    assert backends[:3] == [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.FLASHINFER,
        AttentionBackendEnum.TRITON_ATTN,
    ]
    assert AttentionBackendEnum.FLASH_ATTN_V100 not in backends


def test_sm70_fa2_d256_prefill_env_is_default_on(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_FLASH_V100_FA2_D256_PREFILL", raising=False)
    envs.disable_envs_cache()
    assert envs.VLLM_FLASH_V100_FA2_D256_PREFILL is True

    monkeypatch.setenv("VLLM_FLASH_V100_FA2_D256_PREFILL", "0")
    envs.disable_envs_cache()
    assert envs.VLLM_FLASH_V100_FA2_D256_PREFILL is False


def test_sm70_e5m2_decode_fast_route_envs_are_default_on(monkeypatch):
    import vllm.envs as envs

    names = (
        "VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA",
        "VLLM_FLASH_V100_XQA_E5M2_G6_SPLIT_REDUCE",
        "VLLM_FLASH_V100_XQA_E5M2_PARTITION_PAGE_IDS",
        "VLLM_FLASH_V100_XQA_E5M2_PAIR_LOAD",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_XQA_E5M2_P1024_BEGIN", raising=False)
    envs.disable_envs_cache()
    assert envs.VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA is True
    assert envs.VLLM_FLASH_V100_XQA_E5M2_G6_SPLIT_REDUCE is True
    assert envs.VLLM_FLASH_V100_XQA_E5M2_PARTITION_PAGE_IDS is True
    assert envs.VLLM_FLASH_V100_XQA_E5M2_PAIR_LOAD is True
    assert envs.VLLM_FLASH_V100_XQA_E5M2_P1024_BEGIN == 61633

    for name in names:
        monkeypatch.setenv(name, "0")
    envs.disable_envs_cache()
    assert envs.VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA is False
    assert envs.VLLM_FLASH_V100_XQA_E5M2_G6_SPLIT_REDUCE is False
    assert envs.VLLM_FLASH_V100_XQA_E5M2_PARTITION_PAGE_IDS is False
    assert envs.VLLM_FLASH_V100_XQA_E5M2_PAIR_LOAD is False

    monkeypatch.setenv("VLLM_FLASH_V100_XQA_E5M2_P1024_BEGIN", "49152")
    envs.disable_envs_cache()
    assert envs.VLLM_FLASH_V100_XQA_E5M2_P1024_BEGIN == 49152


def test_sm70_flash_v100_fp8_alias_resolves_to_e5m2(monkeypatch):
    import vllm.engine.arg_utils as arg_utils
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_SM70_FLASH_ATTN_V100", raising=False)
    envs.disable_envs_cache()
    monkeypatch.setattr(
        arg_utils,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            get_device_capability=lambda: SimpleNamespace(major=7, minor=0),
        ),
    )

    assert (
        arg_utils._resolve_sm70_flash_v100_kv_cache_dtype_alias("fp8", "fp8")
        == "fp8_e5m2"
    )
    assert (
        arg_utils._resolve_sm70_flash_v100_kv_cache_dtype_alias("fp8_e4m3", "fp8_e4m3")
        == "fp8_e4m3"
    )
    assert (
        arg_utils._resolve_sm70_flash_v100_kv_cache_dtype_alias("auto", "fp8_e4m3")
        == "fp8_e4m3"
    )


def test_fp8_alias_keeps_upstream_e4m3_semantics_without_sm70_flash(
    monkeypatch,
):
    import vllm.engine.arg_utils as arg_utils
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_SM70_FLASH_ATTN_V100", "0")
    envs.disable_envs_cache()
    monkeypatch.setattr(
        arg_utils,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            get_device_capability=lambda: SimpleNamespace(major=7, minor=0),
        ),
    )

    assert (
        arg_utils._resolve_sm70_flash_v100_kv_cache_dtype_alias("fp8", "fp8") == "fp8"
    )


def test_fp8_alias_is_not_rewritten_outside_nvidia_cuda(monkeypatch):
    import vllm.engine.arg_utils as arg_utils
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_SM70_FLASH_ATTN_V100", raising=False)
    envs.disable_envs_cache()
    monkeypatch.setattr(
        arg_utils,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: False,
            get_device_capability=lambda: pytest.fail(
                "non-CUDA platforms must not query an NVIDIA capability"
            ),
        ),
    )

    assert (
        arg_utils._resolve_sm70_flash_v100_kv_cache_dtype_alias("fp8", "fp8") == "fp8"
    )


def test_sm70_prefill_gather_dense_env_is_default_on(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_FLASH_V100_PREFILL_GATHER_DENSE", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_Q", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_KV", raising=False)
    envs.disable_envs_cache()

    assert envs.VLLM_FLASH_V100_PREFILL_GATHER_DENSE is True
    assert envs.VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_Q == 4096
    assert envs.VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_KV == 8192

    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_GATHER_DENSE", "0")
    envs.disable_envs_cache()
    assert envs.VLLM_FLASH_V100_PREFILL_GATHER_DENSE is False


def test_sm70_gemma_long_prefill_fused_env_is_default_on(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_SM70_GEMMA_LONG_PREFILL_FUSED", raising=False)
    envs.disable_envs_cache()
    assert envs.VLLM_SM70_GEMMA_LONG_PREFILL_FUSED is True

    monkeypatch.setenv("VLLM_SM70_GEMMA_LONG_PREFILL_FUSED", "0")
    envs.disable_envs_cache()
    assert envs.VLLM_SM70_GEMMA_LONG_PREFILL_FUSED is False


def test_sm70_prefill_dense_splitkv3_env_is_default_on(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3_MIN_KV", raising=False)
    envs.disable_envs_cache()

    assert envs.VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3 is True
    assert envs.VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3_MIN_KV == 32768

    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3", "0")
    envs.disable_envs_cache()
    assert envs.VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3 is False


def test_prefill_gather_dense_does_not_require_generic_dense_op(monkeypatch):
    import vllm.envs as envs
    import vllm.v1.attention.backends.flash_attn_v100 as flash_v100

    def paged_prefill(*args, **kwargs):
        raise AssertionError("constructor test must not execute the paged op")

    monkeypatch.setattr(
        flash_v100,
        "_get_flash_ops",
        lambda: (
            None,
            None,
            None,
            None,
            None,
            paged_prefill,
            None,
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        flash_v100,
        "_get_fp8_e5m2_paged_kv_bridge_op",
        lambda: None,
    )
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_GATHER_DENSE", "1")
    envs.disable_envs_cache()
    try:
        impl = flash_v100.FlashAttnV100Impl(
            num_heads=4,
            head_size=256,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )
    finally:
        envs.disable_envs_cache()

    assert impl.use_flash_v100_prefill_paged is True
    assert impl.use_flash_v100_prefill_contig_dense is False
    assert impl.use_flash_v100_prefill_gather_dense is True


def test_flash_v100_normalizes_resolved_float16_cache_dtype(monkeypatch):
    import vllm.v1.attention.backends.flash_attn_v100 as flash_v100

    monkeypatch.setattr(flash_v100, "_get_flash_ops", lambda: (None,) * 9)
    monkeypatch.setattr(
        flash_v100,
        "_get_fp8_e5m2_paged_kv_bridge_op",
        lambda: None,
    )

    impl = flash_v100.FlashAttnV100Impl(
        num_heads=4,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="float16",
    )

    assert impl.kv_cache_dtype == "auto"


def test_prefill_gather_dense_reorders_random_pages_and_reuses_workspace(
    monkeypatch,
):
    import vllm.v1.attention.backends.flash_attn_v100 as flash_v100

    monkeypatch.setattr(flash_v100, "_prefill_gather_dense_workspaces", {})
    key_cache = torch.arange(5 * 4 * 2, dtype=torch.float16).reshape(5, 4, 1, 2)
    value_cache = key_cache + 100
    block_table = torch.tensor([3, 1, 4], dtype=torch.int32)

    first = flash_v100._gather_paged_kv_to_exact_dense(
        key_cache,
        value_cache,
        block_table,
        seq_len=10,
    )
    assert first is not None
    key_dense, value_dense = first
    expected_key = torch.cat((key_cache[3], key_cache[1], key_cache[4]))[:10]
    expected_value = torch.cat((value_cache[3], value_cache[1], value_cache[4]))[:10]
    assert key_dense.shape == (1, 10, 1, 2)
    assert torch.equal(key_dense.squeeze(0), expected_key)
    assert torch.equal(value_dense.squeeze(0), expected_value)
    first_ptr = key_dense.data_ptr()

    second = flash_v100._gather_paged_kv_to_exact_dense(
        key_cache,
        value_cache,
        block_table,
        seq_len=8,
    )
    assert second is not None
    assert second[0].data_ptr() == first_ptr


def test_prefill_gather_dense_accepts_interleaved_kv_views(monkeypatch):
    import vllm.v1.attention.backends.flash_attn_v100 as flash_v100

    monkeypatch.setattr(flash_v100, "_prefill_gather_dense_workspaces", {})
    kv_cache = torch.arange(5 * 2 * 4 * 2, dtype=torch.float16).reshape(5, 2, 4, 1, 2)
    key_cache, value_cache = kv_cache.unbind(1)
    block_table = torch.tensor([3, 1, 4], dtype=torch.int32)

    assert key_cache.is_contiguous() is False
    gathered = flash_v100._gather_paged_kv_to_exact_dense(
        key_cache,
        value_cache,
        block_table,
        seq_len=10,
    )

    assert gathered is not None
    key_dense, value_dense = gathered
    expected_key = torch.cat((key_cache[3], key_cache[1], key_cache[4]))[:10]
    expected_value = torch.cat((value_cache[3], value_cache[1], value_cache[4]))[:10]
    assert torch.equal(key_dense.squeeze(0), expected_key)
    assert torch.equal(value_dense.squeeze(0), expected_value)


def test_prefill_gather_dense_falls_back_when_workspace_is_out_of_memory(
    monkeypatch,
):
    import vllm.v1.attention.backends.flash_attn_v100 as flash_v100

    key_cache = torch.empty((5, 4, 1, 2), dtype=torch.float16)
    monkeypatch.setattr(flash_v100, "_prefill_gather_dense_workspaces", {})
    monkeypatch.setattr(flash_v100, "_warned_prefill_gather_oom", False)

    def raise_oom(*args, **kwargs):
        raise torch.OutOfMemoryError("expected test OOM")

    monkeypatch.setattr(flash_v100.torch, "empty", raise_oom)

    assert flash_v100._get_prefill_gather_dense_workspace(key_cache, 3) is None
    assert flash_v100._warned_prefill_gather_oom is True


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"q_len": 2048}, False),
        ({"seq_len": 4096}, False),
        ({"seq_len": 8224}, False),
        ({"num_seqs": 2}, False),
        ({"causal": False}, False),
    ],
)
def test_prefill_gather_dense_policy_is_evidence_bounded(overrides, expected):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    impl = object.__new__(FlashAttnV100Impl)
    impl.use_flash_v100_prefill_gather_dense = True
    impl.prefill_gather_dense_min_q = 4096
    impl.prefill_gather_dense_min_kv = 8192
    key_cache = torch.empty((12, 784, 1, 256), dtype=torch.float16)
    value_cache = torch.empty_like(key_cache)
    kwargs = {
        "q_len": 4096,
        "seq_len": 8192,
        "head_dim": 256,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "causal": True,
        "window_size": (-1, -1),
        "num_seqs": 1,
        **overrides,
    }

    assert impl._should_use_prefill_gather_dense(**kwargs) is expected


def test_prefix_prefill_prioritizes_gathered_exact_dense_over_paged(
    monkeypatch,
):
    import vllm.v1.attention.backends.flash_attn_v100 as flash_v100
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    impl = object.__new__(FlashAttnV100Impl)
    impl.num_heads = 6
    impl.num_kv_heads = 1
    impl.head_size = 256
    impl.scale = 0.0625
    impl.sliding_window = None
    impl.kv_cache_dtype = "auto"
    impl.use_flash_v100_decode = False
    impl.use_decode_paged_prefill = False
    impl.smallq_decode_max_query_len = 0
    impl.smallq_decode_max_model_len = 0
    impl.use_flash_v100_prefill_paged = True
    impl.use_flash_v100_prefill_bfla = False
    impl.use_flash_v100_prefill_splitkv = False
    impl.use_flash_v100_prefill_contig_dense = True
    impl.prefill_contig_dense_allow_copy = False
    impl.prefill_contig_dense_min_q = 1536
    impl.prefill_contig_dense_min_kv = 8192
    impl.use_flash_v100_prefill_gather_dense = True
    impl.prefill_gather_dense_min_q = 4096
    impl.prefill_gather_dense_min_kv = 8192
    impl.use_fp8_prefill_bridge = False
    impl.flash_attn_prefill_paged_bfla = None
    impl.flash_attn_prefill_paged_splitkv = None
    impl.flash_attn_bhmd_func = None
    impl.flash_attn_func = None

    query_len = 4096
    seq_len = 8192
    page_size = 784
    num_pages = (seq_len + page_size - 1) // page_size
    query = torch.empty((query_len, 6, 256), dtype=torch.float16)
    output = torch.empty_like(query)
    key_cache = torch.empty((num_pages, page_size, 1, 256), dtype=torch.float16)
    value_cache = torch.empty_like(key_cache)
    block_table = torch.arange(num_pages - 1, -1, -1, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    metadata = SimpleNamespace(
        causal=True,
        num_actual_tokens=query_len,
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, query_len], dtype=torch.int32),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        block_table=block_table,
        ddtree_parent_ids=None,
        ddtree_num_tree_tokens_cpu=None,
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    routes: list[str] = []

    def exact_dense(query_arg, key_arg, value_arg, **kwargs):
        assert query_arg.shape == (1, query_len, 6, 256)
        assert key_arg.shape == (1, seq_len, 1, 256)
        assert value_arg.shape == key_arg.shape
        kwargs["out"].fill_(5)
        return kwargs["out"]

    def unexpected_paged(*args, **kwargs):
        raise AssertionError("eligible long prefill must not call the paged kernel")

    monkeypatch.setattr(flash_v100, "_prefill_gather_dense_workspaces", {})
    monkeypatch.setattr(
        flash_v100,
        "_get_sm70_splitd_d256_ops",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(flash_v100, "_try_sm70_fa2_d256_prefill", exact_dense)
    monkeypatch.setattr(flash_v100, "_record_route", routes.append)
    impl.flash_attn_prefill_paged = unexpected_paged

    result = impl._flash_v100_prefill_with_prefix(
        layer,
        query,
        None,
        None,
        (key_cache, value_cache),
        metadata,
        output,
    )

    assert result is output
    assert torch.equal(output, torch.full_like(output, 5))
    assert routes == ["prefill_prefix_gather_splitd_d256"]


def test_sm70_splitd_d256_loader_requires_exact_ops(monkeypatch):
    import vllm.v1.attention.backends.flash_attn_v100 as flash_v100

    fake_interface = types.ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    fake_package = types.ModuleType("vllm.vllm_flash_attn")
    fake_package.__dict__["flash_attn_interface"] = fake_interface
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", fake_package)
    monkeypatch.setitem(
        sys.modules,
        "vllm.vllm_flash_attn.flash_attn_interface",
        fake_interface,
    )

    fake_ops = SimpleNamespace(_vllm_fa2_C=SimpleNamespace())
    warnings: list[str] = []
    monkeypatch.setattr(flash_v100, "torch", SimpleNamespace(ops=fake_ops))
    monkeypatch.setattr(
        flash_v100.logger,
        "warning_once",
        lambda message, *args: warnings.append(message % args),
    )
    monkeypatch.setattr(flash_v100, "_sm70_splitd_d256_ops_checked", False)
    monkeypatch.setattr(flash_v100, "_sm70_splitd_d256_ops", None)
    assert flash_v100._get_sm70_splitd_d256_ops() is None
    assert len(warnings) == 1
    assert "sm70_d256_splitd_n32_dense_fwd" in warnings[0]

    dense = object()
    paged = object()
    fake_ops._vllm_fa2_C.sm70_d256_splitd_n32_dense_fwd = dense
    fake_ops._vllm_fa2_C.sm70_d256_splitd_n32_paged_fwd = paged
    monkeypatch.setattr(flash_v100, "_sm70_splitd_d256_ops_checked", False)
    assert flash_v100._get_sm70_splitd_d256_ops() == (dense, paged, None)

    splitkv3 = object()
    fake_ops._vllm_fa2_C.sm70_d256_splitd_n32_dense_splitkv3_fwd = splitkv3
    monkeypatch.setattr(flash_v100, "_sm70_splitd_d256_ops_checked", False)
    assert flash_v100._get_sm70_splitd_d256_ops() == (dense, paged, splitkv3)


def test_prefill_dense_splitkv3_workspace_reuses_exact_shape(monkeypatch):
    import vllm.v1.attention.backends.flash_attn_v100 as flash_v100

    monkeypatch.setattr(flash_v100, "_prefill_dense_splitkv3_workspaces", {})
    query = torch.empty((1, 2, 3, 4), dtype=torch.float16)

    first = flash_v100._get_prefill_dense_splitkv3_workspace(query)
    second = flash_v100._get_prefill_dense_splitkv3_workspace(query)

    assert first is not None
    assert second is not None
    assert first[0].shape == (3, 1, 2, 3, 4)
    assert first[1].shape == (3, 1, 2, 3)
    assert first[0].data_ptr() == second[0].data_ptr()
    assert first[1].data_ptr() == second[1].data_ptr()


def test_prefill_dense_splitkv3_workspace_oom_falls_back(monkeypatch):
    import vllm.v1.attention.backends.flash_attn_v100 as flash_v100

    query = torch.empty((1, 2, 3, 4), dtype=torch.float16)
    monkeypatch.setattr(flash_v100, "_prefill_dense_splitkv3_workspaces", {})
    monkeypatch.setattr(flash_v100, "_warned_prefill_dense_splitkv3_oom", False)

    def raise_oom(*args, **kwargs):
        raise torch.OutOfMemoryError

    monkeypatch.setattr(flash_v100.torch, "empty", raise_oom)

    assert flash_v100._get_prefill_dense_splitkv3_workspace(query) is None
    assert flash_v100._warned_prefill_dense_splitkv3_oom is True


def test_prefill_dense_splitkv3_policy_is_exact_shape_bounded(monkeypatch):
    import vllm.envs as envs
    import vllm.v1.attention.backends.flash_attn_v100 as flash_v100

    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3", "1")
    envs.disable_envs_cache()
    query = torch.empty((1, 4096, 6, 256), dtype=torch.float16, device="meta")
    key = torch.empty((1, 65536, 1, 256), dtype=torch.float16, device="meta")
    key_32k = torch.empty((1, 32768, 1, 256), dtype=torch.float16, device="meta")
    key_16k = torch.empty((1, 16384, 1, 256), dtype=torch.float16, device="meta")

    assert flash_v100._should_use_prefill_dense_splitkv3(
        query,
        key,
        max_seqlen_q=4096,
        max_seqlen_k=65536,
        splitkv3_op=object(),
    )
    assert flash_v100._should_use_prefill_dense_splitkv3(
        query,
        key_32k,
        max_seqlen_q=4096,
        max_seqlen_k=32768,
        splitkv3_op=object(),
    )
    assert not flash_v100._should_use_prefill_dense_splitkv3(
        query,
        key_16k,
        max_seqlen_q=4096,
        max_seqlen_k=16384,
        splitkv3_op=object(),
    )
    assert not flash_v100._should_use_prefill_dense_splitkv3(
        query[:, :, :4],
        key,
        max_seqlen_q=4096,
        max_seqlen_k=65536,
        splitkv3_op=object(),
    )


def test_dense_prefill_restores_uniform_batch_view_for_exact_splitd(monkeypatch):
    import vllm.v1.attention.backends.flash_attn_v100 as flash_v100

    monkeypatch.setattr(
        flash_v100,
        "_get_flash_ops",
        lambda: (object(), None, None, None, None, None, None, None, None),
    )
    calls: list[tuple[int, ...]] = []

    def fake_splitd(query, key, value, **kwargs):
        calls.append(tuple(query.shape))
        assert query.data_ptr() == flat_query.data_ptr()
        assert key.data_ptr() == flat_key.data_ptr()
        assert value.data_ptr() == flat_value.data_ptr()
        return kwargs["out"]

    routes: list[str] = []
    monkeypatch.setattr(flash_v100, "_try_sm70_fa2_d256_prefill", fake_splitd)
    monkeypatch.setattr(flash_v100, "_record_route", routes.append)

    seq_len = 1024
    flat_query = torch.empty((seq_len, 2, 256), dtype=torch.float16)
    flat_key = torch.empty((seq_len, 2, 256), dtype=torch.float16)
    flat_value = torch.empty((seq_len, 2, 256), dtype=torch.float16)
    output = torch.empty_like(flat_query)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

    result = flash_v100.flash_v100_dense_prefill(
        query=flat_query,
        key=flat_key,
        value=flat_value,
        output=output,
        query_start_loc=cu_seqlens,
        query_start_loc_device=cu_seqlens,
        num_actual_tokens=seq_len,
        softmax_scale=0.125,
    )

    assert result is output
    assert calls == [(1, seq_len, 2, 256)]
    assert routes == ["prefill_dense_splitd_d256"]


def test_flash_v100_prefill_live_token_mismatch_uses_prefix_path(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    monkeypatch.delenv("VLLM_FLASH_V100_PREFILL_USE_TRITON", raising=False)
    monkeypatch.setenv("VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK", "0")
    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16")

    impl = FlashAttnV100Impl(
        num_heads=4,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    impl.use_flash_v100 = True
    impl.use_flash_v100_decode = True

    calls: list[str] = []

    def fail_dense(*args, **kwargs):
        calls.append("dense")
        raise AssertionError("dense prefill path should not be selected")

    def hit_prefix(*args, **kwargs):
        calls.append("prefix")
        return args[-1]

    impl._flash_v100_prefill = fail_dense  # type: ignore[method-assign]
    impl._flash_v100_prefill_with_prefix = hit_prefix  # type: ignore[method-assign]
    impl._maybe_compare_triton_output = lambda *args, **kwargs: None  # type: ignore[method-assign]
    impl._reset_decode_cache = lambda: None  # type: ignore[method-assign]

    query = torch.zeros((1, 4, 256), dtype=torch.float16)
    key = torch.zeros((1, 1, 256), dtype=torch.float16)
    value = torch.zeros((1, 1, 256), dtype=torch.float16)
    output = torch.zeros((1, 4, 256), dtype=torch.float16)
    kv_cache = torch.zeros((2, 1, 16, 1, 256), dtype=torch.float16)

    attn_metadata = SimpleNamespace(
        max_query_len=15,
        max_seq_len=15,
        num_actual_tokens=15,
        query_start_loc=torch.tensor([0, 15], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 15], dtype=torch.int32),
        seq_lens=torch.tensor([15], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([15], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        causal=True,
        max_model_len=262144,
    )
    layer = SimpleNamespace(
        _k_scale_float=1.0,
        _v_scale_float=1.0,
        layer_name="language_model.model.layers.3.self_attn.attn",
    )

    result = impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert calls == ["prefix"]


def test_flash_v100_cudagraph_capture_keeps_cpu_metadata(local_flash_v100_model):
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.v1.attention.backends.flash_attn_v100 import (
        FlashAttnV100MetadataBuilder,
    )

    vllm_config = create_vllm_config(
        model_name=local_flash_v100_model(),
        max_model_len=256,
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder = FlashAttnV100MetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["language_model.model.layers.3.self_attn.attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[15], query_lens=[15]),
        block_size=16,
        device=torch.device("cpu"),
    )

    attn_metadata = builder.build_for_cudagraph_capture(common)

    assert torch.equal(attn_metadata.query_start_loc_cpu, common.query_start_loc_cpu)
    assert torch.equal(attn_metadata.seq_lens_cpu, common.seq_lens_cpu)
    assert attn_metadata.causal is True


def test_flash_v100_decode_shape_hints_stay_backend_local(
    local_flash_v100_model,
):
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.v1.attention.backends.flash_attn_v100 import (
        FlashAttnV100MetadataBuilder,
    )

    vllm_config = create_vllm_config(
        model_name=local_flash_v100_model(),
        max_model_len=2048,
        max_num_seqs=1,
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder = FlashAttnV100MetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["language_model.model.layers.3.self_attn.attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[1025], query_lens=[1]),
        block_size=16,
        device=torch.device("cpu"),
    )

    runtime_metadata = builder.build(0, common)
    assert runtime_metadata.flash_v100_decode_max_seq_len_hint == 1025
    assert runtime_metadata.flash_v100_decode_workspace_seq_capacity_hint is None

    capture_metadata = builder.build_for_cudagraph_capture(common)
    assert capture_metadata.flash_v100_decode_max_seq_len_hint == 1025
    assert capture_metadata.flash_v100_decode_workspace_seq_capacity_hint == 1025
    assert capture_metadata.flash_v100_static_decode_seq_hint == 1025


def test_flash_v100_capture_decode_workspace_covers_graph_max_seq_len(
    local_flash_v100_model,
):
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.v1.attention.backends.flash_attn_v100 import (
        FlashAttnV100MetadataBuilder,
    )

    vllm_config = create_vllm_config(
        model_name=local_flash_v100_model(),
        max_model_len=2048,
        max_num_seqs=1,
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder = FlashAttnV100MetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["language_model.model.layers.3.self_attn.attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[1025], query_lens=[1]),
        block_size=16,
        device=torch.device("cpu"),
    )
    common.max_seq_len = 2048
    common.block_table_tensor = torch.zeros(
        (1, 2048 // 16),
        dtype=torch.int32,
        device=torch.device("cpu"),
    )

    capture_metadata = builder.build_for_cudagraph_capture(common)

    assert capture_metadata.flash_v100_decode_max_seq_len_hint == 1025
    assert capture_metadata.flash_v100_decode_workspace_seq_capacity_hint == 2048
    assert capture_metadata.flash_v100_static_decode_seq_hint == 2048


def test_flash_v100_smallq_cudagraph_metadata_uses_persistent_buffers(
    monkeypatch,
    local_flash_v100_model,
):
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.v1.attention.backends.flash_attn_v100 import (
        FlashAttnV100MetadataBuilder,
    )

    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16")

    vllm_config = create_vllm_config(
        model_name=local_flash_v100_model(),
        max_model_len=256,
        max_num_seqs=2,
    )
    vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4]
    vllm_config.compilation_config.max_cudagraph_capture_size = 4
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder = FlashAttnV100MetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["language_model.model.layers.3.self_attn.attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    capture_common = create_common_attn_metadata(
        BatchSpec(seq_lens=[4], query_lens=[4]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    capture_metadata = builder.build_for_cudagraph_capture(capture_common)
    capture_block_ptr = capture_metadata.smallq_decode_block_table.data_ptr()
    capture_lens_ptr = capture_metadata.smallq_decode_seq_lens.data_ptr()

    assert torch.equal(
        capture_metadata.smallq_decode_seq_lens,
        torch.tensor(
            [1, 2, 3, 4],
            dtype=torch.int32,
            device=capture_metadata.smallq_decode_seq_lens.device,
        ),
    )

    runtime_common = create_common_attn_metadata(
        BatchSpec(seq_lens=[8], query_lens=[4]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    runtime_metadata = builder.build(0, runtime_common)

    assert runtime_metadata.smallq_decode_block_table.data_ptr() == capture_block_ptr
    assert runtime_metadata.smallq_decode_seq_lens.data_ptr() == capture_lens_ptr
    assert torch.equal(
        runtime_metadata.smallq_decode_seq_lens,
        torch.tensor([5, 6, 7, 8], dtype=torch.int32),
    )
    assert torch.equal(
        runtime_metadata.smallq_decode_block_table,
        runtime_common.block_table_tensor.repeat_interleave(
            torch.tensor([4], dtype=torch.int32),
            dim=0,
        ),
    )


def test_flash_v100_smallq_capture_can_bound_workspace_to_context_bucket(
    monkeypatch,
    local_flash_v100_model,
):
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.v1.attention.backends.flash_attn_v100 import (
        FlashAttnV100MetadataBuilder,
    )

    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16")
    vllm_config = create_vllm_config(
        model_name=local_flash_v100_model(),
        max_model_len=2048,
        max_num_seqs=1,
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder = FlashAttnV100MetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["language_model.model.layers.3.self_attn.attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[1024], query_lens=[5]),
        block_size=16,
        device=torch.device("cpu"),
    )
    common.max_seq_len = 1024
    common.block_table_tensor = torch.zeros(
        (1, 2048), dtype=torch.int32, device=torch.device("cpu")
    )

    capture_metadata = builder.build_for_cudagraph_capture(common)

    # Capture normalizes the active small-query rows to q=5; the graph key's
    # bounded common max_seq_len controls the static workspace independently.
    assert capture_metadata.smallq_decode_max_seq_len_hint == 5
    assert capture_metadata.smallq_decode_workspace_seq_capacity_hint == 1024


def test_flash_v100_smallq_context_bucket_can_preserve_partition_size(
    monkeypatch,
    local_flash_v100_model,
):
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.v1.attention.backends.flash_attn_v100 import (
        FlashAttnV100MetadataBuilder,
    )

    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16")
    monkeypatch.setenv("VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE", "1024")
    vllm_config = create_vllm_config(
        model_name=local_flash_v100_model(),
        max_model_len=2048,
        max_num_seqs=1,
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder = FlashAttnV100MetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["language_model.model.layers.3.self_attn.attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[1024], query_lens=[5]),
        block_size=16,
        device=torch.device("cpu"),
    )
    common.max_seq_len = 1024
    common.block_table_tensor = torch.zeros(
        (1, 2048), dtype=torch.int32, device=torch.device("cpu")
    )

    capture_metadata = builder.build_for_cudagraph_capture(common)

    assert capture_metadata.smallq_decode_workspace_seq_capacity_hint == 1024
    assert capture_metadata.smallq_decode_partition_size_hint == 1024


def test_flash_v100_smallq_metadata_masks_cudagraph_padding(
    monkeypatch,
    local_flash_v100_model,
):
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.v1.attention.backends.flash_attn_v100 import (
        FlashAttnV100MetadataBuilder,
    )

    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16")

    vllm_config = create_vllm_config(
        model_name=local_flash_v100_model(),
        max_model_len=256,
        max_num_seqs=3,
    )
    vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 6]
    vllm_config.compilation_config.max_cudagraph_capture_size = 6
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder = FlashAttnV100MetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["language_model.model.layers.3.self_attn.attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[8, 7, 0], query_lens=[3, 2, 0]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    common.num_actual_tokens = 6
    attn_metadata = builder.build(0, common)

    assert torch.equal(
        attn_metadata.smallq_decode_seq_lens,
        torch.tensor(
            [6, 7, 8, 6, 7, 0],
            dtype=torch.int32,
            device=attn_metadata.smallq_decode_seq_lens.device,
        ),
    )
    assert torch.equal(
        attn_metadata.smallq_decode_block_table[-1],
        torch.zeros_like(attn_metadata.smallq_decode_block_table[-1]),
    )


def test_flash_v100_smallq_replay_shape_overflow_fails_fast(
    monkeypatch,
    local_flash_v100_model,
):
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.v1.attention.backends.flash_attn_v100 import (
        FlashAttnV100MetadataBuilder,
    )

    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16")

    vllm_config = create_vllm_config(
        model_name=local_flash_v100_model(),
        max_model_len=256,
        max_num_seqs=1,
    )
    vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2]
    vllm_config.compilation_config.max_cudagraph_capture_size = 2
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder = FlashAttnV100MetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["language_model.model.layers.3.self_attn.attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    capture_common = create_common_attn_metadata(
        BatchSpec(seq_lens=[2], query_lens=[2]),
        block_size=16,
        device=torch.device("cpu"),
    )
    builder.build_for_cudagraph_capture(capture_common)

    runtime_common = create_common_attn_metadata(
        BatchSpec(seq_lens=[20], query_lens=[3]),
        block_size=16,
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="persistent buffer capacity"):
        builder.build(0, runtime_common)


@pytest.mark.parametrize(
    ("max_num_seqs", "expected"),
    [
        (1, [1, 2]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
        (4, [1, 2, 4]),
        (8, [1, 2, 4, 8]),
        (12, [1, 2, 4, 8, 12]),
        (16, [1, 2, 4, 8, 16]),
        (256, [1, 2, 4, 8, 16]),
    ],
)
def test_sm70_nomtp_cudagraph_capture_sizes_cover_concurrency(
    max_num_seqs: int,
    expected: list[int],
):
    from vllm.config.vllm import _sm70_nomtp_cudagraph_capture_sizes

    assert _sm70_nomtp_cudagraph_capture_sizes(max_num_seqs) == expected


def test_flash_v100_decode_query_does_not_attach_smallq_metadata(
    monkeypatch,
    local_flash_v100_model,
):
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.v1.attention.backends.flash_attn_v100 import (
        FlashAttnV100MetadataBuilder,
    )

    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16")

    vllm_config = create_vllm_config(
        model_name=local_flash_v100_model(),
        max_model_len=256,
        max_num_seqs=1,
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder = FlashAttnV100MetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["language_model.model.layers.3.self_attn.attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[9], query_lens=[1]),
        block_size=16,
        device=torch.device("cpu"),
    )
    attn_metadata = builder.build(0, common)

    assert attn_metadata.smallq_decode_block_table is None
    assert attn_metadata.smallq_decode_seq_lens is None
    assert attn_metadata.smallq_query_start_loc is None


def test_flash_v100_smallq_forward_prefers_persistent_decode_metadata():
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    impl = FlashAttnV100Impl(
        num_heads=4,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    persistent_block_table = torch.tensor([[3], [3]], dtype=torch.int32)
    persistent_seq_lens = torch.tensor([8, 9], dtype=torch.int32)
    captured: dict[str, torch.Tensor] = {}

    def fake_decode(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        **kwargs,
    ):
        captured["block_table"] = block_table
        captured["seq_lens"] = seq_lens
        kwargs["out"].fill_(1)

    impl.flash_attn_decode_paged = fake_decode  # type: ignore[method-assign]

    attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([9], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([9], dtype=torch.int32),
        block_table=torch.tensor([[7]], dtype=torch.int32),
        smallq_decode_block_table=persistent_block_table,
        smallq_decode_seq_lens=persistent_seq_lens,
        smallq_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((2, 4, 256), dtype=torch.float16)
    output = torch.zeros((2, 4, 256), dtype=torch.float16)
    key_cache = torch.zeros((4, 16, 1, 256), dtype=torch.float16)
    value_cache = torch.zeros((4, 16, 1, 256), dtype=torch.float16)

    result = impl._flash_v100_small_query_prefill_as_decode(
        layer,
        query,
        key_cache,
        value_cache,
        attn_metadata,
        output,
        attn_metadata.query_start_loc,
        attn_metadata.seq_lens,
    )

    assert result is output
    assert captured["block_table"].data_ptr() == persistent_block_table.data_ptr()
    assert captured["seq_lens"].data_ptr() == persistent_seq_lens.data_ptr()
    assert torch.all(output == 1)


def test_flash_v100_decode_forwards_shape_hints():
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    impl = FlashAttnV100Impl(
        num_heads=4,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    captured: dict[str, int | None] = {}

    def fake_decode(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        **kwargs,
    ):
        captured["max_seq_len_hint"] = kwargs.get("max_seq_len_hint")
        captured["workspace_seq_capacity_hint"] = kwargs.get(
            "workspace_seq_capacity_hint"
        )
        kwargs["out"].fill_(1)

    impl.flash_attn_decode_paged = fake_decode  # type: ignore[method-assign]

    attn_metadata = SimpleNamespace(
        num_actual_tokens=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([4097], dtype=torch.int32),
        flash_v100_decode_max_seq_len_hint=4097,
        flash_v100_decode_workspace_seq_capacity_hint=4097,
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((1, 4, 256), dtype=torch.float16)
    output = torch.zeros((1, 4, 256), dtype=torch.float16)
    kv_cache = torch.zeros((2, 4, 16, 1, 256), dtype=torch.float16)

    result = impl._flash_v100_decode(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert captured["max_seq_len_hint"] == 4097
    assert captured["workspace_seq_capacity_hint"] == 4097
    assert torch.all(output == 1)


def test_flash_v100_decode_uses_xqa_by_default_when_shape_supported(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_USE_XQA", raising=False)

    impl = FlashAttnV100Impl(
        num_heads=6,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    calls: list[str] = []

    def hit_xqa(*args, **kwargs):
        calls.append("xqa")
        kwargs["out"].fill_(1)

    def fail_scalar(*args, **kwargs):
        raise AssertionError("scalar decode should not be selected")

    impl.flash_attn_decode_paged_xqa = hit_xqa  # type: ignore[method-assign]
    impl.flash_attn_decode_paged = fail_scalar  # type: ignore[method-assign]

    attn_metadata = SimpleNamespace(
        num_actual_tokens=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([4097], dtype=torch.int32),
        flash_v100_decode_max_seq_len_hint=4097,
        flash_v100_decode_workspace_seq_capacity_hint=4097,
        flash_v100_decode_active_num_partitions=torch.tensor([17], dtype=torch.int32),
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((1, 6, 256), dtype=torch.float16)
    output = torch.zeros((1, 6, 256), dtype=torch.float16)
    kv_cache = torch.zeros((2, 4, 16, 1, 256), dtype=torch.float16)

    result = impl._flash_v100_decode(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert calls == ["xqa"]
    assert torch.all(output == 1)


@pytest.mark.parametrize(
    ("num_heads", "seq_len", "expected_route"),
    (
        (6, 4096, "scalar"),
        (6, 8192, "scalar"),
        (6, 16383, "scalar"),
        (6, 16384, "xqa"),
        (4, 65536, "scalar"),
        (8, 65536, "xqa"),
    ),
)
def test_flash_v100_fp8_e5m2_decode_xqa_long_context_gate(
    monkeypatch, num_heads, seq_len, expected_route
):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_USE_XQA", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN", raising=False)
    impl = FlashAttnV100Impl(
        num_heads=num_heads,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_e5m2",
    )

    calls: list[str] = []

    def hit_xqa(*args, **kwargs):
        calls.append("xqa")
        kwargs["out"].fill_(1)

    def hit_scalar(*args, **kwargs):
        calls.append("scalar")
        kwargs["out"].fill_(1)

    impl.flash_attn_decode_paged_xqa = hit_xqa  # type: ignore[method-assign]
    impl.flash_attn_decode_paged = hit_scalar  # type: ignore[method-assign]
    attn_metadata = SimpleNamespace(
        num_actual_tokens=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        flash_v100_decode_max_seq_len_hint=seq_len,
        flash_v100_decode_workspace_seq_capacity_hint=262144,
        flash_v100_decode_active_num_partitions=torch.tensor([1], dtype=torch.int32),
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((1, num_heads, 256), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 4, 16, 1, 256), dtype=torch.uint8)

    result = impl._flash_v100_decode(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert calls == [expected_route]
    assert torch.all(output == 1)


def test_flash_v100_fp8_prefill_bridge_accepts_mtp_aligned_tail():
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    impl = object.__new__(FlashAttnV100Impl)
    impl.use_fp8_prefill_bridge = True
    impl.use_flash_v100_prefill_paged = True
    impl.kv_cache_dtype = "fp8_e5m2"
    key_cache = torch.empty((1, 1616, 2, 256), dtype=torch.uint8)
    value_cache = torch.empty_like(key_cache)

    assert impl._should_use_fp8_prefill_bridge(
        q_len=1616,
        head_dim=256,
        key_cache=key_cache,
        value_cache=value_cache,
        causal=True,
        window_size=(-1, -1),
    )


def test_flash_v100_fp8_prefill_bridge_prefers_logical_dense_exact(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as mod
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    impl = object.__new__(FlashAttnV100Impl)
    impl.scale = 256**-0.5
    bridge_calls = []
    exact_calls = []

    def bridge_op(*args):
        bridge_calls.append(args)

    def exact_op(query, key, value, **kwargs):
        exact_calls.append((query, key, value, kwargs))
        kwargs["out"].fill_(3)
        return kwargs["out"]

    impl.fp8_e5m2_paged_kv_to_fp16 = bridge_op
    impl.flash_attn_prefill_paged = lambda *args, **kwargs: pytest.fail(
        "dense exact path unexpectedly fell back to paged prefill"
    )
    key_workspace = torch.empty((1, 784, 1, 256), dtype=torch.float16)
    value_workspace = torch.empty_like(key_workspace)
    output_block_table = torch.zeros((1, 1), dtype=torch.int32)
    monkeypatch.setattr(
        mod,
        "_get_fp8_prefill_bridge_workspace",
        lambda key_cache, required_blocks: (
            key_workspace,
            value_workspace,
            output_block_table,
        ),
    )
    monkeypatch.setattr(
        mod,
        "_uniform_cu_seqlens",
        lambda *args, **kwargs: (
            torch.tensor([0, 64], dtype=torch.int32),
            torch.tensor([0, 64], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(mod, "_try_sm70_fa2_d256_prefill", exact_op)

    query = torch.zeros((1, 64, 6, 256), dtype=torch.float16)
    key_cache = torch.zeros((1, 64, 1, 256), dtype=torch.uint8)
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    seq_lens = torch.tensor([64], dtype=torch.int32)
    out = torch.zeros_like(query)

    result = impl._run_fp8_prefill_bridge(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        seq_len=64,
        k_scale=1.0,
        v_scale=1.0,
        causal=True,
        window_size=(-1, -1),
        out=out,
    )

    assert result is not None
    assert result[0] is out
    assert result[1] is True
    assert len(bridge_calls) == 1
    assert len(exact_calls) == 1
    _, dense_key, dense_value, kwargs = exact_calls[0]
    assert dense_key.shape == (1, 64, 1, 256)
    assert dense_value.shape == dense_key.shape
    assert dense_key.is_contiguous()
    assert dense_value.is_contiguous()
    assert kwargs["cu_seqlens_k"] is not None
    assert kwargs.get("block_table") is None
    assert torch.all(out == 3)


def test_flash_v100_fp8_prefill_bridge_workspace_oom_falls_back(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as mod

    key_cache = torch.zeros((1, 1568, 1, 33), dtype=torch.uint8)
    mod._fp8_prefill_bridge_workspaces.clear()
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(cuda_stream=17),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: (_ for _ in ()).throw(torch.OutOfMemoryError()),
    )

    assert mod._get_fp8_prefill_bridge_workspace(key_cache, 2) is None


@pytest.mark.parametrize(
    ("static_seq_hint", "expected"),
    ((8192, False), (16384, True), (262144, True)),
)
def test_flash_v100_fp8_xqa_graph_capture_uses_static_context_hint(
    monkeypatch, static_seq_hint, expected
):
    from vllm.v1.attention.backends import flash_attn_v100 as mod

    monkeypatch.setattr(mod, "_is_cuda_graph_capturing", lambda query: False)
    metadata = SimpleNamespace(
        flash_v100_cudagraph_capture=True,
        flash_v100_decode_max_seq_len_hint=1,
        flash_v100_static_decode_seq_hint=static_seq_hint,
        flash_v100_decode_workspace_seq_capacity_hint=static_seq_hint,
    )

    assert mod._decode_fp8_xqa_allowed(metadata, torch.empty(1)) is expected


@pytest.mark.parametrize(
    ("routing_enabled", "expected_route"), ((True, "xqa"), (False, "scalar"))
)
def test_flash_v100_fp8_xqa_batch_context_graph_routing(
    monkeypatch,
    routing_enabled,
    expected_route,
):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    impl = FlashAttnV100Impl(
        num_heads=6,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_e5m2",
    )
    calls: list[tuple[str, bool]] = []

    def hit_xqa(*args, **kwargs):
        calls.append(("xqa", kwargs.get("batch_context_routing", False)))
        kwargs["out"].fill_(1)

    def hit_scalar(*args, **kwargs):
        calls.append(("scalar", False))
        kwargs["out"].fill_(1)

    impl.flash_attn_decode_paged_xqa = hit_xqa  # type: ignore[method-assign]
    impl.flash_attn_decode_paged = hit_scalar  # type: ignore[method-assign]
    attn_metadata = SimpleNamespace(
        num_actual_tokens=4,
        block_table=torch.zeros((4, 256), dtype=torch.int32),
        seq_lens=torch.full((4,), 4095, dtype=torch.int32),
        flash_v100_cudagraph_capture=True,
        flash_v100_batch_context_routing=routing_enabled,
        flash_v100_decode_max_seq_len_hint=4095,
        flash_v100_decode_workspace_seq_capacity_hint=4095,
        flash_v100_decode_active_num_partitions=torch.tensor([16], dtype=torch.int32),
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((4, 6, 256), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 2, 16, 1, 256), dtype=torch.uint8)

    result = impl._flash_v100_decode(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert calls == [(expected_route, routing_enabled and expected_route == "xqa")]
    assert torch.all(output == 1)


@pytest.mark.parametrize(
    ("batch_size", "context_capacity", "routing_enabled", "expected"),
    (
        (16, 12287, True, 1024),
        (16, 12288, True, None),
        (12, 2047, True, None),
        (16, 12287, False, None),
    ),
)
def test_flash_v100_batch_context_b16_preserves_p1024_reduction(
    monkeypatch,
    batch_size,
    context_capacity,
    routing_enabled,
    expected,
):
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _batch_context_partition_size_hint,
    )

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    query = torch.empty((batch_size, 6, 256), dtype=torch.float16)
    metadata = SimpleNamespace(
        flash_v100_batch_context_routing=routing_enabled,
        flash_v100_decode_max_seq_len_hint=context_capacity,
        flash_v100_decode_workspace_seq_capacity_hint=context_capacity,
    )

    assert _batch_context_partition_size_hint(query, metadata) == expected


def test_flash_v100_mtp5_dual_cta_partition_policy(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as mod

    monkeypatch.delenv("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", raising=False)
    assert mod._mtp5_xqa_dual_cta_partition_size_hint() == 1024

    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", "0")
    assert mod._mtp5_xqa_dual_cta_partition_size_hint() is None

    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", "false")
    assert mod._mtp5_xqa_dual_cta_partition_size_hint() is None

    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "256")
    assert mod._mtp5_xqa_dual_cta_partition_size_hint() == 256

    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "128")
    with pytest.raises(ValueError, match="must be one of"):
        mod._mtp5_xqa_dual_cta_partition_size_hint()


@pytest.mark.parametrize(("num_heads", "num_kv_heads"), [(6, 1), (12, 2)])
def test_flash_v100_mtp_verifier_uses_xqa_in_long_graph(
    monkeypatch,
    num_heads,
    num_kv_heads,
):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    monkeypatch.delenv("VLLM_FLASH_V100_SMALLQ_DECODE_USE_XQA", raising=False)
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "1024")
    impl = FlashAttnV100Impl(
        num_heads=num_heads,
        head_size=256,
        scale=1.0,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_e5m2",
    )

    calls: list[tuple[str, int | None]] = []

    def hit_xqa(*args, **kwargs):
        calls.append(("xqa", kwargs.get("partition_size_hint")))
        kwargs["out"].fill_(1)

    def fail_scalar(*args, **kwargs):
        raise AssertionError("long-graph MTP verifier should use XQA")

    impl.flash_attn_decode_paged_xqa = hit_xqa  # type: ignore[method-assign]
    impl.flash_attn_decode_paged = fail_scalar  # type: ignore[method-assign]
    smallq_seq_lens = torch.tensor(
        [65532, 65533, 65534, 65535, 65536], dtype=torch.int32
    )
    attn_metadata = SimpleNamespace(
        num_actual_tokens=5,
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        seq_lens=torch.tensor([65536], dtype=torch.int32),
        smallq_decode_block_table=torch.zeros((5, 1), dtype=torch.int32),
        smallq_decode_seq_lens=smallq_seq_lens,
        smallq_query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        smallq_decode_max_seq_len_hint=5,
        smallq_decode_workspace_seq_capacity_hint=262144,
        smallq_decode_partition_size_hint=None,
        flash_v100_cudagraph_capture=True,
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((5, num_heads, 256), dtype=torch.float16)
    output = torch.zeros_like(query)
    key_cache = torch.zeros((1, 1616, num_kv_heads, 256), dtype=torch.uint8)
    value_cache = torch.zeros_like(key_cache)

    result = impl._flash_v100_small_query_prefill_as_decode(
        layer,
        query,
        key_cache,
        value_cache,
        attn_metadata,
        output,
        attn_metadata.smallq_query_start_loc,
        attn_metadata.seq_lens,
    )

    assert result is output
    assert calls == [("xqa", 1024)]
    assert torch.all(output == 1)


def test_flash_v100_mtp_context_bucket_keeps_scalar_verifier(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    monkeypatch.delenv("VLLM_FLASH_V100_SMALLQ_DECODE_USE_XQA", raising=False)
    impl = FlashAttnV100Impl(
        num_heads=6,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_e5m2",
    )

    calls: list[str] = []

    def fail_xqa(*args, **kwargs):
        raise AssertionError("P256 context-bucket verifier must stay scalar")

    def hit_scalar(*args, **kwargs):
        calls.append("scalar")
        kwargs["out"].fill_(1)

    impl.flash_attn_decode_paged_xqa = fail_xqa  # type: ignore[method-assign]
    impl.flash_attn_decode_paged = hit_scalar  # type: ignore[method-assign]
    attn_metadata = SimpleNamespace(
        num_actual_tokens=5,
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        seq_lens=torch.tensor([768], dtype=torch.int32),
        smallq_decode_block_table=torch.zeros((5, 1), dtype=torch.int32),
        smallq_decode_seq_lens=torch.tensor(
            [764, 765, 766, 767, 768], dtype=torch.int32
        ),
        smallq_query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        smallq_decode_max_seq_len_hint=5,
        smallq_decode_workspace_seq_capacity_hint=4096,
        smallq_decode_partition_size_hint=256,
        flash_v100_cudagraph_capture=True,
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((5, 6, 256), dtype=torch.float16)
    output = torch.zeros_like(query)
    key_cache = torch.zeros((1, 1616, 1, 256), dtype=torch.uint8)
    value_cache = torch.zeros_like(key_cache)

    result = impl._flash_v100_small_query_prefill_as_decode(
        layer,
        query,
        key_cache,
        value_cache,
        attn_metadata,
        output,
        attn_metadata.smallq_query_start_loc,
        attn_metadata.seq_lens,
    )

    assert result is output
    assert calls == ["scalar"]
    assert torch.all(output == 1)


def test_flash_v100_decode_uses_xqa_for_qwen35_tp4_long_context(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_USE_XQA", raising=False)

    impl = FlashAttnV100Impl(
        num_heads=4,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    calls: list[str] = []

    def hit_xqa(*args, **kwargs):
        calls.append("xqa")
        kwargs["out"].fill_(1)

    def fail_scalar(*args, **kwargs):
        raise AssertionError("scalar decode should not be selected")

    impl.flash_attn_decode_paged_xqa = hit_xqa  # type: ignore[method-assign]
    impl.flash_attn_decode_paged = fail_scalar  # type: ignore[method-assign]

    attn_metadata = SimpleNamespace(
        num_actual_tokens=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([32769], dtype=torch.int32),
        flash_v100_decode_max_seq_len_hint=1,
        flash_v100_decode_workspace_seq_capacity_hint=65536,
        flash_v100_decode_active_num_partitions=torch.tensor([1], dtype=torch.int32),
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((1, 4, 256), dtype=torch.float16)
    output = torch.zeros((1, 4, 256), dtype=torch.float16)
    kv_cache = torch.zeros((2, 4, 16, 1, 256), dtype=torch.float16)

    result = impl._flash_v100_decode(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert calls == ["xqa"]
    assert torch.all(output == 1)


def test_flash_v100_decode_plans_g6_page784_sawtooth_workspace(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", raising=False)

    impl = FlashAttnV100Impl(
        num_heads=6,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    partition_hints: list[int | None] = []

    def hit_xqa(*args, **kwargs):
        partition_hints.append(kwargs.get("partition_size_hint"))
        kwargs["out"].fill_(1)

    impl.flash_attn_decode_paged_xqa = hit_xqa  # type: ignore[method-assign]
    attn_metadata = SimpleNamespace(
        num_actual_tokens=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([180224], dtype=torch.int32),
        flash_v100_decode_max_seq_len_hint=180224,
        flash_v100_decode_workspace_seq_capacity_hint=262144,
        flash_v100_decode_active_num_partitions=torch.tensor([176], dtype=torch.int32),
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((1, 6, 256), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 2, 784, 1, 256), dtype=torch.float16)

    result = impl._flash_v100_decode(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert partition_hints == [256]
    assert torch.all(output == 1)


def test_flash_v100_decode_sawtooth_rollback_preserves_default_planner(
    monkeypatch,
):
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _g6_aligned_page_partition_size_hint,
    )

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", "0")
    query = torch.zeros((1, 6, 256), dtype=torch.float16)
    key_cache = torch.zeros((1, 784, 1, 256), dtype=torch.float16)

    assert (
        _g6_aligned_page_partition_size_hint(query, key_cache, key_cache, "auto")
        is None
    )


def test_flash_v100_decode_sawtooth_preserves_explicit_partition_override(
    monkeypatch,
):
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _g6_aligned_page_partition_size_hint,
    )

    monkeypatch.setenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", "256")
    monkeypatch.delenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", raising=False)
    query = torch.zeros((1, 6, 256), dtype=torch.float16)
    key_cache = torch.zeros((1, 784, 1, 256), dtype=torch.float16)

    assert (
        _g6_aligned_page_partition_size_hint(query, key_cache, key_cache, "auto")
        is None
    )


def test_flash_v100_decode_sawtooth_workspace_supports_fp8_kv(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _g6_aligned_page_partition_size_hint,
    )

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", raising=False)
    query = torch.zeros((1, 6, 256), dtype=torch.float16)
    key_cache = torch.zeros((1, 784, 1, 256), dtype=torch.float16)

    assert (
        _g6_aligned_page_partition_size_hint(query, key_cache, key_cache, "auto") == 256
    )


def test_flash_v100_decode_e5m2_aligned_page_plans_p256_workspace(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _g6_aligned_page_partition_size_hint,
    )

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", raising=False)
    query = torch.zeros((1, 6, 256), dtype=torch.float16)
    key_cache = torch.zeros((1, 1568, 1, 256), dtype=torch.uint8)

    assert (
        _g6_aligned_page_partition_size_hint(query, key_cache, key_cache, "fp8_e5m2")
        == 256
    )


def test_flash_v100_decode_e5m2_partition_hint_is_page_size_generic(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _g6_aligned_page_partition_size_hint,
    )

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", raising=False)
    query = torch.zeros((1, 6, 256), dtype=torch.float16)
    key_cache = torch.zeros((1, 1616, 1, 256), dtype=torch.uint8)

    assert (
        _g6_aligned_page_partition_size_hint(query, key_cache, key_cache, "fp8_e5m2")
        == 256
    )


def test_flash_v100_decode_e5m2_keeps_small_pages_on_default_planner(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _g6_aligned_page_partition_size_hint,
    )

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", raising=False)
    query = torch.zeros((1, 6, 256), dtype=torch.float16)
    key_cache = torch.zeros((1, 16, 1, 256), dtype=torch.uint8)

    assert (
        _g6_aligned_page_partition_size_hint(query, key_cache, key_cache, "fp8_e5m2")
        is None
    )


def test_flash_v100_decode_e4m3_does_not_use_e5m2_partition_hint(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _g6_aligned_page_partition_size_hint,
    )

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH", raising=False)
    query = torch.zeros((1, 6, 256), dtype=torch.float16)
    key_cache = torch.zeros((1, 1568, 1, 256), dtype=torch.uint8)

    assert (
        _g6_aligned_page_partition_size_hint(query, key_cache, key_cache, "fp8_e4m3")
        is None
    )


def test_flash_v100_decode_keeps_qwen35_tp4_short_context_on_scalar(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_USE_XQA", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_XQA_Q4_MIN_SEQ_LEN", raising=False)

    impl = FlashAttnV100Impl(
        num_heads=4,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    calls: list[str] = []

    def fail_xqa(*args, **kwargs):
        raise AssertionError("q_per_kv=4 short-context decode should stay scalar")

    def hit_scalar(*args, **kwargs):
        calls.append("scalar")
        kwargs["out"].fill_(1)

    impl.flash_attn_decode_paged_xqa = fail_xqa  # type: ignore[method-assign]
    impl.flash_attn_decode_paged = hit_scalar  # type: ignore[method-assign]

    attn_metadata = SimpleNamespace(
        num_actual_tokens=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([4097], dtype=torch.int32),
        flash_v100_decode_max_seq_len_hint=4097,
        flash_v100_decode_workspace_seq_capacity_hint=8192,
        flash_v100_decode_active_num_partitions=torch.tensor([17], dtype=torch.int32),
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((1, 4, 256), dtype=torch.float16)
    output = torch.zeros((1, 4, 256), dtype=torch.float16)
    kv_cache = torch.zeros((2, 4, 16, 1, 256), dtype=torch.float16)

    result = impl._flash_v100_decode(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert calls == ["scalar"]
    assert torch.all(output == 1)


def test_flash_v100_decode_xqa_default_can_be_disabled(monkeypatch):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    monkeypatch.setenv("VLLM_FLASH_V100_DECODE_USE_XQA", "0")

    impl = FlashAttnV100Impl(
        num_heads=6,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    calls: list[str] = []

    def fail_xqa(*args, **kwargs):
        raise AssertionError("xqa decode should be disabled")

    def hit_scalar(*args, **kwargs):
        calls.append("scalar")
        kwargs["out"].fill_(1)

    impl.flash_attn_decode_paged_xqa = fail_xqa  # type: ignore[method-assign]
    impl.flash_attn_decode_paged = hit_scalar  # type: ignore[method-assign]

    attn_metadata = SimpleNamespace(
        num_actual_tokens=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([4097], dtype=torch.int32),
        flash_v100_decode_max_seq_len_hint=4097,
        flash_v100_decode_workspace_seq_capacity_hint=4097,
        flash_v100_decode_active_num_partitions=torch.tensor([17], dtype=torch.int32),
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((1, 6, 256), dtype=torch.float16)
    output = torch.zeros((1, 6, 256), dtype=torch.float16)
    kv_cache = torch.zeros((2, 4, 16, 1, 256), dtype=torch.float16)

    result = impl._flash_v100_decode(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output,
    )

    assert result is output
    assert calls == ["scalar"]
    assert torch.all(output == 1)


def test_flash_v100_smallq_capture_requires_persistent_metadata(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    monkeypatch.setattr(
        flash_attn_v100,
        "_is_cuda_graph_capturing",
        lambda tensor: True,
    )

    impl = FlashAttnV100Impl(
        num_heads=4,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([9], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([9], dtype=torch.int32),
        block_table=torch.tensor([[7]], dtype=torch.int32),
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    query = torch.zeros((2, 4, 256), dtype=torch.float16)
    output = torch.zeros((2, 4, 256), dtype=torch.float16)
    key_cache = torch.zeros((4, 16, 1, 256), dtype=torch.float16)
    value_cache = torch.zeros((4, 16, 1, 256), dtype=torch.float16)

    with pytest.raises(RuntimeError, match="persistent smallq decode metadata"):
        impl._flash_v100_small_query_prefill_as_decode(
            layer,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            output,
            attn_metadata.query_start_loc,
            attn_metadata.seq_lens,
        )


def test_flash_v100_fp8_kv_route_summary_counts_repeated_hits(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as mod

    monkeypatch.setenv("VLLM_FLASH_V100_ROUTE_SUMMARY", "1")
    monkeypatch.setattr(mod.atexit, "register", lambda callback: None)
    monkeypatch.setattr(mod, "_route_counts", {})
    monkeypatch.setattr(mod, "_route_summary_registered", False)
    monkeypatch.setattr(mod, "_logged_fp8_kv_prefill", False)
    monkeypatch.setattr(mod, "_logged_fp8_kv_decode", False)

    mod._log_fp8_kv_cache_route("decode", "fp8_e5m2", "scalar_paged")
    mod._log_fp8_kv_cache_route("decode", "fp8_e5m2", "scalar_paged")
    mod._log_fp8_kv_cache_route("prefill", "fp8_e5m2", "prefix")
    mod._log_fp8_kv_cache_route("decode", "auto", "scalar_paged")

    assert mod._route_counts["fp8_kv_decode"] == 2
    assert mod._route_counts["fp8_kv_decode_scalar_paged"] == 2
    assert mod._route_counts["fp8_kv_prefill"] == 1
    assert mod._route_counts["fp8_kv_prefill_prefix"] == 1
