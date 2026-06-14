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
        torch.tensor([1, 2, 3, 4], dtype=torch.int32),
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
        torch.tensor([6, 7, 8, 6, 7, 0], dtype=torch.int32),
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
        BatchSpec(seq_lens=[4], query_lens=[3]),
        block_size=16,
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="persistent buffer capacity"):
        builder.build(0, runtime_common)


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
