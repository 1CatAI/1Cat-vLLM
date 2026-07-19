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
        BatchSpec(seq_lens=[20], query_lens=[3]),
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
        (6, 8192, "xqa"),
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


def test_flash_v100_fp8_xqa_graph_capture_marker_forces_xqa(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as mod

    monkeypatch.setattr(mod, "_is_cuda_graph_capturing", lambda query: False)
    metadata = SimpleNamespace(
        flash_v100_cudagraph_capture=True,
        flash_v100_decode_max_seq_len_hint=1,
        flash_v100_static_decode_seq_hint=262144,
        flash_v100_decode_workspace_seq_capacity_hint=262144,
    )

    assert mod._decode_fp8_xqa_allowed(metadata, torch.empty(1))


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
