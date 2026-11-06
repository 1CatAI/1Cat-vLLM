# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import threading
from types import SimpleNamespace

import pytest

from vllm.config.execution_policy import FlashV100Policy, GraphPolicy
from vllm.config.flash_v100 import FlashV100Diagnostics, FlashV100Options
from vllm.config.sm70_runtime import RuntimeTraceConfig
from vllm.diagnostics import diagnostics_for

pytestmark = pytest.mark.cpu_test


def prepare(options=None, graph=None, trace=None, dtype="fp8_e4m3"):
    policy = FlashV100Policy(options=options or FlashV100Options())
    policy.resolve()
    graph = graph or GraphPolicy()
    trace = trace or FlashV100Diagnostics()
    policy.options.finalize(graph, trace, dtype)
    return policy, graph, trace


@pytest.mark.parametrize("order", [(False, True), (True, False)])
def test_worker_transfer_keeps_two_engine_policies_and_sources(monkeypatch, order):
    configs = []
    for value in order:
        monkeypatch.setenv("VLLM_FLASH_V100_XQA_G6_QK_PIPELINE", str(int(value)))
        configs.append(prepare()[0])
    receiver, sender = multiprocessing.Pipe(duplex=False)
    thread = threading.Thread(target=sender.send, args=(configs,), daemon=True)
    thread.start()
    try:
        assert receiver.poll(10)
        transferred = receiver.recv()
        thread.join(10)
        assert not thread.is_alive()
    finally:
        receiver.close()
        sender.close()
    monkeypatch.setattr(
        "os.getenv", lambda *a: pytest.fail("worker reread environment")
    )
    for expected, original, actual in zip(order, configs, transferred):
        actual.resolve()
        assert actual.options.xqa_g6_qk_pipeline is expected
        assert actual.compute_hash() == original.compute_hash()
        assert (
            actual.options.sources["xqa_g6_qk_pipeline"]
            == "VLLM_FLASH_V100_XQA_G6_QK_PIPELINE"
        )


@pytest.mark.parametrize("raw", ["", "0", "1", "01", "true", "-1"])
def test_graph_and_native_keep_their_distinct_legacy_parsers(monkeypatch, raw):
    name = "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO"
    monkeypatch.setenv(name, raw)
    policy, graph, _ = prepare()
    from vllm.config.flash_v100 import NATIVE_FIELDS

    index = next(i for i, entry in enumerate(NATIVE_FIELDS) if entry[1] == name)
    assert graph.e4m3_p64_p256_auto is (raw != "0")
    assert policy.options.native_effective[index] == (raw[:1] != "0")
    typed, _, _ = prepare(graph=GraphPolicy(e4m3_p64_p256_auto=False))
    assert typed.options.native_effective[index] == 0
    assert typed.options.native_inputs[index] == "0"


def test_effective_native_projection_participates_in_hash(monkeypatch):
    name = "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO"
    monkeypatch.setenv(name, "01")
    first, first_graph, _ = prepare()
    monkeypatch.setenv(name, "1")
    second, second_graph, _ = prepare()
    assert first_graph.compute_hash() == second_graph.compute_hash()
    assert first.compute_hash() != second.compute_hash()


def test_effective_package_projection_participates_in_hash(monkeypatch):
    name = "VLLM_FLASH_V100_XQA_G6_DUAL_CTA"
    monkeypatch.setenv(name, "1suffix")
    first = prepare()[0]
    monkeypatch.setenv(name, "1")
    second = prepare()[0]
    assert first.options.native_effective == second.options.native_effective
    assert not first.options.python_policy["dual_cta"]
    assert second.options.python_policy["dual_cta"]
    assert first.compute_hash() != second.compute_hash()


def test_trace_unused_format_and_resource_capacity_do_not_change_hash(monkeypatch):
    baseline = prepare(dtype="auto")[0].compute_hash()
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_E4M3_G6_P256_BEGIN", "9999")
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_TRACE", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE", "0")
    assert prepare(dtype="auto")[0].compute_hash() == baseline


def test_old_scalar_alias_has_one_canonical_field(monkeypatch):
    monkeypatch.delenv("VLLM_FLASH_V100_E4M3_SCALAR_FAST", raising=False)
    monkeypatch.setenv("VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST", "0")
    old = prepare()[0]
    assert not old.options.python_policy["scalar_fast"]
    monkeypatch.setenv("VLLM_FLASH_V100_E4M3_SCALAR_FAST", "1")
    main = prepare()[0]
    assert main.options.python_policy["scalar_fast"]
    typed = prepare(FlashV100Options(e4m3_scalar_fast=False))[0]
    assert not typed.options.python_policy["scalar_fast"]


def test_qualified_error_is_bound_without_changing_short_circuit(monkeypatch):
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "bad")
    policy, _, _ = prepare(FlashV100Options(xqa_mtp5_dual_cta=False))
    monkeypatch.setenv("VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE", "512")
    assert policy.options.xqa_mtp5_dual_cta is False
    with pytest.raises(ValueError, match="got 'bad'") as error:
        policy.options.value("xqa_mtp5_partition_size")
    assert isinstance(error.value.__cause__, ValueError)


def test_trace_alias_priority_and_typed_override(monkeypatch):
    monkeypatch.setenv("VLLM_FLASH_V100_ROUTE_SUMMARY", "1")
    monkeypatch.setenv("VLLM_SM70_DEBUG", "")
    trace = FlashV100Diagnostics()
    trace.resolve()
    assert not trace.route_summary
    assert trace.sources["route_summary"] == "VLLM_SM70_DEBUG"
    typed = FlashV100Diagnostics(route_summary=True)
    typed.resolve()
    assert typed.route_summary
    assert typed.sources["route_summary"] == "typed"


def test_attention_observations_are_owned_by_each_engine():
    def engine():
        return SimpleNamespace(
            observability_config=SimpleNamespace(runtime_trace=RuntimeTraceConfig())
        )

    first, second = diagnostics_for(engine()), diagnostics_for(engine())
    first.histories.setdefault("flash_v100_routes", {})["decode"] = 7
    assert second.histories.get("flash_v100_routes", {}) == {}


def test_bound_report_never_evaluates_legacy_getters(monkeypatch):
    policy, _, _ = prepare()
    expected = policy.explain()
    monkeypatch.setattr("os.getenv", lambda *a: pytest.fail("report read environment"))
    assert policy.explain() == expected


def test_smallq_metadata_uses_initialized_policy_after_environment_change(monkeypatch):
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.flash_v100.spec import verify_metadata

    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "5")
    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN", "8192")
    first = prepare()[0]
    second = FlashV100Policy(
        smallq_max_q=16, options=FlashV100Options(smallq_decode_max_model_len=0)
    )
    second.resolve()
    monkeypatch.setattr(
        "os.getenv", lambda *a: pytest.fail("metadata read environment")
    )
    for policy, expected in ((first, (5, 8192)), (second, (16, 0))):
        cfg = SimpleNamespace(attention_config=SimpleNamespace(flash_v100=policy))
        with set_current_vllm_config(cfg):
            assert (
                verify_metadata.configured_smallq_max_query_len(None),
                verify_metadata.configured_smallq_max_model_len(None),
            ) == expected


def test_attention_workspace_shutdown_keeps_other_engine_and_legacy(monkeypatch):
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.runtime_resources import release_runtime_resources
    from vllm.v1.attention.backends.flash_v100 import dense_prefill
    from vllm.v1.attention.ops.sm70_workspaces import workspace_cache

    first, second = SimpleNamespace(), SimpleNamespace()
    key = torch.zeros((1, 256, 1, 8), dtype=torch.uint8)
    legacy: dict = {}
    with set_current_vllm_config(first):
        a = dense_prefill.get_fp8_prefill_bridge_workspace(key, 1)
        workspace_cache("grouped_fp16", legacy)["tensor"] = torch.ones(1)
    with set_current_vllm_config(second):
        b = dense_prefill.get_fp8_prefill_bridge_workspace(key, 1)
        bank = workspace_cache("grouped_fp16", legacy)
        assert bank == {}
        bank["tensor"] = torch.zeros(1)
    assert a[0].data_ptr() != b[0].data_ptr()
    with set_current_vllm_config(first):
        reused = dense_prefill.get_fp8_prefill_bridge_workspace(key, 1)
        assert reused[0].data_ptr() == a[0].data_ptr()
        grown = dense_prefill.get_fp8_prefill_bridge_workspace(key, 3)
        assert grown[0].shape[0] == 3
    dense_prefill.clear_flash_attn_v100_workspaces(first)
    release_runtime_resources(first)
    with set_current_vllm_config(second):
        reused = dense_prefill.get_fp8_prefill_bridge_workspace(key, 1)
        assert reused[0].data_ptr() == b[0].data_ptr()
        assert workspace_cache("grouped_fp16", legacy) is bank
    assert legacy == {}


def test_fp8_bridge_workspace_stays_within_addressable_context():
    import math

    import torch

    from vllm.config import set_current_vllm_config
    from vllm.runtime_resources import release_runtime_resources
    from vllm.v1.attention.backends.flash_v100 import dense_prefill

    engine = SimpleNamespace()
    input_page_size = 1616
    output_page_size = dense_prefill._FP8_PREFILL_BRIDGE_PAGE_SIZE
    max_model_len = 262144
    table_width = math.ceil(max_model_len / input_page_size)
    max_blocks = math.ceil(table_width * input_page_size / output_page_size)
    # Small head dimensions keep the resource-contract test CPU-sized. Page
    # counts and FP16 storage follow the production 262K-context boundary.
    key_cache = torch.empty((1, input_page_size, 1, 2), dtype=torch.uint8)
    max_kv_bytes = max_blocks * output_page_size * 1 * 2 * 2
    try:
        with set_current_vllm_config(engine):
            for seq_len in (180000, max_model_len, 8000):
                active_pages = math.ceil(seq_len / input_page_size)
                required_blocks = math.ceil(
                    active_pages * input_page_size / output_page_size
                )
                workspace = dense_prefill.get_fp8_prefill_bridge_workspace(
                    key_cache, required_blocks, max_blocks=max_blocks
                )
                assert workspace is not None
                key, value, table = workspace
                assert key.shape == (required_blocks, output_page_size, 1, 2)
                assert value.shape == key.shape
                # The backing allocations, not just returned views, must fit
                # the addressable context even after a shorter prior request.
                for tensor in (key, value):
                    assert tensor.untyped_storage().nbytes() <= max_kv_bytes
                    tensor[-1].fill_(7)
                    assert torch.all(tensor[-1] == 7)
                assert table.shape == (1, required_blocks)
                assert table[0, -1].item() == required_blocks - 1
                assert (
                    table.untyped_storage().nbytes()
                    <= max_blocks * table.element_size()
                )
    finally:
        release_runtime_resources(engine)


def test_attention_diagnostic_budgets_and_output_do_not_cross_engines(tmp_path):
    import json

    from vllm.config import set_current_vllm_config
    from vllm.diagnostics import diagnostic_engine_tag, write_json_payload
    from vllm.v1.attention.backends.flash_v100 import config

    paths = []
    for _ in range(2):
        engine = SimpleNamespace(
            observability_config=SimpleNamespace(runtime_trace=RuntimeTraceConfig())
        )
        with set_current_vllm_config(engine):
            assert not config.diagnostic_seen("flash_v100.prefix_dump")
            config.mark_diagnostic("flash_v100.prefix_dump", True)
            assert config.diagnostic_seen("flash_v100.prefix_dump")
            paths.append(
                write_json_payload(
                    str(tmp_path),
                    "compare.json",
                    {"equal": True},
                    diagnostic_engine_tag(),
                )
            )
    assert paths[0] != paths[1]
    for path in paths:
        with open(path) as stream:
            assert json.load(stream) == {"equal": True}


def bridge_engine(
    options, *, context=4097, chunk=65, block_size=1056,
    gpu_memory_budget_bytes=16 * 1024**3,
):
    from vllm.config import set_current_vllm_config
    from vllm.runtime_resources import runtime_resources_for
    from vllm.v1.attention.backends.flash_v100.impl import FlashAttnV100Impl

    policy = prepare(options)[0]
    engine = SimpleNamespace(
        attention_config=SimpleNamespace(flash_v100=policy),
        model_config=SimpleNamespace(max_model_len=context),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=chunk),
        cache_config=SimpleNamespace(block_size=block_size),
    )
    if gpu_memory_budget_bytes is not None:
        runtime_resources_for(engine)["gpu_memory_budget_bytes"] = (
            gpu_memory_budget_bytes
        )
    impl = object.__new__(FlashAttnV100Impl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 24, 6, 256
    impl.kv_cache_dtype = "fp8_e4m3"
    for name in (
        "flash_attn_prefill_paged", "fp8_e4m3_paged_kv_to_fp16",
        "fp8_e5m2_paged_kv_to_fp16", "flash_attn_prefill_paged_splitkv",
        "flash_attn_prefill_paged_bfla", "flash_attn_func",
    ):
        setattr(impl, name, object())
    with set_current_vllm_config(engine):
        impl._initialize_prefill_policy()
    return engine, impl


@pytest.mark.parametrize(
    "budget,groups",
    [
        (32_047_128, 1),  # Exact unsplit fit; tail padding is included.
        (32_047_127, 2),
        (29_884_484, 2),  # Old 256-page projection incorrectly admitted unsplit.
        (16_822_296, 2),
        (16_822_295, 3),  # One byte below the two-group fit.
        (11_214_872, 3),
        (11_214_871, 6),
        (5_607_448, 6),
        (5_607_447, 0),  # No valid divisor fits.
        (0, 0),
    ],
)
def test_bridge_budget_selects_smallest_divisor_and_native_route(
    monkeypatch, budget, groups
):
    import torch

    monkeypatch.delenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS", raising=False)
    _, impl = bridge_engine(FlashV100Options(fp8_bridge_workspace_bytes=budget))
    assert impl.fp8_bridge_head_groups == groups
    key = torch.empty((1, 1056, 6, 256), dtype=torch.uint8)
    assert impl._should_use_fp8_prefill_bridge(
        q_len=65, head_dim=256, key_cache=key, value_cache=key,
        causal=True, window_size=(-1, -1),
    ) is (groups != 0)
    if groups:
        plan = impl.fp8_bridge_workspace_plan
        assert plan.max_kv_tokens == 4704  # Six native 784-token pages.
        assert plan.max_query_tokens == 65
        assert plan.projected_bytes <= budget


@pytest.mark.parametrize("typed", [1, 2, 3, 0, -2])
def test_bridge_explicit_groups_override_budget_and_legacy(monkeypatch, typed):
    monkeypatch.setenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS", "6")
    engine, impl = bridge_engine(
        FlashV100Options(
            fp8_bridge_head_groups=typed, fp8_bridge_workspace_bytes=0
        )
    )
    assert impl.fp8_bridge_head_groups == max(1, typed)
    assert impl.fp8_bridge_workspace_plan is None
    assert engine.attention_config.flash_v100.options.sources[
        "fp8_bridge_head_groups"
    ] == "typed"


def test_bridge_legacy_force_and_budget_errors_remain_engine_owned(monkeypatch):
    monkeypatch.setenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS", "1")
    first, unsplit = bridge_engine(FlashV100Options(fp8_bridge_workspace_bytes=0))
    monkeypatch.delenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS")
    second, native = bridge_engine(FlashV100Options(fp8_bridge_workspace_bytes=0))
    assert unsplit.fp8_bridge_head_groups == 1
    assert native.fp8_bridge_head_groups == 0
    assert first.attention_config.flash_v100.compute_hash() != (
        second.attention_config.flash_v100.compute_hash()
    )
    monkeypatch.setenv("VLLM_FLASH_V100_FP8_BRIDGE_WORKSPACE_BYTES", "-1")
    with pytest.raises(ValueError, match="nonnegative"):
        bridge_engine(FlashV100Options())
    monkeypatch.setenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS", "bad")
    with pytest.raises(ValueError):
        bridge_engine(FlashV100Options())
    # An explicit group count does not evaluate an irrelevant malformed budget.
    monkeypatch.setenv("VLLM_FLASH_V100_FP8_BRIDGE_WORKSPACE_BYTES", "bad")
    _, forced = bridge_engine(FlashV100Options(fp8_bridge_head_groups=1))
    assert forced.fp8_bridge_head_groups == 1


def test_bridge_auto_workspace_storage_fits_after_growth_and_engine_shutdown(
    monkeypatch,
):
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.runtime_resources import release_runtime_resources
    from vllm.v1.attention.backends.flash_v100 import dense_prefill

    monkeypatch.delenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS", raising=False)
    engine, impl = bridge_engine(
        FlashV100Options(fp8_bridge_workspace_bytes=11_214_872)
    )
    other, _ = bridge_engine(FlashV100Options(fp8_bridge_workspace_bytes=0))
    plan = impl.fp8_bridge_workspace_plan
    kv_heads, q_heads = 6 // plan.head_groups, 24 // plan.head_groups
    key = torch.empty((0, 1056, kv_heads, 256), dtype=torch.uint8)
    query = torch.empty((1, 65, q_heads, 256), dtype=torch.float16)
    try:
        with set_current_vllm_config(engine):
            for blocks, tokens in ((4, 48), (6, 65), (2, 32)):
                kv = dense_prefill.get_fp8_prefill_bridge_workspace(
                    key, blocks, max_blocks=6
                )
                scratch = dense_prefill.get_fp8_prefill_bridge_tail_workspace(
                    query, tokens, max_query_len=65, group_scratch=True
                )
                tail = dense_prefill.get_fp8_prefill_bridge_tail_workspace(
                    query, (tokens + 63) // 64 * 64, max_query_len=128
                )
                tensors = (*kv, *scratch, *tail)
                assert sum(
                    tensor.untyped_storage().nbytes() for tensor in tensors
                ) <= plan.projected_bytes <= plan.budget_bytes
                scratch[0].fill_(3)
                assert torch.all(scratch[0] == 3)
            previous_ptr = kv[0].data_ptr()
        with set_current_vllm_config(other):
            other_kv = dense_prefill.get_fp8_prefill_bridge_workspace(key, 2)
            assert other_kv[0].data_ptr() != previous_ptr
        release_runtime_resources(other)
        with set_current_vllm_config(engine):
            assert dense_prefill.get_fp8_prefill_bridge_workspace(
                key, 2, max_blocks=6
            )[0].data_ptr() == previous_ptr
    finally:
        release_runtime_resources(engine)
        release_runtime_resources(other)


@pytest.mark.parametrize("dimension", ["context", "chunk", "block_size"])
def test_bridge_rejects_invalid_configured_dimensions(monkeypatch, dimension):
    monkeypatch.delenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS", raising=False)
    with pytest.raises(ValueError, match="planning dimensions"):
        bridge_engine(
            FlashV100Options(fp8_bridge_workspace_bytes=32 * 1024**2),
            **{dimension: 0},
        )


@pytest.mark.parametrize("query_tokens,table_blocks", [(66, 4), (65, 5)])
def test_bridge_out_of_envelope_uses_native_without_conversion(
    monkeypatch, query_tokens, table_blocks
):
    import torch

    from vllm.v1.attention.backends.flash_v100 import prefill

    monkeypatch.delenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS", raising=False)
    _, impl = bridge_engine(
        FlashV100Options(fp8_bridge_workspace_bytes=11_214_872)
    )
    query = torch.zeros((1, query_tokens, 24, 256), dtype=torch.float16)
    key = torch.empty((0, 1056, 6, 256), dtype=torch.uint8)
    output = torch.full_like(query, float("nan"))
    # The fixture's native callbacks are deliberately noncallable: an invalid
    # request envelope must decline before allocating or launching conversion.
    assert prefill._run_fp8_prefill_bridge(
        impl, query=query, key_cache=key, value_cache=key,
        block_table=torch.zeros((1, table_blocks), dtype=torch.int32),
        seq_lens=torch.tensor([4097], dtype=torch.int32),
        seq_len=4097, k_scale=0.75, v_scale=1.25, causal=True,
        window_size=(-1, -1), out=output,
    ) is None
    assert torch.isnan(output).all()


def test_bridge_default_uses_engine_worker_snapshot_without_runtime_probe(monkeypatch):
    from vllm.config import set_current_vllm_config

    monkeypatch.delenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_FP8_BRIDGE_WORKSPACE_BYTES", raising=False)
    first, grouped = bridge_engine(
        FlashV100Options(), gpu_memory_budget_bytes=32 * 11_214_872
    )
    second, unsplit = bridge_engine(
        FlashV100Options(), gpu_memory_budget_bytes=32 * 32_047_128
    )
    assert grouped.fp8_bridge_head_groups == 3
    assert unsplit.fp8_bridge_head_groups == 1
    assert grouped.fp8_bridge_workspace_plan.budget_bytes == 11_214_872
    assert unsplit.fp8_bridge_workspace_plan.budget_bytes == 32_047_128
    assert first.attention_config.flash_v100.options.sources[
        "fp8_bridge_workspace_bytes"
    ] == "default"
    # Subsequent layers use the worker-owned snapshot, never live free memory.
    monkeypatch.setattr(
        "torch.cuda.mem_get_info",
        lambda *a, **kw: pytest.fail("attention probed live free memory"),
    )
    with set_current_vllm_config(first):
        grouped._initialize_prefill_policy()
    assert grouped.fp8_bridge_head_groups == 3
    assert unsplit.fp8_bridge_head_groups == 1
    _, forced = bridge_engine(
        FlashV100Options(fp8_bridge_workspace_bytes=0),
        gpu_memory_budget_bytes=32 * 32_047_128,
    )
    assert forced.fp8_bridge_head_groups == 0
    # Configuration-only callers cannot accidentally probe a GPU at import or
    # config construction; worker initialization or an explicit cap is required.
    with pytest.raises(RuntimeError, match="worker memory snapshot"):
        bridge_engine(FlashV100Options(), gpu_memory_budget_bytes=None)


@pytest.mark.parametrize("fixed_kv_bytes", [None, 64 * 1024**2])
def test_full_engine_default_cache_and_hybrid_page_reservation(
    monkeypatch, tmp_path, fixed_kv_bytes
):
    import torch
    from transformers import Qwen3Config

    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.runtime_resources import release_runtime_resources, runtime_resources_for
    from vllm.v1.attention.backends.flash_v100 import config, dense_prefill
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    monkeypatch.delenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS", raising=False)
    monkeypatch.delenv("VLLM_FLASH_V100_FP8_BRIDGE_WORKSPACE_BYTES", raising=False)
    hf = Qwen3Config(
        architectures=["Qwen3ForCausalLM"], num_hidden_layers=2,
        hidden_size=6144, intermediate_size=8192, num_attention_heads=24,
        num_key_value_heads=6, head_dim=256, max_position_embeddings=8192,
        vocab_size=64,
    )
    hf.save_pretrained(tmp_path)
    model = ModelConfig(
        model=str(tmp_path), tokenizer=str(tmp_path), dtype="half",
        skip_tokenizer_init=True, enforce_eager=True, max_model_len=4097,
    )
    engine = VllmConfig(
        model_config=model, device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(kv_cache_memory_bytes=fixed_kv_bytes),
        scheduler_config=SchedulerConfig(
            max_model_len=4097, max_num_batched_tokens=65, max_num_seqs=1,
            is_encoder_decoder=False, enable_chunked_prefill=True,
        ),
    )
    assert engine.cache_config.block_size == CacheConfig.DEFAULT_BLOCK_SIZE
    assert not engine.cache_config.user_specified_block_size
    engine.attention_config.flash_v100 = prepare()[0]
    runtime_resources_for(engine)["gpu_memory_budget_bytes"] = 32 * 32_047_128
    _, impl = bridge_engine(FlashV100Options(fp8_bridge_head_groups=1))
    with set_current_vllm_config(engine):
        impl._initialize_prefill_policy()
    assert impl.fp8_bridge_head_groups == 1
    assert impl.fp8_bridge_workspace_plan.max_kv_tokens == 4704

    # Real KV-group data after hybrid page-size equalization. The ordinary
    # CacheConfig default remains 16 and must not override this 1616-token page.
    spec = FullAttentionSpec(
        block_size=1616, num_kv_heads=6, head_size=256, dtype=torch.uint8
    )
    groups = [KVCacheGroupSpec(["attention"], spec)]
    config.configure_fp8_bridge_workspace_plans(
        engine, groups, {"attention": SimpleNamespace(impl=impl)}
    )
    plan = impl.fp8_bridge_workspace_plan
    assert plan.max_kv_tokens == 5488
    assert impl.fp8_bridge_head_groups == 2
    assert engine.cache_config.block_size == CacheConfig.DEFAULT_BLOCK_SIZE
    # The same finalization runs before normal profile_run and before backend
    # initialization when fixed KV bytes skip profiling. Reserve actual buffers,
    # rather than merely asserting that a hook forwarded its arguments.
    key = torch.empty((0, 1616, 3, 256), dtype=torch.uint8)
    query = torch.empty((1, 65, 12, 256), dtype=torch.float16)
    try:
        with set_current_vllm_config(engine):
            kv = dense_prefill.get_fp8_prefill_bridge_workspace(
                key, 7, max_blocks=7
            )
            tail = dense_prefill.get_fp8_prefill_bridge_tail_workspace(
                query, 128, max_query_len=128
            )
            scratch = dense_prefill.get_fp8_prefill_bridge_tail_workspace(
                query, 65, max_query_len=65, group_scratch=True
            )
            assert sum(
                tensor.untyped_storage().nbytes()
                for tensor in (*kv, *tail, *scratch)
            ) == plan.projected_bytes <= plan.budget_bytes
    finally:
        release_runtime_resources(engine)


def test_deferred_input_page_waits_for_resolved_group(monkeypatch):
    import torch

    from vllm.v1.attention.backends.flash_v100.config import (
        configure_fp8_bridge_workspace_plans,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    monkeypatch.delenv("VLLM_FLASH_V100_FP8_BRIDGE_HEAD_GROUPS", raising=False)
    engine, impl = bridge_engine(
        FlashV100Options(fp8_bridge_workspace_bytes=32_047_128), block_size=None
    )
    assert impl.fp8_bridge_workspace_plan is None
    spec = FullAttentionSpec(
        block_size=1616, num_kv_heads=6, head_size=256, dtype=torch.uint8
    )
    configure_fp8_bridge_workspace_plans(
        engine, [KVCacheGroupSpec(["attention"], spec)],
        {"attention": SimpleNamespace(impl=impl)},
    )
    assert impl.fp8_bridge_head_groups == 2
    assert impl.fp8_bridge_workspace_plan.max_kv_tokens == 5488
