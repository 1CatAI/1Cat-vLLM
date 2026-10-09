# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State ownership, initialized routing and graph-visible metadata lifetimes."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.gdn import GdnConfig
from vllm.config.gdn_state import GdnStateConfig, GdnStateTraceConfig
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.runtime_resources import runtime_resources_for
from vllm.v1.attention.backends.gdn_attn import _get_ddtree_gdn_fast_common_buffers
from vllm.v1.attention.ops.gdn_state import (
    GdnMetadataOverride,
    build_state_contract,
    get_registered_gdn_spec_metadata_tensors,
    register_gdn_spec_metadata_tensors,
    state_resources_for,
)


def _context(config):
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        runtime_resources=runtime_resources_for(config),
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_same_layer_names_never_cross_engine_or_reload(reverse):
    configs = [SimpleNamespace(), SimpleNamespace()]
    owners = [state_resources_for(c) for c in configs]
    tensors = [
        tuple(torch.full((2,), i, dtype=torch.int32) for _ in range(9)) for i in (1, 2)
    ]
    order = (1, 0) if reverse else (0, 1)
    for i in order:
        register_gdn_spec_metadata_tensors(["layer.0"], tensors[i], owner=owners[i])
    for i in order:
        with override_forward_context(_context(configs[i])):
            assert (
                get_registered_gdn_spec_metadata_tensors("layer.0", torch.device("cpu"))
                is tensors[i]
            )
            assert not get_registered_gdn_spec_metadata_tensors(
                "missing", torch.device("cpu")
            )[0].numel()
    replacement = tuple(t.clone().fill_(3) for t in tensors[0])
    owners[0].register(["layer.0"], replacement)
    with override_forward_context(_context(configs[0])):
        assert (
            get_registered_gdn_spec_metadata_tensors("layer.0", torch.device("cpu"))
            is replacement
        )
    assert owners[1].get("layer.0", torch.device("cpu")) is tensors[1]


def test_capture_buffers_reused_only_within_owner_and_capacity():
    first, second = (
        state_resources_for(SimpleNamespace()),
        state_resources_for(SimpleNamespace()),
    )
    device = torch.device("cpu")
    a = _get_ddtree_gdn_fast_common_buffers(device, 8, 5, owner=first)
    assert _get_ddtree_gdn_fast_common_buffers(device, 8, 5, owner=first) is a
    b = _get_ddtree_gdn_fast_common_buffers(device, 8, 5, owner=second)
    c = _get_ddtree_gdn_fast_common_buffers(device, 16, 5, owner=first)
    a.spec_query_start_loc.fill_(7)
    b.spec_query_start_loc.fill_(9)
    assert a.spec_query_start_loc[0] == 7
    assert b.spec_query_start_loc[0] == 9
    assert c.spec_query_start_loc.data_ptr() != a.spec_query_start_loc.data_ptr()
    assert not c.spec_sequence_masks.any()


@pytest.mark.parametrize("legacy_slot0", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_reordered_requests_keep_accepted_slots_and_empty_contract(legacy_slot0, empty):
    table = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32)
    order = torch.tensor([2, 0, 1]) if not empty else torch.empty(0, dtype=torch.long)
    table = table[order]
    accepted = torch.tensor([3, 1, 2], dtype=torch.int32)[order]
    mask = torch.tensor([False, True, False])[order]
    state = build_state_contract(
        block_table_tensor=table,
        seq_lens=torch.full((len(order),), 8),
        block_size=16,
        num_spec=2,
        spec_sequence_masks_cpu=mask,
        num_accepted_tokens=accepted,
        current_state_block_ids=None,
        is_mamba_cache_all=False,
        legacy_slot0=legacy_slot0,
        assert_contract=True,
    )
    assert torch.equal(state.spec_state_indices_tensor, table[mask])
    assert torch.equal(state.num_accepted_tokens, accepted[mask])
    offsets = torch.zeros_like(accepted[~mask]) if legacy_slot0 else accepted[~mask] - 1
    expected = table[~mask].gather(1, offsets.long()[:, None]).squeeze(1)
    assert torch.equal(state.non_spec_state_indices_tensor, expected)


def test_metadata_override_restores_nested_state_after_failure():
    original = torch.arange(4)
    metadata = SimpleNamespace(indices=original, accepted=1)
    replacement = original.flip(0)
    with GdnMetadataOverride(metadata, skip_empty=True) as outer:
        outer.set("indices", torch.empty(0))
        assert metadata.indices is original
        outer.set("indices", replacement)
        with pytest.raises(RuntimeError), GdnMetadataOverride(metadata) as inner:
            inner.set("indices", torch.empty(0))
            inner.set("accepted", 3)
            raise RuntimeError("injected computation failure")
        assert metadata.indices is replacement and metadata.accepted == 1
    assert metadata.indices is original and metadata.accepted == 1


def test_state_policy_is_captured_and_diagnostics_do_not_change_hash(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0", "1")
    first = GdnConfig(state=GdnStateConfig(legacy_non_spec_slot0=False))
    first.resolve()
    second = GdnConfig()
    second.resolve()
    assert not first.state.legacy_non_spec_slot0 and second.state.legacy_non_spec_slot0
    monkeypatch.setenv("VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0", "0")
    second.resolve()
    assert second.state.legacy_non_spec_slot0
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    worker_copy = MsgpackDecoder(GdnConfig).decode(MsgpackEncoder().encode(second))
    assert worker_copy.resolved and worker_copy.state.resolved
    worker_copy.resolve()
    assert worker_copy.state.legacy_non_spec_slot0
    assert first.compute_hash() != second.compute_hash()
    before = first.compute_hash()
    monkeypatch.setenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_MAX_DUMPS", "bad")
    monkeypatch.setenv("VLLM_DFLASH_DDTREE_METADATA_PROFILE", "1")
    diagnostic = GdnStateTraceConfig(table_dir="")
    diagnostic.resolve()
    monkeypatch.setenv("VLLM_DFLASH_DDTREE_METADATA_PROFILE", "0")
    diagnostic.resolve()
    assert diagnostic.metadata_profile
    assert diagnostic.table_limit == 32 and before == first.compute_hash()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph state replay")
def test_registered_state_indices_update_on_graph_replay():
    config = SimpleNamespace()
    owner = state_resources_for(config)
    indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    metadata = (indices,) * 9
    owner.register(["layer.0"], metadata)
    state = torch.arange(8, device="cuda", dtype=torch.float32)
    output = torch.empty(2, device="cuda")
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), override_forward_context(_context(config)):
        for _ in range(3):
            selected = get_registered_gdn_spec_metadata_tensors(
                "layer.0", indices.device
            )[0]
            torch.index_select(state, 0, selected, out=output)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph), override_forward_context(_context(config)):
        selected = get_registered_gdn_spec_metadata_tensors("layer.0", indices.device)[
            0
        ]
        torch.index_select(state, 0, selected, out=output)
    indices.copy_(torch.tensor([6, 2], device="cuda", dtype=torch.int32))
    state.add_(10)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(output, torch.tensor([16, 12], device="cuda"))
    assert owner.metadata["layer.0"][0].data_ptr() == indices.data_ptr()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA grouped metadata replay"
)
def test_grouped_metadata_capture_reuses_owner_and_live_inputs(monkeypatch, tmp_path):
    from tests.v1.attention.test_gdn_metadata_builder import (
        _create_grouped_gdn_builders,
        local_gdn_model,
    )
    from vllm.v1.attention.backends.gdn_attn import prepare_dflash2_gdn_group_metadata
    from vllm.v1.attention.ops.gdn_state import compute_common_gdn_attn_metadata

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("retained grouped provider requires SM70")
    monkeypatch.setenv("VLLM_SM70_DFLASH2_FUSED_GDN_METADATA", "1")
    monkeypatch.setenv("VLLM_SM70_DFLASH2_GDN_METADATA_SHADOW", "0")
    # The fixture creates only a local config.json; no model weights are loaded.
    model_dir = local_gdn_model.__wrapped__(tmp_path)
    device = torch.device("cuda")
    builders = _create_grouped_gdn_builders(
        model_dir,
        2,
        num_speculative_tokens=7,
        use_full_cuda_graph=True,
        max_cudagraph_capture_size=16,
        device=device,
    )
    tables = tuple(
        torch.arange(16, device=device, dtype=torch.int32).view(2, 8) + i * 100
        for i in range(2)
    )
    accepted = torch.tensor([2, 5], device=device, dtype=torch.int32)
    query = torch.tensor([0, 8, 16], device=device, dtype=torch.int32)
    common = compute_common_gdn_attn_metadata(
        num_decode_draft_tokens_cpu=torch.tensor([7, 7], dtype=torch.int32),
        query_start_loc=query,
        query_start_loc_cpu=query.cpu(),
        num_spec_state_tokens=7,
        legacy_mixed_decode_routing=False,
    )
    assert common is not None
    kwargs = dict(
        builders_by_group=list(enumerate(builders)),
        block_tables=tables,
        common_gdn_metadata=common,
        num_accepted_tokens=accepted,
        num_actual_tokens=16,
    )
    result = prepare_dflash2_gdn_group_metadata(**kwargs, descriptor=None)
    assert result is not None
    metadata, descriptor = result
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_result = prepare_dflash2_gdn_group_metadata(
            **kwargs, descriptor=descriptor
        )
    assert replay_result is not None and replay_result[1] is descriptor
    for table in tables:
        table.add_(1000)
    accepted.copy_(torch.tensor([7, 3], device=device, dtype=torch.int32))
    graph.replay()
    torch.accelerator.synchronize()
    for i, builder in enumerate(builders):
        row = metadata[id(builder)]
        assert torch.equal(row.spec_state_indices_tensor[:2], tables[i])
        assert (row.spec_state_indices_tensor[2:] == -1).all()
        assert torch.equal(row.num_accepted_tokens[:2], accepted)
        assert (row.num_accepted_tokens[2:] == 1).all()


@pytest.mark.parametrize("family", ["gdn", "mamba", "ple", "plain", "flash"])
def test_backend_declares_only_its_model_state_inputs(family):
    from vllm.v1.attention.backends.flash_v100.metadata import (
        FlashAttnV100MetadataBuilder,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
    from vllm.v1.attention.backends.short_conv_attn import (
        PleShortConvAttentionMetadataBuilder,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata

    cls = {
        "gdn": GDNAttentionMetadataBuilder,
        "mamba": Mamba2AttentionMetadataBuilder,
        "ple": PleShortConvAttentionMetadataBuilder,
        "plain": TritonAttentionMetadataBuilder,
        "flash": FlashAttnV100MetadataBuilder,
    }[family]
    builder = object.__new__(cls)
    prepared, common = object(), object()
    metadata = MambaHybridAttnMetadata(
        is_prefilling=torch.tensor([False, True]),
        num_accepted_tokens=torch.tensor([3, 1]),
        num_decode_draft_tokens_cpu=torch.tensor([2, -1]),
        common_gdn_metadata=common,
        prepared_dflash2_gdn_metadata={id(builder): prepared},
        prepared_dflash2_smallq_metadata={id(builder): prepared},
    )
    kwargs = metadata.get_extra_attn_kwargs(builder, 1)
    if family == "plain":
        assert kwargs == {}
    elif family == "flash":
        assert kwargs == {"prepared_dflash2_smallq_metadata": prepared}
    else:
        assert kwargs["num_accepted_tokens"].tolist() == [3]
        assert kwargs["num_decode_draft_tokens_cpu"].tolist() == [2]
        if family == "gdn":
            assert kwargs["common_gdn_metadata"] is common
            assert kwargs["prepared_dflash2_metadata"] is prepared
        else:
            assert len(kwargs) == 2


def test_state_dump_budgets_and_files_are_per_engine(tmp_path):
    from vllm.v1.attention.backends.gdn_attn import _dump_sm70_gdn_state_table

    policy = GdnStateTraceConfig(table_dir=str(tmp_path), table_limit=1)
    policy.resolve()
    owners = [state_resources_for(SimpleNamespace()) for _ in range(2)]
    paths = [
        _dump_sm70_gdn_state_table(
            {"owner": i}, torch.tensor([16]), 0, 1, policy=policy, owner=owner
        )
        for i, owner in enumerate(owners)
    ]
    assert paths[0] != paths[1]
    assert [torch.load(path, weights_only=True)["owner"] for path in paths] == [0, 1]
    for owner in owners:
        assert (
            _dump_sm70_gdn_state_table(
                {}, torch.tensor([16]), 0, 1, policy=policy, owner=owner
            )
            is None
        )
