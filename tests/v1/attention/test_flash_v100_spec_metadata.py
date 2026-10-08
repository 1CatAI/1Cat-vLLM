# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative metadata preserves calculation bodies and graph buffer identity."""

import ast
import copy
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.attention.backends import flash_attn_v100 as legacy
from vllm.v1.attention.backends.flash_v100 import metadata
from vllm.v1.attention.backends.flash_v100.spec import smallq_metadata
from vllm.v1.attention.backends.flash_v100.spec.hooks import METADATA_HOOKS
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

pytestmark = pytest.mark.cpu_test


_BUFFER_FIELDS = {
    "draft.block_table": "_draft_block_table",
    "draft.seq_lens": "_draft_seq_lens",
    "draft.query_start_loc": "_draft_query_start_loc",
    "draft.shape": "_flash_draft_buffer_shape",
    "smallq.block_table": "_smallq_decode_block_table",
    "smallq.seq_lens": "_smallq_decode_seq_lens",
    "smallq.query_start_loc": "_smallq_query_start_loc",
    "smallq.token_indices": "_smallq_token_indices",
    "smallq.shape": "_smallq_buffer_shape",
}


class _InlineWorkspace(ast.NodeTransformer):
    def __init__(self):
        source = Path(metadata.__file__).with_name("workspace.py")
        self.methods = {
            (cls.name, fn.name): fn
            for cls in ast.parse(source.read_text()).body
            if isinstance(cls, ast.ClassDef)
            for fn in cls.body
            if isinstance(fn, ast.FunctionDef)
        }

    def inline(self, call):
        if not isinstance(call, ast.Call):
            return None
        target = ast.unparse(call.func)
        targets = {
            "self.metadata_workspace.draft.ensure": ("DraftBuffers", "ensure"),
            "self.metadata_workspace.smallq.ensure": ("SmallQueryBuffers", "ensure"),
            "self.metadata_workspace.draft.copy_metadata": (
                "DraftBuffers",
                "copy_metadata",
            ),
        }
        if target not in targets:
            return None
        method = self.methods[targets[target]]
        assert not call.keywords
        assert isinstance(call.func, ast.Attribute)
        assert len(call.args) == len(method.args.args) - 1
        substitutions = dict(
            zip(
                [a.arg for a in method.args.args],
                [call.func.value, *call.args],
            )
        )

        class Substitute(ast.NodeTransformer):
            def visit_Name(self, node):
                return copy.deepcopy(substitutions.get(node.id, node))

        body = copy.deepcopy(method.body)
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        return [Substitute().visit(node) for node in body]

    def visit_Return(self, node):
        return self.inline(node.value) or self.generic_visit(node)

    def visit_Expr(self, node):
        return self.inline(node.value) or self.generic_visit(node)


class _Normalize(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "attach_metadata":
            node.name = "_attach_ddtree_metadata"
        node = self.generic_visit(node)
        if node.args.args and node.args.args[0].arg == "self":
            node.args.args[0].annotation = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body[0].value.value = inspect.cleandoc(node.body[0].value.value)
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        path = ast.unparse(node)
        callbacks = {
            "self.ops.attach_common": "self._attach_common_flash_metadata",
            "self.ops.attach_prefix": "self._attach_prefix_anchored_metadata",
            "self.ops.attach_shape_hints": "self._attach_decode_shape_hints",
            "self.ops.update_active_partitions": (
                "self._update_decode_active_num_partitions"
            ),
            "self.inputs.builder_id": "id(self)",
        }
        if path in callbacks:
            return ast.parse(callbacks[path], mode="eval").body
        prefix = "self.metadata_workspace."
        if path.startswith(prefix) and path[len(prefix) :] in _BUFFER_FIELDS:
            return ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr=_BUFFER_FIELDS[path[len(prefix) :]],
                ctx=node.ctx,
            )
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "_metadata"
            and node.attr == "_as_flash_v100_metadata"
        ):
            return ast.Name(id=node.attr, ctx=node.ctx)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.func) == "metadata_view":
            node.func = ast.Name(id="_as_flash_v100_metadata", ctx=ast.Load())
        if ast.unparse(node.func) == "self.ops.base_build":
            node.func = ast.parse("super().build", mode="eval").body
        if ast.unparse(node.func) == "_config.raw":
            node.func = ast.Attribute(
                value=ast.Name(id="os", ctx=ast.Load()), attr="getenv", ctx=ast.Load()
            )
        if isinstance(node.func, ast.Name) and node.func.id == "super" and node.args:
            assert [ast.unparse(a) for a in node.args] == [
                "_metadata._spec_builder_super_owner",
                "self",
            ]
            node.args = []
        return node


def test_moved_metadata_calculations_match_parent():
    expected = json.loads(
        (
            Path(__file__).parent / "fixtures/flash_v100_metadata_methods.json"
        ).read_text()
    )["methods"]
    actual = {}
    for path in Path(metadata.__file__).parent.joinpath("spec").glob("*.py"):
        for fn in ast.parse(path.read_text()).body:
            if isinstance(fn, ast.FunctionDef) and (
                fn.name in expected
                or (path.name == "tree.py" and fn.name == "attach_metadata")
            ):
                name = (
                    "_attach_ddtree_metadata"
                    if fn.name == "attach_metadata"
                    else fn.name
                )
                assert name not in actual
                actual[name] = hashlib.sha256(
                    ast.dump(_Normalize().visit(_InlineWorkspace().visit(fn))).encode()
                ).hexdigest()
    assert actual == expected


def _builder():
    instance = object.__new__(metadata.FlashAttnV100MetadataBuilder)
    instance.device = torch.device("cpu")
    instance.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        model_config=SimpleNamespace(max_model_len=2048),
    )
    instance._is_speculative_draft_model = False
    METADATA_HOOKS.initialize(instance, None)
    return instance


def test_old_smallq_module_is_the_owner_and_patch_reaches_grouped_function(monkeypatch):
    from vllm.v1.attention.backends.flash_v100 import smallq_metadata as old

    assert old is smallq_metadata
    original = smallq_metadata.DFlash2SmallQGroupDescriptor
    calls = []

    def descriptor(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(legacy, "DFlash2SmallQGroupDescriptor", descriptor)
    monkeypatch.setattr(
        smallq_metadata._sm70_prepare_grouped_smallq_decode_metadata_kernel,
        "run",
        lambda *a, **k: None,
    )
    table = torch.zeros((1, 2), dtype=torch.int32)
    seq = torch.tensor([5], dtype=torch.int32)
    qsl = torch.tensor([0, 2], dtype=torch.int32)
    result = old._sm70_prepare_grouped_smallq_decode_metadata(
        [table.repeat(2, 1)],
        [seq.repeat(2)],
        [qsl.clone()],
        [table],
        seq,
        qsl,
        num_reqs=1,
        num_query_tokens=2,
        real_num_query_tokens=2,
    )
    assert len(calls) == 1 and isinstance(result, original)


def test_tree_capture_restores_authoritative_metadata(monkeypatch):
    instance = _builder()
    attn = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([17, 23], dtype=torch.int32),
    )
    monkeypatch.setattr(legacy, "_is_cuda_graph_capturing", lambda _: True)
    parents = torch.tensor([[-1, 0], [-1, 0]], dtype=torch.int32)
    instance._attach_ddtree_metadata(
        attn,
        ddtree_parent_ids=parents,
        ddtree_num_tree_tokens_cpu=torch.tensor([2, 2], dtype=torch.int32),
    )
    assert attn.ddtree_parent_ids is parents
    assert attn.ddtree_seq_lens_restored_for_triton
    assert attn.ddtree_query_start_loc_restored_for_triton
    assert torch.equal(attn.seq_lens, attn.seq_lens_cpu)
    assert torch.equal(attn.query_start_loc, attn.query_start_loc_cpu)
    with pytest.raises(ValueError, match="required"):
        instance._attach_ddtree_metadata(
            attn, ddtree_parent_ids=parents, ddtree_num_tree_tokens_cpu=None
        )


def test_persistent_draft_buffers_keep_addresses_across_replays():
    instance = _builder()
    common = SimpleNamespace(num_reqs=1)
    attn = SimpleNamespace(
        block_table=torch.tensor([[3, 4]], dtype=torch.int32),
        seq_lens=torch.tensor([9], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    instance._stabilize_draft_graph_metadata(attn, common)
    addresses = [
        getattr(attn, n).data_ptr()
        for n in ("block_table", "seq_lens", "query_start_loc")
    ]
    next_attn = SimpleNamespace(
        block_table=torch.tensor([[5, 6]], dtype=torch.int32),
        seq_lens=torch.tensor([19], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
    )
    instance._stabilize_draft_graph_metadata(next_attn, common)
    assert addresses == [
        getattr(next_attn, n).data_ptr()
        for n in ("block_table", "seq_lens", "query_start_loc")
    ]
    assert next_attn.block_table.tolist() == [[5, 6]]
    assert next_attn.seq_lens.tolist() == [19]
    assert next_attn.query_start_loc.tolist() == [0, 4]
    with pytest.raises(RuntimeError, match="capacity"):
        instance._stabilize_draft_graph_metadata(
            SimpleNamespace(block_table=torch.zeros((1, 3), dtype=torch.int32)), common
        )


def test_feature_build_calls_base_once_and_keeps_hook_order(monkeypatch):
    instance = _builder()
    state = instance.spec_state
    attn = SimpleNamespace(query_start_loc=torch.tensor([0, 2]), max_query_len=2)
    events = []

    def base(self, prefix, common, fast):
        assert self is instance and prefix == 0 and not fast
        events.append("base")
        return attn

    monkeypatch.setattr(TritonAttentionMetadataBuilder, "build", base)
    for name in (
        "_attach_common_flash_metadata",
        "_attach_prefix_anchored_metadata",
        "_attach_ddtree_metadata",
        "_attach_decode_shape_hints",
        "_update_decode_active_num_partitions",
        "_debug_draft_metadata",
    ):
        owner = (
            state
            if name in ("_attach_ddtree_metadata", "_debug_draft_metadata")
            else instance
        )
        setattr(owner, name, lambda *a, _name=name, **kw: events.append(_name))
    state.ops = replace(
        state.ops,
        base_build=base.__get__(instance),
        attach_common=instance._attach_common_flash_metadata,
        attach_prefix=instance._attach_prefix_anchored_metadata,
        attach_shape_hints=instance._attach_decode_shape_hints,
        update_active_partitions=instance._update_decode_active_num_partitions,
    )
    state._update_smallq_decode_metadata = MagicMock()
    common = SimpleNamespace(max_seq_len=17)
    assert instance.build(0, common) is attn
    state._update_smallq_decode_metadata.assert_called_once_with(
        attn, common, workspace_seq_capacity_cap=17
    )
    assert events == [
        "base",
        "_attach_common_flash_metadata",
        "_attach_prefix_anchored_metadata",
        "_attach_ddtree_metadata",
        "_attach_decode_shape_hints",
        "_update_decode_active_num_partitions",
        "_debug_draft_metadata",
    ]


def test_capture_hook_preserves_small_query_lengths_and_bucket_hint(monkeypatch):
    instance = _builder()
    state = instance.spec_state
    attn = SimpleNamespace(max_query_len=3, seq_lens=torch.tensor([1]))
    common = SimpleNamespace(max_seq_len=256)
    state._update_smallq_decode_metadata = MagicMock()
    state._stabilize_draft_graph_metadata = MagicMock()
    monkeypatch.setattr(legacy, "_mtp_context_bucket_partition_size_hint", lambda: 64)
    METADATA_HOOKS.prepare_capture(instance, attn, common)
    assert attn.seq_lens.tolist() == [3]
    state._update_smallq_decode_metadata.assert_called_once_with(
        attn,
        common,
        force=True,
        workspace_seq_capacity_cap=256,
        partition_size_hint=64,
    )
    state._stabilize_draft_graph_metadata.assert_not_called()
    instance._is_dflash_draft_model = True
    METADATA_HOOKS.prepare_capture(instance, attn, common)
    state._stabilize_draft_graph_metadata.assert_called_once_with(attn, common)
