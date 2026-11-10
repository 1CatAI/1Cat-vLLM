# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare unified grouped admission against frozen pre-refactor predicates."""

import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.v1.attention.kv_codecs import FP8_E4M3, FP16, resolve_kv_codec
from vllm.v1.attention.ops import sm70_grouped as grouped

pytestmark = pytest.mark.cpu_test


@dataclass
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    contiguous: bool = True
    pointer: int = 16
    strides: tuple[int, ...] | None = None

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.pointer

    def stride(self, dim=None):
        if self.strides is None:
            strides = []
            step = 1
            for size in reversed(self.shape):
                strides.append(step)
                step *= size
            result = tuple(reversed(strides))
        else:
            result = self.strides
        return result if dim is None else result[dim]


@pytest.fixture
def legacy():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/sm70_grouped_legacy.json").read_text()
    )
    namespace = dict(
        os=os,
        torch=torch,
        envs=envs,
        FP16=FP16,
        FP8_E4M3=FP8_E4M3,
        resolve_kv_codec=resolve_kv_codec,
        MAX_GROUPS=4,
        MAX_CONTEXT=266240,
        GROUP_ROWS=8,
        MAX_GROUPS_PER_CALL=16,
    )
    for source in fixture["predicates"].values():
        exec(source, namespace)
    return namespace


def scenario(codec, groups=1, rows=8, page=None, device="cuda:0", explicit=False):
    if page is None:
        page = 832 if codec is FP16 else 848
    dev = torch.device(device)

    def tensor(shape, dtype):
        return TensorSpec(shape, dtype, dev)

    instance = SimpleNamespace(
        flash_attn_grouped_fp16_fp32_paged=object(),
        flash_attn_grouped_e4m3_fp32_paged=object(),
        kv_cache_dtype=codec.name,
        use_smallq_decode_xqa=True,
        _flash_v100_window_size=lambda **kw: (-1, -1),
    )
    q = tensor((rows, 6, 256), torch.float16)
    k, v = (
        tensor((1, page, 1, 256), codec.storage_dtype),
        tensor((1, page, 1, 256), codec.storage_dtype),
    )
    table = tensor((groups if explicit else rows, 300), torch.int32)
    lengths = tensor((rows,), torch.int32)
    metadata = SimpleNamespace(
        block_table=tensor((groups, 300), torch.int32),
        seq_lens=tensor((groups,), torch.int32),
        causal=True,
    )
    return [instance, q, k, v, table, lengths, metadata, tensor(q.shape, q.dtype)]


def compare(inputs, codec, legacy, *, explicit=False, partition=None, causal=True):
    instance, q, k, v, table, lengths, metadata, out = inputs
    reason = grouped.grouped_fp32_reason(
        codec,
        instance,
        q,
        k,
        v,
        table,
        lengths,
        metadata,
        out=out,
        layout="explicit_groups" if explicit else "request_rows",
        partition_size_hint=partition,
        causal=causal,
    )
    if explicit:
        expected = legacy["grouped_e4m3_fp32_groups_allowed"](
            instance, q, k, v, table, lengths, causal=causal, out=out
        )
        assert (reason is None) == expected
    else:
        name = (
            "grouped_fp16_fp32_reason" if codec is FP16 else "grouped_e4m3_fp32_allowed"
        )
        expected = legacy[name](
            instance,
            q,
            k,
            v,
            table,
            lengths,
            metadata,
            out=out,
            partition_size_hint=partition,
        )
        assert reason == expected if codec is FP16 else (reason is None) == expected
    return reason


@pytest.mark.parametrize(
    "codec,explicit", [(FP16, False), (FP8_E4M3, False), (FP8_E4M3, True)]
)
def test_codec_shape_device_revision_matrix_matches_legacy(
    monkeypatch, legacy, codec, explicit
):
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    envs.disable_envs_cache()
    for available in (False, True):
        monkeypatch.setitem(
            sys.modules,
            "flash_attn_v100",
            SimpleNamespace(
                flash_attn_grouped_e4m3_fp32_available=lambda *args, _ok=available: _ok
            ),
        )
        for groups, rows, page, device in itertools.product(
            (0, 1, 2, 4, 5, 16, 17),
            (1, 2, 8, 16, 32, 40, 128),
            (128, 817, 832, 848, 1024),
            ("cpu", "cuda:0"),
        ):
            inputs = scenario(codec, groups, rows, page, device, explicit)
            compare(inputs, codec, legacy, explicit=explicit)


@pytest.mark.parametrize(
    "codec,explicit", [(FP16, False), (FP8_E4M3, False), (FP8_E4M3, True)]
)
def test_policy_and_layout_rejections_match_legacy(
    monkeypatch, legacy, codec, explicit
):
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "flash_attn_v100",
        SimpleNamespace(flash_attn_grouped_e4m3_fp32_available=lambda *args: True),
    )
    envs.disable_envs_cache()
    changes = [
        "missing",
        "codec",
        "disabled",
        "window",
        "causal",
        "head",
        "kv_ndim",
        "kv_dtype",
        "vshape",
        "capacity",
        "parent_seq",
        "lengths",
        "table_shape",
        "q_dtype",
        "q_contiguous",
        "q_pointer",
        "out_shape",
        "out_dtype",
        "out_device",
        "out_contiguous",
        "kv_device",
        "kv_pointer",
        "kv_strides",
        "metadata_dtype",
        "metadata_device",
        "metadata_contiguous",
    ]
    for change in changes:
        x = scenario(codec, explicit=explicit)
        instance, q, k, v, table, lengths, metadata, out = x
        if change == "missing":
            setattr(instance, grouped.GROUPED_CONTRACTS[codec].operator_attr, None)
        elif change == "codec":
            instance.kv_cache_dtype = "fp8_e5m2"
        elif change == "disabled":
            instance.use_smallq_decode_xqa = False
        elif change == "window":
            instance._flash_v100_window_size = lambda **kw: (8, 0)
        elif change == "causal":
            metadata.causal = False
        elif change == "head":
            q.shape = out.shape = (8, 5, 256)
        elif change == "kv_ndim":
            k.shape = v.shape = (1, 832, 256)
        elif change == "kv_dtype":
            k.dtype = v.dtype = torch.float32
        elif change == "vshape":
            v.shape = (2, 832, 1, 256)
        elif change == "capacity":
            table.shape = metadata.block_table.shape = (1 if explicit else 8, 999)
        elif change == "parent_seq":
            metadata.seq_lens = None
        elif change == "lengths":
            lengths.shape = (9,)
        elif change == "table_shape":
            table.shape = (9, 300)
        elif change == "q_dtype":
            q.dtype = torch.float32
        elif change == "q_contiguous":
            q.contiguous = False
        elif change == "q_pointer":
            q.pointer = 2
        elif change == "out_shape":
            out.shape = (9, 6, 256)
        elif change == "out_dtype":
            out.dtype = torch.float32
        elif change == "out_device":
            out.device = torch.device("cpu")
        elif change == "out_contiguous":
            out.contiguous = False
        elif change == "kv_device":
            k.device = torch.device("cpu")
        elif change == "kv_pointer":
            k.pointer = 2
        elif change == "kv_strides":
            k.strides = (256, 257, 256, 1)
        elif change == "metadata_dtype":
            lengths.dtype = torch.int64
        elif change == "metadata_device":
            lengths.device = torch.device("cpu")
        elif change == "metadata_contiguous":
            lengths.contiguous = False
        compare(
            x,
            codec,
            legacy,
            explicit=explicit,
            causal=change != "causal",
        )
    for raw in (None, "", "0", "64"):
        if raw is None:
            monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
        else:
            monkeypatch.setenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raw)
        for partition in (None,) if explicit else (None, 64):
            compare(
                scenario(codec, explicit=explicit),
                codec,
                legacy,
                explicit=explicit,
                partition=partition if not explicit else None,
            )


def test_scalar_long_old_paths_alias_the_same_owners(monkeypatch):
    from vllm.v1.attention.ops import (
        sm70_e4m3_long,
        sm70_e4m3_scalar,
        sm70_grouped_long,
        sm70_grouped_scalar,
    )

    assert sm70_e4m3_long is sm70_grouped_long
    assert sm70_e4m3_scalar is sm70_grouped_scalar
    assert sm70_grouped_long.logger.name == "vllm.v1.attention.ops.sm70_e4m3_long"
    assert sm70_grouped_scalar.logger.name == "vllm.v1.attention.ops.sm70_e4m3_scalar"
    monkeypatch.setattr(sm70_e4m3_scalar, "BUILTIN_SCALAR_OP", "sentinel")
    assert sm70_grouped_scalar.BUILTIN_SCALAR_OP == "sentinel"


def test_native_loading_workspace_and_scalar_long_calculations_match_parent():
    import ast
    import hashlib

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/sm70_grouped_legacy.json").read_text()
    )

    class Normalize(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "vllm.v1.attention.ops.sm70_grouped_long":
                node.module = "vllm.v1.attention.ops.sm70_e4m3_long"
            return node

    for module, expected in fixture["native_methods"].items():
        path = Path(grouped.__file__).with_name(f"{module}.py")
        actual = {
            node.name: hashlib.sha256(
                ast.dump(Normalize().visit(node)).encode()
            ).hexdigest()
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }
        assert actual == expected


def test_shared_clear_reaches_the_only_fp16_workspace_owner(monkeypatch):
    from vllm.v1.attention.ops import sm70_fp16_grouped

    workspaces = {("test",): [(torch.ones(1), torch.ones(1))]}
    monkeypatch.setattr(sm70_fp16_grouped, "_WORKSPACES", workspaces)
    grouped.clear_grouped_fp16_workspaces()
    assert workspaces == {}


def test_loader_dispatches_declared_native_provider_once(monkeypatch):
    import importlib
    from unittest.mock import MagicMock

    from vllm.v1.attention.kv_codecs import BF16

    for codec, contract in grouped.GROUPED_CONTRACTS.items():
        provider = importlib.import_module(contract.loader_module)
        sentinel = object()
        loader = MagicMock(return_value=sentinel)
        monkeypatch.setattr(provider, contract.loader_name, loader)
        assert grouped.load_grouped_fp32(codec) is sentinel
        loader.assert_called_once_with()
    assert grouped.load_grouped_fp32(BF16) is None


def test_missing_batch_capability_preserves_import_error(monkeypatch, legacy):
    monkeypatch.setitem(sys.modules, "flash_attn_v100", SimpleNamespace())
    inputs = scenario(FP8_E4M3, groups=2, rows=16)
    with pytest.raises(ImportError):
        compare(inputs, FP8_E4M3, legacy)
    # Compare reaches the new admission first, so exercise the frozen original
    # separately to verify its exception category too.
    instance, q, k, v, table, lengths, metadata, out = inputs
    with pytest.raises(ImportError):
        legacy["grouped_e4m3_fp32_allowed"](
            instance,
            q,
            k,
            v,
            table,
            lengths,
            metadata,
            out=out,
            partition_size_hint=None,
        )


@pytest.mark.parametrize("codec,contract", list(grouped.GROUPED_CONTRACTS.items()))
def test_declared_group_limits_generate_admission_cases(monkeypatch, codec, contract):
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    envs.disable_envs_cache()
    if contract.batch_revision is not None:
        module, function, _ = contract.batch_revision
        monkeypatch.setitem(
            sys.modules, module, SimpleNamespace(**{function: lambda *a: True})
        )
    page = (
        contract.page_sizes[0] if contract.page_sizes else contract.page_alignment * 53
    )
    for explicit in (False, True):
        if explicit and not contract.explicit_groups:
            continue
        for groups in (1, contract.max_groups, contract.max_groups + 1):
            inputs = scenario(
                codec,
                groups=groups,
                rows=groups * grouped.GROUP_ROWS,
                page=page,
                explicit=explicit,
            )
            setattr(inputs[0], contract.operator_attr, object())
            instance, q, k, v, table, lengths, metadata, out = inputs
            reason = grouped.grouped_fp32_reason(
                codec,
                instance,
                q,
                k,
                v,
                table,
                lengths,
                metadata,
                out=out,
                layout="explicit_groups" if explicit else "request_rows",
            )
            assert (reason is None) == (groups <= contract.max_groups)
