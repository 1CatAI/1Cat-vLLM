# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check page ABI fallback without importing a CUDA extension on the CPU."""

import ast
import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class Descriptor:
    def __init__(self, shape, dtype=torch.float16, contiguous=True):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = torch.device("cuda:0")
        self.is_cuda = True
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return self

    def stride(self, dim):
        assert dim == -1
        return 1

    def permute(self, *dims):
        return Descriptor(tuple(self.shape[i] for i in dims), self.dtype)


@pytest.mark.parametrize(
    "page,batch,split_available,contiguous,expected",
    [
        (832, 1, True, True, "split"),
        (832, 1, False, True, "fallback"),
        (832, 1, True, False, "fallback"),
        (832, 4, True, True, "fallback"),
        (1024, 1, True, True, "split"),
        (2048, 1, False, True, "native"),
        (1024, 4, True, True, "native"),
        (1648, 1, True, True, "fallback"),
    ],
)
def test_page_abi_dispatch(
    monkeypatch, page, batch, split_available, contiguous, expected
):
    source = (
        Path(__file__).parents[3]
        / "flash-attention-v100/flash_attn_v100/flash_attn_interface.py"
    )
    parsed = ast.parse(source.read_text())
    functions: list[ast.stmt] = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("maybe_contiguous", "flash_attn_prefill_paged")
    ]
    calls = []

    def route(name):
        def run(*args, **kwargs):
            calls.append(name)
            return args[0]

        return run

    extension = SimpleNamespace(
        dflash2_paged_bmhd_fwd=route("native"),
        prefill_paged_fwd=route("fallback"),
    )
    module = ModuleType("flash_attn_v100.sm70_dflash2_split")
    module.__dict__["forward"] = route("split")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "sm70_dflash2_split" and not split_available:
            raise ImportError("Optional split implementation unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
    namespace = {
        "torch": torch,
        "flash_attn_v100_cuda": extension,
        "_copy_bhmd_to_bmhd_out": lambda result, out: out,
        "__package__": "flash_attn_v100",
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    query = Descriptor((batch, 8, 8, 128), contiguous=contiguous)
    cache = Descriptor((40, page, 2, 128))
    namespace["flash_attn_prefill_paged"](
        query,
        cache,
        cache,
        Descriptor((batch, 40), torch.int32),
        Descriptor((batch,), torch.int32),
        out=Descriptor(query.shape),
        causal=False,
        window_size=(2047, 2047),
    )
    assert calls == [expected]
