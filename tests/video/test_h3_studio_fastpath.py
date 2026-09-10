# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.video import fastpath


def test_old_or_unbuilt_kernels_do_not_advertise_fast_profile(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(fastpath, "import_module", missing)
    assert fastpath.studio_capabilities() == {
        "profile": "sm70-dense-v1",
        "available": False,
    }


def test_profile_checks_attention_abi_and_all_packaged_operators(monkeypatch):
    attention = SimpleNamespace(forward=SimpleNamespace(__doc__="forward(query_tile)"))
    modules = {
        "vllm._h3_w8a16_C": SimpleNamespace(scaled_add_=None, ColumnMajorGemmPlan=None),
        "vllm._h3_flashattn_C": attention,
        "vllm._sm70_exact_reduce_C": SimpleNamespace(
            allocate=None, open_handle=None, release=None, run=None
        ),
        "vllm._sm70_sparse_attention_C": SimpleNamespace(forward=None),
    }
    monkeypatch.setattr(fastpath, "import_module", modules.__getitem__)
    assert fastpath.studio_capabilities()["available"]
    attention.forward.__doc__ = "forward(q, k, v, scale)"
    assert not fastpath.studio_capabilities()["available"]


def test_vsa_is_independent_of_dense_and_never_advertises_quality(monkeypatch):
    modules = {
        "vllm._h3_w8a16_C": SimpleNamespace(scaled_add_=None, ColumnMajorGemmPlan=None),
        "vllm._h3_flashattn_C": SimpleNamespace(
            forward=SimpleNamespace(__doc__="forward(query_tile)")
        ),
        "vllm._sm70_exact_reduce_C": SimpleNamespace(
            allocate=None, open_handle=None, release=None, run=None
        ),
        "vllm._sm70_sparse_attention_C": SimpleNamespace(forward=None),
    }

    def load(name):
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(fastpath, "import_module", load)
    capability = fastpath.studio_capabilities()["fasth3"]
    assert capability["vsa_available"]
    assert capability["experimental"]
    assert capability["quality_status"] == "not_accepted"
    assert capability["tasks"] == ["t2va"]
    del modules["vllm._sm70_sparse_attention_C"]
    result = fastpath.studio_capabilities()
    assert result["available"]
    assert not result["fasth3"]["vsa_available"]
